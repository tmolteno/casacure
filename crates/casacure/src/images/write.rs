//! The image write path (issue #14 phase 2): creating CASA image tables
//! (`image(imagename=, shape=, coordsys=)`), writing the raster
//! (`putdata`), copying images (`saveas`) and exporting FITS (`tofits`).
//!
//! Created images store their raster in a one-row table with a
//! TiledShapeStMan array column and carry the `coords`/`imageinfo`/
//! `units`/`miscinfo` keyword records — the layout casacore's image tools
//! read (verified against real python-casacore in `tests/test_images.py`).

use std::io::Write as _;

use crate::images::coordsys::{Coordinate, CoordinateSystem};
use crate::images::image::{Image, ImageError};
use crate::record::{ArrayData, ArrayValue, RecordValue, TableRecord};
use crate::table::WritableTable;
use crate::tabledesc::TableDesc;

/// The keyword records of an image beyond its coordinates.
#[derive(Debug, Clone, Default)]
pub struct ImageMeta {
    pub imageinfo: TableRecord,
    pub units: String,
    pub miscinfo: TableRecord,
}

impl ImageMeta {
    /// The metadata of an opened image (for `saveas` copies).  `unit()`
    /// returns the quoted pyrap form; the keyword stores it unquoted.
    pub fn from_image(img: &Image) -> ImageMeta {
        let units = img.unit();
        ImageMeta {
            imageinfo: img.imageinfo(),
            units: units
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
                .unwrap_or(&units)
                .to_string(),
            miscinfo: img.miscinfo(),
        }
    }

    pub(crate) fn default_info() -> TableRecord {
        let mut info = TableRecord::default();
        info.set("imagetype", RecordValue::String("Intensity".into()));
        info.set("objectname", RecordValue::String(String::new()));
        info
    }
}

/// The `table.info` marker of a PagedImage (casacore writes exactly this).
fn write_table_info(path: &std::path::Path) -> Result<(), ImageError> {
    std::fs::write(path.join("table.info"), "Type = Image\nSubType = \n\n").map_err(|e| {
        ImageError::Other {
            path: path.to_path_buf(),
            msg: format!("cannot write table.info: {e}"),
        }
    })
}

/// Create a CASA image table with a zeroed raster of `shape` (numpy order)
/// and the given coordinates/metadata.  Any existing table at `path` is
/// replaced (the caller's `rm -Rf` in DDFacet becomes unnecessary but
/// harmless).
pub fn create_casa_image(
    path: &std::path::Path,
    shape: &[usize],
    coords: &CoordinateSystem,
    meta: &ImageMeta,
) -> Result<(), ImageError> {
    if path.exists() {
        std::fs::remove_dir_all(path).map_err(|e| ImageError::Other {
            path: path.to_path_buf(),
            msg: format!("cannot replace existing image: {e}"),
        })?;
    }
    // The desc JSON carries the LOGICAL (numpy) shape; the descriptor
    // stores it casa order and the tiled build writes that.
    let logical: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let json = format!(
        r#"{{"map": {{"valueType": "float", "dataManagerType": "TiledShapeStMan",
        "dataManagerGroup": "map", "option": 0, "ndim": {}, "shape": {:?}, "_c_order": true}}}}"#,
        shape.len(),
        logical
    );
    let desc = TableDesc::from_desc_json(&json).map_err(|e| ImageError::Other {
        path: path.to_path_buf(),
        msg: format!("image descriptor: {e}"),
    })?;
    let mut wt = WritableTable::create(path, desc);
    wt.addrows(1);
    let nelem: usize = shape.iter().product();
    // Core-level putcell stores the cell in CASA order (the pyo3 layer
    // does the numpy->casa reversal; here we are below it).
    let casa: Vec<u32> = shape.iter().rev().map(|&d| d as u32).collect();
    wt.putcell(
        0,
        0,
        RecordValue::Array(ArrayValue {
            shape: casa,
            data: ArrayData::Float(vec![0.0; nelem]),
        }),
    )?;
    let keywords: [(&str, RecordValue); 4] = [
        ("coords", RecordValue::Record(coords.raw_record())),
        ("imageinfo", {
            let mut info = meta.imageinfo.clone();
            if info.desc.fields.is_empty() {
                info = ImageMeta::default_info();
            }
            RecordValue::Record(info)
        }),
        ("units", RecordValue::String(meta.units.clone())),
        ("miscinfo", RecordValue::Record(meta.miscinfo.clone())),
    ];
    for (name, value) in keywords {
        wt.putkeyword(name, value);
    }
    // The logtable subtable (an empty TableLogSink table): casacore's
    // image() requires the keyword and the directory to exist.
    let logtable = create_logtable(&path.join("logtable"))?;
    wt.putkeyword(
        "logtable",
        RecordValue::Table(logtable.display().to_string()),
    );
    wt.flush()?;
    drop(wt);
    write_table_info(path)
}

/// Replace the raster of the image table at `path` (pyrap `putdata`).
/// Values are coerced to the raster's float storage the way casacore's
/// `putdata` coerces.
///
/// `data.shape` is in numpy order (pyrap's `putdata` contract) and must
/// match the image's own numpy-order shape.  The mismatch is checked
/// against the stored cell before the tiled encoder runs, so a caller
/// mistake surfaces as [`ImageError::ShapeMismatch`] instead of the
/// storage layer's "unsupported tiled element type" complaint.
pub fn put_data(path: &std::path::Path, data: &ArrayValue) -> Result<(), ImageError> {
    let (read, mut wt) = WritableTable::open_for_update(path)?;
    let raster = wt
        .desc()
        .columns
        .iter()
        .position(|c| c.kind == crate::tabledesc::ColumnKind::Array)
        .ok_or_else(|| ImageError::Other {
            path: path.to_path_buf(),
            msg: "no raster (array) column in the image table".into(),
        })?;
    // The cell comes back in the logical (numpy-order) shape — see
    // `convert.rs`, "cells are returned in the shape they are stored
    // (which, for casacore files, is the logical shape)".  That is exactly
    // the order `putdata` must supply, so no reversal is needed here.
    let want: Vec<u32> = match read.getcell(raster, 0) {
        Ok(RecordValue::Array(a)) => a.shape.clone(),
        _ => Vec::new(),
    };
    if !want.is_empty() && data.shape != want {
        return Err(ImageError::ShapeMismatch {
            got: data.shape.clone(),
            want,
        });
    }
    if !storable_as_float(&data.data) {
        return Err(ImageError::BadElementType {
            kind: array_kind_name(&data.data).to_string(),
        });
    }
    let mut coerced = coerce_float(data);
    coerced.shape.reverse(); // numpy -> casa order for the core write.
    wt.putcell(raster, 0, RecordValue::Array(coerced))?;
    wt.flush()?;
    Ok(())
}

/// An empty TableLogSink table (the image `logtable` subtable): the five
/// columns casacore's logger writes, zero rows.
fn create_logtable(path: &std::path::Path) -> Result<std::path::PathBuf, ImageError> {
    let json = r#"{"TIME": {"valueType": "double", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "SSM", "option": 0, "comment": "MJD in seconds",
        "keywords": {"UNIT": "s", "MEASURE_TYPE": "EPOCH", "MEASURE_REFERENCE": "UTC"}},
        "PRIORITY": {"valueType": "string", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "SSM", "option": 0, "maxlen": 9},
        "MESSAGE": {"valueType": "string", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "SSM", "option": 0},
        "LOCATION": {"valueType": "string", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "SSM", "option": 0},
        "OBJECT_ID": {"valueType": "string", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "SSM", "option": 0}}"#;
    let desc = TableDesc::from_desc_json(json).map_err(|e| ImageError::Other {
        path: path.to_path_buf(),
        msg: format!("logtable descriptor: {e}"),
    })?;
    let mut wt = WritableTable::create(path, desc);
    wt.flush()?;
    drop(wt);
    Ok(path.to_path_buf())
}

/// Coerce an array to float32 storage (casacore's putdata accepts any
/// numeric array for a float image).
pub(crate) fn coerce_float(data: &ArrayValue) -> ArrayValue {
    let flat = match &data.data {
        ArrayData::Float(v) => {
            return ArrayValue {
                shape: data.shape.clone(),
                data: ArrayData::Float(v.clone()),
            }
        }
        ArrayData::Double(v) => v.iter().map(|f| *f as f32).collect(),
        ArrayData::UChar(v) => v.iter().map(|i| f32::from(*i)).collect(),
        ArrayData::Short(v) => v.iter().map(|i| f32::from(*i)).collect(),
        ArrayData::UShort(v) => v.iter().map(|i| f32::from(*i)).collect(),
        ArrayData::Int(v) => v.iter().map(|i| *i as f32).collect(),
        ArrayData::UInt(v) => v.iter().map(|i| *i as f32).collect(),
        ArrayData::Int64(v) => v.iter().map(|i| *i as f32).collect(),
        _ => {
            return ArrayValue {
                shape: data.shape.clone(),
                data: data.data.clone(),
            }
        }
    };
    ArrayValue {
        shape: data.shape.clone(),
        data: ArrayData::Float(flat),
    }
}

/// True when `putdata` can store `data` in the float raster.
///
/// casacore converts *every* numeric raster type to the image's float
/// storage, and rejects the rest with "invalid data type Array<T>".  The
/// rejections matter for the message: passing an unconvertible raster
/// straight at the tiled encoder produced the opaque "encode .map:
/// unsupported tiled element type Float", which names the *column's* type
/// rather than the caller's array.
pub(crate) fn storable_as_float(data: &ArrayData) -> bool {
    matches!(
        data,
        ArrayData::UChar(_)
            | ArrayData::Short(_)
            | ArrayData::UShort(_)
            | ArrayData::Int(_)
            | ArrayData::UInt(_)
            | ArrayData::Int64(_)
            | ArrayData::Float(_)
            | ArrayData::Double(_)
    )
}

/// The casacore type name for an array element type, as casacore's own
/// "invalid data type Array<T>" message spells it.
pub(crate) fn array_kind_name(data: &ArrayData) -> &'static str {
    match data {
        ArrayData::Bool(_) => "Bool",
        ArrayData::UChar(_) => "uChar",
        ArrayData::Short(_) => "Short",
        ArrayData::UShort(_) => "uShort",
        ArrayData::Int(_) => "Int",
        ArrayData::UInt(_) => "uInt",
        ArrayData::Int64(_) => "Int64",
        ArrayData::Float(_) => "Float",
        ArrayData::Double(_) => "Double",
        ArrayData::Complex(_) => "Complex",
        ArrayData::DComplex(_) => "DComplex",
        ArrayData::String(_) => "String",
    }
}

/// Copy an opened image to a new CASA image table (pyrap `saveas`).
pub fn saveas(img: &Image, filename: &std::path::Path) -> Result<(), ImageError> {
    let data = img.getdata()?;
    create_casa_image(
        filename,
        img.shape(),
        img.coordinates(),
        &ImageMeta::from_image(img),
    )?;
    put_data(filename, &data)
}

/// A FITS card under construction.
struct Card(String);

impl Card {
    fn logical(keyword: &str, value: bool) -> Card {
        // FITS logicals are T/F.
        Card(format!(
            "{keyword:<8}= {:>20}",
            if value { "T" } else { "F" }
        ))
    }
    fn integer(keyword: &str, value: i64) -> Card {
        Card(format!("{keyword:<8}= {value:>20}"))
    }
    fn float(keyword: &str, value: f64) -> Card {
        // FITS fixed-format reals: right-justified within the 20-column
        // value field, signed two-digit exponent (Rust gives E-5; FITS
        // wants E-05).  Nine mantissa digits keep the card inside its
        // columns and are far below WCS read noise.
        let mut s = format!("{value:.9E}");
        if let Some(pos) = s.find('E') {
            let (mantissa, exponent) = s.split_at(pos);
            let exp = &exponent[1..];
            let (sign, digits) = exp.strip_prefix('-').map_or(("+", exp), |d| ("-", d));
            s = format!("{mantissa}E{sign}{digits:0>2}");
        }
        Card(format!("{keyword:<8}= {:>20}", s))
    }
    fn string(keyword: &str, value: &str) -> Card {
        // Fixed-format strings open their quote at column 11 and are
        // left-justified inside the value field.
        Card(format!("{keyword:<8}= {:<20}", format!("'{value}'")))
    }
    fn finish(self) -> [u8; 80] {
        let mut card = [b' '; 80];
        let bytes = self.0.as_bytes();
        let n = bytes.len().min(80);
        card[..n].copy_from_slice(&bytes[..n]);
        card
    }
}

/// A direction-coordinate angle in degrees, the unit casacore's `tofits`
/// writes (`CRVALn`/`CDELTn` with `CUNITn = 'deg'`).
///
/// Delegates to the shared [`super::coordsys::angle_to_radians`] so the FITS
/// path and the `toworld`/`topixel` path can never disagree about what a
/// record's `units` mean: an image whose record says `-1'` must write
/// `CDELT = -0.0166667` (and `toworld` must use `-1/60` degrees per pixel),
/// not `-57.29577951`.
fn angle_to_degrees(value: f64, unit: &str) -> f64 {
    super::coordsys::angle_to_radians(value, unit).to_degrees()
}

/// Export an opened image as a FITS primary-image cube (pyrap `tofits`).
/// WCS cards come from the coordinate system (direction angles converted
/// to degrees, CRPIX to 1-based); the beam from the restoring-beam record.
pub fn tofits(img: &Image, filename: &std::path::Path) -> Result<(), ImageError> {
    let data = img.getdata()?;
    let coords = img.coordinates();
    let shape = img.shape();
    let ndim = shape.len();

    // FITS axis order is the reverse of the numpy shape (NAXIS1 fastest).
    let mut cards: Vec<Card> = vec![
        Card::logical("SIMPLE", true),
        Card::integer("BITPIX", -32),
        Card::integer("NAXIS", ndim as i64),
    ];
    for (fits_axis, &d) in shape.iter().rev().enumerate() {
        cards.push(Card::integer(&format!("NAXIS{}", fits_axis + 1), d as i64));
    }
    cards.push(Card::logical("EXTEND", true));
    // Per-axis WCS: casa pixel axis p -> numpy axis ndim-1-p -> FITS axis
    // p+1.  Each axis contributes its own card set (direction axes split
    // into a long RA---/lat DEC-- pair).
    //
    // One card set per FITS axis, whichever coordinate claims it.  This is
    // the invariant `CoordinateSystem::from_fits` assumes: exactly one
    // CTYPE/CRVAL/CDELT/CRPIX group per FITS axis.  Without it a CASA image
    // whose direction coordinate spans non-adjacent pixel axes emits two
    // cards under one axis number and the last write wins, silently
    // rotating/degenerating the WCS on read-back.
    for casa_axis in 0..ndim {
        let fits_axis = casa_axis + 1;
        let mut done = false;
        for c in &coords.coords {
            if done {
                break;
            }
            match c {
                Coordinate::Direction {
                    crval,
                    crpix,
                    cdelt,
                    pixel_axes,
                    projection,
                    units,
                    ..
                } => {
                    if let Some(k) = pixel_axes.iter().position(|&p| p == casa_axis) {
                        done = true;
                        let ctyp = if k == 0 {
                            format!("RA---{projection}")
                        } else {
                            format!("DEC--{projection}")
                        };
                        cards.push(Card::string(&format!("CTYPE{fits_axis}"), &ctyp));
                        cards.push(Card::float(
                            &format!("CRVAL{fits_axis}"),
                            angle_to_degrees(crval[k], &units[k]),
                        ));
                        cards.push(Card::float(
                            &format!("CDELT{fits_axis}"),
                            angle_to_degrees(cdelt[k], &units[k]),
                        ));
                        cards.push(Card::float(&format!("CRPIX{fits_axis}"), crpix[k] + 1.0));
                        // casacore always writes a CUNIT for a direction
                        // axis, and always in degrees.
                        cards.push(Card::string(&format!("CUNIT{fits_axis}"), "deg"));
                    }
                }
                Coordinate::Linear {
                    crval,
                    crpix,
                    cdelt,
                    pixel_axes,
                    ..
                } => {
                    if let Some(k) = pixel_axes.iter().position(|&p| p == casa_axis) {
                        done = true;
                        let (ctype, cunit) = if matches!(c, Coordinate::Linear { name, .. } if name.starts_with("stokes"))
                        {
                            // casacore writes an empty unit for Stokes.
                            ("STOKES".to_string(), "")
                        } else if matches!(c, Coordinate::Linear { name, .. } if name.starts_with("spectral"))
                        {
                            ("FREQ".to_string(), "Hz")
                        } else {
                            (format!("LINEAR{casa_axis}"), "")
                        };
                        cards.push(Card::string(&format!("CTYPE{fits_axis}"), &ctype));
                        cards.push(Card::float(&format!("CRVAL{fits_axis}"), crval[k]));
                        cards.push(Card::float(&format!("CDELT{fits_axis}"), cdelt[k]));
                        cards.push(Card::float(&format!("CRPIX{fits_axis}"), crpix[k] + 1.0));
                        cards.push(Card::string(&format!("CUNIT{fits_axis}"), cunit));
                    }
                }
            }
        }
    }
    let meta = ImageMeta::from_image(img);
    if !meta.units.is_empty() {
        cards.push(Card::string("BUNIT", &meta.units));
    }
    if let Some(RecordValue::Record(beam)) = img.imageinfo().get("restoringbeam") {
        let q = |name: &str| -> Option<f64> {
            match beam.get(name) {
                Some(RecordValue::Record(q)) => match q.get("value") {
                    Some(RecordValue::Double(v)) => Some(*v),
                    _ => None,
                },
                _ => None,
            }
        };
        if let Some(major) = q("major") {
            cards.push(Card::float("BMAJ", major / 3600.0));
        }
        if let Some(minor) = q("minor") {
            cards.push(Card::float("BMIN", minor / 3600.0));
        }
        if let Some(pa) = q("positionangle") {
            cards.push(Card::float("BPA", pa));
        }
    }
    cards.push(Card(String::from("END")));

    // Header block, padded to 2880 bytes.
    let mut header: Vec<u8> = Vec::with_capacity(2880);
    for card in cards {
        header.extend_from_slice(&card.finish());
    }
    while !header.len().is_multiple_of(2880) {
        header.push(b' ');
    }

    // Data: float32, big-endian, FITS axis order (the reverse of numpy's):
    // FITS axis k (0-based, k fastest) is numpy axis ndim-1-k.
    let flat = match &data.data {
        ArrayData::Float(v) => v.clone(),
        ArrayData::Double(v) => v.iter().map(|f| *f as f32).collect(),
        _ => {
            return Err(ImageError::Other {
                path: img.path().to_path_buf(),
                msg: "only numeric rasters export to FITS".into(),
            })
        }
    };
    let reversed: Vec<usize> = shape.iter().rev().copied().collect();
    let mut numpy_strides = vec![1usize; ndim];
    for k in (0..ndim - 1).rev() {
        numpy_strides[k] = numpy_strides[k + 1] * shape[k + 1];
    }
    let mut fits_data = Vec::with_capacity(flat.len() * 4);
    for fits_idx in 0..flat.len() {
        let mut rem = fits_idx;
        let mut numpy_idx = 0usize;
        for k in 0..ndim {
            let c = rem % reversed[k];
            rem /= reversed[k];
            numpy_idx += c * numpy_strides[ndim - 1 - k];
        }
        fits_data.extend_from_slice(&flat[numpy_idx].to_be_bytes());
    }
    while !fits_data.len().is_multiple_of(2880) {
        fits_data.push(0);
    }

    let file = std::fs::File::create(filename).map_err(|e| ImageError::Other {
        path: filename.to_path_buf(),
        msg: format!("cannot create FITS: {e}"),
    })?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&header).map_err(io_err(filename))?;
    w.write_all(&fits_data).map_err(io_err(filename))?;
    w.flush().map_err(io_err(filename))?;
    Ok(())
}

fn io_err(path: &std::path::Path) -> impl Fn(std::io::Error) -> ImageError + '_ {
    |e| ImageError::Other {
        path: path.to_path_buf(),
        msg: format!("FITS write: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card_text(card: Card) -> String {
        String::from_utf8(card.finish().to_vec()).unwrap()
    }

    #[test]
    fn card_logical_and_integer_use_the_fixed_format() {
        assert_eq!(
            card_text(Card::logical("SIMPLE", true)),
            format!("{:<80}", "SIMPLE  =                    T")
        );
        assert_eq!(
            card_text(Card::logical("EXTEND", false)),
            format!("{:<80}", "EXTEND  =                    F")
        );
        assert_eq!(
            card_text(Card::integer("NAXIS", 4)),
            format!("{:<80}", "NAXIS   =                    4")
        );
        assert_eq!(
            card_text(Card::integer("BITPIX", -32)),
            format!("{:<80}", "BITPIX  =                  -32")
        );
    }

    #[test]
    fn every_card_is_exactly_eighty_columns() {
        for card in [
            Card::logical("SIMPLE", true),
            Card::integer("NAXIS", 1),
            Card::float("CRVAL1", 1.75),
            Card::string("CTYPE1", "RA---SIN"),
        ] {
            assert_eq!(card.finish().len(), 80);
        }
    }

    #[test]
    fn card_float_uses_a_signed_two_digit_exponent() {
        // FITS wants E+09 / E-05; Rust's own formatting gives E9 / E-5.
        assert!(card_text(Card::float("CRVAL1", 1.75)).contains("E+00"));
        assert!(card_text(Card::float("CDELT1", -2.5e-5)).contains("E-05"));
        assert!(card_text(Card::float("BMAJ", 3.5e-3)).contains("E-03"));
        // A value past the field width is truncated to 20 columns, never
        // spilling into the next card.
        let wide = card_text(Card::float("X", 1.0e300));
        assert!(wide.contains("E+300"));
        assert_eq!(wide.len(), 80);
    }

    #[test]
    fn card_string_opens_its_quote_in_the_value_field() {
        let text = card_text(Card::string("CTYPE1", "RA---SIN"));
        assert!(text.starts_with("CTYPE1  = 'RA---SIN'"));
        assert_eq!(text.len(), 80);
    }

    #[test]
    fn coerce_float_covers_every_numeric_variant() {
        let cases: Vec<(ArrayData, f32)> = vec![
            (ArrayData::Double(vec![1.5, -2.5]), 1.5),
            (ArrayData::UChar(vec![7]), 7.0),
            (ArrayData::Short(vec![-8]), -8.0),
            (ArrayData::UShort(vec![9]), 9.0),
            (ArrayData::Int(vec![-10]), -10.0),
            (ArrayData::UInt(vec![11]), 11.0),
            (ArrayData::Int64(vec![-12]), -12.0),
            (ArrayData::Float(vec![13.5]), 13.5),
        ];
        for (data, first) in cases {
            let out = coerce_float(&ArrayValue {
                shape: vec![data_len(&data) as u32],
                data,
            });
            match &out.data {
                ArrayData::Float(v) => assert_eq!(v[0], first),
                other => panic!("coerce_float produced {other:?}"),
            }
        }
    }

    fn data_len(data: &ArrayData) -> usize {
        match data {
            ArrayData::Bool(v) => v.len(),
            ArrayData::UChar(v) => v.len(),
            ArrayData::Short(v) => v.len(),
            ArrayData::UShort(v) => v.len(),
            ArrayData::Int(v) => v.len(),
            ArrayData::UInt(v) => v.len(),
            ArrayData::Int64(v) => v.len(),
            ArrayData::Float(v) => v.len(),
            ArrayData::Double(v) => v.len(),
            ArrayData::Complex(v) => v.len(),
            ArrayData::DComplex(v) => v.len(),
            ArrayData::String(v) => v.len(),
        }
    }

    #[test]
    fn coerce_float_passes_non_numeric_types_through() {
        // They must survive to the type check rather than being silently
        // reinterpreted; `storable_as_float` is what rejects them.
        for data in [
            ArrayData::Bool(vec![true]),
            ArrayData::String(vec!["x".into()]),
            ArrayData::Complex(vec![(1.0, 0.0)]),
        ] {
            let out = coerce_float(&ArrayValue {
                shape: vec![1],
                data: data.clone(),
            });
            assert_eq!(out.data, data);
            assert!(!storable_as_float(&out.data));
        }
    }

    #[test]
    fn storable_as_float_matches_casacores_accepted_raster_types() {
        // casacore converts every numeric raster and rejects Bool/String/
        // Complex.  Accepting a wider set is what produced the opaque
        // "unsupported tiled element type Float" instead of a type error.
        assert!(storable_as_float(&ArrayData::UChar(vec![0])));
        assert!(storable_as_float(&ArrayData::Short(vec![0])));
        assert!(storable_as_float(&ArrayData::UShort(vec![0])));
        assert!(storable_as_float(&ArrayData::Int(vec![0])));
        assert!(storable_as_float(&ArrayData::UInt(vec![0])));
        assert!(storable_as_float(&ArrayData::Int64(vec![0])));
        assert!(storable_as_float(&ArrayData::Float(vec![0.0])));
        assert!(storable_as_float(&ArrayData::Double(vec![0.0])));
        assert!(!storable_as_float(&ArrayData::Bool(vec![true])));
        assert!(!storable_as_float(&ArrayData::String(vec!["x".into()])));
        assert!(!storable_as_float(&ArrayData::Complex(vec![(1.0, 0.0)])));
        assert!(!storable_as_float(&ArrayData::DComplex(vec![(1.0, 0.0)])));
    }

    #[test]
    fn array_kind_name_matches_the_casacore_spelling() {
        assert_eq!(array_kind_name(&ArrayData::Bool(vec![])), "Bool");
        assert_eq!(array_kind_name(&ArrayData::UChar(vec![])), "uChar");
        assert_eq!(array_kind_name(&ArrayData::Short(vec![])), "Short");
        assert_eq!(array_kind_name(&ArrayData::UShort(vec![])), "uShort");
        assert_eq!(array_kind_name(&ArrayData::Int(vec![])), "Int");
        assert_eq!(array_kind_name(&ArrayData::UInt(vec![])), "uInt");
        assert_eq!(array_kind_name(&ArrayData::Int64(vec![])), "Int64");
        assert_eq!(array_kind_name(&ArrayData::Float(vec![])), "Float");
        assert_eq!(array_kind_name(&ArrayData::Double(vec![])), "Double");
        assert_eq!(array_kind_name(&ArrayData::Complex(vec![])), "Complex");
        assert_eq!(array_kind_name(&ArrayData::DComplex(vec![])), "DComplex");
        assert_eq!(array_kind_name(&ArrayData::String(vec![])), "String");
    }

    #[test]
    fn angle_to_degrees_honours_the_record_unit() {
        // The default template stores arcmin: -1' is -1/60 degree, not
        // -57.29577951 (the value a radian reading produces).
        assert!((angle_to_degrees(-1.0, "'") - (-1.0 / 60.0)).abs() < 1e-15);
        assert!((angle_to_degrees(1.0, "'") - (1.0 / 60.0)).abs() < 1e-15);
        assert!((angle_to_degrees(-1.0, "rad") - (-1.0f64.to_degrees())).abs() < 1e-12);
        assert!((angle_to_degrees(1.75, "deg") - 1.75).abs() < 1e-15);
        assert!((angle_to_degrees(60.0, "arcsec") - (60.0 / 3600.0)).abs() < 1e-15);
        // An unrecognised unit is radians, a direction record's default.
        assert!((angle_to_degrees(1.0, "") - 1.0f64.to_degrees()).abs() < 1e-12);
    }

    #[test]
    fn angle_to_degrees_agrees_with_the_world_transform() {
        // Whatever unit a record carries, the FITS card and `toworld` must
        // read it the same way.
        for unit in ["rad", "'", "\"", "deg"] {
            let card = angle_to_degrees(2.0, unit);
            let world = super::super::coordsys::angle_to_radians(2.0, unit).to_degrees();
            assert!((card - world).abs() < 1e-15, "{unit}");
        }
    }
}
