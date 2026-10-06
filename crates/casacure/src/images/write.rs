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
pub fn put_data(path: &std::path::Path, data: &ArrayValue) -> Result<(), ImageError> {
    let (_read, mut wt) = WritableTable::open_for_update(path)?;
    let raster = wt
        .desc()
        .columns
        .iter()
        .position(|c| c.kind == crate::tabledesc::ColumnKind::Array)
        .ok_or_else(|| ImageError::Other {
            path: path.to_path_buf(),
            msg: "no raster (array) column in the image table".into(),
        })?;
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
        ArrayData::Int(v) => v.iter().map(|i| *i as f32).collect(),
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
    // p+1.  Each direction axis emits its own pair of cards (long = RA---,
    // lat = DEC--).
    for casa_axis in 0..ndim {
        let fits_axis = casa_axis + 1;
        for c in &coords.coords {
            match c {
                Coordinate::Direction {
                    crval,
                    crpix,
                    cdelt,
                    pixel_axes,
                    projection,
                    ..
                } => {
                    if let Some(k) = pixel_axes.iter().position(|&p| p == casa_axis) {
                        let ctyp = if k == 0 {
                            format!("RA---{projection}")
                        } else {
                            format!("DEC--{projection}")
                        };
                        cards.push(Card::string(&format!("CTYPE{fits_axis}"), &ctyp));
                        cards.push(Card::float(
                            &format!("CRVAL{fits_axis}"),
                            crval[k].to_degrees(),
                        ));
                        cards.push(Card::float(
                            &format!("CDELT{fits_axis}"),
                            cdelt[k].to_degrees(),
                        ));
                        cards.push(Card::float(&format!("CRPIX{fits_axis}"), crpix[k] + 1.0));
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
                        let ctype = if matches!(c, Coordinate::Linear { name, .. } if name.starts_with("stokes"))
                        {
                            "STOKES".to_string()
                        } else if matches!(c, Coordinate::Linear { name, .. } if name.starts_with("spectral"))
                        {
                            "FREQ".to_string()
                        } else {
                            format!("LINEAR{casa_axis}")
                        };
                        cards.push(Card::string(&format!("CTYPE{fits_axis}"), &ctype));
                        cards.push(Card::float(&format!("CRVAL{fits_axis}"), crval[k]));
                        cards.push(Card::float(&format!("CDELT{fits_axis}"), cdelt[k]));
                        cards.push(Card::float(&format!("CRPIX{fits_axis}"), crpix[k] + 1.0));
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
