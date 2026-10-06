//! The image object (issue #14): opens a CASA image table (a one-row table
//! with a tiled raster column and `coords`/`imageinfo` keyword records) or
//! a FITS primary-image file, and serves the pyrap `image` surface DDFacet
//! uses — `getdata`, `coordinates`, `toworld`/`topixel`, `imageinfo`,
//! `shape`, `name`, `unit`, `miscinfo`.

use std::sync::Arc;

use thiserror::Error;

use crate::record::{ArrayValue, RecordValue, TableRecord};
use crate::table::Table;

use super::coordsys::CoordinateSystem;
use super::fits::FitsImage;

#[derive(Debug, Error)]
pub enum ImageError {
    #[error("{path}: no such image (not a CASA table directory or a FITS file)")]
    NoSuchImage { path: std::path::PathBuf },
    #[error(transparent)]
    Table(#[from] crate::table::TableDatError),
    #[error(transparent)]
    Read(#[from] crate::table::TableReadError),
    #[error(transparent)]
    Fits(#[from] super::fits::FitsError),
    #[error(transparent)]
    Coord(#[from] super::coordsys::CoordError),
    #[error("{path}: {msg}")]
    Other {
        path: std::path::PathBuf,
        msg: String,
    },
    #[error(transparent)]
    Write(#[from] crate::table::WriteTableError),
    /// `regrid` given an axis index outside the image's axes — reported
    /// instead of indexing `src_shape[axis]` out of bounds (which panicked
    /// through pyo3 as a `PanicException`).
    #[error("regrid axis {axis} is out of range (the image has {ndim} axes)")]
    BadAxis { axis: usize, ndim: usize },
    /// `putdata` given a raster whose shape is not the image's shape.
    #[error("putdata: array shape {got:?} does not match the image {want:?}")]
    ShapeMismatch { got: Vec<u32>, want: Vec<u32> },
    /// `putdata` given an element type the float raster cannot hold (Bool,
    /// String, Complex).  casacore rejects these too ("invalid data type
    /// Array<T>"); naming the offending type is the whole point, because the
    /// storage layer's own complaint named the *column's* type instead.
    #[error("putdata: invalid data type Array<{kind}> for an image that stores Float")]
    BadElementType { kind: String },
}

/// An opened image: a CASA image table, a FITS file, or an in-memory
/// raster (the result of `regrid`).
pub enum Image {
    Casa(CasaImage),
    Fits(FitsImageInfo),
    Memory(MemoryImage),
}

/// An in-memory raster with its own coordinates/metadata (never persisted
/// until `saveas`).
pub struct MemoryImage {
    pub data: ArrayValue,
    /// The numpy-order shape (mirrors `data.shape` as usize).
    pub shape: Vec<usize>,
    pub coords: CoordinateSystem,
    pub meta: super::write::ImageMeta,
}

/// The CASA-table flavour: the opened snapshot plus the raster column.
pub struct CasaImage {
    pub path: std::path::PathBuf,
    table: Arc<Table>,
    raster: usize,
    shape: Vec<usize>,
    coords: CoordinateSystem,
}

/// The FITS flavour: the mapped file plus its derived coordinates.
pub struct FitsImageInfo {
    pub path: std::path::PathBuf,
    fits: FitsImage,
    coords: CoordinateSystem,
    shape: Vec<usize>,
}

impl Image {
    /// Open a CASA image table directory or a FITS file (pyrap's `image()`
    /// accepts both; it sniffs the path, so do the same).
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Image, ImageError> {
        let path = path.into();
        if path.is_dir() {
            // A directory is an image only when it is a CASA table.  Any
            // other directory is "not an image", and leaking the table
            // layer's io error ("No such file or directory (os error 2)")
            // for a directory that plainly exists sends the caller looking
            // for a missing path instead of a wrong one.  A directory that
            // *is* a table but fails to open still reports the table error.
            if !path.join("table.info").is_file() {
                return Err(ImageError::NoSuchImage { path });
            }
            return Ok(Image::Casa(CasaImage::open(path)?));
        }
        if path.is_file() {
            return Ok(Image::Fits(FitsImageInfo::open(path)?));
        }
        Err(ImageError::NoSuchImage { path })
    }

    pub fn path(&self) -> &std::path::Path {
        match self {
            Image::Casa(c) => &c.path,
            Image::Fits(f) => &f.path,
            Image::Memory(_) => std::path::Path::new(""),
        }
    }

    /// The image shape in pyrap/numpy order (channel, pol, y, x for a
    /// DDFacet cube; the reverse of the CASA on-disk axis order).
    pub fn shape(&self) -> &[usize] {
        match self {
            Image::Casa(c) => &c.shape,
            Image::Fits(f) => &f.shape,
            Image::Memory(m) => &m.shape,
        }
    }

    /// The full raster as one array (pyrap `getdata()`), numpy order.
    pub fn getdata(&self) -> Result<ArrayValue, ImageError> {
        match self {
            Image::Casa(c) => c.getdata(),
            Image::Fits(f) => Ok(ArrayValue {
                shape: f.shape.iter().map(|&d| d as u32).collect(),
                data: f.fits.data_array()?,
            }),
            Image::Memory(m) => Ok(m.data.clone()),
        }
    }

    /// Replace the raster (pyrap `putdata`).  A CASA-table image is
    /// rewritten through a writable open and its snapshot refreshed; an
    /// in-memory image swaps its raster (shape-checked).
    ///
    /// The shape is checked up front for every flavour: casacore rejects a
    /// `putdata` whose shape differs from the image before touching the
    /// storage manager, so a mismatched array must not reach the tiled
    /// encoder (whose "unsupported tiled element type" message is about the
    /// storage, not the caller's mistake).
    pub fn put_data(&mut self, data: &ArrayValue) -> Result<(), ImageError> {
        let want: Vec<u32> = self.shape().iter().map(|&d| d as u32).collect();
        if data.shape != want {
            return Err(ImageError::ShapeMismatch {
                got: data.shape.clone(),
                want,
            });
        }
        // casacore converts any *numeric* raster to the image's float
        // storage and rejects the rest; naming the caller's array type beats
        // the tiled encoder's "unsupported tiled element type Float", which
        // reports the column's type instead.
        if !super::write::storable_as_float(&data.data) {
            return Err(ImageError::BadElementType {
                kind: super::write::array_kind_name(&data.data).to_string(),
            });
        }
        match self {
            Image::Casa(c) => {
                super::write::put_data(&c.path, data)?;
                *self = Image::open(&c.path)?;
                Ok(())
            }
            Image::Memory(m) => {
                m.data = super::write::coerce_float(data);
                Ok(())
            }
            Image::Fits(_) => Err(ImageError::Other {
                path: std::path::PathBuf::new(),
                msg: "cannot putdata into a FITS file".into(),
            }),
        }
    }

    pub fn coordinates(&self) -> &CoordinateSystem {
        match self {
            Image::Casa(c) => &c.coords,
            Image::Fits(f) => &f.coords,
            Image::Memory(m) => &m.coords,
        }
    }

    /// `imageinfo()`: the restoring beam et al. From the `imageinfo`
    /// keyword record (CASA) or the BMAJ/BMIN/BPA cards (FITS).
    pub fn imageinfo(&self) -> TableRecord {
        match self {
            Image::Casa(c) => c.keywords_record("imageinfo").unwrap_or_default(),
            Image::Fits(f) => f.beam_record(),
            Image::Memory(m) => {
                let mut info = m.meta.imageinfo.clone();
                if info.desc.fields.is_empty() {
                    info = super::write::ImageMeta::default_info();
                }
                info
            }
        }
    }

    /// The brightness unit: the `units` keyword (CASA) or BUNIT (FITS).
    /// pyrap's `unit()` wraps it in quotes — matched verbatim.
    ///
    /// Both flavours quote, so `unit()` is symmetric across CASA images,
    /// FITS cubes and in-memory rasters.  A value already carrying the
    /// pyrap quotes (an `ImageMeta` that round-tripped through `unit()`) is
    /// returned as-is rather than double-quoted.
    pub fn unit(&self) -> String {
        let raw = match self {
            Image::Casa(c) => c
                .keyword_value("units")
                .map(|v| match v {
                    RecordValue::String(s) => s,
                    _ => String::new(),
                })
                .unwrap_or_default(),
            Image::Fits(f) => f.fits.string_of("BUNIT").unwrap_or("").to_string(),
            Image::Memory(m) => m.meta.units.clone(),
        };
        if raw.len() >= 2 && raw.starts_with('\'') && raw.ends_with('\'') {
            return raw;
        }
        format!("'{raw}'")
    }

    /// `miscinfo()`: the `miscinfo` keyword record (CASA) or empty (FITS).
    pub fn miscinfo(&self) -> TableRecord {
        match self {
            Image::Casa(c) => c.keywords_record("miscinfo").unwrap_or_default(),
            Image::Fits(_) => TableRecord::default(),
            Image::Memory(m) => m.meta.miscinfo.clone(),
        }
    }
}

impl CasaImage {
    fn open(path: std::path::PathBuf) -> Result<CasaImage, ImageError> {
        let table = Arc::new(Table::open(&path, true)?);
        // The raster column: the single array column of the image table
        // (`map` for a PagedImage; find the first array column).
        let raster = table
            .dat
            .desc
            .columns
            .iter()
            .position(|c| c.kind == crate::tabledesc::ColumnKind::Array)
            .ok_or_else(|| ImageError::Other {
                path: path.clone(),
                msg: "no raster (array) column in the image table".into(),
            })?;
        let cell = table.getcell(raster, 0)?;
        let shape: Vec<usize> = match &cell {
            RecordValue::Array(a) => a.shape.iter().map(|&d| d as usize).collect(),
            _ => {
                return Err(ImageError::Other {
                    path: path.clone(),
                    msg: "the raster cell is not an array".into(),
                })
            }
        };
        let coords = match table.dat.desc.keywords.get("coords") {
            Some(RecordValue::Record(rec)) => CoordinateSystem::from_record(rec, shape.len())?,
            _ => {
                return Err(ImageError::Other {
                    path: path.clone(),
                    msg: "the image table has no coords keyword".into(),
                })
            }
        };
        Ok(CasaImage {
            path,
            table,
            raster,
            shape,
            coords,
        })
    }

    fn keyword_value(&self, name: &str) -> Option<RecordValue> {
        self.table.dat.desc.keywords.get(name).cloned()
    }

    fn keywords_record(&self, name: &str) -> Option<TableRecord> {
        match self.keyword_value(name) {
            Some(RecordValue::Record(rec)) => Some(rec),
            _ => None,
        }
    }

    fn getdata(&self) -> Result<ArrayValue, ImageError> {
        match self.table.getcell(self.raster, 0)? {
            RecordValue::Array(a) => Ok(a),
            _ => Err(ImageError::Other {
                path: self.path.clone(),
                msg: "the raster cell is not an array".into(),
            }),
        }
    }
}

impl FitsImageInfo {
    fn open(path: std::path::PathBuf) -> Result<FitsImageInfo, ImageError> {
        let fits = FitsImage::open(&path)?;
        let shape: Vec<usize> = fits.naxes.iter().rev().copied().collect();
        let coords = CoordinateSystem::from_fits(&fits.header, shape.len());
        Ok(FitsImageInfo {
            path,
            fits,
            coords,
            shape,
        })
    }

    /// The restoring-beam record from the BMAJ/BMIN/BPA cards (degrees),
    /// in casacore's `imageinfo()["restoringbeam"]` layout (major/minor in
    /// arcsec, positionangle in degrees) — the structure DDFacet's
    /// SkyModel tools read.
    fn beam_record(&self) -> TableRecord {
        let mut beam = TableRecord::default();
        let mut entry = |name: &str, value: f64, unit: &str| {
            let mut q = TableRecord::default();
            q.set("value", RecordValue::Double(value));
            q.set("unit", RecordValue::String(unit.into()));
            beam.set(name, RecordValue::Record(q));
        };
        if let Some(major) = self.fits.f64_of("BMAJ") {
            entry("major", major * 3600.0, "arcsec");
        }
        if let Some(minor) = self.fits.f64_of("BMIN") {
            entry("minor", minor * 3600.0, "arcsec");
        }
        if let Some(pa) = self.fits.f64_of("BPA") {
            entry("positionangle", pa, "deg");
        }
        let mut info = TableRecord::default();
        info.set("imagetype", RecordValue::String("Intensity".into()));
        info.set("objectname", RecordValue::String(String::new()));
        if !beam.desc.fields.is_empty() {
            info.set("restoringbeam", RecordValue::Record(beam));
        }
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::write::ImageMeta;
    use crate::record::ArrayData;

    #[test]
    fn error_messages_name_the_cause() {
        // Each of these is the caller's only diagnostic: the shape one used
        // to surface as the storage layer's "unsupported tiled element type
        // Float", and the axis one as a pyo3 PanicException.
        assert_eq!(
            ImageError::NoSuchImage {
                path: "/x/y.image".into()
            }
            .to_string(),
            "/x/y.image: no such image (not a CASA table directory or a FITS file)"
        );
        assert_eq!(
            ImageError::BadAxis { axis: 9, ndim: 4 }.to_string(),
            "regrid axis 9 is out of range (the image has 4 axes)"
        );
        assert_eq!(
            ImageError::ShapeMismatch {
                got: vec![2, 2],
                want: vec![3, 2, 8, 10],
            }
            .to_string(),
            "putdata: array shape [2, 2] does not match the image [3, 2, 8, 10]"
        );
        assert_eq!(
            ImageError::BadElementType {
                kind: "Bool".into()
            }
            .to_string(),
            "putdata: invalid data type Array<Bool> for an image that stores Float"
        );
    }

    fn memory(shape: &[usize], value: f32) -> Image {
        let nelem: usize = shape.iter().product();
        Image::Memory(MemoryImage {
            data: ArrayValue {
                shape: shape.iter().map(|&d| d as u32).collect(),
                data: ArrayData::Float(vec![value; nelem]),
            },
            shape: shape.to_vec(),
            coords: CoordinateSystem::default_for(shape),
            meta: ImageMeta::default(),
        })
    }

    #[test]
    fn shape_and_path_of_an_in_memory_image() {
        let img = memory(&[3, 2, 8, 10], 1.0);
        assert_eq!(img.shape(), &[3, 2, 8, 10]);
        assert_eq!(img.path(), std::path::Path::new(""));
        assert_eq!(img.coordinates().nimaxes, 4);
    }

    #[test]
    fn put_data_checks_the_shape_before_touching_the_raster() {
        let mut img = memory(&[2, 3], 1.0);
        let err = img
            .put_data(&ArrayValue {
                shape: vec![3, 2],
                data: ArrayData::Float(vec![0.0; 6]),
            })
            .unwrap_err();
        assert!(matches!(err, ImageError::ShapeMismatch { .. }), "{err}");
        // The raster is untouched.
        match img.getdata().unwrap().data {
            ArrayData::Float(v) => assert_eq!(v, vec![1.0; 6]),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn put_data_checks_the_element_type() {
        let mut img = memory(&[2, 3], 1.0);
        let err = img
            .put_data(&ArrayValue {
                shape: vec![2, 3],
                data: ArrayData::Bool(vec![true; 6]),
            })
            .unwrap_err();
        match err {
            ImageError::BadElementType { kind } => assert_eq!(kind, "Bool"),
            other => panic!("expected BadElementType, got {other}"),
        }
    }

    #[test]
    fn put_data_into_memory_coerces_to_float32() {
        let mut img = memory(&[2, 2], 0.0);
        img.put_data(&ArrayValue {
            shape: vec![2, 2],
            data: ArrayData::Double(vec![1.5, 2.5, 3.5, 4.5]),
        })
        .unwrap();
        match img.getdata().unwrap().data {
            ArrayData::Float(v) => assert_eq!(v, vec![1.5, 2.5, 3.5, 4.5]),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_fits_flavour_image_is_not_writable() {
        // The FITS arm of `put_data` refuses outright; there is no table to
        // rewrite, and the batch pipelines only ever write CASA products.
        let dir = std::env::temp_dir().join(format!("casacure-img-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.fits");
        let mut body = Vec::new();
        for text in [
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    1",
            "NAXIS1  =                    1",
        ] {
            let mut card = [b' '; 80];
            card[..text.len()].copy_from_slice(text.as_bytes());
            body.extend_from_slice(&card);
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        body.extend_from_slice(&end);
        while body.len() % 2880 != 0 {
            body.push(b' ');
        }
        body.extend_from_slice(&1.0f32.to_be_bytes());
        std::fs::write(&path, &body).unwrap();

        let mut img = Image::open(&path).unwrap();
        let err = img
            .put_data(&ArrayValue {
                shape: vec![1],
                data: ArrayData::Float(vec![2.0]),
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("cannot putdata into a FITS file"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `Image` has no `Debug`, so `unwrap_err` is unavailable.
    fn open_err(path: &std::path::Path) -> ImageError {
        match Image::open(path) {
            Ok(_) => panic!("expected {} to fail to open", path.display()),
            Err(e) => e,
        }
    }

    #[test]
    fn open_reports_a_missing_path_as_no_such_image() {
        let err = open_err(std::path::Path::new("/definitely/not/here.image"));
        assert!(matches!(err, ImageError::NoSuchImage { .. }), "{err}");
        assert_eq!(
            err.to_string(),
            "/definitely/not/here.image: no such image (not a CASA table directory or a FITS file)"
        );
    }

    #[test]
    fn open_reports_a_directory_without_a_table_as_no_such_image() {
        // A directory that exists but is not a CASA table used to surface
        // the table layer's raw "No such file or directory (os error 2)".
        let dir = std::env::temp_dir().join(format!("casacure-nodir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let err = open_err(&dir);
        assert!(matches!(err, ImageError::NoSuchImage { .. }), "{err}");
        assert!(err.to_string().contains("no such image"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_image_meta_has_the_housekeeping_records() {
        // A freshly created image carries no unit (pyrap's `unit()` is
        // then the empty quoted string) and the default imageinfo record.
        let meta = ImageMeta::default();
        assert!(meta.units.is_empty());
        let info = ImageMeta::default_info();
        let names: Vec<&str> = info.desc.fields.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"imagetype"), "{names:?}");
        assert!(names.contains(&"objectname"), "{names:?}");
    }
}
