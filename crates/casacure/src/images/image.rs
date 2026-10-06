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
    pub fn put_data(&mut self, data: &ArrayValue) -> Result<(), ImageError> {
        match self {
            Image::Casa(c) => {
                super::write::put_data(&c.path, data)?;
                *self = Image::open(&c.path)?;
                Ok(())
            }
            Image::Memory(m) => {
                let want: Vec<u32> = m.shape.iter().map(|&d| d as u32).collect();
                let coerced = super::write::coerce_float(data);
                if coerced.shape != want {
                    return Err(ImageError::Other {
                        path: std::path::PathBuf::new(),
                        msg: format!(
                            "putdata: array shape {:?} does not match the image {:?}",
                            coerced.shape, want
                        ),
                    });
                }
                m.data = coerced;
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
