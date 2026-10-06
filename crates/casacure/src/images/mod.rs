//! The image subsystem (issue #14): `casacure.images`, the port of the
//! `pyrap.images` surface DDFacet and killMS use — opening CASA image
//! tables and FITS cubes, raster access, and coordinate conversions.
//!
//! A CASA image is a one-row casacore table (TiledCellStMan raster column)
//! whose coordinate system, restoring beam and units live in keyword
//! records; a DDFacet product is usually the astropy-written FITS file the
//! CASA image was made from.  Both open here through the same `Image`
//! object with the same coordinate conversions.

pub mod coordsys;
pub mod fits;
pub mod image;

pub use coordsys::{CoordError, Coordinate, CoordinateSystem};
pub use fits::{CardValue, FitsError, FitsHeader, FitsImage};
pub use image::{CasaImage, Image, ImageError};
