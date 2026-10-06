//! The `casacure.images` pyo3 module (issue #14): the `pyrap.images`
//! surface DDFacet and killMS consume — the `image` class (open CASA image
//! tables and FITS cubes) and the `coordinates` object it hands back.
//! Registered by `images_submodule` like `quanta`/`measures`.

use std::sync::RwLock;

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

use crate::convert;
use numpy::PyArray1;

use casacure::images as cimg;
use casacure::images::{Coordinate, CoordinateSystem, Image, ImageMeta};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// python-casacore/pyrap-compatible `images.image` object.
#[allow(non_camel_case_types)]
#[pyclass(name = "image")]
pub struct image {
    inner: RwLock<Image>,
    shape: Vec<usize>,
    name: String,
}

#[pymethods]
impl image {
    #[new]
    #[pyo3(signature = (imagename = None, shape = None, coordsys = None, overwrite = true))]
    #[allow(unused_variables)]
    fn new(
        py: Python<'_>,
        imagename: Option<&Bound<'_, PyAny>>,
        shape: Option<Vec<usize>>,
        coordsys: Option<&Bound<'_, PyAny>>,
        overwrite: bool,
    ) -> PyResult<Self> {
        let path_from = |arg: &Bound<'_, PyAny>| -> PyResult<std::path::PathBuf> {
            let s: String = if let Ok(s) = arg.extract::<String>() {
                s
            } else {
                arg.call_method0("__fspath__")?.extract()?
            };
            Ok(casacure::table::absolute_dir(std::path::Path::new(&s)))
        };
        // Create form: image(imagename=, shape= [, coordsys=]).
        if let (Some(name_arg), Some(shape)) = (imagename, &shape) {
            let path = path_from(name_arg)?;
            // pyrap's contract: `overwrite=False` refuses to replace an
            // existing image.  Ignoring the flag silently destroyed a
            // caller's cube; DDFacet's ClassCasaImage passes it explicitly.
            if !overwrite && path.exists() {
                return Err(PyRuntimeError::new_err(format!(
                    "file {} already exists and should not be overwritten",
                    path.display()
                )));
            }
            let csys = match coordsys {
                Some(c) => {
                    let obj: PyRef<'_, coordinates> = c.extract()?;
                    let csys = obj.csys.read().unwrap().clone();
                    csys
                }
                None => CoordinateSystem::default_for(shape),
            };
            cimg::create_casa_image(&path, shape, &csys, &ImageMeta::default()).map_err(err)?;
            let opened = Image::open(&path).map_err(err)?;
            let shape = opened.shape().to_vec();
            let name = opened.path().display().to_string();
            return Ok(image {
                inner: RwLock::new(opened),
                shape,
                name,
            });
        }
        let Some(path_arg) = imagename else {
            return Err(PyValueError::new_err(
                "image() needs a path (or imagename= + shape=)",
            ));
        };
        let path = path_from(path_arg)?;
        let opened = Image::open(&path).map_err(err)?;
        let shape = opened.shape().to_vec();
        let name = opened.path().display().to_string();
        Ok(image {
            inner: RwLock::new(opened),
            shape,
            name,
        })
    }

    /// The raster as a numpy array (pyrap `getdata()`).
    fn getdata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let img = self.inner.read().unwrap();
        let data = img.getdata().map_err(err)?;
        convert::array_to_ndarray(py, &data)
    }

    /// The coordinate system (`coordinates.coordinates`).
    fn coordinates(&self, py: Python<'_>) -> PyResult<Py<coordinates>> {
        let img = self.inner.read().unwrap();
        Py::new(
            py,
            coordinates {
                csys: RwLock::new(img.coordinates().clone()),
            },
        )
    }

    /// Pixel -> world (both tuples in pyrap/numpy axis order).
    fn toworld(&self, py: Python<'_>, pixel: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let img = self.inner.read().unwrap();
        let mut px: Vec<f64> = pixel
            .iter()
            .map(|v| v.extract::<f64>())
            .collect::<PyResult<_>>()?;
        px.reverse();
        let mut world = img.coordinates().to_world(&px).map_err(err)?;
        world.reverse();
        Ok(PyTuple::new(py, world)?.into_any().unbind())
    }

    /// World -> pixel (both tuples in pyrap/numpy axis order).
    fn topixel(&self, py: Python<'_>, world: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let img = self.inner.read().unwrap();
        let mut w: Vec<f64> = world
            .iter()
            .map(|v| v.extract::<f64>())
            .collect::<PyResult<_>>()?;
        w.reverse();
        let mut pixel = img.coordinates().to_pixel(&w).map_err(err)?;
        pixel.reverse();
        Ok(PyTuple::new(py, pixel)?.into_any().unbind())
    }

    /// `imageinfo()`: the restoring beam et al, as a dict.
    fn imageinfo<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let img = self.inner.read().unwrap();
        convert::table_record_to_dict(py, &img.imageinfo())
    }

    /// `miscinfo()`.
    fn miscinfo<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let img = self.inner.read().unwrap();
        convert::table_record_to_dict(py, &img.miscinfo())
    }

    /// The brightness unit (pyrap wraps it in quotes — matched verbatim).
    fn unit(&self) -> PyResult<String> {
        let img = self.inner.read().unwrap();
        Ok(img.unit())
    }

    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.shape.clone())?.into_any().unbind())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("<image '{}' shape {:?}>", self.name, self.shape)
    }

    /// Replace the raster (pyrap `putdata`); the opened snapshot is
    /// refreshed so later reads see the new data.
    fn putdata(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let rec = convert::pyobject_to_record(data.py(), data)?;
        let casacure::record::RecordValue::Array(arr) = rec else {
            return Err(PyValueError::new_err("putdata expects an array"));
        };
        self.inner.write().unwrap().put_data(&arr).map_err(err)
    }

    /// Copy this image to a new CASA image table (pyrap `saveas`).
    fn saveas(&self, filename: &str) -> PyResult<()> {
        let img = self.inner.read().unwrap();
        let path = casacure::table::absolute_dir(std::path::Path::new(filename));
        cimg::saveas(&img, &path).map_err(err)
    }

    /// Export as a FITS primary-image cube (pyrap `tofits`).
    fn tofits(&self, filename: &str) -> PyResult<()> {
        let img = self.inner.read().unwrap();
        let path = casacure::table::absolute_dir(std::path::Path::new(filename));
        cimg::tofits(&img, &path).map_err(err)
    }

    /// Resample the given numpy-order axes onto a target coordinate
    /// system (`img.regrid([2, 3], cMain, outshape=(...))`, ModMosaic's
    /// mosaic stacking).  Returns a new in-memory image.
    #[pyo3(signature = (axes, coordsys, outshape = None))]
    fn regrid(
        &self,
        py: Python<'_>,
        axes: Vec<usize>,
        coordsys: &Bound<'_, PyAny>,
        outshape: Option<Vec<usize>>,
    ) -> PyResult<Py<image>> {
        let img = self.inner.read().unwrap();
        let obj: PyRef<'_, coordinates> = coordsys.extract()?;
        let target = obj.csys.read().unwrap().clone();
        let outshape = outshape.unwrap_or_else(|| img.shape().to_vec());
        let regridded = cimg::regrid(&img, &axes, &target, &outshape).map_err(err)?;
        let shape = regridded.shape().to_vec();
        let name = regridded.path().display().to_string();
        Py::new(
            py,
            image {
                inner: RwLock::new(regridded),
                shape,
                name,
            },
        )
    }
}

/// python-casacore/pyrap-compatible `coordinates` object: the image's
/// coordinate system with the dict()/get/set surface DDFacet touches.
/// `_csys` (the raw record dict MyCasapy2bbs reads) comes through the
/// `__getattr__` fallback below.
#[allow(non_camel_case_types)]
#[pyclass(name = "coordinates")]
pub struct coordinates {
    csys: RwLock<CoordinateSystem>,
}

#[pymethods]
impl coordinates {
    /// The raw coords record under `_csys`
    /// (`coordinates().__dict__["_csys"]["direction0"]["cdelt"]`).
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if name == "_csys" {
            let csys = self.csys.read().unwrap();
            return convert::table_record_to_dict(py, &csys.to_record())
                .map(|d| d.into_any().unbind());
        }
        Err(PyAttributeError::new_err(name.to_string()))
    }

    /// The casacore record layout (`coordinates().dict()`).
    fn dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let csys = self.csys.read().unwrap();
        convert::table_record_to_dict(py, &csys.to_record())
    }

    /// Per-coordinate increments, pyrap's layout: one scalar (single-axis
    /// coordinate) or array per coordinate, in REVERSE coordinate-system
    /// order (spectral, stokes, direction — the order DDFacet's
    /// `incr[-1]` indexing expects).
    fn get_increment(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let csys = self.csys.read().unwrap();
        per_coord_list(py, &csys, |c| match c {
            Coordinate::Direction { cdelt, .. } => vec![cdelt[0], cdelt[1]],
            Coordinate::Linear { cdelt, .. } => cdelt.clone(),
        })
    }

    fn set_increment(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut csys = self.csys.write().unwrap();
        let mut rev = per_coord_values(value)?;
        rev.reverse();
        csys.set_increments(&rev);
        Ok(())
    }

    fn get_referencevalue(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let csys = self.csys.read().unwrap();
        per_coord_list(py, &csys, |c| match c {
            Coordinate::Direction { crval, .. } => vec![crval[0], crval[1]],
            Coordinate::Linear { crval, .. } => crval.clone(),
        })
    }

    fn set_referencevalue(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut csys = self.csys.write().unwrap();
        let mut rev = per_coord_values(value)?;
        rev.reverse();
        csys.set_reference_values(&rev);
        Ok(())
    }

    fn get_referencepixel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let csys = self.csys.read().unwrap();
        per_coord_list(py, &csys, |c| match c {
            Coordinate::Direction { crpix, .. } => vec![crpix[0], crpix[1]],
            Coordinate::Linear { crpix, .. } => crpix.clone(),
        })
    }

    fn set_referencepixel(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut csys = self.csys.write().unwrap();
        let mut rev = per_coord_values(value)?;
        rev.reverse();
        csys.set_reference_pixels(&rev);
        Ok(())
    }
}

/// One scalar-or-array entry per coordinate (reversed CS order), matching
/// the probe's `get_increment()` -> `[2000000.0, array([1.]), array([...])]`
/// layout (spectral, stokes, direction).
fn per_coord_list(
    py: Python<'_>,
    csys: &CoordinateSystem,
    pick: impl Fn(&Coordinate) -> Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let mut entries: Vec<Bound<'_, PyAny>> = Vec::new();
    for c in csys.coords.iter().rev() {
        let v = pick(c);
        // casacore's per-coordinate getters: SpectralCoordinate returns a
        // scalar, the others a vector (a 1-axis Stokes is array([x])).
        let scalar = v.len() == 1
            && matches!(c, Coordinate::Linear { name, .. } if name.starts_with("spectral"));
        entries.push(if scalar {
            v[0].into_pyobject(py).map(|b| b.into_any())?
        } else {
            PyArray1::from_vec(py, v).into_any()
        });
    }
    Ok(PyList::new(py, entries)?.into_any().unbind())
}

/// Normalise pyrap's mixed per-coordinate list (scalars or sequences) into
/// one Vec per coordinate.
fn per_coord_values(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for entry in value.try_iter()? {
        let entry = entry?;
        if let Ok(v) = entry.extract::<f64>() {
            out.push(vec![v]);
        } else {
            out.push(entry.extract::<Vec<f64>>()?);
        }
    }
    Ok(out)
}

/// Build the `casacure.images` submodule (see `quanta_submodule`).
pub(crate) fn images_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "images")?;
    // Free-threading declaration, as the parent module (see `casacure`).
    m.gil_used(false)?;
    m.add_class::<image>()?;
    m.add_class::<coordinates>()?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("casacure.images", &m)?;
    Ok(())
}
