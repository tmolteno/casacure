//! The `casacure.images` pyo3 module (issue #14): the `pyrap.images`
//! surface DDFacet and killMS consume — the `image` class (open CASA image
//! tables and FITS cubes) and the `coordinates` object it hands back.
//! Registered by `images_submodule` like `quanta`/`measures`.

use std::sync::RwLock;

use pyo3::exceptions::{PyAttributeError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

use crate::convert;
use numpy::PyArray1;

use casacure::images::{Coordinate, CoordinateSystem, Image};

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
        imagename: Option<&Bound<'_, PyAny>>,
        shape: Option<Vec<usize>>,
        coordsys: Option<&Bound<'_, PyAny>>,
        overwrite: bool,
    ) -> PyResult<Self> {
        // Creation (imagename + shape [+ coordsys]) is Phase 2 of the port
        // (it needs the write path: raster table + coords records + FITS
        // writer); every live DDFacet read path only opens.
        if shape.is_some() {
            return Err(PyNotImplementedError::new_err(
                "casacure.images: image creation is not implemented yet (issue #14 phase 2)",
            ));
        }
        let Some(path_arg) = imagename else {
            return Err(PyValueError::new_err(
                "image() needs a path (or imagename= + shape=)",
            ));
        };
        let path_str: String = if let Ok(s) = path_arg.extract::<String>() {
            s
        } else {
            path_arg.call_method0("__fspath__")?.extract()?
        };
        let path = casacure::table::absolute_dir(std::path::Path::new(&path_str));
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

    // Phase 2 of the port (write path).
    fn putdata(&self, _data: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "casacure.images: putdata is not implemented yet (issue #14 phase 2)",
        ))
    }

    fn saveas(&self, _filename: &str) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "casacure.images: saveas is not implemented yet (issue #14 phase 2)",
        ))
    }

    fn tofits(&self, _filename: &str) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "casacure.images: tofits is not implemented yet (issue #14 phase 2)",
        ))
    }

    fn regrid(&self, _axes: &Bound<'_, PyAny>, _coordsys: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "casacure.images: regrid is not implemented yet (issue #14 phase 3)",
        ))
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
