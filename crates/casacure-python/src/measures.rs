//! The `casacure.measures` module: a drop-in for the python-casacore
//! `casacore.measures` surface that DDFacet and killMS use
//! (`casacore.measures.measures`), backed by `casacure::measures`.
//!
//! The seven calls exercised by `../DDFacet` and `../killMS` are exposed
//! here: `measures()` constructor, `direction`, `position`, `epoch`,
//! `do_frame`, `posangle`, `measure`, `get_value`. See
//! `PORTING_DDFACET_KILLMS.md` §1.6 for the audit and the semantics pinned
//! to real python-casacore 3.8.1; the J2000 <-> AZEL/AZELGEO accuracy
//! contract (astropy to <1 arcsec over 1926-2126) and the documented
//! casacore divergences are in `MEASURES_ACCURACY.md`.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ::casacure::measures::{Measure, MeasureType, Measures as CoreMeasures, MeasuresError};
use ::casacure::quanta::Quantity as CoreQuantity;

use crate::quanta::Quantity;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn merr(e: MeasuresError) -> PyErr {
    match e {
        MeasuresError::MissingFrame(_) => PyRuntimeError::new_err(e.to_string()),
        MeasuresError::BadReference(_) | MeasuresError::Quantity(_) => {
            PyValueError::new_err(e.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Build a `Measure` dict value (the `{'type', 'refer', 'm0', ...}` shape
/// python-casacore returns).
fn measure_to_dict<'py>(py: Python<'py>, m: &Measure) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let (t, u0, u1, u2) = match m.mtype {
        MeasureType::Direction => ("direction", "rad", Some("rad"), None),
        MeasureType::Position => ("position", "rad", Some("rad"), Some("m")),
        MeasureType::Epoch => ("epoch", "d", None, None),
    };
    d.set_item("type", t)?;
    d.set_item("refer", &m.refer)?;
    let m0 = PyDict::new(py);
    m0.set_item("value", m.values[0])?;
    m0.set_item("unit", u0)?;
    d.set_item("m0", m0)?;
    if let Some(u) = u1 {
        let m1 = PyDict::new(py);
        m1.set_item("value", m.values[1])?;
        m1.set_item("unit", u)?;
        d.set_item("m1", m1)?;
    }
    if let Some(u) = u2 {
        let m2 = PyDict::new(py);
        m2.set_item("value", m.values[2])?;
        m2.set_item("unit", u)?;
        d.set_item("m2", m2)?;
    }
    Ok(d)
}

/// Read a `Measure` from a Python dict (the shape `direction`/`position`/
/// `epoch` return).
fn measure_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Measure> {
    let t: String = d
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("measure dict has no 'type'"))?
        .extract()?;
    let refer: String = d
        .get_item("refer")?
        .ok_or_else(|| PyValueError::new_err("measure dict has no 'refer'"))?
        .extract()?;
    let mtype = match t.as_str() {
        "direction" => MeasureType::Direction,
        "position" => MeasureType::Position,
        "epoch" => MeasureType::Epoch,
        other => return Err(PyValueError::new_err(format!("bad measure type {other}"))),
    };
    let mut values = [0.0f64; 3];
    for (i, key) in ["m0", "m1", "m2"].iter().enumerate() {
        if let Some(md) = d.get_item(*key)? {
            let md = md.cast::<PyDict>()?;
            let v: f64 = md
                .get_item("value")?
                .ok_or_else(|| PyValueError::new_err("measure component has no 'value'"))?
                .extract()?;
            values[i] = v;
        }
    }
    Ok(Measure {
        mtype,
        refer,
        values,
    })
}

/// Read a quantity from a Python value (a `casacure.quanta.Quantity`, a
/// dict `{'value', 'unit'}`, or a bare number in the measure's canonical
/// unit).
fn quantity_from_py(v: &Bound<'_, PyAny>) -> PyResult<CoreQuantity> {
    if let Ok(q) = v.extract::<PyRef<'_, Quantity>>() {
        return Ok(q.as_core());
    }
    if let Ok(d) = v.cast::<PyDict>() {
        let val: f64 = d
            .get_item("value")?
            .ok_or_else(|| PyValueError::new_err("quantity dict has no 'value'"))?
            .extract()?;
        let unit: String = match d.get_item("unit")? {
            Some(u) if !u.is_none() => u.extract()?,
            _ => String::new(),
        };
        return CoreQuantity::new(val, &unit).map_err(err);
    }
    if let Ok(f) = v.extract::<f64>() {
        return CoreQuantity::new(f, "rad").map_err(err);
    }
    Err(PyTypeError::new_err("expected a Quantity, dict or number"))
}

fn py_to_quantity<'py>(py: Python<'py>, q: &CoreQuantity) -> PyResult<Bound<'py, PyAny>> {
    // Return a `casacure.quanta.Quantity` (python-casacore's measures return
    // `Quantity` objects with `get_value([unit])`).
    let qty = Quantity::new(q.clone());
    Ok(qty.into_pyobject(py)?.into_any())
}

/// `casacure.measures.measures` — the measures server (holds the current
/// frame: position + epoch).
#[pyclass(name = "measures")]
pub struct PyMeasures {
    inner: CoreMeasures,
}

#[pymethods]
impl PyMeasures {
    #[new]
    fn new() -> PyMeasures {
        PyMeasures {
            inner: CoreMeasures::new(),
        }
    }

    /// `do_frame(m)` / `doframe(m)` — record a measure as the current frame
    /// component; returns True.
    #[pyo3(name = "do_frame")]
    fn do_frame_py(&mut self, m: &Bound<'_, PyDict>) -> PyResult<bool> {
        let m = measure_from_dict(m)?;
        Ok(self.inner.do_frame(m))
    }

    #[pyo3(name = "doframe")]
    fn doframe_py(&mut self, m: &Bound<'_, PyDict>) -> PyResult<bool> {
        self.do_frame_py(m)
    }

    /// `epoch(refer, quantity)` — a time measure. The quantity is
    /// interpreted as **days since MJD 0** (casacore's `MEpoch` semantics).
    fn epoch(&self, py: Python<'_>, refer: &str, q: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let q = quantity_from_py(q)?;
        let m = self.inner.epoch(refer, &q).map_err(merr)?;
        Ok(measure_to_dict(py, &m)?.into_any().unbind())
    }

    /// `direction(refer, v0, v1)` — a sky direction (quantities in rad, or
    /// raw radians).
    fn direction(
        &self,
        py: Python<'_>,
        refer: &str,
        v0: &Bound<'_, PyAny>,
        v1: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let q0 = quantity_from_py(v0)?;
        let q1 = quantity_from_py(v1)?;
        let m = self.inner.direction(refer, &q0, &q1).map_err(merr)?;
        Ok(measure_to_dict(py, &m)?.into_any().unbind())
    }

    /// `position(refer, v0, v1, v2)` — ITRF Cartesian metres or WGS84
    /// (lon, lat, height).
    fn position(
        &self,
        py: Python<'_>,
        refer: &str,
        v0: &Bound<'_, PyAny>,
        v1: &Bound<'_, PyAny>,
        v2: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let q0 = quantity_from_py(v0)?;
        let q1 = quantity_from_py(v1)?;
        let q2 = quantity_from_py(v2)?;
        let m = self.inner.position(refer, &q0, &q1, &q2).map_err(merr)?;
        Ok(measure_to_dict(py, &m)?.into_any().unbind())
    }

    /// `measure(m, refer)` — convert `m` to `refer` using the frame.
    fn measure(&self, py: Python<'_>, m: &Bound<'_, PyDict>, refer: &str) -> PyResult<Py<PyAny>> {
        let m = measure_from_dict(m)?;
        let out = self.inner.measure(&m, refer).map_err(merr)?;
        Ok(measure_to_dict(py, &out)?.into_any().unbind())
    }

    /// `posangle(m0, m1)` — the position angle at `m0` toward `m1`, as a
    /// quantity in degrees (casacore returns a `Quantity`).
    fn posangle(
        &self,
        py: Python<'_>,
        m0: &Bound<'_, PyDict>,
        m1: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let m0 = measure_from_dict(m0)?;
        let m1 = measure_from_dict(m1)?;
        let pa = self.inner.posangle(&m0, &m1).map_err(merr)?;
        // Return a Quantity in degrees (casacore's posangle returns a
        // quantity whose `get_value('deg')` is the angle).
        let q = CoreQuantity::new(pa.to_degrees(), "deg").map_err(err)?;
        Ok(py_to_quantity(py, &q)?.unbind())
    }

    /// `get_value(m)` — the measure's components as quantities (a list).
    fn get_value(&self, py: Python<'_>, m: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let m = measure_from_dict(m)?;
        let qs = self.inner.get_value(&m);
        let out = PyList::empty(py);
        for q in qs {
            out.append(py_to_quantity(py, &q)?)?;
        }
        Ok(out.into_any().unbind())
    }
}

/// `casacure.measures` submodule registration (mirrors
/// `quanta::quanta_submodule`).
pub fn measures_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "measures")?;
    m.gil_used(false)?;
    m.add_class::<PyMeasures>()?;
    // python-casacore exposes `is_measure` and `measures` from
    // `casacore.measures`; add the former as a trivial type check.
    #[pyfunction]
    fn is_measure(v: &Bound<'_, PyAny>) -> bool {
        v.cast::<PyDict>()
            .map(|d| d.contains("type").unwrap_or(false))
            .unwrap_or(false)
    }
    m.add_function(wrap_pyfunction!(is_measure, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("casacure.measures", &m)?;
    Ok(())
}
