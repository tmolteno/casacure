//! Python bindings for casacure, exposing a python-casacore-compatible
//! `casacure.tables` surface (see `CASACORE_TO_CASA_RS.md` §2-§5): `table`,
//! `taql`, `default_ms` and the type mapping helpers.

mod convert;
mod table;

use ::casacure::ValueType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Map a CASA type name (any alias) to its numpy dtype name.
///
/// Mirrors dask-ms's `_TABLE_TO_PY` mapping.
#[pyfunction]
fn numpy_dtype(casa_type: &str) -> PyResult<String> {
    ValueType::from_casa_name(casa_type)
        .map(|vt| vt.numpy_name().to_string())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Map a numpy dtype name back to the canonical CASA type name.
///
/// Mirrors dask-ms's `_PY_TO_TABLE` mapping.
#[pyfunction]
fn casa_type(numpy_dtype: &str) -> PyResult<String> {
    let vt = ValueType::ALL
        .iter()
        .find(|vt| vt.numpy_name() == numpy_dtype)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("unknown numpy dtype: {numpy_dtype:?}")))?;
    Ok(vt.casa_name().to_string())
}

/// `default_ms(path, tabdesc=None, dminfo=None)` — create the full MS tree
/// (main table + the 12 standard subtables), mirroring casacore. Returns the
/// main table (usable as a context manager).
#[pyfunction]
#[pyo3(signature = (path, tabdesc = None, dminfo = None))]
fn default_ms(
    py: Python<'_>,
    path: &str,
    tabdesc: Option<&Bound<'_, PyAny>>,
    dminfo: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = dminfo;
    let extra = match tabdesc {
        Some(d) if !d.is_none() => {
            if let Ok(dict) = d.downcast::<PyDict>() {
                let rec = convert::dict_to_table_record(py, dict)?;
                Some(rec.to_json_string())
            } else {
                return Err(PyValueError::new_err("tabdesc must be a dict"));
            }
        }
        _ => None,
    };
    ::casacure::ms::default_ms(std::path::Path::new(path), extra.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = table::table(
        py,
        path,
        None,
        0,
        None,
        true,
        true,
        &PyTuple::empty(py),
        None,
    )?;
    Ok(t.into_pyobject(py)?.into_any().unbind())
}

/// `default_ms_subtable(name, path, tabdesc=None, dminfo=None)` — create one
/// MS subtable table.
#[pyfunction]
#[pyo3(signature = (name, path, tabdesc = None, dminfo = None))]
fn default_ms_subtable(
    name: &str,
    path: &str,
    tabdesc: Option<&Bound<'_, PyAny>>,
    dminfo: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let _ = (tabdesc, dminfo);
    ::casacure::ms::default_ms_subtable(name, std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `required_ms_desc(name=None)` -> the descriptor dict.
#[pyfunction]
#[pyo3(signature = (name = None))]
fn required_ms_desc(py: Python<'_>, name: Option<String>) -> PyResult<Py<PyAny>> {
    let desc = ::casacure::ms::required_ms_desc(name.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(table::desc_to_pydict(py, &desc)?.into_any().unbind())
}

/// `complete_ms_desc(name=None)` -> the descriptor dict.
#[pyfunction]
#[pyo3(signature = (name = None))]
fn complete_ms_desc(py: Python<'_>, name: Option<String>) -> PyResult<Py<PyAny>> {
    let desc = ::casacure::ms::complete_ms_desc(name.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(table::desc_to_pydict(py, &desc)?.into_any().unbind())
}

/// The `tables` submodule (drop-in for `casacore.tables`).
fn tables_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "tables")?;
    m.add_class::<table::Table>()?;
    m.add_function(wrap_pyfunction!(table::table, &m)?)?;
    m.add_function(wrap_pyfunction!(table::taql, &m)?)?;
    m.add_function(wrap_pyfunction!(default_ms, &m)?)?;
    m.add_function(wrap_pyfunction!(default_ms_subtable, &m)?)?;
    m.add_function(wrap_pyfunction!(required_ms_desc, &m)?)?;
    m.add_function(wrap_pyfunction!(complete_ms_desc, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("casacure.tables", &m)?;
    Ok(())
}

#[pymodule]
fn casacure(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(numpy_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(casa_type, m)?)?;
    tables_submodule(m)?;
    Ok(())
}
