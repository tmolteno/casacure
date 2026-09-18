//! Python bindings for casacure, exposing a python-casacore-compatible
//! `casacure.tables` surface (see `CASACORE_TO_CASA_RS.md` §2-§3).

use ::casacure::ValueType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

/// casacure: Rust implementation of the CASA table system.
#[pymodule]
fn casacure(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(numpy_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(casa_type, m)?)?;
    Ok(())
}
