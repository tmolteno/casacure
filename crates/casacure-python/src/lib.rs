//! Python bindings for casacure, exposing a python-casacore-compatible
//! `casacure.tables` surface (see `CASACORE_TO_CASA_RS.md` §2-§5): `table`,
//! `taql`, `default_ms` and the type mapping helpers.

mod convert;
mod helpers;
mod selftest;
mod table;

use ::casacure::ValueType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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
    path: &Bound<'_, PyAny>,
    tabdesc: Option<&Bound<'_, PyAny>>,
    dminfo: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let path_str = table::path_string(path)?;
    let _ = dminfo;
    let extra = match tabdesc {
        Some(d) if !d.is_none() => {
            if let Ok(dict) = d.cast::<PyDict>() {
                let rec = convert::dict_to_table_record(py, dict)?;
                Some(rec.to_json_string())
            } else {
                return Err(PyValueError::new_err("tabdesc must be a dict"));
            }
        }
        _ => None,
    };
    ::casacure::ms::default_ms(std::path::Path::new(&path_str), extra.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = table::table(
        py,
        path,
        None,
        0,
        None,
        false, // readonly: the returned main table must be writable
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
    py: Python<'_>,
    name: &str,
    path: &Bound<'_, PyAny>,
    tabdesc: Option<&Bound<'_, PyAny>>,
    dminfo: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _path_str = table::path_string(path)?;
    // Create the subtable at `path`. Like python-casacore's
    // `default_ms_subtable`, a missing tabdesc means "use the standard
    // schema for this subtable" (required_ms_desc(name)); creating a table
    // with no descriptor is not supported.
    let owned_desc = match tabdesc {
        Some(d) if !d.is_none() => None,
        _ => {
            let desc = ::casacure::ms::required_ms_desc(Some(name))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Some(table::desc_to_pydict(py, &desc, None)?)
        }
    };
    let tabdesc: Option<&Bound<'_, PyAny>> = match (&tabdesc, owned_desc.as_ref()) {
        (Some(d), _) => Some(d),
        (None, Some(o)) => Some(o.as_any()),
        (None, None) => None,
    };
    let t = table::table(
        py,
        path,
        tabdesc,
        0,
        dminfo,
        false,
        true,
        &PyTuple::empty(py),
        None,
    )?;
    Ok(t.into_pyobject(py)?.into_any().unbind())
}

/// `required_ms_desc(name=None)` -> the descriptor dict.
#[pyfunction]
#[pyo3(signature = (name = None))]
fn required_ms_desc(py: Python<'_>, name: Option<String>) -> PyResult<Py<PyAny>> {
    let desc = ::casacure::ms::required_ms_desc(name.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(table::desc_to_pydict(py, &desc, None)?.into_any().unbind())
}

/// `complete_ms_desc(name=None)` -> the descriptor dict.
#[pyfunction]
#[pyo3(signature = (name = None))]
fn complete_ms_desc(py: Python<'_>, name: Option<String>) -> PyResult<Py<PyAny>> {
    let desc = ::casacure::ms::complete_ms_desc(name.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(table::desc_to_pydict(py, &desc, None)?.into_any().unbind())
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
    m.add_function(wrap_pyfunction!(tablefromascii, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::makescacoldesc, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::makearrcoldesc, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::makecoldesc, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::maketabdesc, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::makedminfo, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::tableexists, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::tabledelete, &m)?)?;
    m.add_function(wrap_pyfunction!(helpers::tablecopy, &m)?)?;
    // Give the factories a resolvable `__module__` so dask-ms can pickle the
    // TableProxy (which pickles the factory callable).
    for name in ["table", "taql", "default_ms", "default_ms_subtable"] {
        let f = m.getattr(name)?;
        f.setattr("__module__", "casacure.tables")?;
    }
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
    // Keep in sync with `[project] version` in pyproject.toml (the wheel
    // version is 3.8.1.<n>; Cargo's crate version is semver 3.8.1 and cannot
    // carry a fourth part).
    const PACKAGE_VERSION: &str = "3.8.1.1";
    m.add("__version__", PACKAGE_VERSION)?;
    m.add_function(wrap_pyfunction!(numpy_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(casa_type, m)?)?;
    m.add_function(wrap_pyfunction!(selftest::run_tests, m)?)?;
    m.add_function(wrap_pyfunction!(selftest::run_benchmark, m)?)?;
    tables_submodule(m)?;
    Ok(())
}

/// `tablefromascii(path, ascii_desc)` — create a table from a simple ASCII
/// table description (dask-ms's test-only path): a header line of column
/// names, a line of type letters (`R` float, `D` double, `I` int,
/// `X<base>,<digits>` complex), then whitespace-separated data rows.
#[pyfunction]
#[pyo3(signature = (path, ascii_file, ack = true))]
fn tablefromascii(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    ascii_file: &Bound<'_, PyAny>,
    ack: bool,
) -> PyResult<Py<PyAny>> {
    let _ = ack;
    let path = table::path_string(path)?;
    let ascii_file = table::path_string(ascii_file)?;
    use numpy::Complex64;
    let text = std::fs::read_to_string(&ascii_file)
        .map_err(|e| PyValueError::new_err(format!("cannot read {ascii_file}: {e}")))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| PyValueError::new_err("empty ascii table"))?;
    let types = lines
        .next()
        .ok_or_else(|| PyValueError::new_err("missing type line"))?;
    let names: Vec<String> = header.split_whitespace().map(str::to_string).collect();
    let types: Vec<String> = types.split_whitespace().map(str::to_string).collect();
    if names.len() != types.len() {
        return Err(PyValueError::new_err("header/type column count mismatch"));
    }

    // Column descriptors (valueType per type letter).
    let desc = PyDict::new(py);
    for (name, ty) in names.iter().zip(types.iter()) {
        let vt = match ty.chars().next() {
            Some('R') => "float",
            Some('D') => "double",
            Some('I') => "int",
            Some('X') => "dcomplex",
            Some('S') => "string",
            _ => return Err(PyValueError::new_err(format!("unknown ascii type {ty}"))),
        };
        let col = PyDict::new(py);
        col.set_item("valueType", vt)?;
        col.set_item("dataManagerType", "StandardStMan")?;
        col.set_item("dataManagerGroup", "StandardStMan")?;
        col.set_item("option", 0)?;
        col.set_item("maxlen", 0)?;
        col.set_item("comment", "")?;
        let kw = PyDict::new(py);
        col.set_item("keywords", kw)?;
        desc.set_item(name, col)?;
    }
    let tdesc = desc.into_any();
    let t = {
        let name = pyo3::types::PyString::new(py, &path);
        table::table(
            py,
            name.as_any(),
            Some(&tdesc),
            0,
            None,
            false,
            true,
            &PyTuple::empty(py),
            None,
        )?
    };

    // Data rows -> one typed numpy array per column.
    let rows: Vec<&str> = lines.collect();
    let toks: Vec<Vec<&str>> = rows
        .iter()
        .map(|r| r.split_whitespace().collect())
        .collect();
    // Per-column start token offset: complex columns consume two tokens.
    let widths: Vec<usize> = types
        .iter()
        .map(|ty| if ty.starts_with('X') { 2 } else { 1 })
        .collect();
    let mut starts = Vec::with_capacity(widths.len());
    let mut off = 0usize;
    for w in &widths {
        starts.push(off);
        off += w;
    }
    let tobj = t.into_pyobject(py)?;
    if !rows.is_empty() {
        tobj.call_method1("addrows", (rows.len(),))?;
    }
    for (ci, (name, ty)) in names.iter().zip(types.iter()).enumerate() {
        let first = ty.chars().next().unwrap_or('S');
        let start = starts[ci];
        let width = widths[ci];
        let obj: Py<PyAny> = match first {
            'I' => {
                let vals: Vec<i32> = toks.iter().map(|k| k[start].parse().unwrap_or(0)).collect();
                numpy::PyArray1::from_vec(py, vals).into_any().unbind()
            }
            'R' | 'D' => {
                let vals: Vec<f64> = toks
                    .iter()
                    .map(|k| k[start].parse().unwrap_or(0.0))
                    .collect();
                numpy::PyArray1::from_vec(py, vals).into_any().unbind()
            }
            'X' => {
                let vals: Vec<Complex64> = toks
                    .iter()
                    .map(|k| {
                        let re: f64 = k[start].parse().unwrap_or(0.0);
                        let im: f64 = k[start + 1].parse().unwrap_or(0.0);
                        Complex64::new(re, im)
                    })
                    .collect();
                numpy::PyArray1::from_vec(py, vals).into_any().unbind()
            }
            _ => {
                let vals: Vec<String> = toks.iter().map(|k| k[start].to_string()).collect();
                let list = PyList::new(py, vals)?;
                list.into_any().unbind()
            }
        };
        let _ = width;
        tobj.call_method1("putcol", (name, obj, 0, 0))?;
    }
    tobj.call_method0("flush")?;
    Ok(tobj.into_any().unbind())
}
