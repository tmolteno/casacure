//! Module-level `casacore.tables` helper functions that real recipients of
//! the python-casacore API rely on: the column/table-description builders
//! (`makescacoldesc`, `makearrcoldesc`, `makecoldesc`, `maketabdesc`,
//! `makedminfo`) and the table lifecycle helpers (`tableexists`,
//! `tabledelete`, `tablecopy`). Semantics mirror python-casacore 3.8.1's
//! `casacore/tables.py` + `casacore.tables` C++ helpers.

use std::collections::HashMap;

use ::casacure::record::RecordValue;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::table;

/// python-casacore `_value_type_name`: the CASA valueType for a Python value.
fn value_type_name(v: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    // bool must be checked before int (bool is an int subclass).
    if v.is_instance_of::<PyBool>() {
        return Ok("boolean");
    }
    if v.cast::<PyInt>().is_ok() {
        return Ok("int");
    }
    if v.cast::<PyFloat>().is_ok() {
        return Ok("double");
    }
    if v.cast::<PyComplex>().is_ok() {
        return Ok("dcomplex");
    }
    if v.cast::<PyString>().is_ok() {
        return Ok("string");
    }
    if v.cast::<PyDict>().is_ok() {
        return Ok("record");
    }
    Err(PyTypeError::new_err(format!(
        "Value type could not be derived for {}",
        v.repr()?
    )))
}

/// `makescacoldesc(name, value, ...)` — scalar column description dict
/// (mirrors python-casacore's `casacore/tables.py`).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (columnname, value, datamanagertype = "", datamanagergroup = "", options = 0, maxlen = 0, comment = "", valuetype = "", keywords = None))]
pub fn makescacoldesc(
    py: Python<'_>,
    columnname: String,
    value: &Bound<'_, PyAny>,
    datamanagertype: &str,
    datamanagergroup: &str,
    options: i64,
    maxlen: i64,
    comment: &str,
    valuetype: &str,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let vtype = if valuetype.is_empty() {
        value_type_name(value)?.to_string()
    } else {
        valuetype.to_string()
    };
    let rec = PyDict::new(py);
    rec.set_item("valueType", vtype)?;
    rec.set_item("dataManagerType", datamanagertype)?;
    rec.set_item("dataManagerGroup", datamanagergroup)?;
    rec.set_item("option", options)?;
    rec.set_item("maxlen", maxlen)?;
    rec.set_item("comment", comment)?;
    match keywords {
        Some(k) => rec.set_item("keywords", k)?,
        None => rec.set_item("keywords", PyDict::new(py))?,
    }
    let out = PyDict::new(py);
    out.set_item("name", columnname)?;
    out.set_item("desc", rec)?;
    Ok(out.into_any().unbind())
}

/// `makearrcoldesc(name, value, ndim=0, shape=[], ...)` — array column
/// description dict (mirrors python-casacore's `casacore/tables.py`).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (columnname, value, ndim = 0, shape = None, datamanagertype = "", datamanagergroup = "", options = 0, maxlen = 0, comment = "", valuetype = "", keywords = None))]
pub fn makearrcoldesc(
    py: Python<'_>,
    columnname: String,
    value: &Bound<'_, PyAny>,
    ndim: i64,
    shape: Option<&Bound<'_, PyAny>>,
    datamanagertype: &str,
    datamanagergroup: &str,
    options: i64,
    maxlen: i64,
    comment: &str,
    valuetype: &str,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let vtype = if valuetype.is_empty() {
        value_type_name(value)?.to_string()
    } else {
        valuetype.to_string()
    };
    let shape_vec: Vec<i64> = match shape {
        Some(s) => s.extract()?,
        None => Vec::new(),
    };
    let ndim = if shape_vec.is_empty() {
        ndim
    } else if ndim <= 0 {
        shape_vec.len() as i64
    } else {
        ndim
    };
    let rec = PyDict::new(py);
    rec.set_item("valueType", vtype)?;
    rec.set_item("dataManagerType", datamanagertype)?;
    rec.set_item("dataManagerGroup", datamanagergroup)?;
    rec.set_item("ndim", ndim)?;
    rec.set_item("shape", shape_vec.into_pyobject(py)?.into_any())?;
    rec.set_item("_c_order", true)?;
    rec.set_item("option", options)?;
    rec.set_item("maxlen", maxlen)?;
    rec.set_item("comment", comment)?;
    match keywords {
        Some(k) => rec.set_item("keywords", k)?,
        None => rec.set_item("keywords", PyDict::new(py))?,
    }
    let out = PyDict::new(py);
    out.set_item("name", columnname)?;
    out.set_item("desc", rec)?;
    Ok(out.into_any().unbind())
}

/// `makecoldesc(columnname, desc)` — column description from another desc.
#[pyfunction]
pub fn makecoldesc(
    py: Python<'_>,
    columnname: String,
    desc: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    out.set_item("name", columnname)?;
    out.set_item("desc", desc)?;
    Ok(out.into_any().unbind())
}

/// `maketabdesc(descs)` — merge column descriptions into a table description
/// dict (raises `ValueError` on a repeated column name).
#[pyfunction]
#[pyo3(signature = (descs = None))]
pub fn maketabdesc(py: Python<'_>, descs: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    let items: Vec<Bound<'_, PyAny>> = match descs {
        None => Vec::new(),
        Some(d) => {
            if let Ok(_dict) = d.cast::<PyDict>() {
                vec![d.clone()]
            } else if let Ok(list) = d.cast::<PyList>() {
                list.iter().collect()
            } else if let Ok(tup) = d.cast::<PyTuple>() {
                tup.iter().collect()
            } else {
                return Err(PyTypeError::new_err(
                    "maketabdesc expects a dict or a list of column descriptions",
                ));
            }
        }
    };
    for item in items {
        let name: String = item.get_item("name")?.extract()?;
        if out.contains(name.as_str())? {
            return Err(PyValueError::new_err(format!(
                "Column name {name} multiply used in table description"
            )));
        }
        out.set_item(name.as_str(), item.get_item("desc")?)?;
    }
    Ok(out.into_any().unbind())
}

/// `makedminfo(tabdesc, group_spec=None)` — build a data-manager info dict
/// from a table description, grouping columns by `dataManagerGroup`
/// (mirrors python-casacore's `casacore/tables.py`).
#[pyfunction]
#[pyo3(signature = (tabdesc, group_spec = None))]
pub fn makedminfo(
    py: Python<'_>,
    tabdesc: &Bound<'_, PyDict>,
    group_spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    struct Group {
        columns: Vec<String>,
        ty: Option<String>,
        spec: Option<Py<PyAny>>,
    }
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Group> = HashMap::new();
    for (k, v) in tabdesc.iter() {
        let c: String = match k.extract() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if matches!(
            c.as_str(),
            "_define_hypercolumn_" | "_keywords_" | "_private_keywords_"
        ) {
            continue;
        }
        let d = match v.cast::<PyDict>() {
            Ok(dd) => dd,
            Err(_) => continue,
        };
        let mut group: String = d
            .get_item("dataManagerGroup")?
            .and_then(|x| x.extract().ok())
            .unwrap_or_else(|| "StandardStMan".to_string());
        let mut ty: String = d
            .get_item("dataManagerType")?
            .and_then(|x| x.extract().ok())
            .unwrap_or_else(|| "StandardStMan".to_string());
        if group.is_empty() {
            group = "StandardStMan".to_string();
        }
        if ty.is_empty() {
            ty = "StandardStMan".to_string();
        }
        let entry = groups.entry(group.clone()).or_insert_with(|| {
            order.push(group.clone());
            Group {
                columns: Vec::new(),
                ty: None,
                spec: None,
            }
        });
        entry.columns.push(c);
        match &entry.ty {
            None => entry.ty = Some(ty.clone()),
            Some(prev) if prev != &ty => {
                return Err(PyValueError::new_err(format!(
                    "Mismatched dataManagerType '{}' for dataManagerGroup '{}' Previously, the type was '{}'",
                    ty, group, prev
                )))
            }
            _ => {}
        }
        if entry.spec.is_none() {
            entry.spec = group_spec
                .and_then(|gs| gs.get_item(&group).ok().flatten())
                .map(|s| s.unbind());
        }
    }
    let out = PyDict::new(py);
    for (i, gname) in order.iter().enumerate() {
        let g = groups.get(gname).unwrap();
        let dm = PyDict::new(py);
        dm.set_item("COLUMNS", g.columns.clone().into_pyobject(py)?.into_any())?;
        dm.set_item("TYPE", g.ty.as_deref().unwrap_or("StandardStMan"))?;
        dm.set_item("NAME", gname.as_str())?;
        match &g.spec {
            Some(spec) => dm.set_item("SPEC", spec)?,
            None => dm.set_item("SPEC", PyDict::new(py))?,
        }
        dm.set_item("SEQNR", i as i64)?;
        out.set_item(format!("*{}", i + 1), dm)?;
    }
    Ok(out.into_any().unbind())
}

/// `tableexists(tablename)` — whether `name` is a readable CASA table.
#[pyfunction]
pub fn tableexists(name: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = table::path_string(name)?;
    let p = std::path::Path::new(&name);
    Ok(p.is_dir() && p.join("table.dat").is_file())
}

/// `tabledelete(tablename, checksubtables=False, ack=True)` — delete the
/// table directory.
#[pyfunction]
#[pyo3(signature = (tablename, checksubtables = false, ack = true))]
pub fn tabledelete(
    py: Python<'_>,
    tablename: &Bound<'_, PyAny>,
    checksubtables: bool,
    ack: bool,
) -> PyResult<()> {
    let _ = py;
    let _ = checksubtables;
    let _ = ack;
    let name = table::path_string(tablename)?;
    let p = std::path::Path::new(&name);
    if !p.exists() {
        return Err(PyValueError::new_err(format!(
            "Table {} does not exist",
            name
        )));
    }
    std::fs::remove_dir_all(p)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot delete {name}: {e}")))?;
    Ok(())
}

/// Recursively copy a table directory (skipping lock files).
fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> PyResult<()> {
    std::fs::create_dir_all(dst)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot create {dst:?}: {e}")))?;
    for entry in std::fs::read_dir(src)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot read {src:?}: {e}")))?
    {
        let entry = entry.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if fname.starts_with('.') || fname.ends_with(".lock") {
            continue;
        }
        let ty = entry
            .file_type()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dst_p = dst.join(fname.as_ref());
        if ty.is_dir() {
            copy_dir(&entry.path(), &dst_p)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), &dst_p)
                .map_err(|e| PyRuntimeError::new_err(format!("cannot copy to {dst_p:?}: {e}")))?;
        }
    }
    // Normalise the copied row count in `dst/table.dat` (see
    // `casacure::patch_copy_nrow`).  A table written by a legacy writer can
    // carry a stale `0` header row count with the real count only in the
    // lock file's sync record; a byte copy that skips the lock would then
    // reopen the copy as an empty table.  Real casacore's `table.copy`
    // re-writes the row count, so we do the same so the copy is
    // self-consistent.
    if src.join("table.dat").is_file() {
        if let Ok(t) = ::casacure::Table::open(src, true) {
            let _ = ::casacure::patch_copy_nrow(dst, t.nrows());
        }
    }
    Ok(())
}

/// Collect the relative subtable paths referenced by a table's keywords.
fn subtable_paths(t: &::casacure::Table) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(v: &RecordValue, out: &mut Vec<String>) {
        match v {
            RecordValue::Table(s) => out.push(s.clone()),
            RecordValue::Record(r) => {
                for v in &r.values {
                    walk(v, out);
                }
            }
            _ => {}
        }
    }
    for v in &t.dat.desc.keywords.values {
        walk(v, &mut out);
    }
    for v in &t.dat.desc.private_keywords.values {
        walk(v, &mut out);
    }
    out
}

/// `tablecopy(tablename, newtablename, deep=False, valuecopy=False, ...)` —
/// copy a table on disk (a full recursive copy of the table directory, so
/// data is included regardless of the `valuecopy` flag). With `deep`, the
/// subtable directories referenced by `Table:` keywords are copied too.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (tablename, newtablename, deep = false, valuecopy = false, dminfo = None, _endian = "aipsrc", _memorytable = false, _copynorows = false))]
pub fn tablecopy(
    py: Python<'_>,
    tablename: &Bound<'_, PyAny>,
    newtablename: &Bound<'_, PyAny>,
    deep: bool,
    valuecopy: bool,
    dminfo: Option<&Bound<'_, PyAny>>,
    _endian: &str,
    _memorytable: bool,
    _copynorows: bool,
) -> PyResult<()> {
    let _ = py;
    let src = table::path_string(tablename)?;
    let dst = table::path_string(newtablename)?;
    let sp = std::path::Path::new(&src);
    let dp = std::path::Path::new(&dst);
    if !sp.is_dir() {
        return Err(PyValueError::new_err(format!(
            "Table {} does not exist",
            src
        )));
    }
    if dp.exists() {
        return Err(PyRuntimeError::new_err(format!(
            "Table {} already exists",
            dst
        )));
    }
    let _ = valuecopy;
    let _ = dminfo;
    copy_dir(sp, dp)?;
    if deep {
        // Breadth-first copy of every subtable referenced by a `Table:`
        // keyword, into its relative position next to the copy. Subtable
        // paths are stored relative to the table that owns the keyword.
        let mut copied: Vec<std::path::PathBuf> = vec![sp.to_path_buf()];
        let mut queue: Vec<(std::path::PathBuf, std::path::PathBuf)> =
            vec![(sp.to_path_buf(), dp.to_path_buf())];
        while let Some((sdir, ddir)) = queue.pop() {
            let t = match ::casacure::Table::open(&sdir, true) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let sbase = sdir.parent().unwrap_or(std::path::Path::new("/"));
            let dbase = ddir.parent().unwrap_or(std::path::Path::new("/"));
            for sub in subtable_paths(&t) {
                let rel = std::path::Path::new(&sub);
                let ssub = if rel.is_absolute() {
                    rel.to_path_buf()
                } else {
                    sbase.join(rel)
                };
                let dsub = if rel.is_absolute() {
                    rel.to_path_buf()
                } else {
                    dbase.join(rel)
                };
                // Copying a table into the same parent directory leaves the
                // subtable reference pointing at the same location, so the
                // subtable is already in place — copying it onto itself would
                // truncate the data files.
                if ssub == dsub {
                    continue;
                }
                if ssub.is_dir() && !copied.contains(&ssub) {
                    copy_dir(&ssub, &dsub)?;
                    copied.push(ssub.clone());
                    queue.push((ssub, dsub));
                }
            }
        }
    }
    Ok(())
}
