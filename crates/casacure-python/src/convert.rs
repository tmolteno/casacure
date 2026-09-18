//! Conversions between casacure's Rust values and Python objects / numpy
//! arrays, matching the python-casacore / dask-ms surface.

use casacure::record::{ArrayData, ArrayValue, DataType, RecordValue, TableRecord};
use numpy::PyArrayMethods;
use numpy::{Complex32, Complex64};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

// ---------------------------------------------------------------------------
// Shapes / index transposition
// ---------------------------------------------------------------------------

/// Logical (externally visible) shape from the stored CASA shape: reversed.
/// Write a column of cells into a flat C-order buffer. Cells are returned
/// in the shape they are stored (which, for casacore files, is the logical
/// shape; the descriptor shape is reversed metadata only).
pub(crate) fn fill_flat<T: Copy + Default>(
    buf: &mut [T],
    cells: &[RecordValue],
    cell: usize,
    map: impl Fn(&ArrayData, usize) -> T,
    scalar: impl Fn(&RecordValue) -> T,
) -> PyResult<()> {
    for (r, c) in cells.iter().enumerate() {
        if let RecordValue::Array(a) = c {
            let base = r * cell;
            let n = array_len(&a.data);
            for i in 0..cell {
                if base + i < buf.len() && i < n {
                    buf[base + i] = map(&a.data, i);
                }
            }
        } else if r < buf.len() {
            buf[r] = scalar(c);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reading cells into a provided numpy buffer (getcolnp / getcolslicenp)
// ---------------------------------------------------------------------------

fn f64_of(d: &ArrayData, i: usize) -> f64 {
    match d {
        ArrayData::Double(v) => v[i],
        ArrayData::Float(v) => f64::from(v[i]),
        ArrayData::Int(v) => f64::from(v[i]),
        _ => 0.0,
    }
}
fn f32_of(d: &ArrayData, i: usize) -> f32 {
    match d {
        ArrayData::Float(v) => v[i],
        ArrayData::Double(v) => v[i] as f32,
        _ => 0.0,
    }
}
fn i64_of(d: &ArrayData, i: usize) -> i64 {
    match d {
        ArrayData::Int64(v) => v[i],
        ArrayData::Int(v) => i64::from(v[i]),
        ArrayData::Short(v) => i64::from(v[i]),
        ArrayData::UChar(v) => i64::from(v[i]),
        _ => 0,
    }
}
fn i32_of(d: &ArrayData, i: usize) -> i32 {
    match d {
        ArrayData::Int(v) => v[i],
        ArrayData::Int64(v) => v[i] as i32,
        ArrayData::Short(v) => i32::from(v[i]),
        ArrayData::UChar(v) => i32::from(v[i]),
        ArrayData::Bool(v) => i32::from(v[i]),
        _ => 0,
    }
}
fn i16_of(d: &ArrayData, i: usize) -> i16 {
    match d {
        ArrayData::Short(v) => v[i],
        ArrayData::Int(v) => v[i] as i16,
        _ => 0,
    }
}
fn u64_of(d: &ArrayData, i: usize) -> u64 {
    match d {
        ArrayData::UInt(v) => u64::from(v[i]),
        ArrayData::Int(v) => v[i] as u64,
        _ => 0,
    }
}
fn u32_of(d: &ArrayData, i: usize) -> u32 {
    match d {
        ArrayData::UInt(v) => v[i],
        ArrayData::Int(v) => v[i] as u32,
        _ => 0,
    }
}
fn u16_of(d: &ArrayData, i: usize) -> u16 {
    match d {
        ArrayData::UShort(v) => v[i],
        ArrayData::UChar(v) => u16::from(v[i]),
        _ => 0,
    }
}
fn u8_of(d: &ArrayData, i: usize) -> u8 {
    match d {
        ArrayData::UChar(v) => v[i],
        _ => 0,
    }
}
fn bool_of(d: &ArrayData, i: usize) -> bool {
    match d {
        ArrayData::Bool(v) => v[i],
        ArrayData::UChar(v) => v[i] != 0,
        _ => false,
    }
}
fn c64_of(d: &ArrayData, i: usize) -> Complex64 {
    match d {
        ArrayData::DComplex(v) => Complex64::new(v[i].0, v[i].1),
        ArrayData::Complex(v) => Complex64::new(f64::from(v[i].0), f64::from(v[i].1)),
        _ => Complex64::new(0.0, 0.0),
    }
}
fn c32_of(d: &ArrayData, i: usize) -> Complex32 {
    match d {
        ArrayData::Complex(v) => Complex32::new(v[i].0, v[i].1),
        ArrayData::DComplex(v) => Complex32::new(v[i].0 as f32, v[i].1 as f32),
        _ => Complex32::new(0.0, 0.0),
    }
}

/// Fill an existing numpy buffer with a column's cells.
pub(crate) fn fill_buffer_by_dtype(
    buf: &Bound<'_, PyAny>,
    cells: &[RecordValue],
    cell: usize,
) -> PyResult<()> {
    macro_rules! fill_num {
        ($ty:ty, $f:expr, $s:expr) => {{
            if let Ok(arr) = buf.downcast::<numpy::PyArrayDyn<$ty>>() {
                let mut b = arr.readwrite();
                let s = b
                    .as_slice_mut()
                    .map_err(|_| PyValueError::new_err("getcolnp: non-contiguous buffer"))?;
                return fill_flat(s, cells, cell, $f, $s);
            }
        }};
    }
    fill_num!(f64, f64_of, scalar_f64);
    fill_num!(f32, f32_of, scalar_f32);
    fill_num!(i64, i64_of, scalar_i64);
    fill_num!(i32, i32_of, scalar_i32);
    fill_num!(i16, i16_of, scalar_i16);
    fill_num!(u64, u64_of, scalar_u64);
    fill_num!(u32, u32_of, scalar_u32);
    fill_num!(u16, u16_of, scalar_u16);
    fill_num!(u8, u8_of, scalar_u8);
    fill_num!(bool, bool_of, scalar_bool);
    fill_num!(Complex64, c64_of, scalar_c64);
    fill_num!(Complex32, c32_of, scalar_c32);
    Err(PyTypeError::new_err(format!(
        "getcolnp: unsupported buffer dtype {}",
        buf.getattr("dtype")?.str()?.to_str()?
    )))
}

// ---------------------------------------------------------------------------
// Building fresh numpy arrays from columns of cells
// ---------------------------------------------------------------------------

fn reshape_from<T: numpy::Element>(
    py: Python<'_>,
    vals: Vec<T>,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    let arr = numpy::PyArray1::from_vec(py, vals);
    let r = arr.call_method1("reshape", (shape.to_vec(),))?;
    Ok(r.into_any().unbind())
}

/// Build a fresh 1-D numpy array (or list for strings) from scalar cells.
pub(crate) fn scalars_cells_to_array(py: Python<'_>, cells: &[RecordValue]) -> PyResult<Py<PyAny>> {
    if let RecordValue::Bool(_) = cells.first().unwrap_or(&RecordValue::Int(0)) {
        let vec: Vec<bool> = cells
            .iter()
            .map(|v| matches!(v, RecordValue::Bool(true)))
            .collect();
        return Ok(numpy::PyArray1::from_vec(py, vec).into_any().unbind());
    }
    if let RecordValue::Double(_) = cells.first().unwrap_or(&RecordValue::Int(0)) {
        let vec: Vec<f64> = cells
            .iter()
            .map(|v| match v {
                RecordValue::Double(d) => *d,
                RecordValue::Float(f) => f64::from(*f),
                _ => 0.0,
            })
            .collect();
        return Ok(numpy::PyArray1::from_vec(py, vec).into_any().unbind());
    }
    if let RecordValue::Float(_) = cells.first().unwrap_or(&RecordValue::Int(0)) {
        let vec: Vec<f32> = cells
            .iter()
            .map(|v| match v {
                RecordValue::Float(f) => *f,
                _ => 0.0,
            })
            .collect();
        return Ok(numpy::PyArray1::from_vec(py, vec).into_any().unbind());
    }
    if let RecordValue::Int64(_) = cells.first().unwrap_or(&RecordValue::Int(0)) {
        let vec: Vec<i64> = cells
            .iter()
            .map(|v| match v {
                RecordValue::Int64(i) => *i,
                RecordValue::Int(i) => i64::from(*i),
                RecordValue::UChar(u) => i64::from(*u),
                _ => 0,
            })
            .collect();
        return Ok(numpy::PyArray1::from_vec(py, vec).into_any().unbind());
    }
    if let RecordValue::UChar(_) = cells.first().unwrap_or(&RecordValue::Int(0)) {
        let vec: Vec<u8> = cells
            .iter()
            .map(|v| match v {
                RecordValue::UChar(u) => *u,
                _ => 0,
            })
            .collect();
        return Ok(numpy::PyArray1::from_vec(py, vec).into_any().unbind());
    }
    if let RecordValue::String(_) | RecordValue::Table(_) =
        cells.first().unwrap_or(&RecordValue::Int(0))
    {
        let list = string_list(py, cells)?;
        return Ok(list.into_any().unbind());
    }
    let vec: Vec<i32> = cells
        .iter()
        .map(|v| match v {
            RecordValue::Int(i) => *i,
            RecordValue::Int64(i) => *i as i32,
            _ => 0,
        })
        .collect();
    Ok(numpy::PyArray1::from_vec(py, vec).into_any().unbind())
}

fn string_list<'py>(py: Python<'py>, cells: &[RecordValue]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for v in cells {
        let s = match v {
            RecordValue::String(s) | RecordValue::Table(s) => s.clone(),
            other => other.to_json_string(),
        };
        list.append(s)?;
    }
    Ok(list)
}

fn array_len(d: &ArrayData) -> usize {
    match d {
        ArrayData::Bool(v) => v.len(),
        ArrayData::UChar(v) => v.len(),
        ArrayData::Short(v) => v.len(),
        ArrayData::UShort(v) => v.len(),
        ArrayData::Int(v) => v.len(),
        ArrayData::UInt(v) => v.len(),
        ArrayData::Int64(v) => v.len(),
        ArrayData::Float(v) => v.len(),
        ArrayData::Double(v) => v.len(),
        ArrayData::Complex(v) => v.len(),
        ArrayData::DComplex(v) => v.len(),
        ArrayData::String(v) => v.len(),
    }
}

fn scalar_f64(v: &RecordValue) -> f64 {
    match v {
        RecordValue::Double(d) => *d,
        RecordValue::Float(f) => f64::from(*f),
        RecordValue::Int(i) => f64::from(*i),
        RecordValue::Bool(b) => f64::from(*b),
        _ => 0.0,
    }
}
fn scalar_f32(v: &RecordValue) -> f32 {
    match v {
        RecordValue::Float(f) => *f,
        RecordValue::Double(d) => *d as f32,
        RecordValue::Int(i) => *i as f32,
        _ => 0.0,
    }
}
fn scalar_i64(v: &RecordValue) -> i64 {
    match v {
        RecordValue::Int64(i) => *i,
        RecordValue::Int(i) => i64::from(*i),
        RecordValue::UChar(u) => i64::from(*u),
        RecordValue::Bool(b) => i64::from(*b),
        _ => 0,
    }
}
fn scalar_i32(v: &RecordValue) -> i32 {
    match v {
        RecordValue::Int(i) => *i,
        RecordValue::Int64(i) => *i as i32,
        RecordValue::Short(i) => i32::from(*i),
        RecordValue::UChar(u) => i32::from(*u),
        RecordValue::Bool(b) => i32::from(*b),
        _ => 0,
    }
}
fn scalar_i16(v: &RecordValue) -> i16 {
    match v {
        RecordValue::Short(i) => *i,
        RecordValue::Int(i) => *i as i16,
        RecordValue::UChar(u) => i16::from(*u),
        _ => 0,
    }
}
fn scalar_u64(v: &RecordValue) -> u64 {
    match v {
        RecordValue::UInt(u) => u64::from(*u),
        RecordValue::Int(i) => *i as u64,
        _ => 0,
    }
}
fn scalar_u32(v: &RecordValue) -> u32 {
    match v {
        RecordValue::UInt(u) => *u,
        RecordValue::Int(i) => *i as u32,
        _ => 0,
    }
}
fn scalar_u16(v: &RecordValue) -> u16 {
    match v {
        RecordValue::UShort(u) => *u,
        RecordValue::UChar(u) => u16::from(*u),
        RecordValue::Int(i) => *i as u16,
        _ => 0,
    }
}
fn scalar_u8(v: &RecordValue) -> u8 {
    match v {
        RecordValue::UChar(u) => *u,
        RecordValue::Bool(b) => u8::from(*b),
        _ => 0,
    }
}
fn scalar_bool(v: &RecordValue) -> bool {
    match v {
        RecordValue::Bool(b) => *b,
        RecordValue::UChar(u) => *u != 0,
        RecordValue::Int(i) => *i != 0,
        _ => false,
    }
}
fn scalar_c64(v: &RecordValue) -> Complex64 {
    match v {
        RecordValue::DComplex(re, im) => Complex64::new(*re, *im),
        RecordValue::Complex(re, im) => Complex64::new(f64::from(*re), f64::from(*im)),
        _ => Complex64::new(0.0, 0.0),
    }
}
fn scalar_c32(v: &RecordValue) -> Complex32 {
    match v {
        RecordValue::Complex(re, im) => Complex32::new(*re, *im),
        RecordValue::DComplex(re, im) => Complex32::new(*re as f32, *im as f32),
        _ => Complex32::new(0.0, 0.0),
    }
}

/// Build a fresh numpy array for a column of array cells: shape
/// `(nrow, *cell_shape)` where `cell_shape` is the stored (and, for
/// casacore files, logical) cell shape.
#[allow(dead_code)]
pub(crate) fn arrays_to_ndarray(
    py: Python<'_>,
    cells: &[RecordValue],
    cell_shape: &[usize],
) -> PyResult<Py<PyAny>> {
    arrays_to_ndarray_impl(py, cells, cell_shape, false)
}

/// Like `arrays_to_ndarray`, but multidim-string dicts include the row dim
/// in `shape` (the `getcol` form: `(nrow, *cell)`).
pub(crate) fn arrays_to_ndarray_getcol(
    py: Python<'_>,
    cells: &[RecordValue],
    cell_shape: &[usize],
) -> PyResult<Py<PyAny>> {
    arrays_to_ndarray_impl(py, cells, cell_shape, true)
}

fn arrays_to_ndarray_impl(
    py: Python<'_>,
    cells: &[RecordValue],
    cell_shape: &[usize],
    include_row: bool,
) -> PyResult<Py<PyAny>> {
    let nrow = cells.len();
    let cell = cell_shape.iter().product::<usize>().max(1);
    let mut shape = vec![nrow];
    shape.extend_from_slice(cell_shape);
    let kind = cells.iter().find_map(|c| match c {
        RecordValue::Array(a) => Some(&a.data),
        _ => None,
    });
    macro_rules! build {
        ($ty:ty, $f:expr) => {{
            let mut buf: Vec<$ty> = vec![Default::default(); nrow * cell];
            fill_flat(&mut buf, cells, cell, $f, |_| Default::default())?;
            reshape_from(py, buf, &shape)
        }};
    }
    match kind {
        Some(ArrayData::Bool(_)) => build!(bool, bool_of),
        Some(ArrayData::UChar(_)) => build!(u8, u8_of),
        Some(ArrayData::UShort(_)) => build!(u16, u16_of),
        Some(ArrayData::Short(_)) => build!(i16, i16_of),
        Some(ArrayData::Int(_)) => build!(i32, i32_of),
        Some(ArrayData::UInt(_)) => build!(u32, u32_of),
        Some(ArrayData::Int64(_)) => build!(i64, i64_of),
        Some(ArrayData::Float(_)) => build!(f32, f32_of),
        Some(ArrayData::Double(_)) => build!(f64, f64_of),
        Some(ArrayData::Complex(_)) => build!(Complex32, c32_of),
        Some(ArrayData::DComplex(_)) => build!(Complex64, c64_of),
        Some(ArrayData::String(_)) => {
            // Multidim strings come back as {"shape":..., "array":...} dicts;
            // 1-D as a plain list.
            let cell = cell_shape.iter().product::<usize>().max(1);
            let nrow = cells.len();
            let mut flat: Vec<String> = Vec::with_capacity(nrow * cell);
            for c in cells {
                if let RecordValue::Array(a) = c {
                    for i in 0..cell {
                        if let ArrayData::String(v) = &a.data {
                            flat.push(v[i].clone());
                        }
                    }
                }
            }
            let list = PyList::empty(py);
            for s in flat {
                list.append(s)?;
            }
            if include_row || cell_shape.len() > 1 {
                let d = PyDict::new(py);
                if include_row {
                    let mut full = vec![nrow];
                    full.extend_from_slice(cell_shape);
                    d.set_item("shape", full)?;
                } else {
                    d.set_item("shape", cell_shape)?;
                }
                d.set_item("array", list)?;
                Ok(d.into_any().unbind())
            } else {
                Ok(list.into_any().unbind())
            }
        }
        _ => Err(PyValueError::new_err("cannot build array column")),
    }
}

// ---------------------------------------------------------------------------
// Cell values -> Python objects
// ---------------------------------------------------------------------------

fn pybool(py: Python<'_>, b: bool) -> PyResult<Py<PyAny>> {
    Ok(b.into_pyobject(py)?.to_owned().unbind().into())
}

/// One cell to a Python object: numpy arrays for array cells, Python
/// scalars / strings otherwise.
pub(crate) fn cell_to_py(py: Python<'_>, v: &RecordValue) -> PyResult<Py<PyAny>> {
    match v {
        RecordValue::Bool(b) => pybool(py, *b),
        RecordValue::UChar(u) => Ok(PyInt::new(py, u32::from(*u)).into_any().unbind()),
        RecordValue::Short(i) => Ok(PyInt::new(py, *i).into_any().unbind()),
        RecordValue::UShort(u) => Ok(PyInt::new(py, *u).into_any().unbind()),
        RecordValue::Int(i) => Ok(PyInt::new(py, *i).into_any().unbind()),
        RecordValue::UInt(u) => Ok(PyInt::new(py, *u).into_any().unbind()),
        RecordValue::Int64(i) => Ok(PyInt::new(py, *i).into_any().unbind()),
        RecordValue::Float(f) => Ok(PyFloat::new(py, f64::from(*f)).into_any().unbind()),
        RecordValue::Double(d) => Ok(PyFloat::new(py, *d).into_any().unbind()),
        RecordValue::Complex(re, im) => Ok((f64::from(*re), f64::from(*im))
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        RecordValue::DComplex(re, im) => Ok((*re, *im).into_pyobject(py)?.into_any().unbind()),
        RecordValue::String(s) | RecordValue::Table(s) => {
            Ok(PyString::new(py, s).into_any().unbind())
        }
        RecordValue::Array(a) => array_to_ndarray(py, a),
        RecordValue::Record(r) => table_record_to_dict(py, r).map(|d| d.into_any().unbind()),
    }
}

/// A keyword-array value as `{"shape": [..], "array": flat list}`.
pub(crate) fn array_to_dict<'py>(py: Python<'py>, a: &ArrayValue) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let logical: Vec<u32> = a.shape.iter().rev().copied().collect();
    d.set_item("shape", logical)?;
    let mut flat = Vec::new();
    for e in a.elements() {
        flat.push(element_to_py(py, &e)?);
    }
    d.set_item("array", flat)?;
    Ok(d)
}

fn element_to_py(py: Python<'_>, v: &RecordValue) -> PyResult<Py<PyAny>> {
    match v {
        RecordValue::Bool(b) => pybool(py, *b),
        RecordValue::UChar(u) => Ok(PyInt::new(py, u32::from(*u)).into_any().unbind()),
        RecordValue::Short(i) => Ok(PyInt::new(py, *i).into_any().unbind()),
        RecordValue::UShort(u) => Ok(PyInt::new(py, *u).into_any().unbind()),
        RecordValue::Int(i) => Ok(PyInt::new(py, *i).into_any().unbind()),
        RecordValue::UInt(u) => Ok(PyInt::new(py, *u).into_any().unbind()),
        RecordValue::Int64(i) => Ok(PyInt::new(py, *i).into_any().unbind()),
        RecordValue::Float(f) => Ok(PyFloat::new(py, f64::from(*f)).into_any().unbind()),
        RecordValue::Double(d) => Ok(PyFloat::new(py, *d).into_any().unbind()),
        RecordValue::String(s) | RecordValue::Table(s) => {
            Ok(PyString::new(py, s).into_any().unbind())
        }
        RecordValue::Complex(re, im) => Ok((f64::from(*re), f64::from(*im))
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        RecordValue::DComplex(re, im) => Ok((*re, *im).into_pyobject(py)?.into_any().unbind()),
        RecordValue::Array(a) => array_to_ndarray(py, a),
        RecordValue::Record(r) => table_record_to_dict(py, r).map(|d| d.into_any().unbind()),
    }
}

/// Array cell -> numpy ndarray in logical order.
pub(crate) fn array_to_ndarray(py: Python<'_>, a: &ArrayValue) -> PyResult<Py<PyAny>> {
    // A single cell is returned with just the cell shape (no leading row
    // singleton), matching casacore's `getcell`.
    let cell: Vec<usize> = a.shape.iter().map(|&d| d as usize).collect();
    let n = cell.iter().product::<usize>().max(1);
    macro_rules! cell_build {
        ($ty:ty, $f:expr) => {{
            let mut buf: Vec<$ty> = vec![Default::default(); n];
            fill_flat(&mut buf, &[RecordValue::Array(a.clone())], n, $f, |_| {
                Default::default()
            })?;
            reshape_from(py, buf, &cell)
        }};
    }
    match &a.data {
        ArrayData::String(_) => {
            let mut flat: Vec<String> = Vec::with_capacity(n);
            if let ArrayData::String(v) = &a.data {
                flat.extend_from_slice(v);
            }
            let list = PyList::empty(py);
            for s in flat {
                list.append(s)?;
            }
            if cell.len() > 1 {
                let d = PyDict::new(py);
                d.set_item("shape", cell)?;
                d.set_item("array", list)?;
                Ok(d.into_any().unbind())
            } else {
                Ok(list.into_any().unbind())
            }
        }
        ArrayData::Bool(_) => cell_build!(bool, bool_of),
        ArrayData::UChar(_) => cell_build!(u8, u8_of),
        ArrayData::UShort(_) => cell_build!(u16, u16_of),
        ArrayData::Short(_) => cell_build!(i16, i16_of),
        ArrayData::Int(_) => cell_build!(i32, i32_of),
        ArrayData::UInt(_) => cell_build!(u32, u32_of),
        ArrayData::Int64(_) => cell_build!(i64, i64_of),
        ArrayData::Float(_) => cell_build!(f32, f32_of),
        ArrayData::Double(_) => cell_build!(f64, f64_of),
        ArrayData::Complex(_) => cell_build!(Complex32, c32_of),
        ArrayData::DComplex(_) => cell_build!(Complex64, c64_of),
    }
}

// ---------------------------------------------------------------------------
// Record / dict conversions
// ---------------------------------------------------------------------------

/// A `TableRecord` as a nested Python dict.
pub(crate) fn table_record_to_dict<'py>(
    py: Python<'py>,
    rec: &TableRecord,
) -> PyResult<Bound<'py, PyDict>> {
    table_record_to_dict_ctx(py, rec, None)
}

/// Like `table_record_to_dict`, but `TpTable` keyword fields are resolved to
/// the `"Table: <path>"` string python-casacore exposes, with `base` the
/// directory containing the parent table.
pub(crate) fn table_record_to_dict_ctx<'py>(
    py: Python<'py>,
    rec: &TableRecord,
    base: Option<&std::path::Path>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (field, value) in rec.desc.fields.iter().zip(rec.values.iter()) {
        let v = match value {
            RecordValue::Record(sub) => {
                table_record_to_dict_ctx(py, sub, base)?.into_any().unbind()
            }
            RecordValue::Array(a) => array_to_dict(py, a)?.into_any().unbind(),
            RecordValue::Table(name) => {
                let resolved = resolve_subtable_py(name, base);
                PyString::new(py, &resolved).into_any().unbind()
            }
            other => element_to_py(py, other)?,
        };
        d.set_item(&field.name, v)?;
    }
    Ok(d)
}

fn resolve_subtable_py(name: &str, base: Option<&std::path::Path>) -> String {
    match base {
        Some(b) => {
            let path = std::path::Path::new(name);
            let joined = if path.is_absolute() {
                path.to_path_buf()
            } else {
                b.join(path)
            };
            format!(
                "Table: {}",
                casacure::record::lexical_normalize(&joined).display()
            )
        }
        None => name.to_string(),
    }
}

/// Build a `TableRecord` from a Python dict (`putkeywords` etc).
pub(crate) fn dict_to_table_record(py: Python<'_>, d: &Bound<'_, PyDict>) -> PyResult<TableRecord> {
    let mut rec = TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    };
    for (k, v) in d.iter() {
        let name = k
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("keyword names must be strings"))?;
        let value = pyobject_to_record(py, &v)?;
        rec.set(&name, value);
    }
    Ok(rec)
}

/// Convert a generic Python object to a `RecordValue`.
pub(crate) fn pyobject_to_record(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<RecordValue> {
    if v.is_none() {
        return Ok(RecordValue::String(String::new()));
    }
    if v.downcast::<PyBool>().is_ok() {
        return Ok(RecordValue::Bool(v.is_truthy()?));
    }
    if let Ok(s) = v.downcast::<PyString>() {
        return Ok(RecordValue::String(s.to_str()?.to_string()));
    }
    // Integers (python ints and numpy integer scalars, which expose
    // `__index__`).
    if let Ok(n) = v.extract::<i128>() {
        if let Ok(x) = i32::try_from(n) {
            return Ok(RecordValue::Int(x));
        }
        return Ok(RecordValue::Int64(n as i64));
    }
    if let Ok(f) = v.downcast::<PyFloat>() {
        return Ok(RecordValue::Double(f.value()));
    }
    // numpy floating scalars (expose `__float__`).
    if let Ok(f) = v.extract::<f64>() {
        return Ok(RecordValue::Double(f));
    }
    if let Ok(d) = v.downcast::<PyDict>() {
        // The `{"shape": [..], "array": [..]}` multidim-string dict form:
        // only when the array is actually strings.
        if d.contains("shape")? && d.contains("array")? {
            let is_strings = match d.get_item("array")? {
                Some(a) => match a.downcast::<PyList>() {
                    Ok(list) => match list.iter().next() {
                        None => true,
                        Some(e) => {
                            e.is_instance_of::<PyString>()
                                || e.is_instance_of::<numpy::PyArrayDyn<Py<PyAny>>>()
                        }
                    },
                    Err(_) => false,
                },
                None => false,
            };
            if is_strings {
                if let Ok(sv) = py_to_string_array(py, v) {
                    return Ok(sv);
                }
            }
        }
        return Ok(RecordValue::Record(dict_to_table_record(py, d)?));
    }
    if let Ok(list) = v.downcast::<PyList>() {
        return list_to_array(py, list);
    }
    if let Ok(tup) = v.downcast::<PyTuple>() {
        let list = PyList::new(py, tup.iter())?;
        return list_to_array(py, &list);
    }
    // numpy arrays (numeric or object).
    if let Ok(arr) = v.downcast::<numpy::PyArrayDyn<f64>>() {
        let readonly = arr.readonly();
        let shape: Vec<u32> = readonly
            .as_array()
            .shape()
            .to_vec()
            .iter()
            .map(|&d| d as u32)
            .collect();
        let data = readonly.as_array().iter().copied().collect();
        return Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::Double(data),
        }));
    }
    if let Ok(arr) = v.downcast::<numpy::PyArrayDyn<Py<PyAny>>>() {
        let readonly = arr.readonly();
        let shape: Vec<u32> = readonly
            .as_array()
            .shape()
            .to_vec()
            .iter()
            .map(|&d| d as u32)
            .collect();
        let mut s = Vec::with_capacity(readonly.as_array().len());
        for e in readonly.as_array().iter() {
            s.push(e.bind(py).extract::<String>().unwrap_or_default());
        }
        return Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::String(s),
        }));
    }
    Ok(RecordValue::String(v.str()?.to_str()?.to_string()))
}

fn list_to_array(py: Python<'_>, list: &Bound<'_, PyList>) -> PyResult<RecordValue> {
    let items: Vec<RecordValue> = list
        .iter()
        .map(|it| pyobject_to_record(py, &it))
        .collect::<PyResult<_>>()?;
    let shape = vec![items.len() as u32];
    if items
        .iter()
        .all(|v| matches!(v, RecordValue::Int(_) | RecordValue::Int64(_)))
    {
        let vals: Vec<i32> = items
            .iter()
            .map(|v| match v {
                RecordValue::Int(i) => *i,
                RecordValue::Int64(i) => *i as i32,
                _ => 0,
            })
            .collect();
        Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::Int(vals),
        }))
    } else if items
        .iter()
        .all(|v| matches!(v, RecordValue::Double(_) | RecordValue::Float(_)))
    {
        let vals: Vec<f64> = items
            .iter()
            .map(|v| match v {
                RecordValue::Double(d) => *d,
                RecordValue::Float(f) => f64::from(*f),
                _ => 0.0,
            })
            .collect();
        Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::Double(vals),
        }))
    } else if items.iter().all(|v| matches!(v, RecordValue::String(_))) {
        let vals: Vec<String> = items
            .iter()
            .map(|v| match v {
                RecordValue::String(s) => s.clone(),
                _ => String::new(),
            })
            .collect();
        Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::String(vals),
        }))
    } else if items.iter().all(|v| matches!(v, RecordValue::Bool(_))) {
        let vals: Vec<bool> = items
            .iter()
            .map(|v| match v {
                RecordValue::Bool(b) => *b,
                _ => false,
            })
            .collect();
        Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::Bool(vals),
        }))
    } else {
        Err(PyTypeError::new_err(
            "cannot convert mixed list to a keyword array",
        ))
    }
}

/// A `{"shape", "array"}` string dict to a string `ArrayValue`.
pub(crate) fn py_to_string_array(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<RecordValue> {
    if let Ok(d) = v.downcast::<PyDict>() {
        let shape: Vec<u32> = d
            .get_item("shape")?
            .ok_or_else(|| PyValueError::new_err("missing 'shape'"))?
            .extract()?;
        let array: Vec<String> = d
            .get_item("array")?
            .ok_or_else(|| PyValueError::new_err("missing 'array'"))?
            .extract()?;
        return Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::String(array),
        }));
    }
    if let Ok(arr) = v.downcast::<numpy::PyArrayDyn<Py<PyAny>>>() {
        let readonly = arr.readonly();
        let shape: Vec<u32> = readonly
            .as_array()
            .shape()
            .to_vec()
            .iter()
            .map(|&d| d as u32)
            .collect();
        let s: Vec<String> = readonly
            .as_array()
            .iter()
            .map(|e| e.bind(py).extract::<String>().unwrap_or_default())
            .collect();
        return Ok(RecordValue::Array(ArrayValue {
            shape,
            data: ArrayData::String(s),
        }));
    }
    Err(PyValueError::new_err(
        "expected a string {shape,array} dict",
    ))
}

/// The `DataType` of a value (used to type taql-result columns).
pub(crate) fn record_data_type(v: &RecordValue) -> Option<DataType> {
    use DataType as DT;
    Some(match v {
        RecordValue::Bool(_) => DT::Bool,
        RecordValue::UChar(_) => DT::UChar,
        RecordValue::Short(_) => DT::Short,
        RecordValue::UShort(_) => DT::UShort,
        RecordValue::Int(_) => DT::Int,
        RecordValue::UInt(_) => DT::UInt,
        RecordValue::Int64(_) => DT::Int64,
        RecordValue::Float(_) => DT::Float,
        RecordValue::Double(_) => DT::Double,
        RecordValue::Complex(_, _) => DT::Complex,
        RecordValue::DComplex(_, _) => DT::DComplex,
        RecordValue::String(_) | RecordValue::Table(_) => DT::String,
        RecordValue::Array(a) => match &a.data {
            ArrayData::Bool(_) => DT::ArrayBool,
            ArrayData::UChar(_) => DT::ArrayUChar,
            ArrayData::Short(_) => DT::ArrayShort,
            ArrayData::UShort(_) => DT::ArrayUShort,
            ArrayData::Int(_) => DT::ArrayInt,
            ArrayData::UInt(_) => DT::ArrayUInt,
            ArrayData::Int64(_) => DT::ArrayInt64,
            ArrayData::Float(_) => DT::ArrayFloat,
            ArrayData::Double(_) => DT::ArrayDouble,
            ArrayData::Complex(_) => DT::ArrayComplex,
            ArrayData::DComplex(_) => DT::ArrayDComplex,
            ArrayData::String(_) => DT::ArrayString,
        },
        RecordValue::Record(_) => DT::Record,
    })
}

/// Flatten logical coords over a C-order shape.
pub(crate) fn flatten_coords(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    for (c, d) in coords.iter().zip(shape.iter()) {
        idx = idx * (*d).max(1) + c;
    }
    idx
}

/// Unflatten a C-order flat index into logical coords.
pub(crate) fn unflatten(idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    let mut x = idx;
    for k in (0..shape.len()).rev() {
        let d = shape[k].max(1);
        out[k] = x % d;
        x /= d;
    }
    out
}

/// Read a whole numpy array (any supported numeric dtype) into flat
/// `RecordValue`s in C order.
pub(crate) fn numpy_to_record_flat(value: &Bound<'_, PyAny>) -> Option<PyResult<Vec<RecordValue>>> {
    macro_rules! try_num {
        ($ty:ty, $f:expr) => {{
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<$ty>>() {
                let readonly = arr.readonly();
                let mut out = Vec::with_capacity(readonly.as_array().len());
                for e in readonly.as_array().iter() {
                    out.push($f(e));
                }
                return Some(Ok(out));
            }
        }};
    }
    try_num!(f64, |e: &f64| RecordValue::Double(*e));
    try_num!(f32, |e: &f32| RecordValue::Float(*e));
    try_num!(i64, |e: &i64| RecordValue::Int64(*e));
    try_num!(i32, |e: &i32| RecordValue::Int(*e));
    try_num!(u8, |e: &u8| RecordValue::UChar(*e));
    try_num!(u16, |e: &u16| RecordValue::UShort(*e));
    try_num!(bool, |e: &bool| RecordValue::Bool(*e));
    try_num!(Complex64, |e: &Complex64| RecordValue::DComplex(e.re, e.im));
    try_num!(Complex32, |e: &Complex32| RecordValue::Complex(e.re, e.im));
    None
}

/// The flat logical element values of a cell as stored (cells are stored in
/// the shape they were given).
pub(crate) fn cell_logical_flat(cell: &RecordValue) -> Vec<RecordValue> {
    match cell {
        RecordValue::Array(a) => a.elements(),
        other => vec![other.clone()],
    }
}

/// Rebuild a stored-array cell (as-given shape, C order) from flat logical
/// values.
pub(crate) fn cell_from_logical(flat: Vec<RecordValue>, shape: &[usize]) -> RecordValue {
    use casacure::record::{ArrayData, ArrayValue};
    let data = match flat.first() {
        Some(RecordValue::Double(_)) => ArrayData::Double(
            flat.iter()
                .map(|v| match v {
                    RecordValue::Double(d) => *d,
                    _ => 0.0,
                })
                .collect(),
        ),
        Some(RecordValue::Float(_)) => ArrayData::Float(
            flat.iter()
                .map(|v| match v {
                    RecordValue::Float(f) => *f,
                    _ => 0.0,
                })
                .collect(),
        ),
        Some(RecordValue::Int(_)) => ArrayData::Int(
            flat.iter()
                .map(|v| match v {
                    RecordValue::Int(i) => *i,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::Int64(_)) => ArrayData::Int64(
            flat.iter()
                .map(|v| match v {
                    RecordValue::Int64(i) => *i,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::UChar(_)) => ArrayData::UChar(
            flat.iter()
                .map(|v| match v {
                    RecordValue::UChar(u) => *u,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::UShort(_)) => ArrayData::UShort(
            flat.iter()
                .map(|v| match v {
                    RecordValue::UShort(u) => *u,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::Bool(_)) => ArrayData::Bool(
            flat.iter()
                .map(|v| matches!(v, RecordValue::Bool(true)))
                .collect(),
        ),
        Some(RecordValue::Complex(_, _)) => ArrayData::Complex(
            flat.iter()
                .map(|v| match v {
                    RecordValue::Complex(re, im) => (*re, *im),
                    _ => (0.0, 0.0),
                })
                .collect(),
        ),
        Some(RecordValue::DComplex(_, _)) => ArrayData::DComplex(
            flat.iter()
                .map(|v| match v {
                    RecordValue::DComplex(re, im) => (*re, *im),
                    _ => (0.0, 0.0),
                })
                .collect(),
        ),
        Some(RecordValue::String(_)) | Some(RecordValue::Table(_)) => ArrayData::String(
            flat.iter()
                .map(|v| match v {
                    RecordValue::String(s) | RecordValue::Table(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => ArrayData::Double(Vec::new()),
    };
    RecordValue::Array(ArrayValue {
        shape: shape.iter().map(|&d| d as u32).collect(),
        data,
    })
}
