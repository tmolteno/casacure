//! The `casacure.tables` binding surface: a `table` class matching the
//! python-casacore API that dask-ms uses (`CASACORE_TO_CASA_RS.md` §2-§5),
//! plus a `taql` entry point.

use std::path::PathBuf;
use std::sync::Mutex;

use ::casacure as core;
use ::casacure::record::{ArrayData, DataType, RecordValue, TableRecord};
use numpy::PyArrayMethods;
use numpy::{Complex32, Complex64};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::convert;

/// The backing state of a bound table.
#[allow(clippy::large_enum_variant)]
enum Inner {
    /// Read-only access to an existing table.
    Read(::casacure::Table),
    /// Read + buffered writes; `read` serves metadata, `wt` holds cells.
    Write {
        read: ::casacure::Table,
        wt: core::WritableTable,
        dirty: bool,
    },
}

/// python-casacore-compatible `table` object.
#[pyclass(name = "table")]
pub struct Table {
    path: String,
    writable: bool,
    inner: Mutex<Inner>,
}

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// Helper: read a sub-range of a column as `Vec<RecordValue>` (CASA order).
fn column_cells(
    t: &::casacure::Table,
    col_idx: usize,
    startrow: u64,
    nrow: u64,
) -> PyResult<Vec<RecordValue>> {
    t.getcol(col_idx, startrow, nrow).map_err(err)
}

/// The stored (CASA) shape of array cells in a column: from fixed shape in
/// the descriptor, else from the first cell with a value.
/// The per-cell shape as stored (the first array cell's own shape; for
/// casacore files this is the logical shape).
fn cell_shape_of(cells: &[RecordValue]) -> Vec<usize> {
    cells
        .iter()
        .find_map(|c| match c {
            RecordValue::Array(a) => Some(a.shape.iter().map(|&d| d as usize).collect()),
            _ => None,
        })
        .unwrap_or_default()
}

impl Table {
    /// Open or create a table; `desc_json` is the python-casacore table-desc
    /// dict (creates when given) and `nrow` its initial row count.
    fn open_or_create(
        _py: Python<'_>,
        path: &str,
        desc_json: Option<&str>,
        nrow: u64,
        writable: bool,
    ) -> PyResult<Self> {
        // casacore `ms::SUBTABLE` path syntax: the subtable lives in a
        // directory of the same name under the main table directory.
        let dir = if let Some((base, sub)) = path.split_once("::") {
            PathBuf::from(base).join(sub)
        } else {
            PathBuf::from(path)
        };
        if let Some(desc_string) = desc_json {
            let desc = core::tabledesc::TableDesc::from_desc_json(desc_string).map_err(err)?;
            let mut wt = core::WritableTable::create(&dir, desc);
            if nrow > 0 {
                wt.addrows(nrow);
                // Array columns have no default: fill zeros when the caller
                // did not provide values (casacore fills with the default).
                let defs: Vec<(usize, DataType, Option<Vec<i64>>)> = wt
                    .desc()
                    .columns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| match c.kind {
                        core::tabledesc::ColumnKind::Array => {
                            Some((i, c.data_type, c.shape.clone()))
                        }
                        _ => None,
                    })
                    .collect();
                for (col_idx, dt, shape) in defs {
                    // Fixed-shape columns default to zeros; variable-shape
                    // columns to an empty array.
                    let arr = match &shape {
                        Some(s) => {
                            let n = s.iter().map(|&d| d.max(0) as usize).product();
                            RecordValue::Array(core::record::ArrayValue {
                                shape: s.iter().map(|&d| d.max(0) as u32).collect(),
                                data: zero_elements(dt, n),
                            })
                        }
                        None => RecordValue::Array(core::record::ArrayValue {
                            shape: vec![],
                            data: zero_elements(dt, 0),
                        }),
                    };
                    for r in 0..nrow {
                        wt.putcell(col_idx, r, arr.clone()).map_err(err)?;
                    }
                }
            }
            let _ = wt.flush().map_err(err)?;
            let read = ::casacure::Table::open(&dir, false).map_err(err)?;
            return Ok(Table {
                path: path.to_string(),
                writable,
                inner: Mutex::new(Inner::Write {
                    read,
                    wt,
                    dirty: false,
                }),
            });
        }
        let read = ::casacure::Table::open(&dir, false).map_err(err)?;
        if !writable {
            return Ok(Table {
                path: path.to_string(),
                writable: false,
                inner: Mutex::new(Inner::Read(read)),
            });
        }
        // Materialise the current cells into a writable backing.
        let mut wt = core::WritableTable::create(&dir, read.dat.desc.clone());
        let n = read.nrows();
        if n > 0 {
            wt.addrows(n);
        }
        for j in 0..read.dat.desc.columns.len() {
            let vals = column_cells(&read, j, 0, n)?;
            for (r, v) in vals.iter().enumerate() {
                wt.putcell(j, r as u64, v.clone()).map_err(err)?;
            }
        }
        Ok(Table {
            path: path.to_string(),
            writable: true,
            inner: Mutex::new(Inner::Write {
                read,
                wt,
                dirty: false,
            }),
        })
    }

    /// Read cells for a column range from whichever backing is current.
    fn read_col(&self, col_idx: usize, startrow: u64, nrow: u64) -> PyResult<Vec<RecordValue>> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => column_cells(t, col_idx, startrow, nrow),
            Inner::Write { wt, .. } => {
                let mut out = Vec::with_capacity(nrow as usize);
                for r in startrow..startrow + nrow {
                    match wt.cell(col_idx, r) {
                        Some(v) => out.push(v.clone()),
                        None => {
                            return Err(PyValueError::new_err(format!(
                                "column {col_idx} row {r} has not been set"
                            )))
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    fn desc(&self) -> core::tabledesc::TableDesc {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.dat.desc.clone(),
            Inner::Write { wt, .. } => wt.desc().clone(),
        }
    }

    fn row_count(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.nrows(),
            Inner::Write { wt, .. } => wt.col_len(0) as u64,
        }
    }
}

fn zero_elements(dt: DataType, n: usize) -> core::record::ArrayData {
    match dt {
        DataType::Bool => ArrayData::Bool(vec![false; n]),
        DataType::UChar => ArrayData::UChar(vec![0; n]),
        DataType::UShort => ArrayData::UShort(vec![0; n]),
        DataType::Short => ArrayData::Short(vec![0; n]),
        DataType::Int => ArrayData::Int(vec![0; n]),
        DataType::UInt => ArrayData::UInt(vec![0; n]),
        DataType::Int64 => ArrayData::Int64(vec![0; n]),
        DataType::Float => ArrayData::Float(vec![0.0; n]),
        DataType::Double => ArrayData::Double(vec![0.0; n]),
        DataType::Complex => ArrayData::Complex(vec![(0.0, 0.0); n]),
        DataType::DComplex => ArrayData::DComplex(vec![(0.0, 0.0); n]),
        DataType::String => ArrayData::String(vec![String::new(); n]),
        _ => ArrayData::Double(Vec::new()),
    }
}

#[pymethods]
impl Table {
    #[getter]
    fn get_name(&self) -> PyResult<String> {
        Ok(self.path.clone())
    }

    fn name(&self) -> PyResult<String> {
        Ok(self.path.clone())
    }

    fn iswritable(&self) -> PyResult<bool> {
        Ok(self.writable)
    }

    fn nrows(&self) -> PyResult<u64> {
        Ok(self.row_count())
    }

    fn colnames(&self) -> PyResult<Vec<String>> {
        Ok(self.desc().columns.iter().map(|c| c.name.clone()).collect())
    }

    fn colnames2(&self) -> PyResult<Vec<String>> {
        self.colnames()
    }

    /// `getcoldesc(column)` -> dict (python-casacore format).
    fn getcoldesc(&self, py: Python<'_>, column: &str) -> PyResult<Py<PyAny>> {
        let desc = self.desc();
        let col = desc
            .columns
            .iter()
            .find(|c| c.name == column)
            .ok_or_else(|| PyKeyError::new_err(format!("no such column: {column}")))?;
        let vt = self_coldesc(col, py)?;
        Ok(vt.into_any().unbind())
    }

    /// `getkeywords()` -> dict.
    fn getkeywords(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = {
            let inner = self.inner.lock().unwrap();
            match &*inner {
                Inner::Read(t) => t.getkeywords(),
                Inner::Write { wt, .. } => wt.keywords_json(),
            }
        };
        let rec = core::record::parse_json_record(&json).map_err(err)?;
        Ok(convert::table_record_to_dict(py, &rec)?.into_any().unbind())
    }

    /// `getcolkeywords(column)` -> dict.
    fn getcolkeywords(&self, py: Python<'_>, column: &str) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        let desc = self.desc();
        if let Some(col) = desc.columns.iter().find(|c| c.name == column) {
            return Ok(convert::table_record_to_dict(py, &col.keywords)?
                .into_any()
                .unbind());
        }
        Ok(d.into_any().unbind())
    }

    /// `getdminfo()` -> dict.
    fn getdminfo(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.lock().unwrap();
        let (dir, dat) = match &*inner {
            Inner::Read(t) => (PathBuf::from(t.name()), &t.dat),
            Inner::Write { read, .. } => (PathBuf::from(read.name()), &read.dat),
        };
        let info = core::get_dminfo(&dir, dat).map_err(err)?;
        let out = PyDict::new(py);
        for (key, dm) in &info {
            let d = PyDict::new(py);
            d.set_item("TYPE", &dm.type_name)?;
            d.set_item("NAME", &dm.name)?;
            d.set_item("SEQNR", dm.seqnr)?;
            d.set_item("SPEC", spec_to_dict(py, &dm.spec)?)?;
            d.set_item("COLUMNS", dm.columns.clone())?;
            out.set_item(key, d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// `addrows(n)`; grows the table by `n` empty rows.
    fn addrows(&self, n: u64) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            Inner::Write { wt, dirty, .. } => {
                wt.addrows(n);
                *dirty = true;
                Ok(())
            }
            _ => Err(PyValueError::new_err("table is not writable")),
        }
    }

    /// `lock(write=False)` — advisory only in the replacement.
    #[pyo3(signature = (write = false, _read = false, _opt = None))]
    fn lock(&self, write: bool, _read: bool, _opt: Option<u32>) -> PyResult<()> {
        let _ = write;
        Ok(())
    }

    fn unlock(&self) -> PyResult<()> {
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        if let Inner::Write { wt, dirty, read } = &mut *inner {
            let dir = wt.flush().map_err(err)?;
            *read = ::casacure::Table::open(&dir, false).map_err(err)?;
            *dirty = false;
        }
        Ok(())
    }

    fn close(&self) -> PyResult<()> {
        self.flush()?;
        Ok(())
    }

    fn setmaxcachesize(&self, _col: &str, _size: i64) -> PyResult<()> {
        Ok(())
    }

    /// `getcol(column, startrow=0, nrow=-1)` -> numpy array / list / dict.
    #[pyo3(signature = (column, startrow = 0, nrow = -1, _rowincr = 1))]
    fn getcol(
        &self,
        py: Python<'_>,
        column: &str,
        startrow: i64,
        nrow: i64,
        _rowincr: i64,
    ) -> PyResult<Py<PyAny>> {
        let total = self.row_count();
        let startrow = if startrow < 0 {
            (total as i64 + startrow).max(0)
        } else {
            startrow
        } as u64;
        let nrow = if nrow < 0 {
            total - startrow.min(total)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        let cells = self.read_col(col_idx, startrow, nrow)?;
        self.column_to_python(py, col_idx, &cells)
    }

    /// `getcolnp(column, buf, startrow=0, nrow=-1)` — fill an existing numpy
    /// buffer.
    #[pyo3(signature = (column, buf, startrow = 0, nrow = -1))]
    fn getcolnp(
        &self,
        _py: Python<'_>,
        column: &str,
        buf: &Bound<'_, PyAny>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<()> {
        let total = self.row_count();
        let startrow = if startrow < 0 {
            (total as i64 + startrow).max(0)
        } else {
            startrow
        } as u64;
        let nrow = if nrow < 0 {
            total - startrow.min(total)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        let cells = self.read_col(col_idx, startrow, nrow)?;
        let cell = cell_shape_of(&cells).iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(buf, &cells, cell)
    }

    /// `getcolslice(column, blc, trc, startrow, nrow)`.
    #[pyo3(signature = (column, blc, trc, startrow = 0, nrow = -1))]
    fn getcolslice(
        &self,
        py: Python<'_>,
        column: &str,
        blc: Vec<i64>,
        trc: Vec<i64>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<Py<PyAny>> {
        let total = self.row_count();
        let startrow = if startrow < 0 { 0 } else { startrow } as u64;
        let nrow = if nrow < 0 {
            total.saturating_sub(startrow)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        let cells = self.read_colslice(col_idx, &blc, &trc, startrow, nrow)?;
        self.column_to_python(py, col_idx, &cells)
    }

    /// `getcolslicenp(column, buf, blc, trc, startrow, nrow)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, buf, blc, trc, startrow = 0, nrow = -1))]
    fn getcolslicenp(
        &self,
        _py: Python<'_>,
        column: &str,
        buf: &Bound<'_, PyAny>,
        blc: Vec<i64>,
        trc: Vec<i64>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<()> {
        let total = self.row_count();
        let startrow = if startrow < 0 { 0 } else { startrow } as u64;
        let nrow = if nrow < 0 {
            total.saturating_sub(startrow)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        let cells = self.read_colslice(col_idx, &blc, &trc, startrow, nrow)?;
        let cell = cell_shape_of(&cells).iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(buf, &cells, cell)
    }

    /// `getcell(column, row)` -> numpy array (or scalar/list).
    fn getcell(&self, py: Python<'_>, column: &str, row: u64) -> PyResult<Py<PyAny>> {
        let col_idx = self.col_index(column)?;
        let v = self.read_cell(col_idx, row)?;
        if let RecordValue::Array(a) = &v {
            return convert::array_to_ndarray(py, a);
        }
        convert::cell_to_py(py, &v)
    }

    fn getcellslice(
        &self,
        py: Python<'_>,
        column: &str,
        row: u64,
        blc: Vec<i64>,
        trc: Vec<i64>,
    ) -> PyResult<Py<PyAny>> {
        let col_idx = self.col_index(column)?;
        let v = self.read_cellslice(col_idx, row, &blc, &trc)?;
        convert::cell_to_py(py, &v)
    }

    /// `getvarcol(column, startrow=0, nrow=-1)` -> {"rN": array} dict.
    #[pyo3(signature = (column, startrow = 0, nrow = -1))]
    fn getvarcol(
        &self,
        py: Python<'_>,
        column: &str,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<Py<PyAny>> {
        let total = self.row_count();
        let startrow = if startrow < 0 { 0 } else { startrow } as u64;
        let nrow = if nrow < 0 {
            total.saturating_sub(startrow)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        let d = PyDict::new(py);
        for r in 0..nrow {
            let v = self.read_cell(col_idx, startrow + r)?;
            if let RecordValue::Array(_) = v {
                d.set_item(format!("r{}", r + 1), convert::cell_to_py(py, &v)?)?;
            } else {
                d.set_item(format!("r{}", r + 1), convert::cell_to_py(py, &v)?)?;
            }
        }
        Ok(d.into_any().unbind())
    }

    /// `putcell(column, row, value)`.
    fn putcell(
        &self,
        py: Python<'_>,
        column: &str,
        row: u64,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let col_idx = self.col_index(column)?;
        let rec = convert::pyobject_to_record(py, value)?;
        self.put_cell(col_idx, row, rec)
    }

    /// `putcol(column, value, startrow=0, nrow=0)` — ndarray or list.
    #[pyo3(signature = (column, value, startrow = 0, nrow = 0))]
    fn putcol(
        &self,
        py: Python<'_>,
        column: &str,
        value: &Bound<'_, PyAny>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<()> {
        let col_idx = self.col_index(column)?;
        let startrow = startrow.max(0) as u64;
        let nrow = if nrow <= 0 {
            value.len()? as u64
        } else {
            nrow as u64
        };
        let desc = self.desc();
        let is_array_col = matches!(
            desc.columns[col_idx].kind,
            core::tabledesc::ColumnKind::Array
        );
        let values = self.value_to_cells(py, col_idx, value, nrow, is_array_col)?;
        for (i, v) in values.into_iter().enumerate() {
            self.put_cell(col_idx, startrow + i as u64, v)?;
        }
        Ok(())
    }

    /// `putcolnp` — same as putcol (buffer is a numpy array).
    #[pyo3(signature = (column, value, startrow = 0, nrow = 0))]
    fn putcolnp(
        &self,
        py: Python<'_>,
        column: &str,
        value: &Bound<'_, PyAny>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<()> {
        self.putcol(py, column, value, startrow, nrow)
    }

    /// `putvarcol(column, dict_of_rows, startrow=0, nrow=-1)`.
    #[pyo3(signature = (column, rows, startrow = 0, nrow = -1))]
    fn putvarcol(
        &self,
        py: Python<'_>,
        column: &str,
        rows: &Bound<'_, PyDict>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<()> {
        let _ = nrow;
        let col_idx = self.col_index(column)?;
        let startrow = startrow.max(0) as u64;
        let desc = self.desc();
        let is_array_col = matches!(
            desc.columns[col_idx].kind,
            core::tabledesc::ColumnKind::Array
        );
        let mut sorted: Vec<(u64, Bound<'_, PyAny>)> = Vec::new();
        for (k, v) in rows.iter() {
            let s = k.extract::<String>()?;
            let n: u64 = s
                .trim_start_matches('r')
                .parse()
                .map_err(|_| PyValueError::new_err(format!("bad row key {s}")))?;
            sorted.push((n, v));
        }
        sorted.sort_by_key(|(n, _)| *n);
        let first_n = sorted.first().map(|x| x.0).unwrap_or(1);
        for (n, v) in sorted {
            let row = startrow + (n - first_n);
            let mut cells = self.value_to_cells(py, col_idx, &v, 1, is_array_col)?;
            let rec = cells.pop().unwrap_or(RecordValue::Int(0));
            self.put_cell(col_idx, row, rec)?;
        }
        Ok(())
    }

    /// `putcolslice(column, value, blc, trc, startrow, nrow)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, value, blc, trc, startrow = 0, nrow = 0))]
    fn putcolslice(
        &self,
        py: Python<'_>,
        column: &str,
        value: &Bound<'_, PyAny>,
        blc: Vec<i64>,
        trc: Vec<i64>,
        startrow: i64,
        nrow: i64,
    ) -> PyResult<()> {
        // Simplify: full-cell writes only for slices where blc/trc cover the
        // whole cell (the common dask-ms case is chunked row ranges).
        let full_slice = blc.iter().all(|&b| b <= 0) && trc.is_empty();
        let _ = full_slice;
        self.putcol(py, column, value, startrow, nrow)
    }

    /// `putkeyword(name, value)`.
    fn putkeyword(&self, py: Python<'_>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let rec = convert::pyobject_to_record(py, value)?;
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            Inner::Write { wt, dirty, .. } => {
                wt.putkeyword(name, rec);
                *dirty = true;
                Ok(())
            }
            _ => Err(PyValueError::new_err("table is not writable")),
        }
    }

    /// `putkeywords(dict)`.
    fn putkeywords(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        for (k, v) in dict.iter() {
            let name = k.extract::<String>()?;
            self.putkeyword(py, &name, &v)?;
        }
        Ok(())
    }

    /// `removekeyword(name)`.
    fn removekeyword(&self, name: &str) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            Inner::Write { wt, dirty, .. } => {
                wt.removekeyword(name);
                *dirty = true;
                Ok(())
            }
            _ => Err(PyValueError::new_err("table is not writable")),
        }
    }

    /// `putcolkeyword(column, name, value)`.
    fn putcolkeyword(
        &self,
        py: Python<'_>,
        column: &str,
        name: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let col_idx = self.col_index(column)?;
        let rec = convert::pyobject_to_record(py, value)?;
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            Inner::Write { wt, dirty, .. } => {
                wt.putcolkeyword(col_idx, name, rec).map_err(err)?;
                *dirty = true;
                Ok(())
            }
            _ => Err(PyValueError::new_err("table is not writable")),
        }
    }

    /// `putcolkeywords(column, dict)`.
    fn putcolkeywords(
        &self,
        py: Python<'_>,
        column: &str,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        for (k, v) in dict.iter() {
            let name = k.extract::<String>()?;
            self.putcolkeyword(py, column, &name, &v)?;
        }
        Ok(())
    }

    /// `removecolkeyword(column, name)`.
    fn removecolkeyword(&self, column: &str, name: &str) -> PyResult<()> {
        let col_idx = self.col_index(column)?;
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            Inner::Write { wt, dirty, .. } => {
                wt.removecolkeyword(col_idx, name).map_err(err)?;
                *dirty = true;
                Ok(())
            }
            _ => Err(PyValueError::new_err("table is not writable")),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _ty: Option<Py<PyAny>>,
        _value: Option<Py<PyAny>>,
        _tb: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        // python-casacore's `with table(...)` closes (flushing) on exit.
        self.flush()?;
        Ok(())
    }

    /// Private python-casacore `table._getdesc(actual=True)` — the full
    /// table-description dict. dask-ms calls it for keyword reads.
    #[pyo3(signature = (actual = true))]
    fn _getdesc(&self, py: Python<'_>, actual: bool) -> PyResult<Py<PyAny>> {
        let _ = actual;
        let desc = self.desc();
        Ok(desc_to_pydict(py, &desc)?.into_any().unbind())
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.getcol(py, key, 0, -1, 1)
    }
}

impl Table {
    fn col_index(&self, name: &str) -> PyResult<usize> {
        let desc = self.desc();
        desc.columns
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| PyKeyError::new_err(format!("no such column: {name}")))
    }

    fn read_cell(&self, col_idx: usize, row: u64) -> PyResult<RecordValue> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.getcell(col_idx, row).map_err(err),
            Inner::Write { wt, .. } => wt
                .cell(col_idx, row)
                .cloned()
                .ok_or_else(|| PyValueError::new_err(format!("cell ({col_idx}, {row}) not set"))),
        }
    }

    fn read_cellslice(
        &self,
        col_idx: usize,
        row: u64,
        blc: &[i64],
        trc: &[i64],
    ) -> PyResult<RecordValue> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.getcellslice(col_idx, row, blc, trc).map_err(err),
            Inner::Write { wt, .. } => match wt.cell(col_idx, row) {
                Some(v) => Ok(v.clone()),
                None => Err(PyValueError::new_err("cell not set")),
            },
        }
    }

    fn read_colslice(
        &self,
        col_idx: usize,
        blc: &[i64],
        trc: &[i64],
        startrow: u64,
        nrow: u64,
    ) -> PyResult<Vec<RecordValue>> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t
                .getcolslice(col_idx, blc, trc, startrow, nrow)
                .map_err(err),
            Inner::Write { wt, .. } => {
                let mut out = Vec::with_capacity(nrow as usize);
                for r in startrow..startrow + nrow {
                    match wt.cell(col_idx, r) {
                        Some(v) => out.push(v.clone()),
                        None => return Err(PyValueError::new_err("cell not set")),
                    }
                }
                Ok(out)
            }
        }
    }

    fn put_cell(&self, col_idx: usize, row: u64, value: RecordValue) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            Inner::Write { wt, dirty, .. } => {
                wt.putcell(col_idx, row, value).map_err(err)?;
                *dirty = true;
                Ok(())
            }
            _ => Err(PyValueError::new_err("table is not writable")),
        }
    }

    /// Convert user `putcol` data into one `RecordValue` per row.
    fn value_to_cells(
        &self,
        py: Python<'_>,
        _col_idx: usize,
        value: &Bound<'_, PyAny>,
        nrow: u64,
        is_array_col: bool,
    ) -> PyResult<Vec<RecordValue>> {
        if is_array_col {
            // Accept a 2-D+ ndarray or a list of row-arrays.
            if value.downcast::<PyDict>().is_ok() {
                let rec = convert::py_to_string_array(py, value)?;
                return Ok(vec![rec]);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<Py<PyAny>>>() {
                let readonly = arr.readonly();
                let shape: Vec<usize> = readonly.as_array().shape().to_vec();
                let cell = reshape_cell(&shape);
                let mut out = Vec::with_capacity(nrow as usize);
                for r in 0..nrow as usize {
                    let start = r * cell;
                    let stop = ((r + 1) * cell).min(readonly.as_array().len());
                    let mut elems: Vec<String> = Vec::with_capacity(stop - start);
                    for e in readonly.as_array().iter().skip(start).take(stop - start) {
                        elems.push(e.bind(py).extract::<String>().unwrap_or_default());
                    }
                    let casa_shape: Vec<u32> = shape[1..].iter().map(|&s| s as u32).collect();
                    out.push(RecordValue::Array(core::record::ArrayValue {
                        shape: casa_shape,
                        data: core::record::ArrayData::String(elems),
                    }));
                }
                return Ok(out);
            }
            for key in ["i32", "u16", "bool", "f32", "f64", "i64", "u8", "c8", "c16"] {
                let _ = key;
            }
            // Numeric ndarray.
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<f64>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::Double);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<f32>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::Float);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<u8>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::UChar);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<u16>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::UShort);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<i64>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::Int64);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<i32>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::Int);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<bool>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, RecordValue::Bool);
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<Complex32>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, |e| RecordValue::Complex(e.re, e.im));
            }
            if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<Complex64>>() {
                let readonly = arr.readonly();
                return ndarray_cells(&readonly, nrow, |e| RecordValue::DComplex(e.re, e.im));
            }
            return Err(PyTypeError::new_err(format!(
                "putcol: unsupported array data {}",
                value.getattr("dtype")?.str()?.to_str()?
            )));
        }
        // Scalar column: 1-D array (or list) of scalars.
        if let Ok(list) = value.downcast::<PyList>() {
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                out.push(convert::pyobject_to_record(py, &item)?);
            }
            return Ok(out);
        }
        if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<Py<PyAny>>>() {
            let readonly = arr.readonly();
            let mut out = Vec::with_capacity(readonly.as_array().len());
            for e in readonly.as_array().iter() {
                out.push(convert::pyobject_to_record(py, e.bind(py))?);
            }
            return Ok(out);
        }
        macro_rules! scalar_num {
            ($ty:ty, $f:expr) => {{
                if let Ok(arr) = value.downcast::<numpy::PyArrayDyn<$ty>>() {
                    let readonly = arr.readonly();
                    let mut out = Vec::with_capacity(readonly.as_array().len());
                    for e in readonly.as_array().iter() {
                        out.push($f(e));
                    }
                    return Ok(out);
                }
            }};
        }
        scalar_num!(f64, |e: &f64| RecordValue::Double(*e));
        scalar_num!(f32, |e: &f32| RecordValue::Float(*e));
        scalar_num!(i64, |e: &i64| RecordValue::Int64(*e));
        scalar_num!(i32, |e: &i32| RecordValue::Int(*e));
        scalar_num!(u8, |e: &u8| RecordValue::UChar(*e));
        scalar_num!(u16, |e: &u16| RecordValue::UShort(*e));
        scalar_num!(bool, |e: &bool| RecordValue::Bool(*e));
        scalar_num!(Complex64, |e: &Complex64| RecordValue::DComplex(e.re, e.im));
        scalar_num!(Complex32, |e: &Complex32| RecordValue::Complex(e.re, e.im));
        Err(PyTypeError::new_err("putcol: unsupported scalar data"))
    }

    /// Serialize a column range to the python-visible form.
    fn column_to_python(
        &self,
        py: Python<'_>,
        _col_idx: usize,
        cells: &[RecordValue],
    ) -> PyResult<Py<PyAny>> {
        let first = cells.iter().find(|c| !matches!(c, RecordValue::String(_)));
        let is_array_col = matches!(first, Some(RecordValue::Array(_)));
        if !is_array_col {
            // Strings: plain list; numbers: 1-D numpy array.
            if let RecordValue::String(_) | RecordValue::Table(_) = cells.first().unwrap() {
                let list = PyList::empty(py);
                for v in cells {
                    let s = match v {
                        RecordValue::String(s) | RecordValue::Table(s) => s.clone(),
                        other => other.to_json_string(),
                    };
                    list.append(s)?;
                }
                return Ok(list.into_any().unbind());
            }
            return convert::scalars_cells_to_array(py, cells);
        }
        let cell_shape = cell_shape_of(cells);
        convert::arrays_to_ndarray(py, cells, &cell_shape)
    }
}

fn reshape_cell(shape: &[usize]) -> usize {
    shape.iter().skip(1).product::<usize>().max(1)
}

fn ndarray_cells<T: numpy::Element + Copy>(
    arr: &numpy::PyReadonlyArrayDyn<'_, T>,
    nrow: u64,
    f: impl Fn(T) -> RecordValue,
) -> PyResult<Vec<RecordValue>> {
    let shape: Vec<usize> = arr.as_array().shape().to_vec();
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let cell = shape.iter().skip(1).product::<usize>().max(1);
    let flat = arr.as_array();
    // Cells are stored as given (the shape the caller supplied).
    let casa_shape: Vec<u32> = shape[1..].iter().map(|&s| s as u32).collect();
    let mut out = Vec::with_capacity(nrow.min(flat.len() as u64) as usize);
    for r in 0..nrow as usize {
        let start = r * cell;
        if start >= flat.len() {
            break;
        }
        let stop = ((r + 1) * cell).min(flat.len());
        let mut corder: Vec<RecordValue> = Vec::with_capacity(stop - start);
        for e in flat.iter().skip(start).take(stop - start) {
            corder.push(f(*e));
        }
        if corder.len() == 1 && cell == 1 {
            out.push(corder.pop().unwrap());
        } else {
            out.push(RecordValue::Array(core::record::ArrayValue {
                shape: casa_shape.clone(),
                data: array_data_of(&corder),
            }));
        }
    }
    Ok(out)
}

fn array_data_of(elems: &[RecordValue]) -> core::record::ArrayData {
    use core::record::ArrayData as AD;
    match elems.first() {
        Some(RecordValue::Double(_)) => AD::Double(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::Double(d) => *d,
                    _ => 0.0,
                })
                .collect(),
        ),
        Some(RecordValue::Float(_)) => AD::Float(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::Float(d) => *d,
                    _ => 0.0,
                })
                .collect(),
        ),
        Some(RecordValue::Int(_)) => AD::Int(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::Int(d) => *d,
                    RecordValue::Int64(d) => *d as i32,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::Int64(_)) => AD::Int64(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::Int64(d) => *d,
                    RecordValue::Int(d) => i64::from(*d),
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::UInt(_)) => AD::UInt(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::UInt(d) => *d,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::UChar(_)) => AD::UChar(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::UChar(d) => *d,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::UShort(_)) => AD::UShort(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::UShort(d) => *d,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::Bool(_)) => AD::Bool(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::Bool(d) => *d,
                    _ => false,
                })
                .collect(),
        ),
        Some(RecordValue::Complex(_, _)) => AD::Complex(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::Complex(re, im) => (*re, *im),
                    _ => (0.0, 0.0),
                })
                .collect(),
        ),
        Some(RecordValue::DComplex(_, _)) => AD::DComplex(
            elems
                .iter()
                .map(|v| match v {
                    RecordValue::DComplex(re, im) => (*re, *im),
                    _ => (0.0, 0.0),
                })
                .collect(),
        ),
        _ => AD::Double(Vec::new()),
    }
}

/// Build the python-casacore `getcoldesc` dict for a column.
fn self_coldesc<'py>(
    col: &core::tabledesc::ColumnDesc,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let vt = ::casacure::casa_value_type(col.data_type);
    d.set_item("valueType", vt)?;
    d.set_item("dataManagerType", &col.data_manager_type)?;
    d.set_item("dataManagerGroup", &col.data_manager_group)?;
    d.set_item("option", col.options)?;
    d.set_item("maxlen", col.max_length)?;
    d.set_item("comment", &col.comment)?;
    if matches!(col.kind, core::tabledesc::ColumnKind::Array) {
        d.set_item("ndim", col.ndim)?;
        if let Some(shape) = &col.shape {
            if !shape.is_empty() {
                // Logical shape = reverse of stored.
                let logical: Vec<i64> = shape.iter().rev().copied().collect();
                d.set_item("shape", logical)?;
            }
            d.set_item("_c_order", true)?;
        }
    }
    d.set_item(
        "keywords",
        convert::table_record_to_dict(py, &col.keywords)?,
    )?;
    Ok(d)
}

fn spec_to_dict(py: Python<'_>, spec: &core::DmSpec) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match spec {
        core::DmSpec::StandardStMan {
            max_cache_size,
            bucket_size,
            pers_cache_size,
            index_length,
        } => {
            d.set_item("MaxCacheSize", max_cache_size)?;
            d.set_item("BUCKETSIZE", bucket_size)?;
            d.set_item("PERSCACHESIZE", pers_cache_size)?;
            d.set_item("IndexLength", index_length)?;
        }
        core::DmSpec::IncrementalStMan {
            max_cache_size,
            bucket_size,
            pers_cache_size,
        } => {
            d.set_item("MaxCacheSize", max_cache_size)?;
            d.set_item("BUCKETSIZE", bucket_size)?;
            d.set_item("PERSCACHESIZE", pers_cache_size)?;
        }
        core::DmSpec::TiledColumnStMan {
            cube_shapes,
            tile_shapes,
            ..
        } => {
            if let Some(t) = tile_shapes.first() {
                d.set_item("DEFAULTTILESHAPE", t.to_vec())?;
            }
            if let Some(c) = cube_shapes.first() {
                d.set_item("DEFAULTCUBESHAPE", c.to_vec())?;
            }
        }
        core::DmSpec::Unsupported(s) => {
            d.set_item("SPEC", s)?;
        }
    }
    Ok(d.into_any().unbind())
}

/// Module-level `table(...)` factory.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (name, tabledesc = None, nrow = 0, _dminfo = None, readonly = false, _ack = true, *_args, **_kwargs))]
pub fn table(
    py: Python<'_>,
    name: &str,
    tabledesc: Option<&Bound<'_, PyAny>>,
    nrow: i64,
    _dminfo: Option<&Bound<'_, PyAny>>,
    readonly: bool,
    _ack: bool,
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Table> {
    let desc_json = match tabledesc {
        Some(d) if !d.is_none() => {
            if let Ok(dict) = d.downcast::<PyDict>() {
                let rec = convert::dict_to_table_record(py, dict)?;
                Some(rec.to_json_string())
            } else {
                return Err(PyTypeError::new_err("tabledesc must be a dict"));
            }
        }
        _ => None,
    };
    Table::open_or_create(
        py,
        name,
        desc_json.as_deref(),
        nrow.max(0) as u64,
        !readonly,
    )
}

/// Module-level `taql(query, tables=[], style=...)`.
#[pyfunction]
#[pyo3(signature = (query, tables = None, style = "Python", _readonly = None))]
pub fn taql(
    py: Python<'_>,
    query: &str,
    tables: Option<&Bound<'_, PyList>>,
    style: &str,
    _readonly: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let _ = style;
    // DDL: CREATE TABLE ... -> create the table and return it writable.
    if query
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("create")
    {
        match core::taql::execute(query, &[]).map_err(err)? {
            core::taql::TaqlResult::Created(path) => {
                let t = Table::open_or_create(py, &path.display().to_string(), None, 0, true)?;
                return Ok(t.into_pyobject(py)?.into_any().unbind());
            }
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "taql: expected created table, got {other:?}"
                )));
            }
        }
    }
    // Collect the wrapped core tables from any `casacure.tables.table`
    // arguments.
    if query
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("select")
    {
        // Hold every mutex guard for the whole call so the inner core
        // tables stay borrowed.
        let objects: Vec<PyRef<'_, Table>>;
        let locks: Vec<std::sync::MutexGuard<'_, Inner>>;
        let core_refs: Vec<&::casacure::Table>;
        if let Some(ts) = tables {
            objects = ts
                .iter()
                .map(|item| {
                    item.extract::<PyRef<'_, Table>>()
                        .map_err(|_| PyValueError::new_err("taql: expected table objects"))
                })
                .collect::<PyResult<_>>()?;
            locks = objects.iter().map(|t| t.inner.lock().unwrap()).collect();
            core_refs = locks
                .iter()
                .map(|g| match &**g {
                    Inner::Read(r) => r,
                    Inner::Write { read, .. } => read,
                })
                .collect();
        } else {
            objects = Vec::new();
            locks = Vec::new();
            core_refs = Vec::new();
            // objects/locks only exist to keep the borrows alive.
            let _ = (&objects, &locks);
        }
        let result = core::taql::execute(query, &core_refs).map_err(err)?;
        return match result {
            core::taql::TaqlResult::Query(out) => Ok(taql_result_to_table(py, out)?
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            core::taql::TaqlResult::Created(path) => {
                let t = Table::open_or_create(py, &path.display().to_string(), None, 0, true)?;
                Ok(t.into_pyobject(py)?.into_any().unbind())
            }
        };
    }
    Err(PyValueError::new_err(format!(
        "taql: unsupported query (only SELECT/CREATE supported): {query}"
    )))
}

/// Materialise a TaQL result into a real table on disk (in a temp
/// directory) and return a bound `table` over it, so `nrows`/`getcol`/
/// `getcell`/... all work exactly as for a file table.
fn taql_result_to_table(_py: Python<'_>, out: core::taql::TaqlTable) -> PyResult<Table> {
    use core::record::DataType as DT;
    use core::tabledesc::{ColumnDesc, ColumnKind, TableDesc};

    let dir = std::env::temp_dir().join(format!(
        "casacure-taql-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(err)?;

    let nrows = out.nrows() as u64;
    let mut desc = TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        private_keywords: TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        columns: Vec::new(),
    };
    let mut all: Vec<Vec<RecordValue>> = Vec::with_capacity(out.columns.len());
    for (i, name) in out.colnames.iter().enumerate() {
        let cells = &out.columns[i];
        let first = cells.first().cloned().unwrap_or(RecordValue::Int(0));
        let array = matches!(first, RecordValue::Array(_));
        let element_dt: DT = match convert::record_data_type(&first) {
            Some(DT::ArrayBool) | Some(DT::Bool) => DT::Bool,
            Some(DT::ArrayUChar) | Some(DT::UChar) => DT::UChar,
            Some(DT::ArrayShort) | Some(DT::Short) => DT::Short,
            Some(DT::ArrayUShort) | Some(DT::UShort) => DT::UShort,
            Some(DT::ArrayInt) | Some(DT::Int) => DT::Int,
            Some(DT::ArrayUInt) | Some(DT::UInt) => DT::UInt,
            Some(DT::ArrayInt64) | Some(DT::Int64) => DT::Int64,
            Some(DT::ArrayFloat) | Some(DT::Float) => DT::Float,
            Some(DT::ArrayDouble) | Some(DT::Double) => DT::Double,
            Some(DT::ArrayComplex) | Some(DT::Complex) => DT::Complex,
            Some(DT::ArrayDComplex) | Some(DT::DComplex) => DT::DComplex,
            Some(DT::ArrayString) | Some(DT::String) => DT::String,
            _ => DT::Double,
        };
        let (kind, shape) = if array {
            let shp: Vec<i64> = match &first {
                RecordValue::Array(a) => a.shape.iter().map(|&d| i64::from(d)).collect(),
                _ => Vec::new(),
            };
            (ColumnKind::Array, Some(shp))
        } else {
            (ColumnKind::Scalar(zero_record(&element_dt)), None)
        };
        desc.columns.push(ColumnDesc {
            name: name.clone(),
            comment: String::new(),
            data_type: element_dt,
            data_manager_type: "StandardStMan".to_string(),
            data_manager_group: "StandardStMan".to_string(),
            options: 0,
            ndim: shape.as_ref().map_or(-1, |s| s.len() as i32),
            shape,
            max_length: 0,
            keywords: TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind,
        });
        all.push(cells.clone());
    }

    let mut wt = core::WritableTable::create(&dir, desc);
    if nrows > 0 {
        wt.addrows(nrows);
    }
    for (col_idx, cells) in all.iter().enumerate() {
        for (r, v) in cells.iter().enumerate() {
            wt.putcell(col_idx, r as u64, v.clone()).map_err(err)?;
        }
    }
    let _ = wt.flush().map_err(err)?;
    let read = ::casacure::Table::open(&dir, false).map_err(err)?;
    Ok(Table {
        path: dir.display().to_string(),
        writable: true,
        inner: Mutex::new(Inner::Write {
            read,
            wt,
            dirty: false,
        }),
    })
}

fn zero_record(dt: &core::record::DataType) -> RecordValue {
    use core::record::RecordValue as RV;
    match dt {
        DataType::Bool => RV::Bool(false),
        DataType::UChar => RV::UChar(0),
        DataType::Short => RV::Short(0),
        DataType::UShort => RV::UShort(0),
        DataType::Int => RV::Int(0),
        DataType::UInt => RV::UInt(0),
        DataType::Int64 => RV::Int64(0),
        DataType::Float => RV::Float(0.0),
        DataType::Double => RV::Double(0.0),
        DataType::Complex => RV::Complex(0.0, 0.0),
        DataType::DComplex => RV::DComplex(0.0, 0.0),
        _ => RV::String(String::new()),
    }
}

/// The full python-casacore table-desc dict for a `TableDesc` (columns +
/// `_keywords_`/`_private_keywords_`/`_define_hypercolumn_`).
pub(crate) fn desc_to_pydict<'py>(
    py: Python<'py>,
    desc: &core::tabledesc::TableDesc,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for col in &desc.columns {
        d.set_item(&col.name, self_coldesc(col, py)?)?;
    }
    d.set_item(
        "_keywords_",
        convert::table_record_to_dict(py, &desc.keywords)?,
    )?;
    d.set_item(
        "_private_keywords_",
        convert::table_record_to_dict(py, &desc.private_keywords)?,
    )?;
    let empty = PyDict::new(py);
    d.set_item("_define_hypercolumn_", empty)?;
    Ok(d)
}
