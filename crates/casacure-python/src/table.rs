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

/// Resolve a stored subtable reference to an absolute path that opens from
/// any working directory, given the parent table's directory (`table_dir`)
/// and its parent (`base`).
///
/// casacore's convention (`Path::addDirectory` / the real MS and fixture
/// links) is:
/// - `./X` (one `./`) — the subtable is a *sibling* of the table: resolve
///   against the directory containing the table (`/dir/P.tab` stores
///   `./SUB.tab` → `/dir/SUB.tab`);
/// - `././X` (two or more `./`) — the subtable lives *inside* the table's
///   own directory: a real MS stores `././SPECTRAL_WINDOW` and resolves to
///   `<table_dir>/SPECTRAL_WINDOW`;
/// - `./` in front of an absolute path (`.//home/...`) — a legacy writer
///   relativised against a relative table directory; the absolute tail wins;
/// - a bare `X` — resolve against the table's own directory;
/// - bare `MSNAME/SUB` whose first component is the table directory's own
///   basename — a legacy `default_ms` stored the link relative to the parent
///   without the `./`; resolving it against the table directory would double
///   the path.
fn resolve_stored_subtable(
    s: &str,
    table_dir: &std::path::Path,
    _base: &std::path::Path,
) -> String {
    ::casacure::record::resolve_subtable(s, table_dir)
}

/// The backing state of a bound table.
#[allow(clippy::large_enum_variant)]
enum Inner {
    /// Read-only access to an existing table. Holds the opened core table so
    /// repeated reads do not re-read and re-parse the whole table per call.
    Read(std::sync::Arc<::casacure::Table>),
    /// Read + buffered writes backed by the process-shared record for the
    /// table's directory.
    Write {
        shared: std::sync::Arc<std::sync::Mutex<WriteData>>,
    },
}

/// A writable table's shared process-wide state: all handles of a path share
/// one materialised cell store + one read snapshot so concurrent column
/// writes merge instead of one handle's flush clobbering another's.
struct WriteData {
    read: ::casacure::Table,
    wt: core::WritableTable,
    /// Pending in-store writes not yet physically written to disk; write ops
    /// set it, `flush()`/`close()`/mutating taql clear it.
    dirty: bool,
}

/// Live writable backing per table directory (weak: closed tables may be
/// re-materialised by the next writable open).
static WRITE_REGISTRY: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<std::path::PathBuf, std::sync::Arc<std::sync::Mutex<WriteData>>>,
    >,
> = std::sync::OnceLock::new();

fn write_registry() -> &'static std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, std::sync::Arc<std::sync::Mutex<WriteData>>>,
> {
    WRITE_REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Link `dir` to `shared`, replacing any earlier backing (so a table
/// re-created in this process gets fresh state).
fn register_write(dir: &std::path::Path, shared: &std::sync::Arc<std::sync::Mutex<WriteData>>) {
    let mut reg = write_registry().lock().unwrap();
    reg.insert(dir.to_path_buf(), std::sync::Arc::clone(shared));
}

/// The shared backing for `dir` created earlier in this process (strong refs:
/// stays live so later writable handles accumulate into the one cell store).
fn find_write(dir: &std::path::Path) -> Option<std::sync::Arc<std::sync::Mutex<WriteData>>> {
    write_registry().lock().unwrap().get(dir).cloned()
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

/// A `casacure.tables.table` object stored as an MS keyword becomes a table
/// reference (like python-casacore's `TpTable`); anything else is a normal
/// value.
fn keyword_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<RecordValue> {
    if value.is_instance_of::<Table>() {
        let name = value.call_method0("name")?.extract::<String>()?;
        Ok(RecordValue::Table(name))
    } else {
        let rec = convert::pyobject_to_record(py, value)?;
        // A string keyword of the form "Table: <path>" is a subtable
        // reference (casacore stores MS subtable links exactly this way), so
        // it is stored as a TpTable value and surfaces via `getsubtables()`.
        if let RecordValue::String(s) = &rec {
            if let Some(rest) = s.strip_prefix("Table:") {
                return Ok(RecordValue::Table(rest.trim().to_string()));
            }
        }
        Ok(rec)
    }
}

/// Type letter for the ascii dump of a column (matches `tablefromascii`).
fn toascii_type(cells: &[RecordValue]) -> char {
    match cells.first() {
        Some(RecordValue::Bool(_)) => 'S',
        Some(RecordValue::UChar(_)) | Some(RecordValue::UShort(_)) => 'I',
        Some(RecordValue::Int(_)) | Some(RecordValue::Int64(_)) => 'I',
        Some(RecordValue::Float(_)) => 'R',
        Some(RecordValue::Double(_)) => 'D',
        Some(RecordValue::Complex(..)) | Some(RecordValue::DComplex(..)) => 'X',
        Some(RecordValue::String(_)) | Some(RecordValue::Table(_)) => 'S',
        _ => 'S',
    }
}

/// Accept a `str` or any `os.PathLike` (e.g. `pathlib.Path`) and return the
/// filesystem string, like python-casacore's table constructors.
pub(crate) fn path_string(c: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = c.extract::<String>() {
        return Ok(s);
    }
    if let Ok(p) = c.call_method0("__fspath__") {
        return p.extract::<String>();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a str or os.PathLike",
    ))
}

/// Re-materialise a directory's shared writable backing (`WriteData`) from
/// the current on-disk files, after an out-of-band TaQL statement
/// (UPDATE/DELETE/INSERT/ALTER/...) rewrote them. Live handles that share
/// `shared` then read — and, on `close()`, flush — the fresh state instead
/// of a stale pre-statement snapshot (which would otherwise clobber the
/// change back). Returns false when the directory no longer holds a table
/// (e.g. `DROPTABLE`), so the caller can drop the cached entry.
fn refresh_write(
    dir: &std::path::Path,
    shared: &std::sync::Arc<std::sync::Mutex<WriteData>>,
) -> bool {
    let read = match ::casacure::Table::open(dir, false) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let mut wt = core::WritableTable::create(dir.to_path_buf(), read.dat.desc.clone());
    let n = read.nrows();
    if n > 0 {
        wt.addrows(n);
    }
    for j in 0..read.dat.desc.columns.len() {
        let Ok(vals) = read.getcol(j, 0, n) else {
            continue;
        };
        for (r, v) in vals.iter().enumerate() {
            let _ = wt.putcell_loaded(j, r as u64, v.clone());
        }
    }
    let mut s = shared.lock().unwrap();
    s.read = read;
    s.wt = wt;
    s.dirty = false;
    true
}

/// Physically write a shared cell store to disk when it has pending
/// (`dirty`) writes, and refresh its read snapshot from the freshly written
/// files. Write ops buffer into the store and only mark it dirty; the actual
/// table rewrite happens here (explicit `flush()`, on `close()`/context exit,
/// or before a mutating taql statement reads the snapshot). A no-op when the
/// store is clean, so repeated writes do not rewrite the whole table per call.
fn flush_if_dirty(shared: &std::sync::Arc<std::sync::Mutex<WriteData>>) -> PyResult<()> {
    let mut s = shared.lock().unwrap();
    if s.dirty {
        let dir = s.wt.flush().map_err(err)?;
        s.read = ::casacure::Table::open(&dir, false).map_err(err)?;
        s.dirty = false;
    }
    Ok(())
}

/// Read a sub-range of a column as `Vec<RecordValue>` (CASA order).
fn column_cells(
    t: &::casacure::Table,
    col_idx: usize,
    startrow: u64,
    nrow: u64,
) -> PyResult<Vec<RecordValue>> {
    t.getcol(col_idx, startrow, nrow).map_err(err)
}

/// The on-disk twin of a write handle's column `col_idx`, matched by
/// name+type: `addcols`/`removecols` shift the indices within a session, so a
/// positional lookup would read the wrong column (or none).
fn disk_col_index(s: &WriteData, col_idx: usize) -> Option<usize> {
    let cd = s.wt.desc().columns.get(col_idx)?;
    let disk = &s.read.dat.desc.columns;
    if disk
        .get(col_idx)
        .is_some_and(|c| c.name == cd.name && c.data_type == cd.data_type)
    {
        return Some(col_idx);
    }
    disk.iter()
        .position(|c| c.name == cd.name && c.data_type == cd.data_type)
}

/// The value of a row that is not on disk (added since the last flush): the
/// `addrows` default buffered in the cell store — or, for a column the store
/// has no cell for, the schema default.
fn buffered_default_cell(s: &WriteData, col_idx: usize, row: u64) -> RecordValue {
    if let Some(v) = s.wt.cell(col_idx, row) {
        return v.clone();
    }
    s.wt.desc()
        .columns
        .get(col_idx)
        .and_then(::casacure::default_cell_value)
        // Unreachable: every column kind has a default.
        .unwrap_or(RecordValue::Bool(false))
}

/// One cell of a merged write-handle read: a pending write, else the on-disk
/// value when the snapshot has the row, else the `addrows` default.
fn merged_cell(s: &WriteData, col_idx: usize, row: u64) -> PyResult<RecordValue> {
    if let Some(v) = s.wt.pending_cell(col_idx, row) {
        return Ok(v.clone());
    }
    if row < s.read.nrows() {
        if let Some(di) = disk_col_index(s, col_idx) {
            return s.read.getcell(di, row).map_err(err);
        }
    }
    Ok(buffered_default_cell(s, col_idx, row))
}

/// `ValueError` for a row the table does not have (the casacure contract for
/// read violations; casacore raises its own "no such row" `RuntimeError`).
fn row_out_of_range(row: u64, nrow: u64) -> PyErr {
    PyValueError::new_err(format!("row {row} is out of range (table has {nrow} rows)"))
}

/// `ValueError` for a row range that runs past the end of the table.
fn range_out_of_range(startrow: u64, end: u64, nrow: u64) -> PyErr {
    PyValueError::new_err(format!(
        "row range {startrow}..{end} is outside the table ({nrow} rows)"
    ))
}

/// Apply a cell sub-array slice (0-based inclusive corners; scalar cells are
/// returned unchanged).
fn slice_cell(cell: RecordValue, blc: &[i64], trc: &[i64]) -> PyResult<RecordValue> {
    if blc.is_empty() && trc.is_empty() {
        return Ok(cell);
    }
    Ok(match cell {
        RecordValue::Array(a) => {
            RecordValue::Array(::casacure::slice_array_value(&a, blc, trc).map_err(err)?)
        }
        other => other,
    })
}

/// A write-handle column range.
///
/// The read snapshot only covers the rows the table had at the last flush:
/// rows added since — and columns added this session — have no on-disk value,
/// so handing the whole range to the snapshot raises `row N not covered by
/// any indexed bucket` (python `test_check_putdata` / `test_tableascii`).
/// Rows the disk does have are read in one range; every other row comes from
/// the buffer; pending writes overlay both.
fn merged_col_cells(
    s: &WriteData,
    col_idx: usize,
    startrow: u64,
    nrow: u64,
) -> PyResult<Vec<RecordValue>> {
    let total = s.wt.col_len(col_idx) as u64;
    let end = startrow.saturating_add(nrow);
    if end > total {
        return Err(range_out_of_range(startrow, end, total));
    }
    let mut out: Vec<RecordValue> = Vec::with_capacity(nrow as usize);
    // Rows the read snapshot covers — none when the column is not on disk
    // (it was added this session).
    let mut row = startrow;
    if let Some(di) = disk_col_index(s, col_idx) {
        let disk_hi = end.min(s.read.nrows());
        if disk_hi > startrow {
            out.extend(column_cells(&s.read, di, startrow, disk_hi - startrow)?);
            row = disk_hi;
        }
    }
    for r in row..end {
        out.push(buffered_default_cell(s, col_idx, r));
    }
    for (i, r) in (startrow..end).enumerate() {
        if let Some(v) = s.wt.pending_cell(col_idx, r) {
            out[i] = v.clone();
        }
    }
    Ok(out)
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

/// A short human name for a `RecordValue` variant (for error messages).
fn value_name(v: &RecordValue) -> &'static str {
    match v {
        RecordValue::Bool(_) => "boolean",
        RecordValue::UChar(_) => "uchar",
        RecordValue::Short(_) => "short",
        RecordValue::UShort(_) => "ushort",
        RecordValue::Int(_) => "int",
        RecordValue::UInt(_) => "uint",
        RecordValue::Int64(_) => "int64",
        RecordValue::Float(_) => "float",
        RecordValue::Double(_) => "double",
        RecordValue::Complex(_, _) => "complex",
        RecordValue::DComplex(_, _) => "dcomplex",
        RecordValue::String(_) | RecordValue::Table(_) => "string",
        RecordValue::Record(_) => "record",
        RecordValue::Array(_) => "array",
    }
}

/// Whether an `ArrayData` buffer matches a column's element type exactly.
fn array_data_matches(dt: DataType, d: &ArrayData) -> bool {
    use ArrayData as A;
    matches!(
        (dt, d),
        (DataType::Bool, A::Bool(_))
            | (DataType::UChar, A::UChar(_))
            | (DataType::Short, A::Short(_))
            | (DataType::UShort, A::UShort(_))
            | (DataType::Int, A::Int(_))
            | (DataType::UInt, A::UInt(_))
            | (DataType::Int64, A::Int64(_))
            | (DataType::Float, A::Float(_))
            | (DataType::Double, A::Double(_))
            | (DataType::Complex, A::Complex(_))
            | (DataType::DComplex, A::DComplex(_))
            | (DataType::String, A::String(_))
    )
}

/// Whether a `RecordValue` may be stored in `col`: the element type must
/// match exactly (the putcol coercion already produced the column's type),
/// and array columns must hold `Array` cells.
fn record_fits_column(col: &core::tabledesc::ColumnDesc, v: &RecordValue) -> bool {
    let scalar_ok = matches!(
        (col.data_type, v),
        (DataType::Bool, RecordValue::Bool(_))
            | (DataType::UChar, RecordValue::UChar(_))
            | (DataType::Short, RecordValue::Short(_))
            | (DataType::UShort, RecordValue::UShort(_))
            | (DataType::Int, RecordValue::Int(_))
            | (DataType::UInt, RecordValue::UInt(_))
            | (DataType::Int64, RecordValue::Int64(_))
            | (DataType::Float, RecordValue::Float(_))
            | (DataType::Double, RecordValue::Double(_))
            | (DataType::Complex, RecordValue::Complex(_, _))
            | (DataType::DComplex, RecordValue::DComplex(_, _))
            | (DataType::String, RecordValue::String(_))
    );
    match &col.kind {
        core::tabledesc::ColumnKind::Array => match v {
            RecordValue::Array(a) => array_data_matches(col.data_type, &a.data),
            _ => false,
        },
        core::tabledesc::ColumnKind::Record => matches!(v, RecordValue::Record(_)),
        _ => scalar_ok,
    }
}

/// Fill a caller's numpy buffer straight from the mapped data files (the
/// [`::casacure::Table::getcol_raw`] path), for a column
/// `raw_column_supported` accepts. `Ok(false)` means the buffer's dtype,
/// size or layout does not fit and the caller must take the generic path.
fn fill_numpy_raw(
    py: Python<'_>,
    t: &::casacure::Table,
    col_idx: usize,
    startrow: u64,
    nrow: u64,
    buf: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let desc = &t.dat.desc.columns[col_idx];
    let is_array = matches!(desc.kind, core::tabledesc::ColumnKind::Array);
    let count = if is_array {
        match &desc.shape {
            Some(s) if !s.is_empty() => s.iter().rev().map(|&d| d.max(0) as usize).product(),
            _ => return Ok(false), // variable-shape arrays keep the old path
        }
    } else {
        1
    };
    let le = !t.dat.header.big_endian;
    let native = le == cfg!(target_endian = "little");
    let short = |got: usize, want: usize| core::table::TableReadError::UnsupportedRaw {
        name: desc.name.clone(),
        reason: format!("stored cell has {got} bytes, expected {want}"),
    };

    macro_rules! typed {
        ($ty:ty, $from:expr) => {{
            if let Ok(arr) = buf.cast::<numpy::PyArrayDyn<$ty>>() {
                let mut rw = arr.readwrite();
                let Ok(slice) = rw.as_slice_mut() else {
                    return Ok(false);
                };
                if slice.len() as u64 != nrow * count as u64 {
                    return Ok(false);
                }
                let sz = std::mem::size_of::<$ty>();
                let mut row = 0usize;
                // Decode straight from the mapped file into the caller's
                // buffer with the GIL released (see fill_buffer_by_dtype), so
                // the dask scheduler can overlap independent column reads.
                py.allow_threads(|| {
                    t.getcol_raw(col_idx, startrow, nrow, |_, bytes| {
                        let dst = &mut slice[row * count..(row + 1) * count];
                        if bytes.len() < count * sz {
                            return Err(short(bytes.len(), count * sz));
                        }
                        if native {
                            // Same byte order as the host: the stored cell is
                            // the numpy element layout, one memcpy per row.
                            // SAFETY: `dst` is `count` plain numeric elements
                            // (`sz` bytes each, no padding, any bit pattern
                            // valid), so viewing it as `count * sz` bytes is
                            // sound; the ranges cannot overlap (`bytes` borrows
                            // the mapped file, `dst` the numpy buffer).
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(
                                    dst.as_mut_ptr().cast::<u8>(),
                                    count * sz,
                                )
                            };
                            out.copy_from_slice(&bytes[..count * sz]);
                        } else {
                            for (d, b) in dst.iter_mut().zip(bytes.chunks_exact(sz)) {
                                *d = $from(b, le);
                            }
                        }
                        row += 1;
                        Ok(())
                    })
                    .map_err(err)
                })?;
                return Ok(true);
            } else {
                return Ok(false); // buffer dtype/contiguity mismatch -> fallback
            }
        }};
    }

    if desc.data_type == DataType::Bool && !is_array {
        // IncrementalStMan Bool scalars: one byte per cell. Converted, not
        // copied: a stored byte other than 0/1 is not a valid `bool`.
        if let Ok(arr) = buf.cast::<numpy::PyArrayDyn<bool>>() {
            let mut rw = arr.readwrite();
            let Ok(slice) = rw.as_slice_mut() else {
                return Ok(false);
            };
            if slice.len() as u64 != nrow {
                return Ok(false);
            }
            let mut row = 0usize;
            t.getcol_raw(col_idx, startrow, nrow, |_, bytes| {
                let Some(&b) = bytes.first() else {
                    return Err(short(0, 1));
                };
                slice[row] = b != 0;
                row += 1;
                Ok(())
            })
            .map_err(err)?;
            return Ok(true);
        }
        return Ok(false);
    }

    if desc.data_type == DataType::Bool && is_array {
        // Tiled Bool cells are bit-packed (LSB-first) in the tile file.
        if let Ok(arr) = buf.cast::<numpy::PyArrayDyn<bool>>() {
            let mut rw = arr.readwrite();
            let Ok(slice) = rw.as_slice_mut() else {
                return Ok(false);
            };
            if slice.len() as u64 != nrow * count as u64 {
                return Ok(false);
            }
            let mut row = 0usize;
            py.allow_threads(|| {
                t.getcol_raw_bits(col_idx, startrow, nrow, |bytes, skip, nelem| {
                    if nelem != count || (skip + nelem).div_ceil(8) > bytes.len() {
                        return Err(short(bytes.len(), (skip + count).div_ceil(8)));
                    }
                    let dst = &mut slice[row * count..(row + 1) * count];
                    ::casacure::tsm::decode_bits_into(bytes, skip, dst);
                    row += 1;
                    Ok(())
                })
                .map_err(err)
            })?;
            return Ok(true);
        }
        return Ok(false);
    }

    match desc.data_type {
        DataType::UChar => typed!(u8, |b: &[u8], _le: bool| b[0]),
        DataType::Short => typed!(i16, |b: &[u8], le: bool| if le {
            i16::from_le_bytes(b.try_into().unwrap())
        } else {
            i16::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::UShort => typed!(u16, |b: &[u8], le: bool| if le {
            u16::from_le_bytes(b.try_into().unwrap())
        } else {
            u16::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::Int => typed!(i32, |b: &[u8], le: bool| if le {
            i32::from_le_bytes(b.try_into().unwrap())
        } else {
            i32::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::UInt => typed!(u32, |b: &[u8], le: bool| if le {
            u32::from_le_bytes(b.try_into().unwrap())
        } else {
            u32::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::Int64 => typed!(i64, |b: &[u8], le: bool| if le {
            i64::from_le_bytes(b.try_into().unwrap())
        } else {
            i64::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::Float => typed!(f32, |b: &[u8], le: bool| if le {
            f32::from_le_bytes(b.try_into().unwrap())
        } else {
            f32::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::Double => typed!(f64, |b: &[u8], le: bool| if le {
            f64::from_le_bytes(b.try_into().unwrap())
        } else {
            f64::from_be_bytes(b.try_into().unwrap())
        }),
        DataType::Complex => typed!(numpy::Complex32, |b: &[u8], le: bool| {
            let (re, im) = if le {
                (
                    f32::from_le_bytes(b[0..4].try_into().unwrap()),
                    f32::from_le_bytes(b[4..8].try_into().unwrap()),
                )
            } else {
                (
                    f32::from_be_bytes(b[0..4].try_into().unwrap()),
                    f32::from_be_bytes(b[4..8].try_into().unwrap()),
                )
            };
            numpy::Complex32::new(re, im)
        }),
        DataType::DComplex => typed!(numpy::Complex64, |b: &[u8], le: bool| {
            let (re, im) = if le {
                (
                    f64::from_le_bytes(b[0..8].try_into().unwrap()),
                    f64::from_le_bytes(b[8..16].try_into().unwrap()),
                )
            } else {
                (
                    f64::from_be_bytes(b[0..8].try_into().unwrap()),
                    f64::from_be_bytes(b[8..16].try_into().unwrap()),
                )
            };
            numpy::Complex64::new(re, im)
        }),
        _ => Ok(false),
    }
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
        // Table paths are handled absolute (mirrors casacore, whose table
        // names are absolute): subtable links are stored relative to the
        // table's parent and resolved against it, which only round-trips
        // from an absolute directory.
        let abs = if let Some((base, sub)) = path.split_once("::") {
            format!(
                "{base}::{sub}",
                base = core::table::absolute_dir(std::path::Path::new(base)).display(),
            )
        } else {
            core::table::absolute_dir(std::path::Path::new(path))
                .display()
                .to_string()
        };
        // casacore `ms::SUBTABLE` path syntax: the subtable lives in a
        // directory of the same name under the main table directory.
        let dir = if let Some((base, sub)) = abs.split_once("::") {
            PathBuf::from(base).join(sub)
        } else {
            PathBuf::from(&abs)
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
            let shared = std::sync::Arc::new(std::sync::Mutex::new(WriteData {
                read,
                wt,
                dirty: false,
            }));
            register_write(&dir, &shared);
            return Ok(Table {
                path: abs.clone(),
                writable,
                inner: Mutex::new(Inner::Write { shared }),
            });
        }
        if !writable {
            // A read-only open must see pending (unflushed) writes of any
            // live writable backing for this directory: write ops are
            // buffered, but a fresh handle (and python-casacore, whose SM
            // writes are file-visible) reads the files. Flush a dirty
            // backing before opening the files so this is not a stale
            // snapshot.
            if let Some(shared) = find_write(&dir) {
                if shared.lock().unwrap().dirty {
                    flush_if_dirty(&shared)?;
                }
            }
            let read = ::casacure::Table::open(&dir, false).map_err(err)?;
            return Ok(Table {
                path: abs.clone(),
                writable: false,
                inner: Mutex::new(Inner::Read(std::sync::Arc::new(read))),
            });
        }
        // Reuse a live shared backing for this directory so concurrent
        // writable handles accumulate into one cell store (a second flush of
        // a stale snapshot must not clobber the first handle's writes).
        if let Some(shared) = find_write(&dir) {
            return Ok(Table {
                path: abs.clone(),
                writable: true,
                inner: Mutex::new(Inner::Write { shared }),
            });
        }
        // A writable open of an existing table is LAZY: no column is
        // materialised, so a changed-columns write holds only the rows it
        // actually writes instead of per-cell RecordValues for the whole
        // table (~3 GiB on a 1.6 GB MS).
        let (read, wt) = core::WritableTable::open_for_update(&dir).map_err(err)?;
        let shared = std::sync::Arc::new(std::sync::Mutex::new(WriteData {
            read,
            wt,
            dirty: false,
        }));
        register_write(&dir, &shared);
        Ok(Table {
            path: abs.clone(),
            writable: true,
            inner: Mutex::new(Inner::Write { shared }),
        })
    }

    /// The resolved table directory (handles `ms::SUBTABLE` syntax).
    fn dir_of(&self) -> std::path::PathBuf {
        if let Some((base, sub)) = self.path.split_once("::") {
            PathBuf::from(base).join(sub)
        } else {
            PathBuf::from(&self.path)
        }
    }

    /// A fresh read-only core table for running TaQL against the current
    /// on-disk state.
    fn core_running(&self) -> PyResult<::casacure::Table> {
        // The on-disk state must include this handle's pending (unflushed)
        // writes; `query()`/`sort()`/`select_run()` run against the files.
        if let Inner::Write { shared, .. } = &*self.inner.lock().unwrap() {
            flush_if_dirty(shared)?;
        }
        ::casacure::Table::open(self.dir_of(), false).map_err(err)
    }

    /// Read cells for a column range from whichever backing is current.
    ///
    /// A write handle merges its unflushed writes over the on-disk state;
    /// rows the table did not have at the last flush read back as their
    /// `addrows` default (see [`merged_col_cells`]).
    fn read_col(&self, col_idx: usize, startrow: u64, nrow: u64) -> PyResult<Vec<RecordValue>> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => {
                // Clone the Arc out and drop the lock before the decode, so
                // concurrent reads through this handle are not serialised by
                // the inner mutex (reads on the core table are pure &-reads).
                let t = std::sync::Arc::clone(t);
                drop(inner);
                column_cells(&t, col_idx, startrow, nrow)
            }
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                merged_col_cells(&s, col_idx, startrow, nrow)
            }
        }
    }

    fn desc(&self) -> core::tabledesc::TableDesc {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.dat.desc.clone(),
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                s.wt.desc().clone()
            }
        }
    }

    fn row_count(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.nrows(),
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                s.wt.col_len(0) as u64
            }
        }
    }
}

/// A zero-filled numpy array of `dims` with the numpy dtype of a numeric or
/// Bool column (`None` for strings, records and other non-numeric types).
fn zeros_for_column(
    py: Python<'_>,
    desc: &core::tabledesc::ColumnDesc,
    dims: &[usize],
) -> PyResult<Option<Py<PyAny>>> {
    use numpy::PyArrayDyn;
    macro_rules! z {
        ($ty:ty) => {
            PyArrayDyn::<$ty>::zeros(py, dims, false)
                .into_any()
                .unbind()
        };
    }
    Ok(Some(match desc.data_type {
        DataType::Bool => z!(bool),
        DataType::UChar => z!(u8),
        DataType::Short => z!(i16),
        DataType::UShort => z!(u16),
        DataType::Int => z!(i32),
        DataType::UInt => z!(u32),
        DataType::Int64 => z!(i64),
        DataType::Float => z!(f32),
        DataType::Double => z!(f64),
        DataType::Complex => z!(numpy::Complex32),
        DataType::DComplex => z!(numpy::Complex64),
        _ => return Ok(None),
    }))
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
        let base = std::path::Path::new(&self.path)
            .parent()
            .map(|p| p.to_path_buf());
        let vt = self_coldesc(py, col, base.as_deref())?;
        Ok(vt.into_any().unbind())
    }

    /// Run `sql` (a `SELECT * FROM $1 ...`) against this table and return
    /// the result as a new `table`.
    fn select_run(&self, py: Python<'_>, sql: &str) -> PyResult<Py<PyAny>> {
        let core_t = self.core_running()?;
        match core::taql::execute(sql, &[&core_t]).map_err(err)? {
            core::taql::TaqlResult::Query(out) => Ok(taql_result_to_table(py, out)?
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            other => Err(PyRuntimeError::new_err(format!(
                "taql: expected a SELECT result, got {other:?}"
            ))),
        }
    }

    /// `table.query(query)` — select rows matching a TaQL selection
    /// expression (casacore `Table::query`), returned as a new table.
    /// DDFacet uses e.g. `t.query("FIELD_ID==1")`.
    #[pyo3(signature = (query, _options = None))]
    fn query(
        &self,
        py: Python<'_>,
        query: &str,
        _options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.select_run(py, &format!("SELECT * FROM $1 WHERE {query}"))
    }

    /// `table.select(query)` — alias of `query` (casacore returns a
    /// TableIterator; a filtered table is sufficient for DDFacet's usage).
    #[pyo3(signature = (query, _sort = false, _only_unnamed = false))]
    fn select(
        &self,
        py: Python<'_>,
        query: &str,
        _sort: bool,
        _only_unnamed: bool,
    ) -> PyResult<Py<PyAny>> {
        self.query(py, query, None)
    }

    /// `table.sort(column)` — return the table sorted by `column` (ascending,
    /// like casacore `Table::sort`). DDFacet: `t.query(...).sort("TIME")`.
    #[pyo3(signature = (column, _addtoprefix = false))]
    fn sort(&self, py: Python<'_>, column: &str, _addtoprefix: bool) -> PyResult<Py<PyAny>> {
        self.select_run(py, &format!("SELECT * FROM $1 ORDERBY {column}"))
    }

    /// `getkeyword(name)` — a single keyword value (None when absent).
    fn getkeyword(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let rec = {
            let inner = self.inner.lock().unwrap();
            match &*inner {
                Inner::Read(t) => t.dat.desc.keywords.clone(),
                Inner::Write { shared, .. } => shared.lock().unwrap().wt.desc().keywords.clone(),
            }
        };
        let base = self.dir_of();
        let d = convert::table_record_to_dict_ctx(py, &rec, Some(&base))?;
        match d.get_item(name)? {
            Some(v) => Ok(v.unbind()),
            None => Ok(py.None()),
        }
    }

    /// `getkeywords()` -> dict.
    fn getkeywords(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.lock().unwrap();
        let rec = match &*inner {
            Inner::Read(t) => t.dat.desc.keywords.clone(),
            Inner::Write { shared, .. } => shared.lock().unwrap().wt.desc().keywords.clone(),
        };
        let base = self.dir_of();
        Ok(convert::table_record_to_dict_ctx(py, &rec, Some(&base))?
            .into_any()
            .unbind())
    }

    /// `getcolkeywords(column)` -> dict.
    fn getcolkeywords(&self, py: Python<'_>, column: &str) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        let desc = self.desc();
        let base = self.dir_of();
        if let Some(col) = desc.columns.iter().find(|c| c.name == column) {
            return Ok(
                convert::table_record_to_dict_ctx(py, &col.keywords, Some(&base))?
                    .into_any()
                    .unbind(),
            );
        }
        Ok(d.into_any().unbind())
    }

    /// `getdminfo()` -> dict.
    fn getdminfo(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.lock().unwrap();
        let info = match &*inner {
            Inner::Read(t) => {
                let dir = std::path::PathBuf::from(t.name());
                core::get_dminfo(&dir, &t.dat).map_err(err)?
            }
            Inner::Write { shared, .. } => {
                // Descriptors may carry unflushed structural changes
                // (addcols/removecols/keywords), so persist before reading
                // the on-disk metadata snapshot.
                flush_if_dirty(shared)?;
                let s = shared.lock().unwrap();
                let dir = PathBuf::from(s.read.name());
                core::get_dminfo(&dir, &s.read.dat).map_err(err)?
            }
        };
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

    /// `addcols(coldesc_dict, dminfo=None)` — append columns (each a
    /// python-casacore column-desc dict) to the writable table.
    #[pyo3(signature = (coldesc, dminfo = None))]
    fn addcols(
        &self,
        py: Python<'_>,
        coldesc: &Bound<'_, PyDict>,
        dminfo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let _ = dminfo;
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    // Pull the desc's columns out (they live in a dict of
                    // {colname: coldesc}).
                    let json = {
                        let rec = convert::dict_to_table_record(py, coldesc)?;
                        rec.to_json_string()
                    };

                    let parsed = core::tabledesc::TableDesc::from_desc_json(&json).map_err(err)?;
                    let mut s = shared.lock().unwrap();
                    for cd in parsed.columns {
                        s.wt.addcol(cd);
                    }
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
    }

    /// `removecols(names)` — drop columns (and their data) from the table.
    fn removecols(&self, py: Python<'_>, columns: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = py;
        // Accept a single column name or a sequence of names.
        let names: Vec<String> = if let Ok(name) = columns.extract::<String>() {
            vec![name]
        } else {
            columns.extract()?
        };
        // Resolve column indices against the current descriptor before taking
        // the write lock (col_index would re-lock the same mutex).
        let desc = self.desc();
        let mut indices: Vec<usize> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            let col_idx = desc
                .columns
                .iter()
                .position(|c| &c.name == name)
                .ok_or_else(|| PyKeyError::new_err(format!("no such column: {name}")))?;
            if seen.insert(col_idx) {
                indices.push(col_idx);
            }
        }
        indices.sort_unstable();
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    let mut s = shared.lock().unwrap();
                    for idx in indices.iter().rev() {
                        s.wt.removecol(*idx);
                    }
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
    }

    /// `removecol(name)` — drop a single column (and its data).
    fn removecol(&self, py: Python<'_>, column: &str) -> PyResult<()> {
        let s = pyo3::types::PyString::new(py, column);
        self.removecols(py, s.as_any())
    }

    /// `addrows(n)`; grows the table by `n` empty rows.
    fn addrows(&self, n: u64) -> PyResult<()> {
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    let mut s = shared.lock().unwrap();
                    s.wt.addrows(n);
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
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
        if let Inner::Write { shared } = &mut *inner {
            flush_if_dirty(shared)?;
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
        // Clone the Arc out of the lock so concurrent reads through this
        // handle are not serialised on the inner mutex.
        let read_table = {
            let inner = self.inner.lock().unwrap();
            match &*inner {
                Inner::Read(t) => Some(std::sync::Arc::clone(t)),
                _ => None,
            }
        };
        if let Some(t) = &read_table {
            if nrow == 0 {
                // casacore returns an empty 1-D array of the column type.
                if let Some(arr) = zeros_for_column(py, &t.dat.desc.columns[col_idx], &[0])? {
                    return Ok(arr);
                }
            }
            // Typed fast path: allocate the result array and fill it
            // straight from the data files (see `getcolnp`).
            if startrow.saturating_add(nrow) <= t.nrows() && t.raw_column_supported(col_idx) {
                let desc = &t.dat.desc.columns[col_idx];
                let mut dims = vec![nrow as usize];
                if matches!(desc.kind, core::tabledesc::ColumnKind::Array) {
                    if let Some(s) = &desc.shape {
                        dims.extend(s.iter().rev().map(|&d| d.max(0) as usize));
                    }
                }
                if let Some(arr) = zeros_for_column(py, desc, &dims)? {
                    if fill_numpy_raw(py, t, col_idx, startrow, nrow, arr.bind(py))? {
                        return Ok(arr);
                    }
                }
            }
        }
        // Per-cell decode fallback; pure Rust, so run it with the GIL
        // released to let the scheduler overlap independent reads.
        let cells = py.allow_threads(|| self.read_col(col_idx, startrow, nrow))?;
        self.column_to_python(py, col_idx, &cells)
    }

    /// `taql(query)` — run a TaQL query against this table (the `$1`/`$t`
    /// reference), like python-casacore's `table.taql`. Returns the result
    /// table.
    #[pyo3(signature = (query))]
    fn taql(slf: &Bound<'_, Self>, py: Python<'_>, query: &str) -> PyResult<Py<PyAny>> {
        let tables = PyList::empty(py);
        tables.append(slf.clone().into_any())?;
        crate::table::taql(py, query, Some(&tables), "Python", None)
    }

    /// `getsubtables()` — the subtable reference strings held by the table's
    /// `Table:` keywords, as stored (matches python-casacore, which returns
    /// e.g. `["./[ANTENNA]"]` for a keyword `"Table: ./ANTENNA"`).
    ///
    /// [FIXME]: The docstring is kept deliberately simple; the returned
    /// strings are exactly what was written.
    #[pyo3(signature = ())]
    fn getsubtables(&self, _py: Python<'_>) -> PyResult<Vec<String>> {
        let desc = self.desc();
        // The table's own directory, made absolute, so the returned subtable
        // paths open from any working directory (like python-casacore).
        let name: String = match &*self.inner.lock().unwrap() {
            Inner::Read(t) => t.name().to_string(),
            _ => self.name()?,
        };
        let table_dir = std::fs::canonicalize(&name).unwrap_or_else(|_| name.clone().into());
        let base = table_dir.parent().unwrap_or(std::path::Path::new("."));
        let mut out: Vec<String> = Vec::new();
        fn walk(v: &RecordValue, out: &mut Vec<String>) {
            match v {
                RecordValue::Table(s) => {
                    if !out.contains(s) {
                        out.push(s.clone());
                    }
                }
                RecordValue::Record(r) => {
                    for v in &r.values {
                        walk(v, out);
                    }
                }
                _ => {}
            }
        }
        for v in &desc.keywords.values {
            walk(v, &mut out);
        }
        for v in &desc.private_keywords.values {
            walk(v, &mut out);
        }
        // Resolve each reference to an absolute path that opens from any
        // working directory (like python-casacore).
        let resolved: Vec<String> = out
            .into_iter()
            .map(|s| resolve_stored_subtable(&s, &table_dir, base))
            .collect();
        Ok(resolved)
    }

    /// `copy(newtablename, deep=False, ...)` — copy this table on disk; `deep`
    /// also copies the subtable directories referenced by `Table:` keywords
    /// (like python-casacore's `table.copy`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (new_table_name, deep = false, valuecopy = false, dminfo = None, _endian = "aipsrc", _memorytable = false, _copynorows = false))]
    fn copy(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        new_table_name: &Bound<'_, PyAny>,
        deep: bool,
        valuecopy: bool,
        dminfo: Option<&Bound<'_, PyAny>>,
        _endian: &str,
        _memorytable: bool,
        _copynorows: bool,
    ) -> PyResult<()> {
        let own: String = slf.call_method0("name")?.extract()?;
        let src = pyo3::types::PyString::new(py, &own);
        crate::helpers::tablecopy(
            py,
            src.as_any(),
            new_table_name,
            deep,
            valuecopy,
            dminfo,
            "aipsrc",
            false,
            false,
        )
    }

    /// `toascii(filename, columnnames=None)` — write the table (or the given
    /// columns) to an ascii file in the `tablefromascii` format.
    #[pyo3(signature = (filename, columnnames = None))]
    fn toascii(
        &self,
        _py: Python<'_>,
        filename: &Bound<'_, PyAny>,
        columnnames: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let filename = path_string(filename)?;
        let cols: Vec<String> = match columnnames {
            Some(c) if !c.is_none() => c.extract()?,
            _ => self.colnames()?,
        };
        let total = self.row_count();
        let mut name_line = Vec::with_capacity(cols.len());
        let mut type_line = Vec::with_capacity(cols.len());
        let mut col_cells: Vec<Vec<RecordValue>> = Vec::with_capacity(cols.len());
        for name in &cols {
            let col_idx = self.col_index(name)?;
            let cells = self.read_col(col_idx, 0, total)?;
            name_line.push(name.clone());
            type_line.push(toascii_type(&cells).to_string());
            col_cells.push(cells);
        }
        // tablefromascii splits name/type lines on whitespace.
        let mut lines = vec![name_line.join(" "), type_line.join(" ")];
        for r in 0..total as usize {
            let mut row = Vec::new();
            for cells in &col_cells {
                let cell = &cells[r];
                match cell {
                    RecordValue::Bool(b) => row.push(if *b { "1" } else { "0" }.to_string()),
                    RecordValue::UChar(b) => row.push(b.to_string()),
                    RecordValue::UShort(b) => row.push(b.to_string()),
                    RecordValue::Int(i) => row.push(i.to_string()),
                    RecordValue::Int64(i) => row.push(i.to_string()),
                    RecordValue::Float(f) => row.push(format!("{f}")),
                    RecordValue::Double(d) => row.push(format!("{d}")),
                    RecordValue::Complex(re, im) => {
                        row.push(format!("{re}"));
                        row.push(format!("{im}"));
                    }
                    RecordValue::DComplex(re, im) => {
                        row.push(format!("{re}"));
                        row.push(format!("{im}"));
                    }
                    RecordValue::String(s) | RecordValue::Table(s) => {
                        row.push(s.clone());
                    }
                    other => row.push(format!("{other:?}")),
                }
            }
            lines.push(row.join(" "));
        }
        std::fs::write(&filename, format!("{}\n", lines.join("\n")))
            .map_err(|e| PyValueError::new_err(format!("cannot write {filename}: {e}")))?;
        Ok(())
    }

    /// `getcolnp(column, buf, startrow=0, nrow=-1)` — fill an existing numpy
    /// buffer.
    #[pyo3(signature = (column, buf, startrow = 0, nrow = -1))]
    fn getcolnp(
        &self,
        py: Python<'_>,
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
        // Typed-buffer fast path: for a read handle over a StandardStMan
        // numeric column, decode each cell straight from the mapped data
        // file into `buf` — no per-cell `RecordValue`/`ArrayData` Vec. This
        // holds only the numpy result buffer plus a small read window
        // (mapped pages are dropped as the scan progresses) instead of the
        // result buffer + a full per-cell copy + the whole mapped file.
        // Clone the Arc out of the lock before the (GIL-released) decode so
        // that concurrent reads through this handle are not serialised by the
        // inner mutex.
        let read_table = {
            let inner = self.inner.lock().unwrap();
            match &*inner {
                Inner::Read(t) => Some(std::sync::Arc::clone(t)),
                Inner::Write { .. } => None,
            }
        };
        if let Some(t) = &read_table {
            if t.raw_column_supported(col_idx)
                && fill_numpy_raw(py, t, col_idx, startrow, nrow, buf)?
            {
                return Ok(());
            }
        }
        // The per-cell decode is pure Rust; release the GIL so the dask
        // scheduler can overlap this read with independent work.
        let cells = py.allow_threads(|| self.read_col(col_idx, startrow, nrow))?;
        let cell = cell_shape_of(&cells).iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(py, buf, &cells, cell)
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
        let cells = py.allow_threads(|| self.read_colslice(col_idx, &blc, &trc, startrow, nrow))?;
        self.column_to_python(py, col_idx, &cells)
    }

    /// `getcolslicenp(column, buf, blc, trc, startrow, nrow)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, buf, blc, trc, startrow = 0, nrow = -1))]
    fn getcolslicenp(
        &self,
        py: Python<'_>,
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
        let cells = py.allow_threads(|| self.read_colslice(col_idx, &blc, &trc, startrow, nrow))?;
        let cell = cell_shape_of(&cells).iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(py, buf, &cells, cell)
    }

    /// `getcell(column, row)` -> numpy array (or scalar/list).
    fn getcell(&self, py: Python<'_>, column: &str, row: u64) -> PyResult<Py<PyAny>> {
        let col_idx = self.col_index(column)?;
        let v = self.read_cell(col_idx, row)?;
        if let RecordValue::Array(a) = &v {
            // Array cells keep every stored dimension for `getcol` — casacore
            // returns (1, 79) for a 1-row 79-channel column; the leading-row
            // singleton trim is `getcell`/`getvarcol` semantics only, and
            // producing a bare (79,) here broke skarabina's CHAN_FREQ read
            // (`chan_freq[0]` collapsing to a scalar).
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
    /// Number of rows the shared writable store holds for `col_idx` (0 on a
    /// read-only handle).  Used to tell a full-column `putcol` (which can be
    /// written without reading the old values) from a partial one.
    fn store_col_len(&self, col_idx: usize) -> u64 {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Write { shared } => shared.lock().unwrap().wt.col_len(col_idx) as u64,
            _ => 0,
        }
    }

    fn putcell(
        &self,
        py: Python<'_>,
        column: &str,
        row: u64,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let col_idx = self.col_index(column)?;
        let rec = keyword_value(py, value)?;
        self.put_cell(col_idx, row, rec)?;
        Ok(())
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
        // Multidim-string dict form: `{"shape": [nrow, *cell], "array": [...]}`
        // (dask-ms's multidim string writes) — split into per-row cells.
        if value.cast::<PyDict>().is_ok()
            && value.cast::<PyDict>().unwrap().contains("shape")?
            && value.cast::<PyDict>().unwrap().contains("array")?
        {
            use casacure::record::ArrayData;
            let av = match convert::pyobject_to_record(py, value)? {
                RecordValue::Array(a) => a,
                other => {
                    return Err(PyTypeError::new_err(format!(
                        "expected a string array dict, got {other:?}"
                    )));
                }
            };
            let ArrayData::String(strings) = &av.data else {
                return Err(PyTypeError::new_err("expected string array data"));
            };
            if av.shape.is_empty() {
                return Err(PyValueError::new_err("string array dict has no shape"));
            }
            let nrow = av.shape[0] as usize;
            let cell_shape: Vec<u32> = av.shape[1..].to_vec();
            let cell = cell_shape.iter().product::<u32>() as usize;
            for r in 0..nrow {
                let from = r * cell;
                let to = ((r + 1) * cell).min(strings.len());
                let rec = RecordValue::Array(casacure::record::ArrayValue {
                    shape: cell_shape.clone(),
                    data: ArrayData::String(strings[from..to].to_vec()),
                });
                self.put_cell(col_idx, startrow + r as u64, rec)?;
            }
            return Ok(());
        }
        // Dict form: `{"rN": value, ...}` — per-row scalar/array writes
        // (dask-ms writes scalar varcols this way).
        if value.cast::<PyDict>().is_ok() {
            let mut rows: Vec<(u64, Bound<'_, PyAny>)> = Vec::new();
            for (k, v) in value.cast::<PyDict>().unwrap().iter() {
                let s = k.extract::<String>()?;
                if let Some(n) = s.strip_prefix('r') {
                    let idx: u64 = n
                        .parse()
                        .map_err(|_| PyValueError::new_err(format!("bad row key {s}")))?;
                    rows.push((idx, v));
                }
            }
            rows.sort_by_key(|(n, _)| *n);
            for (i, (row_offset, v)) in rows.into_iter().enumerate() {
                let row = startrow + row_offset;
                let desc = self.desc();
                let is_array_col = matches!(
                    desc.columns[col_idx].kind,
                    core::tabledesc::ColumnKind::Array
                );
                let mut cells = self.value_to_cells(py, col_idx, &v, 1, is_array_col, true)?;
                let rec = cells.pop().unwrap_or(RecordValue::Int(0));
                self.put_cell(col_idx, row, rec)?;
                let _ = i;
            }
            return Ok(());
        }
        let nrow = if nrow <= 0 {
            value.len()? as u64
        } else {
            nrow as u64
        };
        // A putcol covering the whole column replaces it without reading the
        // old values; a partial write overlays only its rows' cells, leaving
        // every other row on disk until the next flush.
        let total = self.store_col_len(col_idx);
        let full = startrow == 0 && nrow >= total;
        let desc = self.desc();
        let is_array_col = matches!(
            desc.columns[col_idx].kind,
            core::tabledesc::ColumnKind::Array
        );
        let values = self.value_to_cells(py, col_idx, value, nrow, is_array_col, false)?;
        // Whole-batch store: lock the shared writable store once and reuse
        // the core table's column write (no per-cell re-locking). Runs with
        // the GIL released so the dask scheduler can overlap independent
        // work. Per-cell type/shape validation is still applied (parity with
        // upstream's `put_cells`).
        self.put_cells_batch(py, col_idx, startrow, values)?;
        let _ = full;
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
            let mut cells = self.value_to_cells(py, col_idx, &v, 1, is_array_col, true)?;
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
        let _ = py;
        let col_idx = self.col_index(column)?;
        let startrow = startrow.max(0) as u64;
        let nrow = if nrow <= 0 {
            value.len()? as u64
        } else {
            nrow as u64
        };
        // Full-cell slice (blc=0.., trc=-1) -> plain putcol.
        let full =
            blc.iter().all(|&b| b <= 0) && trc.len() == blc.len() && trc.iter().all(|&t| t < 0);
        if full {
            return self.putcol(py, column, value, startrow as i64, nrow as i64);
        }
        // Overlay `value[nrow, sub...]` into each fixed cell at logical
        // blc..trc. The numpy array is `(nrow, s0, s1, ...)` in C order.
        let flat = convert::numpy_to_record_flat(value)
            .ok_or_else(|| PyTypeError::new_err("putcolslice: unsupported element type"))??;
        let shape_obj = value.getattr("shape")?;
        let shape: Vec<usize> = shape_obj.extract()?;
        if shape.is_empty() {
            return Err(PyValueError::new_err("putcolslice: empty value"));
        }
        let sub: Vec<usize> = shape[1..].to_vec();
        let sub_cell = sub.iter().product::<usize>().max(1);
        let starts: Vec<usize> = blc.iter().map(|&b| b.max(0) as usize).collect();
        // inclusive full-cell ends; -1 means "to the end of the sub-slice".
        let ends: Vec<usize> = trc
            .iter()
            .zip(starts.iter())
            .zip(sub.iter())
            .map(|((&t, &s), &subd)| {
                if t < 0 {
                    s + subd - 1
                } else {
                    t.max(0) as usize
                }
            })
            .collect();

        for r in 0..nrow as usize {
            let row = startrow + r as u64;
            let mut cell = match self.read_cell(col_idx, row) {
                Ok(c) => c,
                Err(_) => RecordValue::Int(0),
            };
            // Cell's stored (as-given) shape; default for a missing cell.
            let cshape: Vec<usize> = match &cell {
                RecordValue::Array(a) => a.shape.iter().map(|&d| d as usize).collect(),
                _ => {
                    // Build a zero default of the fixed shape when present.
                    let desc = self.desc();
                    let fixed: Vec<usize> = desc.columns[col_idx]
                        .shape
                        .clone()
                        .map(|s| s.iter().rev().map(|&d| d.max(0) as usize).collect())
                        .unwrap_or_default();
                    let n = fixed.iter().product::<usize>().max(1);
                    let dt = desc.columns[col_idx].data_type;
                    cell = RecordValue::Array(core::record::ArrayValue {
                        shape: fixed.iter().map(|&d| d as u32).collect(),
                        data: zero_elements(dt, n),
                    });
                    let _ = n;
                    fixed
                }
            };
            let mut grid = convert::cell_logical_flat(&cell);
            let base = r * sub_cell;
            for idx in 0..sub_cell {
                let coords = convert::unflatten(idx, &sub);
                let mut loc = vec![0usize; coords.len()];
                for k in 0..coords.len() {
                    let s = starts[k];
                    let span = ends[k].saturating_sub(s) + 1;
                    if coords[k] >= span {
                        continue;
                    }
                    loc[k] = s + coords[k];
                }
                let lf = convert::flatten_coords(&loc, &cshape);
                if lf < grid.len() && base + idx < flat.len() {
                    grid[lf] = flat[base + idx].clone();
                }
            }
            let stored = convert::cell_from_logical(grid, &cshape);
            self.put_cell(col_idx, row, stored)?;
        }
        Ok(())
    }

    /// `putkeyword(name, value)`.
    fn putkeyword(&self, py: Python<'_>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let rec = keyword_value(py, value)?;
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    let mut s = shared.lock().unwrap();
                    s.wt.putkeyword(name, rec);
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
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
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    let mut s = shared.lock().unwrap();
                    s.wt.removekeyword(name);
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
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
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    let mut s = shared.lock().unwrap();
                    s.wt.putcolkeyword(col_idx, name, rec).map_err(err)?;
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
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
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    let mut s = shared.lock().unwrap();
                    s.wt.removecolkeyword(col_idx, name).map_err(err)?;
                    s.dirty = true;
                }
                _ => return Err(PyValueError::new_err("table is not writable")),
            }
        }
        Ok(())
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
        let base = self.dir_of();
        Ok(desc_to_pydict(py, &desc, Some(&base))?.into_any().unbind())
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
            Inner::Read(t) => {
                let n = t.nrows();
                // Row bounds are checked here so an out-of-range row is the
                // documented ValueError, not whatever the storage manager
                // happens to raise (python-casacore raises "no such row";
                // casacure's contract is ValueError for read violations).
                if row >= n {
                    return Err(row_out_of_range(row, n));
                }
                t.getcell(col_idx, row).map_err(err)
            }
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                let n = s.wt.col_len(col_idx) as u64;
                if row >= n {
                    return Err(row_out_of_range(row, n));
                }
                merged_cell(&s, col_idx, row)
            }
        }
    }

    fn read_cellslice(
        &self,
        col_idx: usize,
        row: u64,
        blc: &[i64],
        trc: &[i64],
    ) -> PyResult<RecordValue> {
        // python-casacore's getcellslice blc/trc are 1-based inclusive with
        // -1 meaning the last element; dask-ms uses the (-1,..) idiom for
        // "the whole cell", which the core 0-based slicer rejects.  Any -1
        // corner means "take the whole cell" (equivalent to the core's
        // empty-slice form).
        if blc.iter().any(|&b| b < 0) || trc.iter().any(|&t| t < 0) {
            return self.read_cell(col_idx, row);
        }
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t.getcellslice(col_idx, row, blc, trc).map_err(err),
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                let n = s.wt.col_len(col_idx) as u64;
                if row >= n {
                    return Err(row_out_of_range(row, n));
                }
                let cell = merged_cell(&s, col_idx, row)?;
                slice_cell(cell, blc, trc)
            }
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
        // Same 1-based/-1 "whole cell" idiom as getcellslice.
        if blc.iter().any(|&b| b < 0) || trc.iter().any(|&t| t < 0) {
            return self.read_col(col_idx, startrow, nrow);
        }
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => t
                .getcolslice(col_idx, blc, trc, startrow, nrow)
                .map_err(err),
            Inner::Write { shared, .. } => {
                // Per-row merged read (the on-disk snapshot does not have the
                // rows added since the last flush), then the sub-array slice.
                let s = shared.lock().unwrap();
                let total = s.wt.col_len(col_idx) as u64;
                let end = startrow.saturating_add(nrow);
                if end > total {
                    return Err(range_out_of_range(startrow, end, total));
                }
                let mut out = Vec::with_capacity(nrow as usize);
                for r in startrow..end {
                    let cell = merged_cell(&s, col_idx, r)?;
                    out.push(slice_cell(cell, blc, trc)?);
                }
                Ok(out)
            }
        }
    }

    fn put_cell(&self, col_idx: usize, row: u64, value: RecordValue) -> PyResult<()> {
        self.put_cells(col_idx, row, std::iter::once(value))
    }

    /// Write consecutive cells from `startrow`, taking the handle and store
    /// locks once for the whole batch (a `putcol` chunk), not per cell.
    fn put_cells(
        &self,
        col_idx: usize,
        startrow: u64,
        values: impl IntoIterator<Item = RecordValue>,
    ) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        let Inner::Write { shared, .. } = &mut *inner else {
            return Err(PyValueError::new_err("table is not writable"));
        };
        let mut s = shared.lock().unwrap();
        for (row, value) in (startrow..).zip(values) {
            // Validate before writing: the putcol coercion has already
            // produced the column's exact type, so a mismatch here is an
            // incompatible write (e.g. a string into a double column) that
            // casacore rejects rather than silently storing — and a
            // fixed-shape array column must receive exactly its declared
            // cell shape. The desc is borrowed inside the shared lock (no
            // per-cell clone on the hot putcol path).
            if let Some(col) = s.wt.desc().columns.get(col_idx) {
                if !record_fits_column(col, &value) {
                    return Err(PyTypeError::new_err(format!(
                        "putcol/putcell: {} cannot be stored in column {} (valueType {})",
                        value_name(&value),
                        col.name,
                        ::casacure::casa_value_type(col.data_type)
                    )));
                }
                if let (Some(fixed), RecordValue::Array(a)) = (&col.shape, &value) {
                    // Logical cell shape = the reversed stored shape.
                    let matches = a.shape.len() == fixed.len()
                        && a.shape
                            .iter()
                            .zip(fixed.iter().rev())
                            .all(|(&got, &want)| got == want.max(0) as u32);
                    if !fixed.is_empty() && !matches {
                        let logical: Vec<u32> =
                            fixed.iter().rev().map(|&d| d.max(0) as u32).collect();
                        return Err(PyValueError::new_err(format!(
                            "putcol: cell shape {:?} does not match column {} fixed shape {:?}",
                            a.shape, col.name, logical
                        )));
                    }
                }
            }
            s.wt.putcell(col_idx, row, value).map_err(err)?;
            s.dirty = true;
        }
        Ok(())
    }

    /// Store a whole `putcol` batch of cells (one `RecordValue` per row).
    ///
    /// Like [`Table::put_cells`]: the handle and shared-store locks are taken
    /// once for the whole batch and per-cell type/shape validation is kept,
    /// but the store loop runs with the GIL released so the dask scheduler can
    /// overlap independent writes.
    fn put_cells_batch(
        &self,
        py: Python<'_>,
        col_idx: usize,
        startrow: u64,
        values: Vec<RecordValue>,
    ) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        let Inner::Write { shared, .. } = &mut *inner else {
            return Err(PyValueError::new_err("table is not writable"));
        };
        let mut s = shared.lock().unwrap();
        // Validate against the column's declared type and fixed shape before
        // writing (parity with the per-cell `put_cell` checks).  The desc is
        // cloned so the whole validation + store loop can run GIL-free.
        let desc = s.wt.desc().clone();
        let wt = &mut s.wt;
        py.allow_threads(|| -> PyResult<()> {
            for (row, value) in (startrow..).zip(values) {
                if let Some(col) = desc.columns.get(col_idx) {
                    if !record_fits_column(col, &value) {
                        return Err(PyTypeError::new_err(format!(
                            "putcol/putcell: {} cannot be stored in column {} (valueType {})",
                            value_name(&value),
                            col.name,
                            ::casacure::casa_value_type(col.data_type)
                        )));
                    }
                    if let (Some(fixed), RecordValue::Array(a)) = (&col.shape, &value) {
                        // Logical cell shape = the reversed stored shape.
                        let matches = a.shape.len() == fixed.len()
                            && a.shape
                                .iter()
                                .zip(fixed.iter().rev())
                                .all(|(&got, &want)| got == want.max(0) as u32);
                        if !fixed.is_empty() && !matches {
                            let logical: Vec<u32> =
                                fixed.iter().rev().map(|&d| d.max(0) as u32).collect();
                            return Err(PyValueError::new_err(format!(
                                "putcol: cell shape {:?} does not match column {} fixed shape {:?}",
                                a.shape, col.name, logical
                            )));
                        }
                    }
                }
                wt.putcell(col_idx, row, value).map_err(err)?;
            }
            Ok(())
        })?;
        s.dirty = true;
        Ok(())
    }

    /// Convert user `putcol` data into one `RecordValue` per row.
    fn value_to_cells(
        &self,
        py: Python<'_>,
        col_idx: usize,
        value: &Bound<'_, PyAny>,
        nrow: u64,
        is_array_col: bool,
        varcol: bool,
    ) -> PyResult<Vec<RecordValue>> {
        // The column's declared element type, for scalar (list/tuple) values:
        // casacore casts every element to it on putcol.
        let col_dt = self.desc().columns.get(col_idx).map(|c| c.data_type);
        // Normalize numpy unicode/byte string arrays to nested Python lists
        // so the string handling paths (the scalar list branch and the array
        // list->typed-ndarray normalization) see them.
        let stringified: Option<Bound<'_, PyAny>> = if value.getattr("dtype").is_ok() {
            let kind: String = value.getattr("dtype")?.getattr("kind")?.extract()?;
            if kind == "U" || kind == "S" {
                Some(value.call_method0("tolist")?)
            } else {
                None
            }
        } else {
            None
        };
        let value = match &stringified {
            Some(s) => s,
            None => value,
        };
        // Coerce numeric ndarrays (scalar AND array columns) to the column's
        // element type (casacore casts, e.g. complex64 -> dcomplex when the
        // column is C8, int64/int16 -> int32 for an Int column). String
        // arrays are already handled by `stringified` above, so skip them.
        let mut coerced: Option<Bound<'_, PyAny>> = None;
        if stringified.is_none() {
            if let Some(npd) = core::record::data_type_to_np(
                self.desc().columns.get(col_idx).map(|c| &c.data_type),
            ) {
                // Any ndarray (typed ndarrays don't all downcast to the
                // `PyArrayDyn<PyAny>` form, e.g. complex64).
                if value.getattr("dtype").is_ok()
                    && value.cast::<PyList>().is_err()
                    && value.cast::<PyDict>().is_err()
                {
                    let dtype = value.getattr("dtype")?;
                    let kind: String = dtype.getattr("kind")?.extract()?;
                    let itemsize: i64 = dtype.getattr("itemsize")?.extract()?;
                    let have = core::record::np_kind_itemsize(&kind, itemsize);
                    if have != Some(npd) {
                        coerced = Some(value.call_method1("astype", (npd,))?);
                    }
                }
            }
        }
        let value = match &coerced {
            Some(c) => c,
            None => value,
        };
        if is_array_col {
            // Accept a 2-D+ ndarray or a list of row-arrays.
            if value.cast::<PyDict>().is_ok() {
                let rec = convert::py_to_string_array(py, value)?;
                return Ok(vec![rec]);
            }
            // Plain Python list/tuple (putvarcol row values, or putcol with
            // lists-of-rows): normalize to a typed ndarray of the column's
            // element type and fall through to the ndarray handling below.
            let mut normalized: Vec<Bound<'_, PyAny>> = Vec::new();
            let mut value = value;
            if value.cast::<PyList>().is_ok() || value.cast::<PyTuple>().is_ok() {
                let mut arr = py.import("numpy")?.getattr("asarray")?.call1((value,))?;
                // String columns have no numpy mapping in `data_type_to_np`;
                // unicode/byte arrays must become object arrays for the
                // string-cell handler to see them.
                let npd = core::record::data_type_to_np(
                    self.desc().columns.get(col_idx).map(|c| &c.data_type),
                )
                .or_else(|| {
                    matches!(
                        self.desc().columns.get(col_idx).map(|c| c.data_type),
                        Some(core::record::DataType::String)
                    )
                    .then_some("object")
                });
                if let Some(npd) = npd {
                    let dt = arr.getattr("dtype")?;
                    let kind: String = dt.getattr("kind")?.extract()?;
                    let itemsize: i64 = dt.getattr("itemsize")?.extract()?;
                    if core::record::np_kind_itemsize(&kind, itemsize) != Some(npd) {
                        arr = arr.call_method1("astype", (npd,))?;
                    }
                }
                normalized.push(arr);
                value = normalized.last().unwrap();
            }
            // Complex arrays (the coercion above has already matched the
            // array to the column precision); numpy 0.26 names: Complex32 =
            // c64 (float32 complex), Complex64 = c128 (float64 complex).
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<numpy::Complex32>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, |v| {
                    core::record::ArrayData::Complex(v.into_iter().map(|c| (c.re, c.im)).collect())
                });
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<numpy::Complex64>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, |v| {
                    core::record::ArrayData::DComplex(v.into_iter().map(|c| (c.re, c.im)).collect())
                });
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<Py<PyAny>>>() {
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
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<f64>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(
                    &readonly,
                    nrow,
                    varcol,
                    core::record::ArrayData::Double,
                );
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<f32>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(
                    &readonly,
                    nrow,
                    varcol,
                    core::record::ArrayData::Float,
                );
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<u8>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(
                    &readonly,
                    nrow,
                    varcol,
                    core::record::ArrayData::UChar,
                );
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<i16>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(
                    &readonly,
                    nrow,
                    varcol,
                    core::record::ArrayData::Short,
                );
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<u32>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, core::record::ArrayData::UInt);
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<u16>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(
                    &readonly,
                    nrow,
                    varcol,
                    core::record::ArrayData::UShort,
                );
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<i64>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(
                    &readonly,
                    nrow,
                    varcol,
                    core::record::ArrayData::Int64,
                );
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<i32>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, core::record::ArrayData::Int);
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<bool>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, core::record::ArrayData::Bool);
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<Complex32>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, |v| {
                    core::record::ArrayData::Complex(v.into_iter().map(|c| (c.re, c.im)).collect())
                });
            }
            if let Ok(arr) = value.cast::<numpy::PyArrayDyn<Complex64>>() {
                let readonly = arr.readonly();
                return ndarray_cells_typed(&readonly, nrow, varcol, |v| {
                    core::record::ArrayData::DComplex(v.into_iter().map(|c| (c.re, c.im)).collect())
                });
            }
            return Err(PyTypeError::new_err(format!(
                "putcol: unsupported array data {}",
                value.getattr("dtype")?.str()?.to_str()?
            )));
        }
        // Scalar column: 1-D array (or list/tuple) of scalars. Each element
        // is cast to the column's declared type (byte-identical semantics).
        if let Ok(list) = value.cast::<PyList>() {
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                let rec = convert::pyobject_to_record(py, &item)?;
                out.push(match col_dt.as_ref() {
                    Some(dt) => convert::cast_scalar_to(dt, &rec),
                    None => rec,
                });
            }
            return Ok(out);
        }
        if let Ok(tup) = value.cast::<PyTuple>() {
            let mut out = Vec::with_capacity(tup.len());
            for item in tup.iter() {
                let rec = convert::pyobject_to_record(py, &item)?;
                out.push(match col_dt.as_ref() {
                    Some(dt) => convert::cast_scalar_to(dt, &rec),
                    None => rec,
                });
            }
            return Ok(out);
        }
        if let Ok(arr) = value.cast::<numpy::PyArrayDyn<Py<PyAny>>>() {
            let readonly = arr.readonly();
            let mut out = Vec::with_capacity(readonly.as_array().len());
            for e in readonly.as_array().iter() {
                out.push(convert::pyobject_to_record(py, e.bind(py))?);
            }
            return Ok(out);
        }
        macro_rules! scalar_num {
            ($ty:ty, $f:expr) => {{
                if let Ok(arr) = value.cast::<numpy::PyArrayDyn<$ty>>() {
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
        scalar_num!(i16, |e: &i16| RecordValue::Short(*e));
        scalar_num!(u32, |e: &u32| RecordValue::UInt(*e));
        scalar_num!(u8, |e: &u8| RecordValue::UChar(*e));
        scalar_num!(u16, |e: &u16| RecordValue::UShort(*e));
        scalar_num!(bool, |e: &bool| RecordValue::Bool(*e));
        scalar_num!(Complex64, |e: &Complex64| RecordValue::DComplex(e.re, e.im));
        scalar_num!(Complex32, |e: &Complex32| RecordValue::Complex(e.re, e.im));
        // Fall back to a single-value conversion (numpy scalars, etc.).
        let rec = convert::pyobject_to_record(py, value)?;
        if matches!(rec, RecordValue::Array(_)) {
            return Err(PyTypeError::new_err(
                "putcol: unexpected array for a scalar column",
            ));
        }
        Ok(vec![rec])
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
            // Empty cells (e.g. a taql result with no rows): return empty.
            if let Some(RecordValue::String(_) | RecordValue::Table(_)) = cells.first() {
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
        convert::arrays_to_ndarray_getcol(py, cells, &cell_shape)
    }
}

fn reshape_cell(shape: &[usize]) -> usize {
    shape.iter().skip(1).product::<usize>().max(1)
}

fn ndarray_cells_typed<T: numpy::Element + Copy>(
    arr: &numpy::PyReadonlyArrayDyn<'_, T>,
    nrow: u64,
    varcol: bool,
    build: impl Fn(Vec<T>) -> ArrayData,
) -> PyResult<Vec<RecordValue>> {
    let shape: Vec<usize> = arr.as_array().shape().to_vec();
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    // `putcol` arrays are (nrow, *cell) so the cell shape drops the first
    // dim; `putvarcol` values are full per-row cells (keep the whole shape).
    let cell_shape: &[usize] = if varcol { &shape } else { &shape[1..] };
    let cell = cell_shape.iter().product::<usize>().max(1);
    let view = arr.as_array();
    // Row-major element order. A C-contiguous array (the usual numpy input)
    // is borrowed as one slice; anything else is copied once into logical
    // order. Rows are then plain sub-slices: re-walking the ndarray iterator
    // from element 0 for every row (`iter().skip(start)`) made a putcol
    // quadratic in the chunk size.
    let owned: Vec<T>;
    let flat: &[T] = match arr.as_slice() {
        Ok(s) => s,
        Err(_) => {
            owned = view.iter().copied().collect();
            &owned
        }
    };
    let casa_shape: Vec<u32> = cell_shape.iter().map(|&s| s as u32).collect();
    let mut out = Vec::with_capacity(nrow.min(flat.len() as u64) as usize);
    for r in 0..nrow as usize {
        let start = r * cell;
        if start >= flat.len() {
            break;
        }
        let stop = ((r + 1) * cell).min(flat.len());
        // Build each row's typed element buffer directly from the borrowed
        // numpy data (one copy, no intermediate per-element `RecordValue`).
        let elems: Vec<T> = flat[start..stop].to_vec();
        // Array columns always store `RecordValue::Array` cells, even for a
        // 1-element cell (e.g. an ncorr=1 CORR_TYPE column in an MS): the
        // storage managers and readers expect an Array value there.
        out.push(RecordValue::Array(core::record::ArrayValue {
            shape: casa_shape.clone(),
            data: build(elems),
        }));
    }
    Ok(out)
}

/// Build the python-casacore `getcoldesc` dict for a column.
fn self_coldesc<'py>(
    py: Python<'py>,
    col: &core::tabledesc::ColumnDesc,
    _base: Option<&std::path::Path>,
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
        }
        d.set_item("_c_order", true)?;
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
        core::DmSpec::TiledShapeStMan {
            default_tile_shape,
            seqnr,
            hypercubes,
        } => {
            d.set_item("MaxCacheSize", 0)?;
            d.set_item("DEFAULTTILESHAPE", default_tile_shape.clone())?;
            d.set_item("MAXIMUMCACHESIZE", 0)?;
            let cubes = PyDict::new(py);
            for (i, c) in hypercubes.iter().enumerate() {
                let cd = PyDict::new(py);
                cd.set_item("CubeShape", c.cube_shape.clone())?;
                cd.set_item("TileShape", c.tile_shape.clone())?;
                cd.set_item("CellShape", c.cell_shape.clone())?;
                cd.set_item("BucketSize", c.bucket_size)?;
                cd.set_item("ID", PyDict::new(py))?;
                cubes.set_item(format!("*{}", i + 1), cd)?;
            }
            d.set_item("HYPERCUBES", cubes)?;
            d.set_item("SEQNR", seqnr)?;
            d.set_item("IndexSize", hypercubes.len())?;
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
    name: &Bound<'_, PyAny>,
    tabledesc: Option<&Bound<'_, PyAny>>,
    nrow: i64,
    _dminfo: Option<&Bound<'_, PyAny>>,
    readonly: bool,
    _ack: bool,
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Table> {
    let name = path_string(name)?;
    let desc_json = match tabledesc {
        Some(d) if !d.is_none() => {
            if let Ok(dict) = d.cast::<PyDict>() {
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
        &name,
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
    // Collect the wrapped core tables from any `casacure.tables.table`
    // arguments and hold every mutex guard for the whole call so the inner
    // core tables stay borrowed. All statements (SELECT, UPDATE, DELETE,
    // INSERT, ALTER, DROPTABLE, SHOW/HELP, CALC, COUNT, CREATE TABLE) may
    // reference `$N` tables; CREATE TABLE takes only an on-disk path, so
    // the `tables` list is optional for it.
    let objects: Vec<PyRef<'_, Table>>;
    let locks: Vec<std::sync::MutexGuard<'_, Inner>>;
    let shared_guards: Vec<Option<std::sync::MutexGuard<'_, WriteData>>>;
    let core_refs: Vec<&::casacure::Table>;
    if let Some(ts) = tables {
        // Persist any pending (unflushed) writes so the core snapshots below
        // (and any statement) run against the current cell state; write ops
        // buffer into the shared store and only flush on request.
        for item in ts.iter() {
            let obj = item
                .extract::<PyRef<'_, Table>>()
                .map_err(|_| PyValueError::new_err("taql: expected table objects"))?;
            let inner = obj.inner.lock().unwrap();
            if let Inner::Write { shared, .. } = &*inner {
                flush_if_dirty(shared)?;
            }
        }
        objects = ts
            .iter()
            .map(|item| {
                item.extract::<PyRef<'_, Table>>()
                    .map_err(|_| PyValueError::new_err("taql: expected table objects"))
            })
            .collect::<PyResult<_>>()?;
        locks = objects.iter().map(|t| t.inner.lock().unwrap()).collect();
        shared_guards = locks
            .iter()
            .map(|g| match &**g {
                Inner::Write { shared, .. } => Some(shared.lock().unwrap()),
                _ => None,
            })
            .collect();
        core_refs = locks
            .iter()
            .zip(shared_guards.iter())
            .map(|(g, sg)| match &**g {
                Inner::Read(t) => t.as_ref(),
                Inner::Write { .. } => &sg.as_ref().expect("write table has a shared guard").read,
            })
            .collect();
    } else {
        objects = Vec::new();
        locks = Vec::new();
        shared_guards = Vec::new();
        core_refs = Vec::new();
        // objects/locks/guards only exist to keep borrows alive.
        let _ = (&objects, &locks, &shared_guards);
    }
    let mut touched: Vec<std::path::PathBuf> = Vec::new();
    let result = core::taql::execute_into(query, &core_refs, &mut touched).map_err(err)?;
    // A mutating statement (UPDATE/DELETE/INSERT/ALTER/DROPTABLE, CREATE,
    // SELECT INTO) rewrote the on-disk tables; refresh any materialised
    // writable state cached for those directories so live handles and the
    // next open see the fresh files instead of a stale snapshot (a stale
    // handle's `close()` would otherwise regenerate the old cells and
    // clobber the change). A directory that no longer holds a table
    // (`DROPTABLE`) drops its cached entry instead.
    drop(shared_guards); // release the WriteData locks before refresh() re-locks them
    if !touched.is_empty() {
        let mut reg = write_registry().lock().unwrap();
        for dir in &touched {
            if let Some(shared) = reg.get(dir) {
                if !refresh_write(dir, shared) {
                    reg.remove(dir);
                }
            }
        }
    }
    Ok(match result {
        core::taql::TaqlResult::Query(out) => taql_result_to_table(py, out)?
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        core::taql::TaqlResult::Created(path) => {
            let t = Table::open_or_create(py, &path.display().to_string(), None, 0, true)?;
            t.into_pyobject(py)?.into_any().unbind()
        }
    })
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
    let shared = std::sync::Arc::new(std::sync::Mutex::new(WriteData {
        read,
        wt,
        dirty: false,
    }));
    register_write(&dir, &shared);
    Ok(Table {
        path: dir.display().to_string(),
        writable: true,
        inner: Mutex::new(Inner::Write { shared }),
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
    base: Option<&std::path::Path>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for col in &desc.columns {
        d.set_item(&col.name, self_coldesc(py, col, base)?.into_any())?;
    }
    d.set_item(
        "_keywords_",
        convert::table_record_to_dict_ctx(py, &desc.keywords, base)?,
    )?;
    d.set_item(
        "_private_keywords_",
        convert::table_record_to_dict_ctx(py, &desc.private_keywords, base)?,
    )?;
    let empty = PyDict::new(py);
    d.set_item("_define_hypercolumn_", empty)?;
    Ok(d)
}

#[cfg(test)]
mod subtable_resolve_tests {
    use super::resolve_stored_subtable;

    const TABLE_DIR: &str = "/data/ms_cure.ms";
    const BASE: &str = "/data";

    #[test]
    fn casacore_dot_form_resolves_against_parent() {
        // One `./` is casacore's sibling form: stored `./SUB.tab` by a table
        // at <dir>/P.tab resolves to <dir>/SUB.tab (the containing
        // directory) — a real casacore fixture and casacore's
        // `Path::addDirectory` agree.
        assert_eq!(
            resolve_stored_subtable("./ANTENNA", TABLE_DIR.as_ref(), BASE.as_ref()),
            "/data/ANTENNA"
        );
    }

    #[test]
    fn casacore_double_dot_form_resolves_against_table_dir() {
        // Two (or more) `./` is casacore's in-table-dir form: a real MS
        // stores its SPECTRAL_WINDOW link as `././SPECTRAL_WINDOW` and
        // casacore's getsubtables() returns `<table_dir>/SPECTRAL_WINDOW`.
        assert_eq!(
            resolve_stored_subtable("././SPECTRAL_WINDOW", TABLE_DIR.as_ref(), BASE.as_ref()),
            "/data/ms_cure.ms/SPECTRAL_WINDOW"
        );
    }

    #[test]
    fn absolute_stored_path_is_kept() {
        assert_eq!(
            resolve_stored_subtable("/other/ms/SOURCE", TABLE_DIR.as_ref(), BASE.as_ref()),
            "/other/ms/SOURCE"
        );
    }

    #[test]
    fn legacy_dot_absolute_form_uses_absolute_tail() {
        // Written by a relativisation against a relative table directory.
        assert_eq!(
            resolve_stored_subtable(
                ".//data/ms_cure.ms/SOURCE",
                TABLE_DIR.as_ref(),
                BASE.as_ref()
            ),
            "/data/ms_cure.ms/SOURCE"
        );
    }

    #[test]
    fn legacy_bare_parent_relative_form_does_not_double() {
        // Written by a default_ms given a relative MS path.
        assert_eq!(
            resolve_stored_subtable("ms_cure.ms/ANTENNA", TABLE_DIR.as_ref(), BASE.as_ref()),
            "/data/ms_cure.ms/ANTENNA"
        );
    }

    #[test]
    fn other_bare_names_are_table_dir_relative() {
        assert_eq!(
            resolve_stored_subtable("SUB", TABLE_DIR.as_ref(), BASE.as_ref()),
            "/data/ms_cure.ms/SUB"
        );
    }
}
