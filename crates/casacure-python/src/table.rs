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
    /// The table directory this backing belongs to (see [`DirPin`]).
    pin: DirPin,
    /// The column names on disk as of the last sync (backing creation,
    /// flush, or out-of-band refresh) — NOT the live `wt` desc, which
    /// `addcols` extends before the flush. Compared with a fresh parse of
    /// `table.dat` when a later open reuses the backing, so a column
    /// another writer added (python-casacore, dask-ms; issue #30) makes
    /// the backing re-materialise instead of serving the stale schema —
    /// and a *dirty* backing is refused rather than flushed over the
    /// foreign column, which would rewrite `table.dat` without it.
    disk_columns: Vec<String>,
    /// Set for a TaQL query result materialised under the temp dir: the
    /// directory is removed when the last handle on this backing drops.
    /// Declared LAST so `read`/`wt` are closed before it is removed.
    scratch: Option<ScratchDir>,
}

/// The column names in `table.dat` right now (`None` if unreadable — the
/// directory no longer holds a table).
fn disk_column_names(dir: &std::path::Path) -> Option<Vec<String>> {
    let bytes = std::fs::read(dir.join("table.dat")).ok()?;
    let dat = core::parse_table_dat(&bytes).ok()?;
    Some(dat.desc.columns.iter().map(|c| c.name.clone()).collect())
}

/// Whether `table.dat`'s column set differs from `names` (the backing's
/// last-synced view): another writer added, removed or renamed a column
/// out of band (issue #30). Unreadable `table.dat` counts as changed (the
/// callers re-materialise from disk or drop the entry anyway).
fn disk_columns_differ(dir: &std::path::Path, names: &[String]) -> bool {
    match disk_column_names(dir) {
        Some(now) => now != names,
        None => true,
    }
}

/// A TaQL result table's temp directory, removed when dropped (casacore keeps
/// query results in memory or as scratch tables deleted on close; casacure
/// materialises them on disk, so it has to delete them itself).  Every live one
/// is also in [`SCRATCH_DIRS`], which the `atexit` hook sweeps: dask-ms caches
/// table proxies for the life of the process, and objects still referenced at
/// interpreter shutdown are not reliably dropped.
struct ScratchDir(std::path::PathBuf);

static SCRATCH_DIRS: std::sync::OnceLock<Mutex<std::collections::HashSet<std::path::PathBuf>>> =
    std::sync::OnceLock::new();

fn scratch_dirs() -> &'static Mutex<std::collections::HashSet<std::path::PathBuf>> {
    SCRATCH_DIRS.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

impl ScratchDir {
    fn new(dir: std::path::PathBuf) -> ScratchDir {
        scratch_dirs().lock().unwrap().insert(dir.clone());
        ScratchDir(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        if let Ok(mut set) = scratch_dirs().lock() {
            set.remove(&self.0);
        }
    }
}

impl Drop for Table {
    /// A TaQL result's backing is also held (strongly) by the write registry,
    /// so it would otherwise live -- and keep its temp directory -- for the
    /// whole process.  When this is the last handle (its reference + the
    /// registry's), take it out of the registry; dropping this handle's
    /// reference then drops the backing and its [`ScratchDir`].
    fn drop(&mut self) {
        let Ok(inner) = self.inner.get_mut() else {
            return;
        };
        if let Inner::Write { shared } = inner {
            let scratch = shared.lock().map(|w| w.scratch.is_some()).unwrap_or(false);
            if scratch && std::sync::Arc::strong_count(shared) <= 2 {
                if let Ok(mut reg) = write_registry().lock() {
                    reg.remove(std::path::Path::new(&self.path));
                }
            }
        }
    }
}

/// Remove every TaQL result directory this process still holds (the `atexit`
/// hook, `casacure.tables._cleanup_scratch`).  Returns how many were removed.
#[pyfunction]
pub(crate) fn _cleanup_scratch() -> usize {
    let dirs: Vec<std::path::PathBuf> = match scratch_dirs().lock() {
        Ok(mut set) => set.drain().collect(),
        Err(_) => return 0,
    };
    dirs.iter()
        .filter(|d| std::fs::remove_dir_all(d).is_ok())
        .count()
}

/// Once per process, before the first TaQL result is written: remove
/// `casacure-taql-<pid>-<nanos>` directories left in the temp dir by processes
/// that no longer exist (a crash or a kill skips both the drop and the atexit
/// hook).  Conservative on purpose -- a directory goes only if its pid is not
/// alive on this host AND it is more than a day old, so a /tmp shared across
/// pid namespaces (docker) cannot lose a live table.  Linux only (/proc).
fn sweep_stale_scratch() {
    static DONE: std::sync::Once = std::sync::Once::new();
    DONE.call_once(|| {
        if !std::path::Path::new("/proc/self").exists() {
            return;
        }
        let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
            return;
        };
        let day = std::time::Duration::from_secs(24 * 3600);
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(rest) = name.to_str().and_then(|n| n.strip_prefix("casacure-taql-")) else {
                continue;
            };
            let Some(pid) = rest.split('-').next().and_then(|p| p.parse::<u32>().ok()) else {
                continue;
            };
            if pid == std::process::id() || std::path::Path::new(&format!("/proc/{pid}")).exists() {
                continue;
            }
            let old = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > day);
            if old {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    });
}

/// Identifies the on-disk table directory a [`WriteData`] was built for, so
/// a table deleted and re-created at the same path (a new directory) is not
/// served the old table's cells. The open directory fd keeps the old inode
/// allocated while the backing lives, so its number cannot be recycled for
/// the new directory and the (dev, inode) comparison is reliable.
struct DirPin(Option<std::fs::File>);

impl DirPin {
    fn new(dir: &std::path::Path) -> DirPin {
        #[cfg(unix)]
        {
            DirPin(std::fs::File::open(dir).ok())
        }
        #[cfg(not(unix))]
        {
            let _ = dir;
            DirPin(None)
        }
    }

    /// Whether `dir` is still the directory this pin was taken on. Without a
    /// pin (non-unix, or the open failed) the backing is trusted, as before.
    fn is_current(&self, dir: &std::path::Path) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let Some(pinned) = &self.0 else {
                return true;
            };
            match (pinned.metadata(), std::fs::metadata(dir)) {
                (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
                _ => false,
            }
        }
        #[cfg(not(unix))]
        {
            let _ = dir;
            true
        }
    }
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
/// A backing whose directory has since been deleted (and perhaps re-created)
/// is dropped from the registry instead: its cells belong to the old table.
fn find_write(dir: &std::path::Path) -> Option<std::sync::Arc<std::sync::Mutex<WriteData>>> {
    let mut reg = write_registry().lock().unwrap();
    let shared = reg.get(dir).cloned()?;
    let current = shared.lock().unwrap().pin.is_current(dir);
    if current {
        Some(shared)
    } else {
        reg.remove(dir);
        None
    }
}

/// Return free glibc arena memory to the OS.  `SELECT`/`sort` results are
/// materialised as heap `RecordValue` trees (GiBs on a real MS); those
/// allocations land in the *calling thread's* glibc arena, whose freed pages
/// glibc otherwise never gives back -- with DDFacet's pool threads each
/// sorting, the retained high-water marks summed to tens of GiB (2026-10-08:
/// 5 threads x 1 sort each retained 10.2 GiB where the same work on one
/// thread reused 2 GiB; `MALLOC_ARENA_MAX=2` bounded it to 4.1 GiB, which is
/// how the arenas were identified).  Trimming after each materialising
/// selection and on close hands the freed pages back; live allocations are
/// untouched.
#[cfg(target_os = "linux")]
fn trim_heap() {
    extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    unsafe { malloc_trim(0) };
}
#[cfg(not(target_os = "linux"))]
fn trim_heap() {}

/// python-casacore-compatible `table` object.
#[pyclass(name = "table")]
pub struct Table {
    path: String,
    writable: bool,
    inner: Mutex<Inner>,
    /// The `lockoptions` this handle was opened with, as given (for
    /// `table.lockoptions()`).
    lock_options: core::lockfile::LockOptions,
    /// `AutoLocking` yield bookkeeping for read handles (see
    /// [`Table::auto_tick`]).
    auto_state: Mutex<AutoLockState>,
    /// Whether this handle is a **reference table** (a `query()`/`sort()`
    /// result: a row order over the source table, no copied cells). Kept as
    /// a plain flag so the write paths can explain their refusal without
    /// re-locking `inner` (see [`Table::not_writable`]).
    view: bool,
}

#[derive(Default)]
struct AutoLockState {
    released: bool,
}

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Map a core dminfo error to the python-casacore exception classes:
/// structural mistakes are `TypeError`, a named column the description
/// lacks is `KeyError`, and unsupported managers / SPEC fields are
/// `ValueError` (the fail-loud contract of issue #20).
fn dm_err(e: core::tabledesc::DmInfoError) -> PyErr {
    match e {
        core::tabledesc::DmInfoError::BadRecord => PyTypeError::new_err(e.to_string()),
        core::tabledesc::DmInfoError::UnknownColumn { .. } => PyKeyError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// The `dminfo` argument of table creation / `addcols` / deep `tablecopy`
/// as the core record `apply_dminfo` consumes.
pub(crate) fn dminfo_record(py: Python<'_>, dminfo: &Bound<'_, PyAny>) -> PyResult<TableRecord> {
    match convert::pyobject_to_record(py, dminfo)? {
        RecordValue::Record(rec) => Ok(rec),
        _ => Err(PyTypeError::new_err("dminfo must be a dict")),
    }
}

/// Normalise the three documented `addcols` descriptor forms to the
/// `{colname: coldesc}` form. Mirrors python-casacore's `table.addcols`
/// wrapper (`casacore/tables/table.py`), which unwraps `{'name', 'desc'}`
/// and `getcoldesc`-with-`'name'` dicts via `maketabdesc`/`makecoldesc`.
fn normalize_coldesc<'py>(
    py: Python<'py>,
    coldesc: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    // Form 1: `maketabdesc` — `{colname: coldesc, ...}` (no 'name' key).
    if !coldesc.contains("name")? {
        return Ok(coldesc.clone());
    }
    let name_val = coldesc
        .get_item("name")?
        .ok_or_else(|| PyValueError::new_err("addcols: descriptor has no 'name'"))?;
    let name: String = name_val
        .extract()
        .map_err(|_| PyTypeError::new_err("addcols: 'name' must be a string"))?;
    // Form 2: `make*desc` output — exactly {'name', 'desc'}.
    if coldesc.len() == 2 && coldesc.contains("desc")? {
        let inner = coldesc
            .get_item("desc")?
            .ok_or_else(|| PyValueError::new_err("addcols: descriptor has no 'desc'"))?;
        let out = PyDict::new(py);
        out.set_item(name, inner)?;
        return Ok(out);
    }
    // Form 3: `getcoldesc` output with 'name' added — the dict itself is the
    // column descriptor (minus the 'name' key).
    if coldesc.contains("valueType")? {
        let out = PyDict::new(py);
        let rec = PyDict::new(py);
        for (k, v) in coldesc.iter() {
            let key: String = k.extract()?;
            if key == "name" {
                continue;
            }
            rec.set_item(key, v)?;
        }
        // A cloned tiled column needs its own data-manager group: two
        // array columns in one TiledColumnStMan/TiledShapeStMan group are
        // rejected ("one array column per group"). python-casacore's
        // `msutil.addImagingColumns` names them `modeldata`/`correcteddata`.
        // Only override when the source group is a tiled manager's group
        // (a StandardStMan group is shareable).
        let dm_type: String = coldesc
            .get_item("dataManagerType")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_default();
        let dm_group: String = coldesc
            .get_item("dataManagerGroup")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_default();
        if (dm_type.starts_with("Tiled") || dm_type == "TiledShapeStMan") && !dm_group.is_empty() {
            rec.set_item("dataManagerGroup", name.to_ascii_lowercase())?;
        }
        out.set_item(name, rec)?;
        return Ok(out);
    }
    Err(PyValueError::new_err(
        "addcols: unrecognized descriptor form (expected a maketabdesc, make*desc or getcoldesc dict)",
    ))
}

/// Apply the `dminfo` argument to the columns being added (`addcols`), so
/// the flush that regenerates the table puts each new column in the manager
/// the caller asked for (`create_table` groups columns by the descriptor's
/// data-manager type/group, and writes one block file per group).
///
/// Both shapes callers use are accepted, matching casacore's
/// `Table::addColumns(desc, dminfo)`:
/// * a flat record `{TYPE, NAME, SPEC}` (python-casacore's documented form)
///   applies to every column of this `addcols` call;
/// * a `getdminfo()`-shaped mapping `{field: {TYPE, NAME, COLUMNS, ...}}` —
///   what dask-ms's descriptor builders pass — applies each record to the
///   columns it names in `COLUMNS`.
///
/// `SPEC.DEFAULTTILESHAPE` is honoured (a tiled column is written with the
/// requested tiles), `SEQNR` and the cache-sizing SPEC fields are ignored,
/// and an unsupported manager or SPEC field raises `ValueError` — a silent
/// fallback to another layout is the bug of issue #20. Validation and
/// application live in the core (`tabledesc::apply_dminfo`), shared with
/// the `table()`/`default_ms()` creation paths.
fn apply_dminfo(
    py: Python<'_>,
    dminfo: &Bound<'_, PyAny>,
    cols: &mut [core::tabledesc::ColumnDesc],
) -> PyResult<()> {
    let rec = dminfo_record(py, dminfo)?;
    core::tabledesc::apply_dminfo(&rec, cols).map_err(dm_err)
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
/// (UPDATE/DELETE/INSERT/ALTER/...) rewrote them — or after another
/// *process* grew the table while this one's backing was idle. Live handles
/// that share `shared` then read — and, on `close()`, flush — the fresh
/// state instead of a stale pre-statement snapshot (which would otherwise
/// clobber the change back). Returns false when the directory no longer
/// holds a table (e.g. `DROPTABLE`), so the caller can drop the cached
/// entry.
fn refresh_write(
    dir: &std::path::Path,
    shared: &std::sync::Arc<std::sync::Mutex<WriteData>>,
    options: core::lockfile::LockOptions,
) -> bool {
    let read = match ::casacure::Table::open(dir, false) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let mut wt = match core::WritableTable::create_with_lock(
        dir.to_path_buf(),
        read.dat.desc.clone(),
        options,
    ) {
        Ok(wt) => wt,
        Err(_) => return false,
    };
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
    s.pin = DirPin::new(dir);
    s.disk_columns = s
        .read
        .dat
        .desc
        .columns
        .iter()
        .map(|c| c.name.clone())
        .collect();
    true
}

/// Physically write a shared cell store to disk when it has pending
/// (`dirty`) writes, and refresh its read snapshot from the freshly written
/// files. Write ops buffer into the store and only mark it dirty; the actual
/// table rewrite happens here (explicit `flush()`, on `close()`/context exit,
/// or before a mutating taql statement reads the snapshot). A no-op when the
/// store is clean, so repeated writes do not rewrite the whole table per call.
fn flush_if_dirty(
    dir: &std::path::Path,
    shared: &std::sync::Arc<std::sync::Mutex<WriteData>>,
) -> PyResult<()> {
    let mut s = shared.lock().unwrap();
    if s.dirty {
        // Another writer changed the columns on disk since the last sync:
        // flushing would rewrite `table.dat` from this backing's (now
        // stale) desc and silently drop the foreign column (issue #30's
        // write-over variant). Refuse instead — the caller surfaces it.
        if disk_columns_differ(dir, &s.disk_columns) {
            return Err(PyRuntimeError::new_err(format!(
                "{}: the table's columns changed on disk (another writer's \
                 addcols/removecols) while this handle holds unflushed \
                 writes; the write would drop the foreign column",
                dir.display()
            )));
        }
        let flushed = s.wt.flush().map_err(err)?;
        s.read = ::casacure::Table::open(&flushed, false).map_err(err)?;
        s.disk_columns = s.wt.desc().columns.iter().map(|c| c.name.clone()).collect();
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
    if s.wt.added_since_flush(&cd.name) {
        return None; // no on-disk twin, whatever a removed column of that name held
    }
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

/// The error a write op raises on a handle with no writable backing.
///
/// A **reference table** (a `query()`/`sort()` result) gets a specific
/// message: casacore routes its writes through the row order (probed against
/// python-casacore 3.8.1: `putcol` on a sorted result lands in the source
/// rows it maps to), casacure does not yet, so it is refused loudly rather
/// than sent nowhere — which is what the pre-reference-table implementation
/// did: such writes went into a discarded temp copy of the selection.
fn not_writable(view: bool) -> PyErr {
    if view {
        return PyValueError::new_err(
            "this is a reference table (a query()/sort() result): casacure does not route \
             writes through its row order yet (casacore does), so it is read-only — write \
             through the source table instead",
        );
    }
    PyValueError::new_err("table is not writable")
}

/// Apply a cell sub-array slice (0-based inclusive corners; scalar cells are
/// returned unchanged).
fn slice_cell(cell: RecordValue, blc: &[i64], trc: &[i64], inc: &[i64]) -> PyResult<RecordValue> {
    if blc.is_empty() && trc.is_empty() {
        return Ok(cell);
    }
    Ok(match cell {
        RecordValue::Array(a) => {
            RecordValue::Array(::casacure::slice_array_value_inc(&a, blc, trc, inc).map_err(err)?)
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

/// The shape conflict that stops `cells` being stacked into one array, if any:
/// the first defined shape seen, and the first cell whose shape differs from
/// it. Empty (undefined) cells and non-array cells never take part (they read
/// as defaults, issue #8). Kept pure — free of `PyErr`, whose construction
/// needs a linked Python — so the decision table is unit-tested directly;
/// `check_cells_conform` below only renders the verdict.
fn first_shape_conflict(cells: &[RecordValue]) -> Option<(&[u32], &[u32])> {
    let mut first: Option<&[u32]> = None;
    for cell in cells {
        let RecordValue::Array(a) = cell else {
            continue;
        };
        if a.shape.is_empty() || a.shape.contains(&0) {
            continue;
        }
        match first {
            None => first = Some(&a.shape),
            Some(shape) if shape != a.shape.as_slice() => return Some((shape, &a.shape)),
            Some(_) => {}
        }
    }
    None
}

/// One stacked array can only hold cells of one shape: refuse a variable-shape
/// column whose rows differ, as casacore does, rather than cut every row to the
/// first row's shape. Empty (undefined) cells stay exempt: they read as defaults.
fn check_cells_conform(cells: &[RecordValue]) -> PyResult<()> {
    match first_shape_conflict(cells) {
        None => Ok(()),
        Some((shape, other)) => Err(PyRuntimeError::new_err(format!(
            "getcol: cells of shape {shape:?} and {other:?} cannot be stacked; \
             use getvarcol for a column whose rows differ in shape"
        ))),
    }
}

/// Whether `cells` holds string arrays. python-casacore's `getcol` never
/// refuses a ragged string column: it stacks it, cutting (or padding) every
/// row to the first row's shape and returning the `{"array", "shape"}`
/// dict — an MS `HISTORY.CLI_COMMAND` column comes back as
/// `[nrow, 1]` with each row's first element. The conform check must
/// therefore not apply to string arrays; the numeric variable-shape case
/// still raises, as casacore does (`StManIndArray::get shapes not
/// conforming`).
fn is_string_array_column(cells: &[RecordValue]) -> bool {
    cells.iter().any(|c| {
        matches!(
            c,
            RecordValue::Array(a) if matches!(a.data, ArrayData::String(_))
        )
    })
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
                // Decode straight from the mapped file into the caller's
                // buffer with the GIL released (see fill_buffer_by_dtype), so
                // the dask scheduler can overlap independent column reads.
                // `off` is the visit's row offset in this call: a reference
                // table (query/sort result) gathers its rows out of source
                // order, so the buffer slot cannot be a visit counter.
                py.detach(|| {
                    t.getcol_raw(col_idx, startrow, nrow, |off, _, bytes| {
                        let dst = &mut slice[off * count..(off + 1) * count];
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
            t.getcol_raw(col_idx, startrow, nrow, |off, _, bytes| {
                let Some(&b) = bytes.first() else {
                    return Err(short(0, 1));
                };
                // `off` covers `0..nrow` exactly once (the row offset within
                // this call), so every slot of the buffer is written.
                slice[off] = b != 0;
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
            py.detach(|| {
                t.getcol_raw_bits(col_idx, startrow, nrow, |off, bytes, skip, nelem| {
                    if nelem != count || (skip + nelem).div_ceil(8) > bytes.len() {
                        return Err(short(bytes.len(), (skip + count).div_ceil(8)));
                    }
                    let dst = &mut slice[off * count..(off + 1) * count];
                    ::casacure::tsm::decode_bits_into(bytes, skip, dst);
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
    /// Parse python-casacore's `lockoptions` argument: one of the option
    /// words (`default, auto, autonoread, user, usernoread, permanent,
    /// permanentwait`) or a dict `{'option': <word>, 'interval': <seconds>,
    /// 'maxwait': <attempts>}` (`TableProxy::makeLockOptions`).
    pub(crate) fn parse_lockoptions(
        lo: &Bound<'_, PyAny>,
    ) -> PyResult<core::lockfile::LockOptions> {
        let unknown = |w: &str| {
            PyRuntimeError::new_err(format!(
                "'{w}' is an unknown lock option; valid are default,auto,autonoread,\
user,usernoread,permanent,permanentwait"
            ))
        };
        let (mode, interval, maxwait) = if let Ok(word) = lo.extract::<String>() {
            let mode = core::lockfile::LockMode::parse(&word).ok_or_else(|| unknown(&word))?;
            (mode, None, None)
        } else {
            let dict = lo
                .cast::<PyDict>()
                .map_err(|_| PyRuntimeError::new_err("lockoptions must be a string or a dict"))?;
            let word: String = match dict.get_item("option")? {
                Some(v) => v.extract().map_err(err)?,
                None => "default".to_string(),
            };
            let mode = core::lockfile::LockMode::parse(&word).ok_or_else(|| unknown(&word))?;
            let interval = match dict.get_item("interval")? {
                Some(v) => Some(v.extract::<f64>().map_err(err)?),
                None => None,
            };
            let maxwait = match dict.get_item("maxwait")? {
                Some(v) => Some(v.extract::<u32>().map_err(err)?),
                None => None,
            };
            (mode, interval, maxwait)
        };
        Ok(core::lockfile::LockOptions {
            mode,
            // casacore keeps the interval as a double in seconds; sub-second
            // intervals round up to a whole second here.
            interval: interval.map(|i| i.ceil() as u32).unwrap_or(5),
            maxwait: maxwait.unwrap_or(0),
        })
    }

    /// Open or create a table; `desc_json` is the python-casacore table-desc
    /// dict (creates when given) and `nrow` its initial row count.
    /// `dminfo` (the parsed `dminfo` argument, creation only) picks the new
    /// columns' storage managers before the table is written
    /// (`tabledesc::apply_dminfo`); it is ignored when opening.
    ///
    /// Runs without the GIL where the caller detaches: opening takes the
    /// mode's open lock, which may block on another process.
    fn open_or_create(
        path: &str,
        desc_json: Option<&str>,
        dminfo: Option<&core::record::TableRecord>,
        nrow: u64,
        writable: bool,
        options: core::lockfile::LockOptions,
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
            let mut desc = core::tabledesc::TableDesc::from_desc_json(desc_string).map_err(err)?;
            if let Some(dm) = dminfo {
                core::tabledesc::apply_dminfo(dm, &mut desc.columns).map_err(dm_err)?;
            }
            let mut wt = core::WritableTable::create_with_lock(&dir, desc, options).map_err(err)?;
            // Write the empty table, then grow it: the growth appends
            // default rows in place (zeroed tiles, default buckets, empty or
            // zeroed array cells -- casacore's defaults), where buffering
            // one default cell per row and array column before the first
            // flush cost memory and time in the table's size.
            if nrow > 0 {
                let _ = wt.flush().map_err(err)?;
                wt.addrows(nrow);
            }
            let _ = wt.flush().map_err(err)?;
            let read = ::casacure::Table::open_with_lock(&dir, false, options).map_err(err)?;
            let disk_columns = read
                .dat
                .desc
                .columns
                .iter()
                .map(|c| c.name.clone())
                .collect();
            let shared = std::sync::Arc::new(std::sync::Mutex::new(WriteData {
                read,
                wt,
                dirty: false,
                pin: DirPin::new(&dir),
                disk_columns,
                scratch: None,
            }));
            register_write(&dir, &shared);
            return Ok(Table {
                path: abs.clone(),
                writable,
                inner: Mutex::new(Inner::Write { shared }),
                lock_options: options,
                auto_state: Mutex::new(AutoLockState::default()),
                view: false,
            });
        }
        if !writable {
            // A read-only open must see pending (unflushed) writes of any
            // live writable backing for this directory: write ops are
            // buffered, but a fresh handle (and python-casacore, whose SM
            // writes are file-visible) reads the files. Flush a dirty
            // backing before opening the files so this is not a stale
            // snapshot.
            //
            // If the flush fails — a read-only block, e.g. the hard-linked
            // blocks of a `--write-changed-only` output — the writer's own
            // flush()/close() has already reported that error and the
            // writes stay pending on the backing; a reader must not be
            // refused a read it had no part in failing, so it falls back to
            // the files: the last state the writer actually persisted.
            if let Some(shared) = find_write(&dir) {
                if shared.lock().unwrap().dirty {
                    let _ = flush_if_dirty(&dir, &shared);
                }
            }
            let read = ::casacure::Table::open_with_lock(&dir, true, options).map_err(err)?;
            return Ok(Table {
                path: abs.clone(),
                writable: false,
                inner: Mutex::new(Inner::Read(std::sync::Arc::new(read))),
                lock_options: options,
                auto_state: Mutex::new(AutoLockState::default()),
                view: false,
            });
        }
        // Reuse a live shared backing for this directory so concurrent
        // writable handles accumulate into one cell store (a second flush of
        // a stale snapshot must not clobber the first handle's writes).
        // casacore `DefaultLocking` follows the backing's options; an
        // explicit permanent request still has to acquire (it then conflicts
        // with another *process* holding the lock — fcntl locks do not
        // conflict within one process).
        if let Some(shared) = find_write(&dir) {
            // The backing is stale if another process grew the table or
            // changed its columns (addcols/removecols/rename; issue #30 —
            // python-casacore and dask-ms 0.2.x create columns out of band)
            // since this one's last sync (nothing pending here): adopt the
            // on-disk state, keeping the caller's locking options.
            let (stale, schema_conflict) = {
                let s = shared.lock().unwrap();
                let schema_changed = disk_columns_differ(&dir, &s.disk_columns);
                let rows_grew = ::casacure::lock_sync_nrrow(&dir)
                    .filter(|&n| n != 0)
                    .is_some_and(|n| n > s.wt.nrows());
                (
                    !s.dirty && (rows_grew || schema_changed),
                    s.dirty && schema_changed,
                )
            };
            // A schema change with pending writes must NOT be flushed over:
            // the flush rewrites `table.dat` from this backing's stale desc
            // and silently drops the foreign column. Refuse instead (the
            // same message `flush()` itself raises via `flush_if_dirty`).
            if schema_conflict {
                return Err(PyRuntimeError::new_err(format!(
                    "{}: the table's columns changed on disk (another \
                     writer's addcols/removecols) while this handle holds \
                     unflushed writes; the write would drop the foreign \
                     column",
                    dir.display()
                )));
            }
            if stale && !refresh_write(&dir, &shared, options) {
                return Err(err(format!(
                    "{}: table changed on disk and could not be reloaded",
                    dir.display()
                )));
            }
            let eff = options.effective();
            if eff.is_permanent() {
                let nattempts = if eff.mode == core::lockfile::LockMode::PermanentLocking {
                    1
                } else {
                    0
                };
                if shared.lock().unwrap().wt.lock(true, nattempts).is_err() {
                    return Err(PyRuntimeError::new_err(format!(
                        "Permanent lock on table {} could not be acquired",
                        dir.display()
                    )));
                }
            }
            return Ok(Table {
                path: abs.clone(),
                writable: true,
                inner: Mutex::new(Inner::Write { shared }),
                lock_options: options,
                auto_state: Mutex::new(AutoLockState::default()),
                view: false,
            });
        }
        // A writable open of an existing table is LAZY: no column is
        // materialised, so a changed-columns write holds only the rows it
        // actually writes instead of per-cell RecordValues for the whole
        // table (~3 GiB on a 1.6 GB MS).
        let (read, wt) =
            core::WritableTable::open_for_update_with_lock(&dir, options).map_err(err)?;
        let shared = std::sync::Arc::new(std::sync::Mutex::new(WriteData {
            disk_columns: read
                .dat
                .desc
                .columns
                .iter()
                .map(|c| c.name.clone())
                .collect(),
            read,
            wt,
            dirty: false,
            pin: DirPin::new(&dir),
            scratch: None,
        }));
        register_write(&dir, &shared);
        Ok(Table {
            path: abs.clone(),
            writable: true,
            inner: Mutex::new(Inner::Write { shared }),
            lock_options: options,
            auto_state: Mutex::new(AutoLockState::default()),
            view: false,
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

    /// casacore's `AutoLocking` yield for read handles: the read lock taken
    /// at open is released once another process is waiting (checked with
    /// `LockFile::inspect`'s 25-call/interval throttle), and the next
    /// operation re-acquires it — blocking until the writer is done, then
    /// resyncing from the freshly written state. A released handle reads
    /// its (stale) snapshot until then, exactly as casacore continues on
    /// its in-memory caches after an auto release.
    fn auto_tick(&self) -> PyResult<()> {
        let mut st = self.auto_state.lock().unwrap();
        if st.released {
            st.released = false;
            let t = ::casacure::Table::open_with_lock(self.dir_of(), true, self.lock_options)
                .map_err(err)?;
            *self.inner.lock().unwrap() = Inner::Read(std::sync::Arc::new(t));
            return Ok(());
        }
        // `LockFile::inspect_has_waiter` owns casacore's throttle (every
        // 25th call, at most once per interval; interval 0 inspects every
        // call).
        let waiter = {
            let inner = self.inner.lock().unwrap();
            let Inner::Read(arc) = &*inner else {
                return Ok(());
            };
            let eff = arc.lock_options();
            if eff.mode != core::lockfile::LockMode::AutoLocking || !eff.read_locking {
                return Ok(());
            }
            match arc.lock_file() {
                Some(lf) => lf.lock().unwrap().inspect_has_waiter(false).map_err(err)?,
                None => false,
            }
        };
        if waiter {
            st.released = true;
            // Drop the locked snapshot for an unlocked one: the registry
            // keeps the fd (and therefore the read lock) only while some
            // handle still references it.
            let unlocked = ::casacure::Table::open(self.dir_of(), true).map_err(err)?;
            *self.inner.lock().unwrap() = Inner::Read(std::sync::Arc::new(unlocked));
        }
        Ok(())
    }

    /// The core table TaQL runs against this handle's current state, as an
    /// `Arc`: the call holds it without this handle's mutex (a long
    /// `WHERE`/`ORDERBY` scan must not serialise other threads using the
    /// same handle) and a reference table can keep its source open
    /// afterwards.
    ///
    /// A read handle contributes its own snapshot — which for a
    /// `query()`/`sort()` result *is* the reference table, so a chained
    /// selection composes its row order instead of re-opening the source
    /// directory and losing it. A write handle is flushed and re-opened
    /// from disk, since its cell store is not a `Table`.
    fn core_source(&self) -> PyResult<std::sync::Arc<::casacure::Table>> {
        // Refresh a released auto-lock snapshot first (no-op for a
        // reference table: it has no lock file of its own).
        self.auto_tick()?;
        let inner = self.inner.lock().unwrap();
        Ok(match &*inner {
            Inner::Read(t) => std::sync::Arc::clone(t),
            Inner::Write { shared, .. } => {
                // The on-disk state must include this handle's pending
                // (unflushed) writes; `query()`/`sort()`/`select_run()`
                // run against the files.
                flush_if_dirty(&self.dir_of(), shared)?;
                std::sync::Arc::new(::casacure::Table::open(self.dir_of(), false).map_err(err)?)
            }
        })
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

    /// Validate a normalized slice request against casacore's rules
    /// (issue #38): the column must be an array column — every casacore
    /// slice entry point refuses a scalar one, with any corners — and the
    /// corners must not be empty: an `IPosition` of length 0 never matches
    /// a cell's `ndim()`, so casacore raises its `Slicer` rank error where
    /// casacure's core would read the empty slice as "the whole cell".
    /// The whole-cell reading stays an *internal* idiom — the `-1` corner
    /// conversion in `read_cellslice` feeds it and dask-ms's `(-1, ..)`
    /// whole-cell idiom relies on it — only the Python-visible empty
    /// corners are refused here.
    fn check_slice_request(
        &self,
        name: &str,
        column: &str,
        col_idx: usize,
        blc: &[i64],
        trc: &[i64],
    ) -> PyResult<()> {
        let desc = self.desc();
        if !matches!(
            desc.columns[col_idx].kind,
            core::tabledesc::ColumnKind::Array
        ) {
            return Err(PyRuntimeError::new_err(format!(
                "{name}: column {column} is not an array column \
                 (casacore: TableProxy::getColumnSlice: column {column} is not an array column)"
            )));
        }
        if blc.is_empty() || trc.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "{name}: empty blc/trc do not match the cell rank \
                 (casacore: Slicer error: Shape IPosition-lengths differ from ndim())"
            )));
        }
        Ok(())
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
                _ => return Err(not_writable(self.view)),
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
                _ => return Err(not_writable(self.view)),
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
                _ => return Err(not_writable(self.view)),
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
                _ => return Err(not_writable(self.view)),
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

    /// `t[key]` — python-casacore's `table.__getitem__`: an integer key is a
    /// row number (returns the row's values as a dict `{colname: value}`);
    /// a string key is a column name (returns the column as an array).
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Column-name form: t["DATA"].
        if let Ok(name) = key.extract::<String>() {
            return self.getcol(py, &name, 0, -1, 1);
        }
        // Row form: t[0] -> {colname: value} for that row.
        let row: i64 = key.extract().map_err(|_| {
            PyTypeError::new_err("table key must be a column name (str) or a row number (int)")
        })?;
        let desc = self.desc();
        let d = PyDict::new(py);
        for (i, cd) in desc.columns.iter().enumerate() {
            let v = self.read_cell(i, row.max(0) as u64)?;
            let base = self.dir_of();
            let v = convert::cell_to_py(py, &v)?;
            let _ = &base;
            d.set_item(&cd.name, v)?;
        }
        Ok(d.into_any().unbind())
    }

    /// `len(t)` — the number of rows (python-casacore's `table.__len__`).
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.nrows()? as usize)
    }

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
        self.auto_tick()?;
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

    /// Run `sql` against this table and return the result as a new `table`.
    ///
    /// A statement that is a pure **row selection** of this table
    /// (`SELECT * FROM $1 [WHERE ...] [ORDERBY ...] [LIMIT/OFFSET]`) comes
    /// back as a casacore-style *reference table*: the selected row order is
    /// kept and no cell data is copied (issue #16 — the materialised form
    /// held ~2 GiB of heap per distinct selection on a real MS, for the
    /// lifetime of the handle). Anything else — computed columns, `UNIQUE`,
    /// `GROUPBY`, `SELECT ... INTO`, mutating statements — materialises
    /// through TaQL exactly as before.
    fn select_run(&self, py: Python<'_>, sql: &str) -> PyResult<Py<PyAny>> {
        let source = self.core_source()?;
        // Row order first: parse and evaluate only the row-mode half of the
        // statement (filter/sort/paginate), no projection.
        if let Some(rows) = core::taql::execute_row_order(sql, &[&*source]).map_err(err)? {
            let view = core::Table::row_order(std::sync::Arc::clone(&source), rows).map_err(err)?;
            let path = view.name().to_string();
            let out = Table {
                path,
                // casacore's query/sort results are read-only reference
                // tables (`table.iswritable()` is False there too).
                writable: false,
                inner: Mutex::new(Inner::Read(std::sync::Arc::new(view))),
                lock_options: core::lockfile::LockOptions::no_locking(),
                auto_state: Mutex::new(AutoLockState::default()),
                view: true,
            };
            // The row-mode evaluation still caches its key/WHERE columns as
            // `TqValue`s; hand those arena pages back like the materialising
            // path does.
            trim_heap();
            return Ok(out.into_pyobject(py)?.into_any().unbind());
        }
        match core::taql::execute(sql, &[&*source]).map_err(err)? {
            core::taql::TaqlResult::Query(out) => {
                let out = taql_result_to_table(py, out)?
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                trim_heap();
                Ok(out)
            }
            other => Err(PyRuntimeError::new_err(format!(
                "taql: expected a SELECT result, got {other:?}"
            ))),
        }
    }

    /// A materialised copy of this handle's rows as a temp-dir table —
    /// `copy()` on a reference table has to copy the *selected rows in their
    /// order*, where `tablecopy` over `name()` would copy the whole source
    /// table. Only reached for `copy`, so the materialisation cost is paid
    /// on demand.
    fn materialise_result(&self, py: Python<'_>) -> PyResult<Table> {
        let source = self.core_source()?;
        match core::taql::execute("SELECT * FROM $1", &[&*source]).map_err(err)? {
            core::taql::TaqlResult::Query(out) => {
                let t = taql_result_to_table(py, out)?;
                trim_heap();
                Ok(t)
            }
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

    /// `info()` -> dict `{'type', 'subType', 'readme'}` — the table's
    /// `table.info` file (casacore `Table::info()`). A table without the
    /// file (written before casacure created it, or deleted afterwards)
    /// reads as empty strings, like casacore.
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dir = self.dir_of();
        let info = core::tableinfo::TableInfo::read(&dir).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("type", info.r#type)?;
        d.set_item("subType", info.sub_type)?;
        d.set_item("readme", info.readme)?;
        Ok(d.into_any().unbind())
    }

    /// `putinfo(dict)` — set any of `type`, `subType`, `readme` on the
    /// table's `table.info` (casacore `TableProxy::putinfo`, probed against
    /// python-casacore 3.8.1): fields absent from the dict keep their
    /// value, the readme is stored as whole lines (a missing trailing
    /// newline is appended), and an unknown field is the casacore
    /// `RuntimeError` naming the accepted keys.
    fn putinfo(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let dict = value.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err("putinfo takes a dict with the fields type, subType, readme")
        })?;
        // Validate the whole dict before touching anything (casacore
        // converts the record first, so a bad shape fails even on a
        // read-only table).
        let mut new_type: Option<String> = None;
        let mut new_sub_type: Option<String> = None;
        let mut new_readme: Option<String> = None;
        for (k, v) in dict.iter() {
            let key = k.extract::<String>()?;
            let target = match key.as_str() {
                "type" => &mut new_type,
                "subType" => &mut new_sub_type,
                "readme" => &mut new_readme,
                other => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Name of field {other} must be type, subType or readme"
                    )))
                }
            };
            *target = Some(v.extract().map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "RecordRep::get_pointer - field {key} must be a string"
                ))
            })?);
        }
        let dir = self.dir_of();
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Write { .. } => {
                // table.info is metadata beside the data files (like the
                // keywords' table.dat home, but a plain text file): write
                // it through, no cell flush involved.
                let mut info = core::tableinfo::TableInfo::read(&dir).map_err(err)?;
                if let Some(t) = new_type {
                    info.r#type = t;
                }
                if let Some(s) = new_sub_type {
                    info.sub_type = s;
                }
                if let Some(r) = new_readme {
                    info.readme = r;
                }
                info.write(&dir).map_err(err)?;
                Ok(())
            }
            _ => Err(not_writable(self.view)),
        }
    }

    /// `addreadmeline(line)` — append a line to the table info's readme.
    fn addreadmeline(&self, line: &str) -> PyResult<()> {
        let dir = self.dir_of();
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Write { .. } => {
                let mut info = core::tableinfo::TableInfo::read(&dir).map_err(err)?;
                info.readme.push_str(line);
                info.readme.push('\n');
                info.write(&dir).map_err(err)?;
                Ok(())
            }
            _ => Err(not_writable(self.view)),
        }
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
                flush_if_dirty(&self.dir_of(), shared)?;
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

    /// `addcols(coldesc_dict, dminfo=None)` — append columns to the writable
    /// table. `dminfo` selects the storage manager of the new columns
    /// (`apply_dminfo`).
    ///
    /// Accepts the three forms python-casacore's `table.addcols` documents
    /// (see `casacore/tables/table.py`):
    /// * a `maketabdesc` dict `{colname: coldesc, ...}` (multiple columns);
    /// * a `make*desc` dict `{'name': colname, 'desc': {...}}` (one column);
    /// * a `getcoldesc` dict with a `'name'` key added (one column).
    #[pyo3(signature = (coldesc, dminfo = None))]
    fn addcols(
        &self,
        py: Python<'_>,
        coldesc: &Bound<'_, PyDict>,
        dminfo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        {
            let mut inner = self.inner.lock().unwrap();
            match &mut *inner {
                Inner::Write { shared, .. } => {
                    // Normalise the three documented forms to the
                    // `{colname: coldesc}` form before parsing.
                    let normalized = normalize_coldesc(py, coldesc)?;
                    // Pull the desc's columns out (they live in a dict of
                    // {colname: coldesc}).
                    let json = {
                        let rec = convert::dict_to_table_record(py, &normalized)?;
                        rec.to_json_string()
                    };

                    let parsed = core::tabledesc::TableDesc::from_desc_json(&json).map_err(err)?;
                    let mut s = shared.lock().unwrap();
                    let mut columns = parsed.columns;
                    if let Some(dminfo) = dminfo {
                        apply_dminfo(py, dminfo, &mut columns)?;
                    }
                    for cd in columns {
                        s.wt.addcol(cd);
                    }
                    s.dirty = true;
                }
                _ => return Err(not_writable(self.view)),
            }
        }
        Ok(())
    }

    /// Add MODEL_DATA, CORRECTED_DATA and IMAGING_WEIGHT to this MS through
    /// the shared writable backing (python-casacore's
    /// `casacore.tables.addImagingColumns`, reimplemented on the binding so
    /// live `table` handles see the change). Returns the names added.
    pub(crate) fn add_imaging_columns_impl(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        // Read the current column names and DATA's descriptor through this
        // handle (so the shared WriteData is used).
        let names = self.colnames()?;
        if !names.iter().any(|n| n == "DATA") {
            return Err(PyValueError::new_err("Column DATA does not exist"));
        }
        let data_desc_obj = self.getcoldesc(py, "DATA")?;
        let data_desc = data_desc_obj.bind(py).cast::<PyDict>()?;
        // Whether DATA is tiled (python-casacore checks `dminfo['TYPE'][:5]
        // == 'Tiled'`).
        let data_dm = data_desc
            .get_item("dataManagerType")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let data_tiled = data_dm.starts_with("Tiled");

        let mut added: Vec<String> = Vec::new();
        for (name, comment) in [
            ("MODEL_DATA", "The model data column"),
            ("CORRECTED_DATA", "The corrected data column"),
        ] {
            if names.iter().any(|n| n == name) {
                continue;
            }
            // Clone DATA's descriptor under the new name/comment/group.
            let desc = data_desc
                .clone()
                .unbind()
                .bind(py)
                .cast::<PyDict>()?
                .clone();
            let _ = desc.del_item("keywords");
            desc.set_item("name", name)?;
            desc.set_item("comment", comment)?;
            // Each cloned column needs its own data-manager group (casacore's
            // `msutil.py` names them `modeldata`/`correcteddata`/`imagingweight`;
            // a shared TiledColumnStMan group is rejected by the storage
            // manager: "one array column per group").
            desc.set_item("dataManagerGroup", name.to_ascii_lowercase())?;
            if !data_tiled {
                desc.set_item("dataManagerType", "TiledShapeStMan")?;
            }
            let cols = PyDict::new(py);
            cols.set_item(name, desc)?;
            self.addcols(py, &cols, None)?;
            added.push(name.to_string());
        }
        if !names.iter().any(|n| n == "IMAGING_WEIGHT") {
            // 1-dim float, shape [nchan] (the first element of DATA's shape).
            let shape = data_desc
                .get_item("shape")?
                .map(|s| s.extract::<Vec<i64>>().unwrap_or_default())
                .unwrap_or_default();
            let nchan = shape.first().copied().unwrap_or(0).max(0);
            let desc = PyDict::new(py);
            desc.set_item("valueType", "float")?;
            desc.set_item("dataManagerType", "TiledShapeStMan")?;
            desc.set_item("dataManagerGroup", "imagingweight")?;
            desc.set_item("option", 4)?;
            desc.set_item("maxlen", 0)?;
            desc.set_item("comment", "")?;
            desc.set_item("ndim", 1)?;
            desc.set_item("shape", vec![nchan])?;
            desc.set_item("_c_order", true)?;
            desc.set_item("keywords", PyDict::new(py))?;
            let cols = PyDict::new(py);
            cols.set_item("IMAGING_WEIGHT", desc)?;
            self.addcols(py, &cols, None)?;
            added.push("IMAGING_WEIGHT".to_string());
        }
        // MODEL_DATA's CHANNEL_SELECTION: int32 [[0, nch], ...] per SPW.
        if added.iter().any(|n| n == "MODEL_DATA") {
            let chans = self.channel_selection_rows(py)?;
            self.putcolkeyword(py, "MODEL_DATA", "CHANNEL_SELECTION", chans.bind(py))?;
        }
        self.flush()?;
        Ok(added)
    }

    /// Remove MODEL_DATA, CORRECTED_DATA and IMAGING_WEIGHT (whichever
    /// exist) and flush. Returns the names removed.
    pub(crate) fn remove_imaging_columns_impl(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let names = self.colnames()?;
        let mut removed: Vec<String> = Vec::new();
        for name in ["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"] {
            if names.iter().any(|n| n == name) {
                removed.push(name.to_string());
            }
        }
        if !removed.is_empty() {
            self.removecols(py, &PyList::new(py, removed.clone())?.into_any())?;
            self.flush()?;
        }
        Ok(removed)
    }

    /// `MODEL_DATA`'s `CHANNEL_SELECTION` value: an int32 `[[0, nch], ...]`
    /// array, one pair per spectral window (from `SPECTRAL_WINDOW.NUM_CHAN`).
    fn channel_selection_rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let base = self.dir_of();
        let spw_dir = base.join("SPECTRAL_WINDOW");
        if !spw_dir.join("table.dat").exists() {
            let np = py.import("numpy")?;
            let zeros = np.getattr("zeros")?.call1(((1, 2), "int32"))?;
            return Ok(zeros.unbind());
        }
        let spw_path = spw_dir.display().to_string();
        let spw_name = pyo3::types::PyString::new(py, &spw_path);
        let spw = table(
            py,
            &spw_name.into_any(),
            None,
            0,
            None,
            true,
            true,
            None,
            &PyTuple::empty(py),
            None,
        )?;
        let names = spw.colnames()?;
        let Some(idx) = names.iter().position(|n| n == "NUM_CHAN") else {
            let np = py.import("numpy")?;
            let zeros = np.getattr("zeros")?.call1(((1, 2), "int32"))?;
            return Ok(zeros.unbind());
        };
        let n = spw.row_count();
        let cells = spw.read_col(idx, 0, n)?;
        let mut flat: Vec<i32> = Vec::with_capacity(cells.len() * 2);
        for c in &cells {
            let nch = match c {
                RecordValue::Int(i) => *i,
                RecordValue::Int64(i) => *i as i32,
                _ => 0,
            };
            flat.push(0);
            flat.push(nch);
        }
        if flat.is_empty() {
            flat.extend_from_slice(&[0, 0]);
        }
        // A plain 2-D int32 numpy array (python-casacore's
        // `CHANNEL_SELECTION` value is `np.int32([[0, nch], ...])`).
        let np = py.import("numpy")?;
        let nrows = (flat.len() / 2) as i64;
        let arr = np.getattr("asarray")?.call1((flat,))?;
        let arr = arr.call_method1("reshape", ((nrows, 2),))?;
        let arr = arr.call_method1("astype", ("int32",))?;
        Ok(arr.unbind())
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
                _ => return Err(not_writable(self.view)),
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
                _ => return Err(not_writable(self.view)),
            }
        }
        Ok(())
    }

    /// `removerows(rownrs)` — remove rows (python-casacore parity). The row
    /// numbers address the table as it stands and may come in any order;
    /// survivors renumber in order, persisted at the next flush (like every
    /// buffered write here: explicit `flush()` or `close()`).
    ///
    /// A lazy writable open buffers only written rows, so every row is first
    /// materialised into the cell store as a session write — the
    /// [`casacure::WritableTable::from_table`] pattern the taql `DELETE`
    /// path uses. The store has to be the whole table before rows go: it
    /// must out-rank the pre-removal on-disk rows for merged reads and for
    /// `materialize_all` at flush, both of which would otherwise answer
    /// with rows at their old positions. This costs one table's worth of
    /// memory for the duration — row removal rewrites the table anyway.
    fn removerows(&self, rownrs: &Bound<'_, PyAny>) -> PyResult<()> {
        let rows: Vec<u64> = match rownrs.extract::<Vec<u64>>() {
            Ok(rows) => rows,
            Err(_) => {
                let not_ints = || PyTypeError::new_err("rownrs must be a sequence of integers");
                let iter = rownrs.try_iter().map_err(|_| not_ints())?;
                let mut rows = Vec::new();
                for item in iter {
                    let item = item.map_err(|_| not_ints())?;
                    rows.push(item.extract::<u64>().map_err(|_| not_ints())?);
                }
                rows
            }
        };
        let mut inner = self.inner.lock().unwrap();
        let Inner::Write { shared, .. } = &mut *inner else {
            return Err(not_writable(self.view));
        };
        let mut s = shared.lock().unwrap();
        let nrow = s.wt.nrows();
        if let Some(&row) = rows.iter().find(|&&row| row >= nrow) {
            return Err(row_out_of_range(row, nrow));
        }
        if rows.is_empty() {
            return Ok(());
        }
        // Materialise every row as a session write (see the doc comment).
        let ncols = s.wt.desc().columns.len();
        for col in 0..ncols {
            let cd = s.wt.desc().columns[col].clone();
            let missing = (0..nrow).any(|row| s.wt.cell(col, row).is_none());
            let disk_vals = if missing {
                disk_col_index(&s, col).and_then(|di| s.read.getcol(di, 0, s.read.nrows()).ok())
            } else {
                None
            };
            for row in 0..nrow {
                let value =
                    s.wt.cell(col, row)
                        .cloned()
                        .or_else(|| {
                            disk_vals
                                .as_ref()
                                .and_then(|v| v.get(row as usize).cloned())
                        })
                        .or_else(|| ::casacure::default_cell_value(&cd));
                let Some(value) = value else {
                    continue;
                };
                s.wt.putcell(col, row, value).map_err(err)?;
            }
        }
        s.wt.drop_rows(&rows);
        s.dirty = true;
        Ok(())
    }

    /// `lock(write=True, nattempts=0)`: acquire the table's lock (a real
    /// fcntl lock on `table.lock`; `nattempts == 0` waits indefinitely,
    /// otherwise that many one-second-apart attempts are made). Locks do
    /// not nest: an already-held write lock satisfies a read request.
    #[pyo3(signature = (write = true, nattempts = 0))]
    fn lock(&self, py: Python<'_>, write: bool, nattempts: u32) -> PyResult<()> {
        self.auto_state.lock().unwrap().released = false;
        py.detach(|| {
            let mut inner = self.inner.lock().unwrap();
            match &*inner {
                // A read snapshot is immutable, so the lock is driven on the
                // directory's shared lock file; the resync is the re-open
                // below (its attach re-enters the same LockFile, where the
                // fresh acquire is a no-op while held).
                Inner::Read(arc) => {
                    let Some(lf) = arc.lock_file() else {
                        return Ok(());
                    };
                    let typ = if write {
                        core::lockfile::LockType::Write
                    } else {
                        core::lockfile::LockType::Read
                    };
                    let ok = lf.lock().unwrap().acquire(typ, nattempts).map_err(err)?;
                    if !ok {
                        return Err(err(format!(
                            "Error (gave up acquiring the lock) when acquiring {}-lock on {}",
                            if write { "write" } else { "read" },
                            self.dir_of().display()
                        )));
                    }
                    let t =
                        ::casacure::Table::open_with_lock(self.dir_of(), true, self.lock_options)
                            .map_err(err)?;
                    *inner = Inner::Read(std::sync::Arc::new(t));
                    Ok(())
                }
                Inner::Write { shared } => {
                    {
                        let mut s = shared.lock().unwrap();
                        s.wt.lock(write, nattempts).map_err(err)?;
                    }
                    // A fresh acquire may follow another process's write:
                    // adopt its rows into the read snapshot (the writable
                    // read path answers rows beyond the snapshot with
                    // defaults).
                    let mut s = shared.lock().unwrap();
                    if s.wt.nrows() > s.read.nrows() {
                        s.read = ::casacure::Table::open_with_lock(
                            self.dir_of(),
                            false,
                            core::lockfile::LockOptions::no_locking(),
                        )
                        .map_err(err)?;
                    }
                    Ok(())
                }
            }
        })
    }

    /// `unlock()`: flush pending writes and release the lock. Nothing is
    /// done when no lock is held.
    fn unlock(&self) -> PyResult<()> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(arc) => {
                if let Some(lf) = arc.lock_file() {
                    lf.lock().unwrap().release_read().map_err(err)?;
                }
                Ok(())
            }
            Inner::Write { shared } => {
                let mut s = shared.lock().unwrap();
                s.wt.unlock().map_err(err)?;
                // casacore's `unlock()` physically flushes pending writes so
                // other processes see them once the lock drops. Disk now
                // holds this backing's own descriptor — refresh the snapshot
                // the schema guard compares against, or the handle's next
                // `flush_if_dirty` reads its own unlock-flush as a foreign
                // addcols/removecols and refuses to write (dask-ms's
                // add-column path hits exactly this).
                s.disk_columns = s.wt.desc().columns.iter().map(|c| c.name.clone()).collect();
                s.dirty = false;
                Ok(())
            }
        }
    }

    /// `haslock(write=True)`: whether the write (or read) lock is held.
    #[pyo3(signature = (write = true))]
    fn haslock(&self, write: bool) -> PyResult<bool> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            // The shared lock file's `held` is the process truth (a write
            // lock held anywhere in the process covers every read handle).
            Inner::Read(arc) => Ok(match arc.lock_file().map(|lf| lf.lock().unwrap().held) {
                Some(Some(core::lockfile::LockType::Write)) => true,
                Some(Some(_)) => !write,
                _ => false,
            }),
            Inner::Write { shared } => Ok(shared.lock().unwrap().wt.has_lock(write)),
        }
    }

    /// `lockoptions()`: the option word, inspection interval (seconds) and
    /// maximum wait this handle was opened with.
    fn lockoptions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("option", self.lock_options.mode.as_str())?;
        d.set_item("interval", self.lock_options.interval)?;
        d.set_item("maxwait", self.lock_options.maxwait)?;
        Ok(d.into_any().unbind())
    }

    /// `ismultiused()`: whether another process has the table open.
    fn ismultiused(&self) -> PyResult<bool> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(arc) => arc.is_multi_used().map_err(err),
            Inner::Write { shared } => shared.lock().unwrap().wt.is_multi_used().map_err(err),
        }
    }

    fn flush(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        if let Inner::Write { shared } = &mut *inner {
            flush_if_dirty(&self.dir_of(), shared)?;
        }
        Ok(())
    }

    fn close(&self) -> PyResult<()> {
        self.flush()?;
        // Release any lock this handle holds (the Drop of the core handle
        // would anyway; this is the explicit casacore close semantics).
        let inner = self.inner.lock().unwrap();
        if let Inner::Read(arc) = &*inner {
            if let Some(lf) = arc.lock_file() {
                lf.lock().unwrap().release_read().map_err(err)?;
            }
        }
        drop(inner);
        // A closed handle may have been the last owner of a materialised
        // store (query/sort trees, a writable cell store); give its arena
        // pages back to the OS (see trim_heap).
        trim_heap();
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
        self.auto_tick()?;
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
        if nrow == 0 {
            // casacore returns an empty 1-D array of the column type (a
            // string column: an empty list), whatever the handle kind.
            let desc = self.desc();
            if let Some(cd) = desc.columns.get(col_idx) {
                if let Some(arr) = zeros_for_column(py, cd, &[0])? {
                    return Ok(arr);
                }
                if cd.data_type == core::record::DataType::String {
                    return Ok(PyList::empty(py).into_any().unbind());
                }
            }
        }
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
        let cells = py.detach(|| self.read_col(col_idx, startrow, nrow))?;
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
    ///
    /// A reference table (`query()`/`sort()` result) copies its **selected
    /// rows in their order**: it is materialised first, because `name()`
    /// (which `tablecopy` takes) is the source table's directory and copying
    /// that would copy every source row instead of the result.
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
        let materialised: Option<Table> = {
            let this = slf.borrow();
            let is_view = matches!(&*this.inner.lock().unwrap(), Inner::Read(t) if t.is_view());
            if is_view {
                Some(this.materialise_result(py)?)
            } else {
                None
            }
        };
        let own: String = match &materialised {
            Some(temp) => temp.path.clone(),
            None => slf.call_method0("name")?.extract()?,
        };
        let src = pyo3::types::PyString::new(py, &own);
        let out = crate::helpers::tablecopy(
            py,
            src.as_any(),
            new_table_name,
            deep,
            valuecopy,
            dminfo,
            "aipsrc",
            false,
            false,
        );
        // The temp materialisation (and its scratch directory) goes away now
        // that the copy is on disk.
        drop(materialised);
        out
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
        self.auto_tick()?;
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
        let cells = py.detach(|| self.read_col(col_idx, startrow, nrow))?;
        let cell = cell_shape_of(&cells).iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(py, buf, &cells, cell)
    }

    /// `getcolslice(column, blc, trc, inc=[], startrow=0, nrow=-1, rowincr=1)`
    /// (python-casacore's signature). The legacy 5-argument casacure form
    /// `(column, blc, trc, startrow, nrow)` is also accepted (a positional
    /// `inc` that is an integer means `startrow`). `blc`/`trc` may be a
    /// bare int, as in python-casacore: a length-1 `IPosition`, which
    /// selects a whole 1-D cell and fails in the slicer (a `RuntimeError`)
    /// on any other rank (issue #36).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, blc, trc, inc = None, startrow = 0, nrow = -1, rowincr = 1))]
    fn getcolslice(
        &self,
        py: Python<'_>,
        column: &str,
        blc: &Bound<'_, PyAny>,
        trc: &Bound<'_, PyAny>,
        inc: Option<Bound<'_, PyAny>>,
        mut startrow: i64,
        mut nrow: i64,
        rowincr: i64,
    ) -> PyResult<Py<PyAny>> {
        let blc = slice_arg(blc, "getcolslice blc")?;
        let trc = slice_arg(trc, "getcolslice trc")?;
        let inc_v = split_inc_startrow_nrow(inc, &mut startrow, &mut nrow)?;
        let _ = rowincr;
        self.auto_tick()?;
        let total = self.row_count();
        let startrow = if startrow < 0 { 0 } else { startrow } as u64;
        let nrow = if nrow < 0 {
            total.saturating_sub(startrow)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        self.check_slice_request("getcolslice", column, col_idx, &blc, &trc)?;
        let cells =
            py.detach(|| self.read_colslice(col_idx, &blc, &trc, &inc_v, startrow, nrow))?;
        self.column_to_python(py, col_idx, &cells)
    }

    /// `getcolslicenp(column, buf, blc, trc, inc=[], startrow=0, nrow=-1,
    /// rowincr=1)` (python-casacore's signature). The legacy 5-argument
    /// casacure form `(column, buf, blc, trc, startrow, nrow)` is also
    /// accepted (a positional `inc` that is an integer means `startrow`).
    /// `blc`/`trc` may be a bare int, as in python-casacore (issue #36).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, buf, blc, trc, inc = None, startrow = 0, nrow = -1, rowincr = 1))]
    fn getcolslicenp(
        &self,
        py: Python<'_>,
        column: &str,
        buf: &Bound<'_, PyAny>,
        blc: &Bound<'_, PyAny>,
        trc: &Bound<'_, PyAny>,
        inc: Option<Bound<'_, PyAny>>,
        mut startrow: i64,
        mut nrow: i64,
        rowincr: i64,
    ) -> PyResult<()> {
        let blc = slice_arg(blc, "getcolslicenp blc")?;
        let trc = slice_arg(trc, "getcolslicenp trc")?;
        let inc_v = split_inc_startrow_nrow(inc, &mut startrow, &mut nrow)?;
        let _ = rowincr;
        self.auto_tick()?;
        let total = self.row_count();
        let startrow = if startrow < 0 { 0 } else { startrow } as u64;
        let nrow = if nrow < 0 {
            total.saturating_sub(startrow)
        } else {
            nrow as u64
        };
        let col_idx = self.col_index(column)?;
        self.check_slice_request("getcolslicenp", column, col_idx, &blc, &trc)?;
        // Whole-cell fast path (DDFacet's chunk reader lives here): a slice
        // that covers the fixed cell on a raw-supported column fills the
        // buffer straight from the data files, one memcpy per row.  The
        // per-cell path below costs ~20x on a 435-row chunk read: it builds a
        // heap RecordValue per row and then converts element-by-element
        // through fill_flat's dtype match (measured 2026-10-08, ssd0000.MS
        // DATA 4x4 complex64: 0.70 ms/call vs python-casacore's 0.03).
        let read_table = {
            let inner = self.inner.lock().unwrap();
            match &*inner {
                Inner::Read(t) => Some(std::sync::Arc::clone(t)),
                _ => None,
            }
        };
        if let Some(t) = &read_table {
            let desc = &t.dat.desc.columns[col_idx];
            if slice_is_whole_cell(&blc, &trc, &inc_v, desc) {
                // A whole-cell slice fills (nrow, cell...) logical: pin the
                // buffer to exactly that shape before the raw write
                // (casacore's conformance, issue #38 — it raises for a
                // buffer too big, too small, or with the wrong row count,
                // where casacure used to fill whatever overlapped).
                if let Some(s) = &desc.shape {
                    let mut expected = vec![nrow as usize];
                    expected.extend(s.iter().rev().map(|&d| d.max(0) as usize));
                    check_buf_shape(buf, &expected, "getcolslicenp", column)?;
                }
                if t.raw_column_supported(col_idx)
                    && fill_numpy_raw(py, t, col_idx, startrow, nrow, buf)?
                {
                    return Ok(());
                }
            }
        }
        let cells =
            py.detach(|| self.read_colslice(col_idx, &blc, &trc, &inc_v, startrow, nrow))?;
        // The same conformance for the per-cell path (issue #38).
        let mut expected = vec![nrow as usize];
        expected.extend(cell_shape_of(&cells));
        check_buf_shape(buf, &expected, "getcolslicenp", column)?;
        let cell = cell_shape_of(&cells).iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(py, buf, &cells, cell)
    }

    /// `getcell(column, row)` -> numpy array (or scalar/list).
    fn getcell(&self, py: Python<'_>, column: &str, row: u64) -> PyResult<Py<PyAny>> {
        self.auto_tick()?;
        let col_idx = self.col_index(column)?;
        let v = self.read_cell(col_idx, row)?;
        if let RecordValue::Array(mut a) = v {
            // Array cells keep every stored dimension for `getcol` — casacore
            // returns (1, 79) for a 1-row 79-channel column; the leading-row
            // singleton trim is `getcell`/`getvarcol` semantics only, and
            // producing a bare (79,) here broke skarabina's CHAN_FREQ read
            // (`chan_freq[0]` collapsing to a scalar).
            //
            // python-casacore's `getcell` does conform the cell to the
            // column's *declared* ndim by dropping leading length-1 axes: a
            // (1, N) cell stored in an NDIM=1 column reads back as (N,)
            // while `getvarcol` keeps the stored (1, N) shape (issue #19 —
            // dask-ms's exemplar check compares `getcell` against the
            // descriptor's ndim). ndim <= 0 is casacore's "no declared
            // dimensionality" — nothing to conform to — and fixed-shape
            // columns are unaffected: their stored shape already equals the
            // declaration.
            let ndim = {
                let desc = self.desc();
                desc.columns[col_idx].ndim
            };
            while ndim >= 1 && a.shape.len() as i32 > ndim && a.shape.first() == Some(&1) {
                a.shape.remove(0);
            }
            return convert::array_to_ndarray(py, &a);
        }
        convert::cell_to_py(py, &v)
    }

    /// `iscelldefined(column, rownr)` — whether the cell holds a value
    /// (issue #32). A scalar or fixed-shape-array cell is defined in every
    /// row; a variable-shape array cell is defined only when an array was
    /// written to it (an undefined one reads as its default per issue #8,
    /// and this is the only way to tell the two apart). A pending buffered
    /// write counts as defined; a row added but never written reads as the
    /// schema default, which for a variable-shape column is "no array".
    fn iscelldefined(&self, column: &str, row: u64) -> PyResult<bool> {
        self.auto_tick()?;
        let col_idx = self.col_index(column)?;
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => {
                let n = t.nrows();
                if row >= n {
                    return Err(row_out_of_range(row, n));
                }
                t.is_cell_defined(col_idx, row).map_err(err)
            }
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                let n = s.wt.col_len(col_idx) as u64;
                if row >= n {
                    return Err(row_out_of_range(row, n));
                }
                if s.wt.pending_cell(col_idx, row).is_some() {
                    return Ok(true);
                }
                let desc = &s.wt.desc().columns[col_idx];
                let variable_array = matches!(desc.kind, core::tabledesc::ColumnKind::Array)
                    && !desc.shape.as_deref().is_some_and(|sh| !sh.is_empty());
                if row < s.read.nrows() {
                    if let Some(di) = disk_col_index(&s, col_idx) {
                        return s.read.is_cell_defined(di, row).map_err(err);
                    }
                }
                // Not on disk and not pending: the schema default — a real
                // value for a scalar or fixed-shape column, "no array" for
                // a variable-shape one.
                Ok(!variable_array)
            }
        }
    }

    /// `getcellslice(column, rownr, blc, trc, inc=[])` — python-casacore's
    /// signature, `inc` included (issue #36: the 5th argument existed only
    /// in python-casacore until now). `blc`/`trc`/`inc` may be bare ints,
    /// as in python-casacore: length-1 `IPosition`s that work on a 1-D
    /// cell and fail in the slicer (a `RuntimeError`) on any other rank.
    /// There is no legacy form to disambiguate here, so an integer `inc`
    /// is the stride, like casacore's.
    #[pyo3(signature = (column, row, blc, trc, inc = None))]
    fn getcellslice(
        &self,
        py: Python<'_>,
        column: &str,
        row: u64,
        blc: &Bound<'_, PyAny>,
        trc: &Bound<'_, PyAny>,
        inc: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let blc = slice_arg(blc, "getcellslice blc")?;
        let trc = slice_arg(trc, "getcellslice trc")?;
        let inc_v = match inc {
            Some(v) => slice_arg(&v, "getcellslice inc")?,
            None => Vec::new(),
        };
        self.auto_tick()?;
        let col_idx = self.col_index(column)?;
        self.check_slice_request("getcellslice", column, col_idx, &blc, &trc)?;
        let v = self.read_cellslice(col_idx, row, &blc, &trc, &inc_v)?;
        convert::cell_to_py(py, &v)
    }

    /// `getcellslicenp(column, buf, rownr, blc, trc, inc=[])` —
    /// python-casacore's signature (issue #37: it did not exist at all
    /// until now): fill `buf` from one cell's slice and return `None`.
    /// The buffer carries the *slice's* shape — no leading row axis, unlike
    /// `getcolslicenp` — and must match it exactly (casacore's
    /// `ArrayColumn::getSlice` conformance, issue #38) and be C-contiguous
    /// (`ValueError`, as `getcolslicenp` already raises).  `blc`/`trc`/
    /// `inc` take the scalar-corner rules of issue #36.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, buf, row, blc, trc, inc = None))]
    fn getcellslicenp(
        &self,
        py: Python<'_>,
        column: &str,
        buf: &Bound<'_, PyAny>,
        row: u64,
        blc: &Bound<'_, PyAny>,
        trc: &Bound<'_, PyAny>,
        inc: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let blc = slice_arg(blc, "getcellslicenp blc")?;
        let trc = slice_arg(trc, "getcellslicenp trc")?;
        let inc_v = match inc {
            Some(v) => slice_arg(&v, "getcellslicenp inc")?,
            None => Vec::new(),
        };
        self.auto_tick()?;
        let col_idx = self.col_index(column)?;
        self.check_slice_request("getcellslicenp", column, col_idx, &blc, &trc)?;
        // One cell's slice: the same read as getcellslice, so every corner
        // rank/range refusal (and the -1 whole-cell idiom) comes with it.
        let v = py.detach(|| self.read_cellslice(col_idx, row, &blc, &trc, &inc_v))?;
        let expected: Vec<usize> = match &v {
            // An array column always yields an array cell (even an
            // undefined one reads as the default, issue #8); the shape is
            // exactly what `getcellslice` would return, so the buffer
            // contract and that method's output agree by construction.
            RecordValue::Array(a) => a.shape.iter().map(|&d| d as usize).collect(),
            _ => Vec::new(),
        };
        check_buf_shape(buf, &expected, "getcellslicenp", column)?;
        let cell = expected.iter().product::<usize>().max(1);
        convert::fill_buffer_by_dtype(py, buf, std::slice::from_ref(&v), cell)
    }

    /// `putcellslice(column, rownr, value, blc, trc, inc=[])` —
    /// python-casacore's signature (issue #37: it did not exist at all
    /// until now).  `value` must be exactly the cell's slice shape — no
    /// leading row axis and no broadcast, which casacore's conformance
    /// check enforces (issue #38) — and lands in one cell; an integer
    /// `inc` is the stride, as in casacore (there is no legacy form to
    /// disambiguate here, unlike `putcolslice`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, row, value, blc, trc, inc = None))]
    fn putcellslice(
        &self,
        py: Python<'_>,
        column: &str,
        row: u64,
        value: &Bound<'_, PyAny>,
        blc: &Bound<'_, PyAny>,
        trc: &Bound<'_, PyAny>,
        inc: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let blc = slice_arg(blc, "putcellslice blc")?;
        let trc = slice_arg(trc, "putcellslice trc")?;
        let inc_v = match inc {
            Some(v) => slice_arg(&v, "putcellslice inc")?,
            None => Vec::new(),
        };
        let col_idx = self.col_index(column)?;
        self.check_slice_request("putcellslice", column, col_idx, &blc, &trc)?;
        // Promote to the `putcolslice` value layout — a leading axis of 1
        // for this one row — the same trick `putcell` uses for `putcol`, so
        // the shared overlay below is the single implementation of both.
        let np = py.import("numpy")?;
        let arr = np.getattr("asanyarray")?.call1((value,))?;
        let shape: Vec<i64> = arr.getattr("shape")?.extract()?;
        let mut promoted = vec![1i64];
        promoted.extend(shape.iter().copied());
        let promoted = arr.call_method1("reshape", (promoted,))?;
        self.put_slice_overlay(
            py,
            column,
            col_idx,
            &promoted,
            &blc,
            &trc,
            &inc_v,
            row,
            1,
            "putcellslice",
        )
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
            // python-casacore numbers the keys from row 0 of the table
            // (`getvarcol("V", 1, 2)` -> r2, r3), not from the window's
            // start. (The dict-form putcol/putvarcol input is a different
            // convention: dask-ms builds {"r%d" % i} keys 0-based —
            // conftest.py — so its "rN" means row N.)
            d.set_item(
                format!("r{}", startrow + r + 1),
                convert::cell_to_py(py, &v)?,
            )?;
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
        let is_array_col = matches!(
            self.desc().columns[col_idx].kind,
            core::tabledesc::ColumnKind::Array
        );
        let rec = if is_array_col {
            // One cell of an array column: route through the shared putcol
            // cell conversion (dtype coercion to the column's element type,
            // numpy string arrays, string-array dicts) by promoting the cell
            // to a single-row batch. putcell used to go through
            // `keyword_value`, which cannot convert numeric or string
            // ndarrays, so every array value except already-matching Python
            // scalars-in-lists was rejected where python-casacore accepts
            // them (issue #29: float, int64, short, complex, dcomplex —
            // and numpy string arrays / nested lists).
            let is_seq = value.cast::<PyList>().is_ok() || value.cast::<PyTuple>().is_ok();
            let is_ndarray = !is_seq && value.getattr("dtype").is_ok();
            if is_seq || is_ndarray {
                let np = py.import("numpy")?;
                let arr = if is_seq {
                    np.getattr("asanyarray")?.call1((value,))?
                } else {
                    value.clone()
                };
                let shape: Vec<i64> = arr.getattr("shape")?.extract()?;
                let mut promoted = vec![1i64];
                promoted.extend(shape.iter().copied());
                let promoted = arr.call_method1("reshape", (promoted,))?;
                let mut cells = self.value_to_cells(py, col_idx, &promoted, 1, true, false)?;
                cells.pop().unwrap_or(RecordValue::Int(0))
            } else {
                keyword_value(py, value)?
            }
        } else if matches!(
            self.desc().columns[col_idx].kind,
            core::tabledesc::ColumnKind::Scalar(_)
        ) {
            // A lone scalar, cast to the column's type like every putcol
            // element is (a Python 42.0 into a float column stores as
            // float32, not rejected).
            let rec = keyword_value(py, value)?;
            match self.desc().columns[col_idx].data_type {
                core::record::DataType::Table | core::record::DataType::Record => rec,
                dt => convert::cast_scalar_to(&dt, &rec),
            }
        } else {
            keyword_value(py, value)?
        };
        self.put_cell(col_idx, row, rec)?;
        Ok(())
    }

    /// `rownumbers(other=None)` — with no argument, the base-table row
    /// answering each row of this table, in this table's order
    /// (python-casacore's `table.rownumbers()`); a table that is its own
    /// source answers 0..nrows.  With another table, each of this table's
    /// rows as its row number in `other` (-1 where the row is absent, the
    /// `sel.rownumbers(t)` / `t.rownumbers(sel)` pair).
    #[pyo3(signature = (other = None))]
    fn rownumbers(&self, py: Python<'_>, other: Option<&Table>) -> PyResult<Py<PyAny>> {
        if let Some(o) = other {
            let (base, mine) = self.rooted_rows()?;
            let (o_base, theirs) = o.rooted_rows()?;
            if base != o_base {
                return Err(PyValueError::new_err(
                    "rownumbers: tables do not share a base table",
                ));
            }
            let mut index = std::collections::HashMap::with_capacity(theirs.len());
            for (i, &r) in theirs.iter().enumerate() {
                index.insert(r, i as i64);
            }
            let out: Vec<i64> = mine
                .iter()
                .map(|&r| index.get(&r).copied().unwrap_or(-1))
                .collect();
            return Ok(numpy::PyArray1::from_slice(py, &out).into_any().unbind());
        }
        let total = self.row_count();
        let rows: Vec<u64> = {
            let inner = self.inner.lock().unwrap();
            match &*inner {
                Inner::Read(t) => match t.row_numbers() {
                    Some(r) => r.to_vec(),
                    None => (0..total).collect(),
                },
                Inner::Write { .. } => (0..total).collect(),
            }
        };
        Ok(numpy::PyArray1::from_slice(py, &rows).into_any().unbind())
    }

    /// This handle's rows rooted at the ultimate base table.
    fn rooted_rows(&self) -> PyResult<(std::path::PathBuf, Vec<u64>)> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            Inner::Read(t) => Ok(t.rooted_row_numbers()),
            Inner::Write { shared, .. } => {
                let (total, dir) = {
                    let s = shared.lock().unwrap();
                    (s.wt.col_len(0) as u64, s.wt.dir().to_path_buf())
                };
                Ok((dir, (0..total).collect()))
            }
        }
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
                let cd = &desc.columns[col_idx];
                let is_array_col = matches!(cd.kind, core::tabledesc::ColumnKind::Array);
                let mut cells = self.value_to_cells(py, col_idx, &v, 1, is_array_col, true)?;
                let rec = cells.pop().unwrap_or(RecordValue::Int(0));
                // A lone scalar is cast to the column's type like every
                // other putcol element (a plain 99.0 into a float column
                // stores as float, not rejected).
                let rec = if matches!(cd.kind, core::tabledesc::ColumnKind::Scalar(_)) {
                    match cd.data_type {
                        core::record::DataType::Table | core::record::DataType::Record => rec,
                        dt => convert::cast_scalar_to(&dt, &rec),
                    }
                } else {
                    rec
                };
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

    /// `putcolslice(column, value, blc, trc, inc=[], startrow=0, nrow=-1,
    /// rowincr=1)` — python-casacore's signature, `inc` and `rowincr`
    /// included (issue #36). The legacy 6-argument casacure form
    /// `(column, value, blc, trc, startrow, nrow)` keeps working: a
    /// positional integer in the `inc` slot means `startrow`, exactly as
    /// in `getcolslice` (`tests/test_subcell_tiles.py` uses it).
    /// `blc`/`trc` may be bare ints, as in python-casacore. `rowincr` is
    /// accepted but not honoured, like casacure's other slice methods.
    /// Everything casacore refuses — a non-array column, empty corners, a
    /// value whose rows or slice extent differ from what would be written
    /// — is refused before a single cell changes (issues #37, #38).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, value, blc, trc, inc = None, startrow = 0, nrow = 0, rowincr = 1))]
    fn putcolslice(
        &self,
        py: Python<'_>,
        column: &str,
        value: &Bound<'_, PyAny>,
        blc: &Bound<'_, PyAny>,
        trc: &Bound<'_, PyAny>,
        inc: Option<Bound<'_, PyAny>>,
        mut startrow: i64,
        mut nrow: i64,
        rowincr: i64,
    ) -> PyResult<()> {
        let blc = slice_arg(blc, "putcolslice blc")?;
        let trc = slice_arg(trc, "putcolslice trc")?;
        let inc_v = split_inc_startrow_nrow(inc, &mut startrow, &mut nrow)?;
        let _ = rowincr;
        let col_idx = self.col_index(column)?;
        self.check_slice_request("putcolslice", column, col_idx, &blc, &trc)?;
        let startrow = startrow.max(0) as u64;
        self.put_slice_overlay(
            py,
            column,
            col_idx,
            value,
            &blc,
            &trc,
            &inc_v,
            startrow,
            nrow,
            "putcolslice",
        )
    }
}

/// One corner (`blc`, `trc`) or stride (`inc`) argument of the slice APIs.
/// python-casacore's binding takes an `IPosition` and converts a bare
/// integer to a length-1 one — which selects a whole 1-D cell and, on any
/// other rank, fails inside the `Slicer` with a `RuntimeError`
/// (`Shape IPosition-lengths differ from ndim()`), never a `TypeError`
/// (issue #36: casacure used to reject the scalar before any slicing
/// happened, which casacore callers catching `RuntimeError` never see).  A
/// sequence is taken as-is; anything that is neither (float, str, None) is
/// an argument error, as in casacore — whose `ArgumentError` subclasses
/// `TypeError`.
fn slice_arg(arg: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    if let Ok(seq) = arg.extract::<Vec<i64>>() {
        return Ok(seq);
    }
    if let Ok(n) = arg.extract::<i64>() {
        return Ok(vec![n]);
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a sequence of ints or an int \
         (python-casacore converts a bare int to a length-1 IPosition)"
    )))
}

/// A caller's buffer must be exactly the shape casacore's conformance
/// check asks for (issue #38): too big, too small and wrong-row buffers all
/// raise `ArrayColumn::getColumn ... Table array conformance error` there,
/// while casacure filled whatever overlapped — `fill_flat` is bounds-checked,
/// so no memory was at risk, the caller just silently got partial data.
fn check_buf_shape(
    buf: &Bound<'_, PyAny>,
    expected: &[usize],
    name: &str,
    column: &str,
) -> PyResult<()> {
    let shape: Vec<usize> = buf.getattr("shape")?.extract()?;
    if shape != expected {
        return Err(PyRuntimeError::new_err(format!(
            "{name}: buffer shape {shape:?} does not match the slice shape {expected:?} \
             (casacore: ArrayColumn::getColumn in column {column}: \
             Table array conformance error)"
        )));
    }
    Ok(())
}

/// Interpret the `inc` argument of `getcolslice`/`getcolslicenp`/
/// `putcolslice`, which is either python-casacore's per-dimension step
/// (`inc=[]`, a sequence of ints) or — in the legacy casacure form
/// `(column, blc, trc, startrow, nrow)` — an integer that is really
/// `startrow`. Returns the step sequence and, for the legacy form, shifts
/// `startrow`/`nrow` into their python-casacore positions.
fn split_inc_startrow_nrow(
    inc: Option<Bound<'_, PyAny>>,
    startrow: &mut i64,
    nrow: &mut i64,
) -> PyResult<Vec<i64>> {
    let Some(v) = inc else {
        return Ok(Vec::new());
    };
    if let Ok(seq) = v.extract::<Vec<i64>>() {
        // python-casacore form: `inc` is a per-dimension step.
        return Ok(seq);
    }
    if let Ok(n) = v.extract::<i64>() {
        // Legacy form: this integer is `startrow`; the next positional is
        // `nrow`. `*startrow` currently holds `nrow` (the 4th arg), so shift.
        let legacy_nrow = *startrow;
        *startrow = n;
        *nrow = legacy_nrow;
        return Ok(Vec::new());
    }
    Err(PyTypeError::new_err(
        "slice inc must be a sequence of ints (python-casacore) or an int (legacy casacure startrow)",
    ))
}

/// True when the `getcolslice`/`getcolslicenp` arguments select every element
/// of the cell, so the read can go through the raw bulk path
/// (`fill_numpy_raw`) instead of per-cell decoding.  Matches
/// `Table::read_colslice`'s idiom: any negative `blc`/`trc` means the whole
/// cell, an empty `blc`/`trc` pair too, and scalar columns ignore the slice.
/// `desc.shape` is stored order, so compare against its reverse
/// (python-casacore's logical order, as `fill_numpy_raw` also does).
fn slice_is_whole_cell(
    blc: &[i64],
    trc: &[i64],
    inc: &[i64],
    desc: &core::tabledesc::ColumnDesc,
) -> bool {
    use core::tabledesc::ColumnKind;
    if !matches!(desc.kind, ColumnKind::Array) {
        return true; // scalar column: the slice is ignored by read_colslice
    }
    if (blc.is_empty() && trc.is_empty())
        || blc.iter().any(|&b| b < 0)
        || trc.iter().any(|&t| t < 0)
    {
        return true;
    }
    let Some(s) = &desc.shape else {
        return false; // variable-shape array: no fixed whole to cover
    };
    if blc.len() != s.len() || trc.len() != s.len() {
        return false;
    }
    blc.iter()
        .zip(trc.iter())
        .zip(s.iter().rev())
        .enumerate()
        .all(|(d, ((&b, &t), &stored))| {
            let n = stored.max(0);
            let step = inc.get(d).copied().unwrap_or(1).max(1);
            b == 0 && step == 1 && t == n - 1
        })
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
#[pyo3(signature = (name, tabledesc = None, nrow = 0, dminfo = None, readonly = false, _ack = true, lockoptions = None, *_args, **_kwargs))]
pub fn table(
    py: Python<'_>,
    name: &Bound<'_, PyAny>,
    tabledesc: Option<&Bound<'_, PyAny>>,
    nrow: i64,
    dminfo: Option<&Bound<'_, PyAny>>,
    readonly: bool,
    _ack: bool,
    lockoptions: Option<&Bound<'_, PyAny>>,
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Table> {
    let mut name = path_string(name)?;
    // python-casacore's `_remove_prefix`: a `"Table: <path>"` keyword value
    // passed to `table()` opens that path (see `casacore/tables/tablehelper.py`).
    if let Some(rest) = name.strip_prefix("Table: ") {
        name = rest.to_string();
    }
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
    // python-casacore only consults `dminfo` when the tabledesc creates a
    // new table; it is inert on an open.
    let dminfo_rec = match (desc_json.as_deref(), dminfo) {
        (Some(_), Some(dm)) if !dm.is_none() => Some(dminfo_record(py, dm)?),
        _ => None,
    };
    // python-casacore's default: `lockoptions='default'` (AutoLocking).
    let options = match lockoptions {
        Some(lo) if !lo.is_none() => Table::parse_lockoptions(lo)?,
        _ => core::lockfile::LockOptions::locking_default(),
    };
    // Opening acquires the mode's lock, which may wait on another process:
    // do that without the GIL so other Python threads keep running.
    py.detach(|| {
        Table::open_or_create(
            &name,
            desc_json.as_deref(),
            dminfo_rec.as_ref(),
            nrow.max(0) as u64,
            !readonly,
            options,
        )
    })
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
                flush_if_dirty(&obj.dir_of(), shared)?;
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
                // The taql path owns these tables' writes; keep their
                // replacement backing lock-free as before.
                if !refresh_write(dir, shared, core::lockfile::LockOptions::no_locking()) {
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
            // A TaQL-created scratch table: fresh directory, single handle,
            // no locking (also keeps lock files out of temp dirs).
            let t = Table::open_or_create(
                &path.display().to_string(),
                None,
                None,
                0,
                true,
                core::lockfile::LockOptions::no_locking(),
            )?;
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
    sweep_stale_scratch();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(err)?;
    // from here on the directory is owned by this guard: an error below
    // removes it again, and success hands it to the table's backing
    let scratch = ScratchDir::new(dir.clone());

    let nrows = out.nrows() as u64;
    let mut desc = TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        // The source table's keywords (python-casacore's reference-table
        // semantics: `t.query(...).getkeyword('ANTENNA')` returns the source
        // table's keyword).
        keywords: out.source_keywords.clone(),
        private_keywords: TableRecord {
            desc: Default::default(),
            record_type: ::casacure::record::RECORD_VARIABLE,
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
                record_type: ::casacure::record::RECORD_VARIABLE,
                values: Vec::new(),
            },
            kind,
            tile_shape: None,
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
    let disk_columns = read
        .dat
        .desc
        .columns
        .iter()
        .map(|c| c.name.clone())
        .collect();
    let shared = std::sync::Arc::new(std::sync::Mutex::new(WriteData {
        read,
        wt,
        dirty: false,
        pin: DirPin::new(&dir),
        disk_columns,
        scratch: Some(scratch),
    }));
    register_write(&dir, &shared);
    Ok(Table {
        path: dir.display().to_string(),
        writable: true,
        inner: Mutex::new(Inner::Write { shared }),
        // TaQL result tables are fresh temp-dir tables with one handle.
        lock_options: core::lockfile::LockOptions::no_locking(),
        auto_state: Mutex::new(AutoLockState::default()),
        view: false,
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

/// Unit tests for the `getcol` stacking check (issue #25): one stacked array
/// holds one shape, and empty (undefined) cells stay exempt because they read
/// as defaults (issue #8).  The user-visible refusal — `getcol` raising
/// instead of cutting every row to the first row's shape, with the hint to use
/// `getvarcol` — is exercised end to end in `tests/test_core_tables_e2e.py`
/// (the refusal itself needs a linked Python to build its `PyErr`); this pins
/// the decision table itself, including the corners a table rarely produces.
#[cfg(test)]
mod getcol_conform_tests {
    use super::first_shape_conflict;
    use ::casacure::record::{ArrayData, ArrayValue, RecordValue};

    /// A double array cell whose element count follows from `shape`.
    fn array(shape: &[u32]) -> RecordValue {
        let n = shape.iter().map(|&d| d as usize).product();
        RecordValue::Array(ArrayValue {
            shape: shape.to_vec(),
            data: ArrayData::Double(vec![0.0; n]),
        })
    }

    #[test]
    fn rows_of_one_shape_stack() {
        assert!(first_shape_conflict(&[array(&[2]), array(&[2]), array(&[2])]).is_none());
    }

    #[test]
    fn a_single_row_or_no_rows_always_stacks() {
        assert!(first_shape_conflict(&[array(&[3])]).is_none());
        assert!(first_shape_conflict(&[]).is_none());
    }

    #[test]
    fn rows_of_different_shapes_report_both_shapes() {
        // The issue #25 shape: getcol must refuse, not cut every row to the
        // first row's shape. The verdict carries the two shapes, which is
        // exactly what the error message renders.
        let cells = [array(&[1]), array(&[2]), array(&[3])];
        assert_eq!(
            first_shape_conflict(&cells),
            Some((&[1u32][..], &[2u32][..]))
        );
        // Where the difference appears does not matter: a divergent second
        // row or a divergent last one, either way.
        let late = [array(&[2]), array(&[2]), array(&[3])];
        assert_eq!(
            first_shape_conflict(&late),
            Some((&[2u32][..], &[3u32][..]))
        );
        let reversed = [array(&[3]), array(&[2])];
        assert_eq!(
            first_shape_conflict(&reversed),
            Some((&[3u32][..], &[2u32][..]))
        );
    }

    #[test]
    fn undefined_and_empty_cells_stay_exempt() {
        // #8: a never-written cell reads as the default.  It must not make
        // the column ragged next to the one real shape.
        let cells = vec![
            array(&[2]),
            array(&[]),               // no shape at all: never written
            array(&[0]),              // a zero dimension: zero elements
            RecordValue::Double(0.0), // a scalar among arrays is skipped
            array(&[2]),
        ];
        assert!(first_shape_conflict(&cells).is_none());
    }

    #[test]
    fn all_undefined_cells_stack_as_defaults() {
        assert!(first_shape_conflict(&[array(&[]), array(&[0]), RecordValue::Int(0)]).is_none());
    }

    #[test]
    fn exempt_cells_do_not_hide_a_real_difference() {
        // The exemption covers exempt cells only: two real, different shapes
        // still refuse with undefined rows interleaved between them.
        assert_eq!(
            first_shape_conflict(&[array(&[]), array(&[2]), array(&[3])]),
            Some((&[2u32][..], &[3u32][..]))
        );
        assert_eq!(
            first_shape_conflict(&[array(&[2]), array(&[0]), array(&[3])]),
            Some((&[2u32][..], &[3u32][..]))
        );
    }
}

/// Unit tests for the out-of-band schema check (issue #30): a later open
/// reusing this process's live backing must notice that another writer
/// added, removed or replaced `table.dat` — or that it became unreadable —
/// instead of serving the stale schema. The end-to-end scenarios (a column
/// python-casacore/dask-ms added, an external `removecols`, a dirty backing
/// refusing to flush over a foreign column) are pinned in
/// `tests/test_cross_writer_visibility.py`; this covers the decision
/// itself, including the unreadable-file corner no table produces on
/// purpose.
#[cfg(test)]
mod cross_writer_column_tests {
    use super::{disk_column_names, disk_columns_differ};
    use ::casacure::record::{DataType, RecordValue, TableRecord};
    use ::casacure::{create_table, ColumnDesc, ColumnKind, TableDesc};
    use std::path::PathBuf;

    /// `<temp>/<name>`, removed first and again on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let p = std::env::temp_dir()
                .join(format!("casacure-py-diskcols-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Scratch(p)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn empty_record() -> TableRecord {
        TableRecord {
            desc: Default::default(),
            record_type: ::casacure::record::RECORD_VARIABLE,
            values: Vec::new(),
        }
    }

    /// One scalar double column per name — enough to shape `table.dat`'s
    /// column list, which is all the check reads.
    fn desc_with(names: &[&str]) -> TableDesc {
        TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: names
                .iter()
                .map(|n| ColumnDesc {
                    name: (*n).into(),
                    comment: String::new(),
                    data_type: DataType::Double,
                    data_manager_type: "StandardStMan".into(),
                    data_manager_group: "SSM".into(),
                    options: 0,
                    ndim: -1,
                    shape: None,
                    max_length: 0,
                    keywords: empty_record(),
                    kind: ColumnKind::Scalar(RecordValue::Double(0.0)),
                    tile_shape: None,
                })
                .collect(),
        }
    }

    /// A real table directory holding exactly `names`, in order.
    fn table_with(tag: &str, names: &[&str]) -> Scratch {
        let dir = Scratch::new(tag);
        let values: Vec<Vec<RecordValue>> = names
            .iter()
            .map(|_| vec![RecordValue::Double(1.0)])
            .collect();
        create_table(dir.0.as_path(), &desc_with(names), &values).unwrap();
        dir
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn the_backing_owners_own_column_set_is_not_a_change() {
        let dir = table_with("same", &["VALUE"]);
        assert_eq!(disk_column_names(&dir.0), Some(names(&["VALUE"])));
        assert!(!disk_columns_differ(&dir.0, &names(&["VALUE"])));
    }

    #[test]
    fn a_column_added_out_of_band_is_a_change() {
        // What a foreign writer's addcols leaves behind: the disk column
        // list holds a column this backing never synced (issue #30's
        // dask-ms case).
        let dir = table_with("added", &["VALUE", "EXTRA"]);
        assert_eq!(disk_column_names(&dir.0), Some(names(&["VALUE", "EXTRA"])));
        assert!(disk_columns_differ(&dir.0, &names(&["VALUE"])));
    }

    #[test]
    fn a_column_removed_out_of_band_is_a_change() {
        // The backing still believes EXTRA exists; disk says otherwise.
        let dir = table_with("removed", &["VALUE"]);
        assert!(disk_columns_differ(&dir.0, &names(&["VALUE", "EXTRA"])));
    }

    #[test]
    fn an_unreadable_table_dat_counts_as_changed() {
        // The callers re-materialise from disk or drop the entry when this
        // reports "changed", so a directory that is no longer a table must
        // not be served from the stale backing either.
        let dir = Scratch::new("nodat");
        assert_eq!(disk_column_names(&dir.0), None);
        assert!(disk_columns_differ(&dir.0, &names(&["VALUE"])));

        let dir = table_with("corrupt", &["VALUE"]);
        std::fs::write(dir.0.join("table.dat"), b"not a table").unwrap();
        assert_eq!(disk_column_names(&dir.0), None);
        assert!(disk_columns_differ(&dir.0, &names(&["VALUE"])));
    }
}

impl Table {
    /// The shared write half of `putcolslice`/`putcellslice` (issues #36,
    /// #37, #38): overlay `value`, shaped `(nrow, sub...)`, into each row's
    /// cell at logical `blc..trc` stepping `inc`. Every mismatch casacore
    /// refuses is refused first — before a single cell is written, so a
    /// refusal never leaves a half-written window behind: the value's row
    /// count against the rows actually written, a window past the last row,
    /// the corner ranks (pair, cell, value), and the value's per-dimension
    /// extent against the slice.
    #[allow(clippy::too_many_arguments)]
    fn put_slice_overlay(
        &self,
        py: Python<'_>,
        column: &str,
        col_idx: usize,
        value: &Bound<'_, PyAny>,
        blc: &[i64],
        trc: &[i64],
        inc_v: &[i64],
        startrow: u64,
        nrow: i64,
        name: &str,
    ) -> PyResult<()> {
        // casacore resolves its nrow=-1 to the rest of the column and the
        // value must carry exactly those rows: casacure's old value-driven
        // default silently wrote a prefix (issue #38).
        let total = self.row_count();
        let resolved = if nrow <= 0 {
            total.saturating_sub(startrow)
        } else {
            nrow as u64
        };
        // Overlay `value[nrow, sub...]` into each fixed cell at logical
        // blc..trc stepping inc. The numpy array is `(nrow, s0, s1, ...)` in
        // C order.
        let flat = convert::numpy_to_record_flat(value)
            .ok_or_else(|| PyTypeError::new_err(format!("{name}: unsupported element type")))??;
        let shape_obj = value.getattr("shape")?;
        let shape: Vec<usize> = shape_obj.extract()?;
        if shape.is_empty() {
            return Err(PyValueError::new_err(format!("{name}: empty value")));
        }
        let sub: Vec<usize> = shape[1..].to_vec();
        // Exactly the rows to write (#38): casacore's putColumnCells
        // conformance.
        let rows = value.len()? as u64;
        if rows != resolved {
            return Err(PyRuntimeError::new_err(format!(
                "{name}: value has {rows} rows but {resolved} start at row {startrow} \
                 (casacore: ArrayColumn::putColumnCells: Table array conformance error)"
            )));
        }
        if startrow >= total || startrow + resolved > total {
            let bad = startrow.max(total);
            return Err(row_out_of_range(bad, total));
        }
        // casacore's Slicer refuses corners whose ranks differ from each
        // other or from the cell ("Shape IPosition-lengths differ from
        // ndim()").  Without this the starts[k]/ends[k] indexing below
        // panics across the FFI boundary when the value carries more
        // dimensions than the corners — a wrong-rank putcolslice was a
        // `PanicException`, not an error (issue #36).
        if blc.len() != trc.len() {
            return Err(PyRuntimeError::new_err(format!(
                "{name}: blc {blc:?} and trc {trc:?} lengths differ \
                 (casacore: Slicer error: IPosition-lengths differ)"
            )));
        }
        // Fixed cells pin the corner rank and bounds (#38): casacore's
        // Slicer compares the IPositions against ndim(), and a corner past
        // the cell fails the way a read slice fails.
        let cell_dims: Option<Vec<i64>> = {
            let desc = self.desc();
            desc.columns[col_idx]
                .shape
                .as_ref()
                .map(|s| s.iter().rev().map(|&d| d.max(0)).collect())
        };
        if let Some(logical) = &cell_dims {
            if blc.len() != logical.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "{name}: corner rank {} does not match the cell rank {} \
                     (casacore: Slicer error: Shape IPosition-lengths differ from ndim())",
                    blc.len(),
                    logical.len()
                )));
            }
            for k in 0..blc.len() {
                let start = blc[k].max(0);
                let end = trc[k];
                if start >= logical[k] || (end >= 0 && end >= logical[k]) {
                    return Err(PyRuntimeError::new_err(format!(
                        "{name}: slice {blc:?}..{trc:?} does not fit the cell shape {logical:?}"
                    )));
                }
            }
        }
        // Full-cell slice (blc<=0.., trc<0, unit stride) -> plain putcol.
        let full = blc.iter().all(|&b| b <= 0)
            && trc.iter().all(|&t| t < 0)
            && (inc_v.is_empty() || inc_v.iter().all(|&s| s == 1));
        if full {
            return self.putcol(py, column, value, startrow as i64, resolved as i64);
        }
        if sub.len() != blc.len() {
            return Err(PyRuntimeError::new_err(format!(
                "{name}: value slice rank {} does not fit the corner rank {} \
                 (casacore: Slicer error: Shape IPosition-lengths differ from ndim())",
                sub.len(),
                blc.len()
            )));
        }
        if !inc_v.is_empty() && inc_v.len() != blc.len() {
            return Err(PyRuntimeError::new_err(format!(
                "{name}: inc {inc_v:?} and blc {blc:?} lengths differ \
                 (casacore: Slicer error: IPosition-lengths differ)"
            )));
        }
        // Per-dimension stride: empty inc means 1 everywhere, as in
        // python-casacore. Indexed by the value's sub-rank, which the
        // checks above bind to the corner rank.
        let steps: Vec<usize> = (0..sub.len())
            .map(|k| inc_v.get(k).map(|&s| s.max(1) as usize).unwrap_or(1))
            .collect();
        let sub_cell = sub.iter().product::<usize>().max(1);
        let starts: Vec<usize> = blc.iter().map(|&b| b.max(0) as usize).collect();
        // inclusive ends reached by stepping from the start; a negative trc
        // means "to the end of the sub-slice" (the extent check below has
        // already pinned that length against the cell where the cell shape
        // is known).
        let ends: Vec<usize> = trc
            .iter()
            .zip(starts.iter())
            .zip(sub.iter())
            .zip(steps.iter())
            .map(|(((&t, &s), &subd), &step)| {
                if t < 0 {
                    s + (subd - 1) * step
                } else {
                    t.max(0) as usize
                }
            })
            .collect();
        // Exactly the slice's elements (#38): casacore's putSlice /
        // putColumn(slicer) conformance, per dimension. A fixed cell's own
        // end resolves a negative trc; variable-shape cells cannot be
        // resolved before the cell is read and stay value-driven.
        for k in 0..sub.len() {
            let end = if trc[k] >= 0 {
                Some(trc[k])
            } else {
                cell_dims.as_ref().map(|l| l[k] - 1)
            };
            let Some(e) = end else { continue };
            let start = blc[k].max(0);
            if e < start {
                return Err(PyRuntimeError::new_err(format!(
                    "{name}: slice {blc:?}..{trc:?} is not ascending"
                )));
            }
            let extent = (e - start) / steps[k] as i64 + 1;
            if sub[k] != extent as usize {
                return Err(PyRuntimeError::new_err(format!(
                    "{name}: value has {} elements in dimension {k}, the slice has {extent} \
                     (casacore: Table array conformance error (shape=[{}], expected [{extent}]))",
                    sub[k], sub[k]
                )));
            }
        }

        for r in 0..resolved as usize {
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
                    let step = steps[k];
                    let span = ends[k].saturating_sub(s) / step + 1;
                    if coords[k] >= span {
                        continue;
                    }
                    loc[k] = s + coords[k] * step;
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
}

impl Table {
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
            return Err(not_writable(self.view));
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
}

impl Table {
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
        inc: &[i64],
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
            Inner::Read(t) => t.getcellslice_inc(col_idx, row, blc, trc, inc).map_err(err),
            Inner::Write { shared, .. } => {
                let s = shared.lock().unwrap();
                let n = s.wt.col_len(col_idx) as u64;
                if row >= n {
                    return Err(row_out_of_range(row, n));
                }
                let cell = merged_cell(&s, col_idx, row)?;
                slice_cell(cell, blc, trc, inc)
            }
        }
    }

    fn read_colslice(
        &self,
        col_idx: usize,
        blc: &[i64],
        trc: &[i64],
        inc: &[i64],
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
                .getcolslice_inc(col_idx, blc, trc, inc, startrow, nrow)
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
                    out.push(slice_cell(cell, blc, trc, inc)?);
                }
                Ok(out)
            }
        }
    }
}

impl Table {
    /// Store a whole `putcol` batch of cells (one `RecordValue` per row).
    ///
    /// Like [`Table::put_cells`]: the shared-store lock is taken once for the
    /// whole batch and per-cell type/shape validation is kept, but the store
    /// loop runs with the GIL released so the dask scheduler can overlap
    /// independent writes.
    ///
    /// NO Rust lock may be held across the detach: re-attaching the GIL at
    /// its end while holding the store lock deadlocks against any thread
    /// that holds the GIL and wants the store (`col_index` -> `desc`, a
    /// concurrent `flush`, another handle's `open`).  The backing is cloned
    /// out and locked inside the detach instead.
    fn put_cells_batch(
        &self,
        py: Python<'_>,
        col_idx: usize,
        startrow: u64,
        values: Vec<RecordValue>,
    ) -> PyResult<()> {
        let shared = {
            let inner = self.inner.lock().unwrap();
            let Inner::Write { shared, .. } = &*inner else {
                return Err(not_writable(self.view));
            };
            std::sync::Arc::clone(shared)
        };
        py.detach(|| -> PyResult<()> {
            let mut s = shared.lock().unwrap();
            // Validate against the column's declared type and fixed shape
            // before writing (parity with the per-cell `put_cell` checks).
            let desc = s.wt.desc().clone();
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
                s.wt.putcell(col_idx, row, value).map_err(err)?;
                // Mark the store dirty per stored cell: a batch that fails
                // mid-way keeps the cells it already wrote pending for the
                // next flush instead of silently dropping them.
                s.dirty = true;
            }
            Ok(())
        })
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
                value.getattr("dtype")?.str()?.to_cow()?
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
        if !is_string_array_column(cells) {
            check_cells_conform(cells)?;
        }
        let cell_shape = cell_shape_of(cells);
        convert::arrays_to_ndarray_getcol(py, cells, &cell_shape)
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
}
