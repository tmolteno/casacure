//! The cost of a small incremental write must track the rows written, not
//! the size of the table.
//!
//! dask-ms writes a changed column chunk by chunk and flushes after each
//! chunk. A flush patches only the pending cells in place, but the write
//! buffer used to hold one slot per *table* row for every written column:
//! `WritableTable::grow_cells` sized it to the whole table on the first
//! write after each flush and `clear_pending` freed it again, so every
//! one-row `putcell` + `flush` cost O(table rows) in allocation and
//! initialisation (~61% of a one-row putcol + flush in a py-spy profile,
//! 100k-row table).
//!
//! Timing tests would be flaky, so these count the bytes allocated (a
//! deterministic quantity) during one-row write + flush rounds on a small
//! and a large table of identical layout, and require the per-flush
//! allocation to grow by less than one byte per extra table row. (What
//! remains is metadata the flush re-reads, e.g. the StandardStMan bucket
//! index at one entry per bucket, ~0.6 B per row here.)
//!
//! This is its own test binary because it installs a global allocator; the
//! counter is thread-local so tests running in parallel do not see each
//! other's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use casacure::record::{ArrayData, ArrayValue, DataType, RecordValue, TableRecord};
use casacure::{ColumnDesc, ColumnKind, TableDesc, WritableTable};

struct CountingAlloc;

thread_local! {
    static ALLOCATED: Cell<u64> = const { Cell::new(0) };
}

fn count(bytes: usize) {
    // `try_with`: allocations during thread teardown must not panic.
    let _ = ALLOCATED.try_with(|c| c.set(c.get() + bytes as u64));
}

// SAFETY: every call is forwarded unchanged to the system allocator; the
// wrapper only records the requested sizes.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Bytes allocated on this thread while running `f`.
fn allocated_by(f: impl FnOnce()) -> u64 {
    let before = ALLOCATED.with(Cell::get);
    f();
    ALLOCATED.with(Cell::get) - before
}

const SMALL_ROWS: u64 = 4_096;
const LARGE_ROWS: u64 = 32 * SMALL_ROWS;
/// Measured one-row write + flush rounds (after one warm-up round).
const ROUNDS: u64 = 8;
/// Allowed growth of the per-flush allocation per extra table row.
const MAX_BYTES_PER_EXTRA_ROW: f64 = 1.0;

fn empty_record() -> TableRecord {
    TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    }
}

fn scalar(name: &str, dt: DataType, default: RecordValue, dm: &str, group: &str) -> ColumnDesc {
    ColumnDesc {
        name: name.into(),
        comment: String::new(),
        data_type: dt,
        data_manager_type: dm.into(),
        data_manager_group: group.into(),
        options: 0,
        ndim: -1,
        shape: None,
        max_length: 0,
        keywords: empty_record(),
        kind: ColumnKind::Scalar(default),
    }
}

/// A fixed-shape array column; `casa_shape` is the stored (reversed) shape.
fn array(name: &str, dt: DataType, casa_shape: Vec<i64>, dm: &str, group: &str) -> ColumnDesc {
    ColumnDesc {
        name: name.into(),
        comment: String::new(),
        data_type: dt,
        data_manager_type: dm.into(),
        data_manager_group: group.into(),
        options: 4, // FixedShape
        ndim: casa_shape.len() as i32,
        shape: Some(casa_shape),
        max_length: 0,
        keywords: empty_record(),
        kind: ColumnKind::Array,
    }
}

/// A cut-down MS main table: TIME on IncrementalStMan, ANTENNA2 + FLAG_ROW +
/// UVW sharing one StandardStMan (as in a real MS), FLAG and WEIGHT on
/// TiledShapeStMan.
fn ms_like_desc() -> TableDesc {
    TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: empty_record(),
        private_keywords: empty_record(),
        columns: vec![
            scalar(
                "TIME",
                DataType::Double,
                RecordValue::Double(0.0),
                "IncrementalStMan",
                "ISMData",
            ),
            scalar(
                "ANTENNA2",
                DataType::Int,
                RecordValue::Int(0),
                "StandardStMan",
                "SSMData",
            ),
            scalar(
                "FLAG_ROW",
                DataType::Bool,
                RecordValue::Bool(false),
                "StandardStMan",
                "SSMData",
            ),
            array("UVW", DataType::Double, vec![3], "StandardStMan", "SSMData"),
            array(
                "FLAG",
                DataType::Bool,
                vec![4, 16],
                "TiledShapeStMan",
                "TiledFlag",
            ),
            array(
                "WEIGHT",
                DataType::Float,
                vec![4],
                "TiledShapeStMan",
                "TiledWeight",
            ),
        ],
    }
}

const TIME: usize = 0;
const ANTENNA2: usize = 1;
const FLAG_ROW: usize = 2;
const UVW: usize = 3;
const FLAG: usize = 4;
const WEIGHT: usize = 5;

fn flag_cell(set: bool) -> RecordValue {
    RecordValue::Array(ArrayValue {
        shape: vec![16, 4],
        data: ArrayData::Bool(vec![set; 64]),
    })
}

fn weight_cell(w: f32) -> RecordValue {
    RecordValue::Array(ArrayValue {
        shape: vec![4],
        data: ArrayData::Float(vec![w; 4]),
    })
}

fn uvw_cell(r: u64) -> RecordValue {
    RecordValue::Array(ArrayValue {
        shape: vec![3],
        data: ArrayData::Double(vec![r as f64, 0.5, -1.0]),
    })
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "casacure-flushcost-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Create an `nrows`-row MS-like table with every column written, on disk.
fn build_table(tag: &str, nrows: u64) -> std::path::PathBuf {
    let dir = temp_dir(tag);
    let mut wt = WritableTable::create(&dir, ms_like_desc());
    wt.addrows(nrows);
    for r in 0..nrows {
        wt.putcell(TIME, r, RecordValue::Double((r / 64) as f64))
            .unwrap();
        wt.putcell(ANTENNA2, r, RecordValue::Int((r % 63) as i32))
            .unwrap();
        wt.putcell(FLAG_ROW, r, RecordValue::Bool(false)).unwrap();
        wt.putcell(UVW, r, uvw_cell(r)).unwrap();
        wt.putcell(FLAG, r, flag_cell(false)).unwrap();
        wt.putcell(WEIGHT, r, weight_cell(1.0)).unwrap();
    }
    wt.flush().unwrap();
    dir
}

/// Average bytes allocated by one `putcell` of a single row of `col` plus
/// the `flush` that persists it, on a writable (dask-ms style) handle of an
/// `nrows`-row table. One unmeasured warm-up round comes first.
fn one_row_flush_bytes(
    tag: &str,
    nrows: u64,
    col: usize,
    value: impl Fn(u64) -> RecordValue,
) -> f64 {
    let dir = build_table(tag, nrows);
    let (_snapshot, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    // Spread the rows over the table so the written cells land in
    // different buckets / tiles.
    let row = |i: u64| (i * 997) % nrows;
    wt.putcell(col, row(0), value(0)).unwrap();
    wt.flush().unwrap();
    let bytes = allocated_by(|| {
        for i in 1..=ROUNDS {
            wt.putcell(col, row(i), value(i)).unwrap();
            wt.flush().unwrap();
        }
    });
    drop(wt);
    let _ = std::fs::remove_dir_all(&dir);
    bytes as f64 / ROUNDS as f64
}

/// Assert the one-row write + flush of `col` costs (nearly) the same
/// allocation on a small and a 32x larger table.
fn assert_flush_cost_independent_of_table_rows(
    what: &str,
    col: usize,
    value: impl Fn(u64) -> RecordValue + Copy,
) {
    let small = one_row_flush_bytes(&format!("{what}-small"), SMALL_ROWS, col, value);
    let large = one_row_flush_bytes(&format!("{what}-large"), LARGE_ROWS, col, value);
    let per_extra_row = (large - small) / (LARGE_ROWS - SMALL_ROWS) as f64;
    assert!(
        per_extra_row < MAX_BYTES_PER_EXTRA_ROW,
        "one-row {what} putcell+flush allocates {small:.0} B on a {SMALL_ROWS}-row table \
         but {large:.0} B on a {LARGE_ROWS}-row table: {per_extra_row:.2} B per extra \
         table row (limit {MAX_BYTES_PER_EXTRA_ROW}) — the write buffer is sized to the \
         table, not to the rows written"
    );
}

#[test]
fn one_row_flush_of_ssm_bool_scalar_does_not_scale_with_table_rows() {
    // FLAG_ROW: the StandardStMan in-place bit patch.
    assert_flush_cost_independent_of_table_rows("flag_row", FLAG_ROW, |i| {
        RecordValue::Bool(i % 2 == 1)
    });
}

#[test]
fn one_row_flush_of_ssm_int_scalar_does_not_scale_with_table_rows() {
    // ANTENNA2: the StandardStMan in-place byte patch.
    assert_flush_cost_independent_of_table_rows("antenna2", ANTENNA2, |i| {
        RecordValue::Int(100 + i as i32)
    });
}

#[test]
fn one_row_flush_of_tsm_bool_array_does_not_scale_with_table_rows() {
    // FLAG: the TiledShapeStMan bit-packed in-place patch.
    assert_flush_cost_independent_of_table_rows("flag", FLAG, |i| flag_cell(i % 2 == 1));
}

#[test]
fn one_row_flush_of_tsm_float_array_does_not_scale_with_table_rows() {
    // WEIGHT: the TiledShapeStMan byte-aligned in-place patch.
    assert_flush_cost_independent_of_table_rows("weight", WEIGHT, |i| weight_cell(2.0 + i as f32));
}

/// The measurement itself must be sound: the written values really reach
/// the table, so the tests above measure a working incremental flush (not,
/// say, a write that silently went nowhere).
#[test]
fn one_row_flushes_persist_the_written_cells() {
    let nrows = SMALL_ROWS;
    let dir = build_table("persist", nrows);
    {
        let (_snapshot, mut wt) = WritableTable::open_for_update(&dir).unwrap();
        for i in 0..ROUNDS {
            let r = (i * 997) % nrows;
            wt.putcell(FLAG_ROW, r, RecordValue::Bool(true)).unwrap();
            wt.putcell(FLAG, r, flag_cell(true)).unwrap();
            wt.flush().unwrap();
        }
    }
    let t = casacure::Table::open(&dir, false).unwrap();
    for r in 0..nrows {
        let written = (0..ROUNDS).any(|i| (i * 997) % nrows == r);
        assert_eq!(
            t.getcell(FLAG_ROW, r).unwrap(),
            RecordValue::Bool(written),
            "FLAG_ROW row {r}"
        );
        assert_eq!(
            t.getcell(FLAG, r).unwrap(),
            flag_cell(written),
            "FLAG row {r}"
        );
        assert_eq!(
            t.getcell(ANTENNA2, r).unwrap(),
            RecordValue::Int((r % 63) as i32)
        );
        assert_eq!(t.getcell(UVW, r).unwrap(), uvw_cell(r));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
