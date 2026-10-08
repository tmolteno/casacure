//! Casacore-style **reference tables** (`table.query()` / `table.sort()`
//! results, issue #16): a row order over the source table with no cells
//! copied.
//!
//! A reference table has to answer every read API with exactly the values
//! the source rows it maps to hold — in its own row order — while exposing
//! the source's descriptor and no data files of its own. These tests build
//! an MS-like table on every storage manager casacore uses (IncrementalStMan
//! scalars, StandardStMan scalars and arrays, tiled Bool/Float arrays) and
//! hold a shuffled order (with a repeated row) against it.

#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use std::sync::Arc;

use casacure::record::{ArrayData, ArrayValue, DataType, RecordValue, TableRecord};
use casacure::table::TableReadError;
use casacure::taql::{execute, execute_row_order, TaqlResult};
use casacure::{ColumnDesc, ColumnKind, Table, TableDesc, WritableTable};

fn empty_record() -> TableRecord {
    TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    }
}

fn scalar(name: &str, dt: DataType, default: RecordValue, dm: &str) -> ColumnDesc {
    ColumnDesc {
        name: name.into(),
        comment: String::new(),
        data_type: dt,
        data_manager_type: dm.into(),
        data_manager_group: dm.into(),
        options: 0,
        ndim: -1,
        shape: None,
        max_length: 0,
        keywords: empty_record(),
        kind: ColumnKind::Scalar(default),
    }
}

/// A fixed-shape array column; `casa_shape` is the stored (reversed) shape.
fn array(name: &str, dt: DataType, casa_shape: Vec<i64>, dm: &str) -> ColumnDesc {
    ColumnDesc {
        name: name.into(),
        comment: String::new(),
        data_type: dt,
        data_manager_type: dm.into(),
        // A group per tiled column: TiledShapeStMan holds one array column
        // per group (the MS layout: FLAG and WEIGHT each have their own).
        data_manager_group: format!("{dm}-{name}"),
        options: 4, // FixedShape
        ndim: casa_shape.len() as i32,
        shape: Some(casa_shape),
        max_length: 0,
        keywords: empty_record(),
        kind: ColumnKind::Array,
    }
}

const TIME: usize = 0; // IncrementalStMan scalar (the per-row-resolve case)
const ANTENNA1: usize = 1; // StandardStMan scalar
const UVW: usize = 2; // StandardStMan array
const FLAG: usize = 3; // TiledShapeStMan Bool array (bit-packed raw path)
const WEIGHT: usize = 4; // TiledShapeStMan Float array

fn desc() -> TableDesc {
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
            ),
            scalar(
                "ANTENNA1",
                DataType::Int,
                RecordValue::Int(0),
                "StandardStMan",
            ),
            array("UVW", DataType::Double, vec![3], "StandardStMan"),
            array("FLAG", DataType::Bool, vec![4, 16], "TiledShapeStMan"),
            array("WEIGHT", DataType::Float, vec![4], "TiledShapeStMan"),
        ],
    }
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = testdir::TestDir::new(format!(
        "casacure-reftable-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&*dir).unwrap();
    // by value: the guard is kept until the test's thread exits (testdir.rs)
    dir.into()
}

const NROWS: u64 = 16;

/// An `NROWS`-row table whose cells are all distinct per row, so a wrong row
/// mapping cannot go unnoticed.
fn build(tag: &str) -> std::path::PathBuf {
    let dir = temp_dir(tag);
    let mut wt = WritableTable::create(&dir, desc());
    wt.addrows(NROWS);
    for r in 0..NROWS {
        // TIME changes every row (worst case for IncrementalStMan's
        // interval index: a per-row resolve re-parses the bucket).
        wt.putcell(TIME, r, RecordValue::Double(1000.0 + r as f64))
            .unwrap();
        wt.putcell(ANTENNA1, r, RecordValue::Int((r % 4) as i32))
            .unwrap();
        wt.putcell(
            UVW,
            r,
            RecordValue::Array(ArrayValue {
                shape: vec![3],
                data: ArrayData::Double(vec![r as f64, -(r as f64), 0.5]),
            }),
        )
        .unwrap();
        wt.putcell(
            FLAG,
            r,
            RecordValue::Array(ArrayValue {
                shape: vec![16, 4],
                data: ArrayData::Bool((0..64).map(|i| (i as u64 + r).is_multiple_of(3)).collect()),
            }),
        )
        .unwrap();
        wt.putcell(
            WEIGHT,
            r,
            RecordValue::Array(ArrayValue {
                shape: vec![4],
                data: ArrayData::Float(vec![r as f32 * 0.25; 4]),
            }),
        )
        .unwrap();
    }
    wt.flush().unwrap();
    dir
}

/// The test row order: shuffled, with row 2 twice (a repeated source row is
/// legal — a view indexes the source, it is not a partition of it) and row 15
/// dropped (a selection).
const ORDER: [u64; 10] = [7, 2, 13, 0, 2, 9, 4, 11, 6, 1];

/// A reference table over a freshly opened source.
fn view_of(tag: &str) -> (std::path::PathBuf, Table) {
    let dir = build(tag);
    let source = Table::open(&dir, true).unwrap();
    let view = Table::row_order(Arc::new(source), ORDER.to_vec()).unwrap();
    (dir, view)
}

fn col(t: &Table, col: usize) -> Vec<RecordValue> {
    t.getcol(col, 0, t.nrows()).unwrap()
}

fn doubles(v: &[RecordValue]) -> Vec<f64> {
    v.iter()
        .map(|v| match v {
            RecordValue::Double(d) => *d,
            other => panic!("not a double: {other:?}"),
        })
        .collect()
}

#[test]
fn a_reference_table_is_a_read_only_snapshot_of_the_source() {
    let (dir, view) = view_of("shape");
    let source = Table::open(&dir, true).unwrap();

    assert!(view.is_view(), "row_order must produce a reference table");
    assert!(!source.is_view());
    assert_eq!(view.nrows(), ORDER.len() as u64);
    assert!(
        !view.is_writable(),
        "casacore's query/sort results are read-only"
    );
    assert_eq!(view.name(), source.name(), "name() is the source directory");
    assert!(
        view.lock_file().is_none(),
        "a reference table holds no lock of its own"
    );
    // The descriptor travels with the reference table: same columns, same
    // keywords — every non-row API reads the source's metadata.
    assert_eq!(view.colnames(), source.colnames());
    assert_eq!(view.dat.desc.columns.len(), source.dat.desc.columns.len());
    assert_eq!(
        view.raw_column_supported(WEIGHT),
        source.raw_column_supported(WEIGHT)
    );
    // Its own data-file sets stay empty: reads must map rows first.
    assert!(view.ssm_files.is_empty() && view.ism_files.is_empty() && view.tsm_files.is_empty());
    // Paging the mapped pages of a reference table is the source's business.
    view.drop_data_file_pages();
}

#[test]
fn every_read_maps_through_the_row_order() {
    let (dir, view) = view_of("reads");
    let source = Table::open(&dir, true).unwrap();

    for col_idx in 0..source.colnames().len() {
        let expected: Vec<RecordValue> = ORDER
            .iter()
            .map(|&r| source.getcell(col_idx, r).unwrap())
            .collect();

        // getcol over the whole view, and a sub-range of it.
        assert_eq!(col(&view, col_idx), expected, "getcol col {col_idx}");
        let part = view.getcol(col_idx, 2, 4).unwrap();
        assert_eq!(part, expected[2..6].to_vec(), "getcol range col {col_idx}");

        // getcell per view row.
        for (i, &src_row) in ORDER.iter().enumerate() {
            assert_eq!(
                view.getcell(col_idx, i as u64).unwrap(),
                source.getcell(col_idx, src_row).unwrap(),
                "getcell col {col_idx} row {i}"
            );
        }

        // getvarcol is getcol over every row.
        assert_eq!(view.getvarcol(col_idx).unwrap(), expected, "getvarcol");
    }

    // Array slicing: the slice of the mapped cell, not of some other row.
    let blc = [1];
    let trc = [2];
    let sliced = view
        .getcolslice(WEIGHT, &blc, &trc, 0, view.nrows())
        .unwrap();
    let expected: Vec<RecordValue> = ORDER
        .iter()
        .map(|&r| source.getcellslice(WEIGHT, r, &blc, &trc).unwrap())
        .collect();
    assert_eq!(sliced, expected);
    assert_eq!(
        view.getcellslice(UVW, 3, &[0], &[1]).unwrap(),
        source.getcellslice(UVW, ORDER[3], &[0], &[1]).unwrap()
    );
}

#[test]
fn out_of_range_rows_report_the_view_row() {
    let (_dir, view) = view_of("bounds");
    let err = view.getcell(TIME, ORDER.len() as u64).unwrap_err();
    let TableReadError::RowOutOfRange { row, .. } = err else {
        panic!("expected RowOutOfRange, got {err:?}");
    };
    assert_eq!(
        row,
        ORDER.len() as u64,
        "the view's own row, not the source's"
    );

    // A range past the end of the *view* is refused even though the source
    // could serve it.
    let err = view.getcol(TIME, ORDER.len() as u64 - 1, 4).unwrap_err();
    assert!(
        matches!(err, TableReadError::RowOutOfRange { .. }),
        "{err:?}"
    );
    // An empty range at the end is fine, as on a real table.
    assert!(view.getcol(TIME, ORDER.len() as u64, 0).unwrap().is_empty());
}

#[test]
fn raw_reads_visit_in_the_callers_row_order() {
    let (dir, view) = view_of("raw");
    let source = Table::open(&dir, true).unwrap();

    for col_idx in [TIME, ANTENNA1, UVW, WEIGHT] {
        // The visits carry the caller's row offset (a reference table reads
        // its source rows in ascending order, so they arrive scattered) and
        // each slot must be filled exactly once.
        let mut slots: Vec<Option<Vec<u8>>> = vec![None; ORDER.len()];
        view.getcol_raw(col_idx, 0, view.nrows(), |off, _shape, bytes| {
            assert!(off < slots.len(), "visit offset {off} outside the call");
            assert!(
                slots[off].replace(bytes.to_vec()).is_none(),
                "row {off} was visited twice"
            );
            Ok(())
        })
        .unwrap();
        let seen: Vec<Vec<u8>> = slots
            .into_iter()
            .map(|s| s.expect("visited once"))
            .collect();
        assert_eq!(seen.len(), ORDER.len());
        for (i, &src_row) in ORDER.iter().enumerate() {
            let mut at_source: Vec<(usize, Vec<u8>)> = Vec::new();
            source
                .getcol_raw(col_idx, src_row, 1, |off, _shape, bytes| {
                    at_source.push((off, bytes.to_vec()));
                    Ok(())
                })
                .unwrap();
            assert_eq!(seen[i], at_source[0].1, "raw col {col_idx} row {i}");
        }
    }

    // The tiled Bool column goes through the bit-packed raw path; its visits
    // carry the caller's row offset too (they arrive in source order).
    let mut slots: Vec<Option<(Vec<u8>, usize, usize)>> = vec![None; ORDER.len()];
    view.getcol_raw_bits(FLAG, 0, view.nrows(), |off, bytes, skip, nelem| {
        assert!(off < slots.len(), "visit offset {off} outside the call");
        assert!(
            slots[off].replace((bytes.to_vec(), skip, nelem)).is_none(),
            "row {off} was visited twice"
        );
        Ok(())
    })
    .unwrap();
    let decode = |v: &(Vec<u8>, usize, usize)| {
        let mut out = vec![false; v.2];
        casacure::tsm::decode_bits_into(&v.0, v.1, &mut out);
        out
    };
    for (i, &src_row) in ORDER.iter().enumerate() {
        let mut at_source = Vec::new();
        source
            .getcol_raw_bits(FLAG, src_row, 1, |_off, bytes, skip, nelem| {
                at_source.push((bytes.to_vec(), skip, nelem));
                Ok(())
            })
            .unwrap();
        let got = slots[i].take().expect("visited once");
        assert_eq!(decode(&got), decode(&at_source[0]), "FLAG row {i}");
    }
}

#[test]
fn chained_reference_tables_compose_their_row_orders() {
    let dir = build("chain");
    let source = Table::open(&dir, true).unwrap();
    let first = Table::row_order(Arc::new(source), ORDER.to_vec()).unwrap();
    assert!(first.is_view());

    // Re-order the *view*: row i of `second` is row `local[i]` of `first`.
    let local: Vec<u64> = (0..ORDER.len() as u64).rev().collect();
    let second = Table::row_order(Arc::new(first), local.clone()).unwrap();
    assert!(second.is_view());
    assert_eq!(second.nrows(), ORDER.len() as u64);
    assert_eq!(second.name(), Table::open(&dir, true).unwrap().name());

    let source = Table::open(&dir, true).unwrap();
    for (i, &l) in local.iter().enumerate() {
        let want = source.getcell(TIME, ORDER[l as usize]).unwrap();
        assert_eq!(second.getcell(TIME, i as u64).unwrap(), want, "row {i}");
    }
    // The composed reads still match a single gather of the source.
    let expected: Vec<RecordValue> = local
        .iter()
        .map(|&l| source.getcell(UVW, ORDER[l as usize]).unwrap())
        .collect();
    assert_eq!(col(&second, UVW), expected);
}

#[test]
fn a_reference_table_row_order_reflects_the_source_order_it_was_built_from() {
    // `execute_row_order` + `row_order` is the query()/sort() path: the rows
    // TaQL selects are the rows the reference table serves, in order.
    let dir = build("taqlpath");
    let source = Table::open(&dir, true).unwrap();
    let rows = execute_row_order(
        "SELECT * FROM $1 WHERE ANTENNA1 == 2 ORDERBY TIME",
        &[&source],
    )
    .unwrap()
    .expect("pure row selection");
    let want: Vec<u64> = (0..NROWS).filter(|r| r % 4 == 2).collect();
    assert_eq!(rows, want);

    let view = Table::row_order(Arc::new(source), rows.clone()).unwrap();
    assert_eq!(view.nrows(), rows.len() as u64);
    let times = doubles(&col(&view, TIME));
    assert!(times.windows(2).all(|w| w[0] <= w[1]), "ORDERBY applied");
    for (i, &src_row) in rows.iter().enumerate() {
        assert_eq!(
            view.getcell(TIME, i as u64).unwrap(),
            Table::open(&dir, true)
                .unwrap()
                .getcell(TIME, src_row)
                .unwrap()
        );
    }
}

#[test]
fn mutating_taql_refuses_a_reference_table() {
    let (_dir, view) = view_of("guards");
    let tables = [&view];
    for q in [
        "UPDATE $1 SET TIME = 0",
        "DELETE FROM $1 WHERE ANTENNA1 == 0",
        "ALTER TABLE $1 ADD COLUMN extra double",
    ] {
        let err = execute(q, &tables).expect_err("must refuse a reference table");
        let msg = err.to_string();
        assert!(
            msg.contains("reference table"),
            "{q} should name the reference table: {msg}"
        );
    }
    // The source itself is untouched.
    let dir = view.name().to_string();
    let source = Table::open(&dir, true).unwrap();
    assert_eq!(source.nrows(), NROWS);
    assert_eq!(doubles(&col(&source, TIME))[0], 1000.0);
}

#[test]
fn taql_selects_over_a_reference_table_answer_in_view_rows() {
    let (_dir, view) = view_of("taqlread");
    let out = match execute("SELECT * FROM $1 WHERE ANTENNA1 == 1", &[&view]).unwrap() {
        TaqlResult::Query(t) => t,
        other => panic!("expected a query result, got {other:?}"),
    };
    // The selection addresses the view: row ids are view rows, and the
    // values are the source rows the view maps to.
    let want: Vec<u64> = ORDER
        .iter()
        .enumerate()
        .filter(|(_, &r)| r % 4 == 1)
        .map(|(i, _)| i as u64)
        .collect();
    assert_eq!(out.nrows(), want.len());
    let times = doubles(out.getcol("TIME").unwrap());
    let expect: Vec<f64> = want
        .iter()
        .map(|&i| 1000.0 + ORDER[i as usize] as f64)
        .collect();
    assert_eq!(times, expect);
}
