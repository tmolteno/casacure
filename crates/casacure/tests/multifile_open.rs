//! The MultiFile container open error (issue #33): a table written with
//! casacore's `table.storage.option: multifile` keeps every data manager's
//! bytes inside one `table.mf` file (`table.mfh5` for the HDF5 flavour), so
//! the per-manager `table.f*` files this reader mmaps do not exist.  The
//! open used to surface a bare `table.f0: No such file or directory` os
//! error, which reads like corruption; it must name the container instead.
//!
//! These tests are hermetic: they shape a directory the way a multifile
//! table looks on disk (real `table.dat`, no `table.f*`, one container),
//! without needing python-casacore to create the fixture.

#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use casacure::record::{DataType, RecordValue, TableRecord};
use casacure::{create_table, ColumnDesc, ColumnKind, Table, TableDatError};
use testdir::TestDir;

fn empty_record() -> TableRecord {
    TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    }
}

fn one_scalar_column() -> casacure::TableDesc {
    casacure::TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: empty_record(),
        private_keywords: empty_record(),
        columns: vec![ColumnDesc {
            name: "VALUE".into(),
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
        }],
    }
}

/// A real casacure table directory (`table.dat` + `table.f0` + `table.f0i`).
fn real_table(tag: &str) -> TestDir {
    let dir = TestDir::new(format!("casacure-multifile-{tag}-{}", std::process::id()));
    let values: Vec<Vec<RecordValue>> = vec![vec![RecordValue::Double(1.0)]];
    create_table(&dir, &one_scalar_column(), &values).unwrap();
    dir
}

/// Shape `dir` the way casacore's multifile storage leaves it: the per-manager
/// data files are gone (their bytes live in the container) and the container
/// file sits beside `table.dat`.
fn make_multifile(dir: &std::path::Path, container: &str) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("table.f") {
            std::fs::remove_file(&path).unwrap();
        }
    }
    std::fs::write(dir.join(container), b"container stand-in").unwrap();
}

fn open_error(dir: &TestDir) -> String {
    match Table::open(dir, true) {
        Ok(_) => panic!("opening a MultiFile table must fail"),
        Err(e) => {
            let e: TableDatError = e;
            e.to_string()
        }
    }
}

#[test]
fn a_multifile_table_names_the_container_not_a_missing_data_file() {
    // The issue #33 shape: table.dat and table.mf exist, table.f0 does not.
    // Before the fix this surfaced `.../table.f0: No such file or directory`,
    // which reads like file corruption rather than an unsupported container.
    let dir = real_table("mf");
    make_multifile(&dir, "table.mf");

    let msg = open_error(&dir);
    assert!(msg.contains("MultiFile"), "{msg}");
    assert!(msg.contains("table.mf"), "{msg}");
    assert!(
        !msg.contains("No such file"),
        "the bare missing-file os error must not surface: {msg}"
    );
}

#[test]
fn the_hdf5_flavour_of_the_container_is_named_too() {
    // Same container, HDF5 flavour: casacore writes `table.mfh5` with
    // `table.storage.option: multifilehdf5`. The message must name both the
    // container kind and the file.
    let dir = real_table("mfh5");
    make_multifile(&dir, "table.mfh5");

    let msg = open_error(&dir);
    assert!(msg.contains("MultiFile"), "{msg}");
    assert!(msg.contains("table.mfh5"), "{msg}");
    assert!(
        !msg.contains("No such file"),
        "the bare missing-file os error must not surface: {msg}"
    );
}

#[test]
fn without_a_container_a_missing_data_file_is_still_reported_as_such() {
    // The container check must not hijack unrelated failures: the same
    // directory minus the container is a plain missing-data-file error, and
    // must not claim MultiFile.
    let dir = real_table("bare");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("table.f") {
            std::fs::remove_file(&path).unwrap();
        }
    }

    let msg = open_error(&dir);
    // Exactly what issue #33 was filed against, kept for a table that is
    // not multifile: the per-manager data file is missing, by name.
    assert!(msg.contains("table.f0"), "{msg}");
    assert!(msg.contains("No such file"), "{msg}");
    assert!(
        !msg.contains("MultiFile"),
        "no container is present, so MultiFile must not be named: {msg}"
    );
}

#[test]
fn a_missing_directory_is_not_reported_as_multifile() {
    // Nothing exists at all: still not a MultiFile error.
    let dir = TestDir::new(format!("casacure-multifile-absent-{}", std::process::id()));
    let msg = open_error(&dir);
    assert!(msg.contains("No such file"), "{msg}");
    assert!(
        !msg.contains("MultiFile"),
        "an absent directory must not be named a MultiFile table: {msg}"
    );
}
