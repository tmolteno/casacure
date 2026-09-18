//! Manifest-driven comparison tests against real casacore-written tables.
//!
//! `tests/make_fixtures.py` (run with python-casacore installed) writes
//! `tests/fixtures/manifest.json` describing the tables casacore created and
//! the dtypes its `getcol` returned. These tests pin casacure's type system
//! to that observed behaviour without needing the pyo3 bindings.
//!
//! Run `cargo test -p casacure --test compat_fixtures` after regenerating
//! fixtures. The tests are skipped (with a message) when the manifest is
//! absent, e.g. on a fresh clone.

use std::collections::BTreeMap;
use std::path::PathBuf;

use casacure::ValueType;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Manifest {
    casacore_version: String,
    tables: BTreeMap<String, TableFixture>,
}

#[derive(Debug, Deserialize)]
struct TableFixture {
    path: String,
    nrows: u64,
    big_endian: bool,
    columns: BTreeMap<String, ColumnFixture>,
}

#[derive(Debug, Deserialize)]
struct ColumnFixture {
    value_type: String,
    /// numpy dtype `.str` as returned by casacore's `getcol`, or `"list"`
    /// for 1-D string columns.
    getcol_dtype: String,
}

fn manifest_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/casacure; fixtures live at the repo root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/manifest.json")
}

fn load_manifest() -> Option<Manifest> {
    let path = manifest_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!(
                "skipping compat fixtures: cannot read {} ({e}); \
                 run `.venv/bin/python tests/make_fixtures.py`",
                path.display()
            );
            return None;
        }
    };
    Some(serde_json::from_str(&text).expect("manifest.json is not valid"))
}

/// numpy dtype `.str` that casacore's `getcol` produces for `vt`, including
/// the documented quirk that `uchar` columns come back as `uint16`
/// (`tests/test_types_compat.py`). 1-D string columns arrive as Python lists.
fn expected_getcol_dtype(vt: ValueType) -> &'static str {
    match vt {
        ValueType::Bool => "|b1",
        ValueType::Byte => "<u2", // casacore promotes uchar to uint16
        ValueType::Short => "<i2",
        ValueType::UShort => "<u2",
        ValueType::Int => "<i4",
        ValueType::UInt => "<u4",
        ValueType::Float => "<f4",
        ValueType::Double => "<f8",
        ValueType::Complex => "<c8",
        ValueType::DComplex => "<c16",
        ValueType::String => "list",
    }
}

#[test]
fn fixture_value_types_parse_and_match_getcol_dtypes() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    assert!(!manifest.tables.is_empty(), "manifest lists no tables");

    for (table_name, table) in &manifest.tables {
        assert!(!table.columns.is_empty(), "{table_name}: no columns");
        for (col_name, col) in &table.columns {
            let vt = ValueType::from_casa_name(&col.value_type)
                .unwrap_or_else(|e| panic!("{table_name}.{col_name}: unparseable value_type: {e}"));
            assert_eq!(
                expected_getcol_dtype(vt),
                col.getcol_dtype,
                "{table_name}.{col_name}: casacore getcol dtype mismatch \
                 (value_type {:?} -> {})",
                col.value_type,
                vt.numpy_name(),
            );
        }
    }
}

#[test]
fn fixture_tables_exist_on_disk() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    for (name, table) in &manifest.tables {
        let table_dat = fixtures_dir.join(&table.path).join("table.dat");
        assert!(
            table_dat.is_file(),
            "fixture table {name}: {} missing",
            table_dat.display()
        );
    }
}

#[test]
fn manifest_reports_casacore_version() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    assert!(
        !manifest.casacore_version.is_empty(),
        "manifest has no casacore_version"
    );
}

#[test]
fn fixture_table_dat_headers_parse() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    for (name, table) in &manifest.tables {
        let buf = std::fs::read(fixtures_dir.join(&table.path).join("table.dat"))
            .expect("cannot read table.dat");
        let hdr = casacure::parse_table_header(&buf)
            .unwrap_or_else(|e| panic!("{name}: table.dat header failed to parse: {e}"));
        assert_eq!(hdr.version, 2, "{name}: unexpected Table version");
        assert_eq!(hdr.nrow, table.nrows, "{name}: wrong row count");
        assert_eq!(
            hdr.big_endian, table.big_endian,
            "{name}: wrong endianness flag"
        );
        assert_eq!(hdr.kind, "PlainTable", "{name}: wrong table kind");
    }
}

#[test]
fn fixture_table_descs_parse() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    for (name, table) in &manifest.tables {
        let buf = std::fs::read(fixtures_dir.join(&table.path).join("table.dat"))
            .expect("cannot read table.dat");
        let dat = casacure::parse_table_dat(&buf)
            .unwrap_or_else(|e| panic!("{name}: table.dat failed to parse: {e}"));
        assert_eq!(dat.header.nrow, table.nrows, "{name}: wrong row count");
        assert_eq!(
            dat.desc.columns.len(),
            table.columns.len(),
            "{name}: wrong column count"
        );
        for (col_name, col) in &table.columns {
            let desc = dat
                .desc
                .column(col_name)
                .unwrap_or_else(|| panic!("{name}.{col_name}: column not in descriptor"));
            let expected = ValueType::from_casa_name(&col.value_type).unwrap();
            assert_eq!(
                desc.value_type(),
                Some(expected),
                "{name}.{col_name}: wrong value type"
            );
            assert_eq!(
                desc.data_manager_type, "StandardStMan",
                "{name}.{col_name}: wrong data manager"
            );
            assert!(
                matches!(desc.kind, casacure::ColumnKind::Scalar(_)),
                "{name}.{col_name}: expected scalar column"
            );
        }
    }
}
