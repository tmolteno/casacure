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
    /// Optional per-column cell values (long-string / ISM tables).
    #[serde(default)]
    values: BTreeMap<String, Vec<serde_json::Value>>,
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

/// Ground-truth column offsets inside `typed.tab/table.f0` read off the real
/// casacore-written SSM spec (one per column, including the value sizes).
const TYPED_COLUMN_OFFSETS: [u32; 10] = [0, 4, 36, 100, 228, 356, 484, 740, 996, 1508];

/// The sample values written by `tests/make_fixtures.py` (`COLUMN_CASES`).
fn typed_values() -> [(&'static str, casacure::record::RecordValue); 10] {
    [
        ("COL_B", casacure::record::RecordValue::Bool(true)),
        ("COL_U1", casacure::record::RecordValue::UChar(7)),
        ("COL_I2", casacure::record::RecordValue::Short(-300)),
        ("COL_I4", casacure::record::RecordValue::Int(-70000)),
        ("COL_U4", casacure::record::RecordValue::UInt(4_000_000_000)),
        ("COL_R4", casacure::record::RecordValue::Float(1.5)),
        ("COL_R8", casacure::record::RecordValue::Double(1.5e300)),
        ("COL_C4", casacure::record::RecordValue::Complex(1.5, 2.5)),
        (
            "COL_C8",
            casacure::record::RecordValue::DComplex(1.5e300, 2.5e300),
        ),
        (
            "COL_S",
            casacure::record::RecordValue::String("hello".into()),
        ),
    ]
}

/// The exact decoded values read back out of the real casacore-written
/// StandardStMan data file `typed.tab/table.f0` (little-endian host).
#[test]
fn fixture_standard_stman_column_values_read() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    for (name, table) in &manifest.tables {
        // Pins the scalar `typed.tab` values; the array table is covered by
        // `fixture_array_column_read`.
        if name != "typed" {
            continue;
        }
        let buf = std::fs::read(fixtures_dir.join(&table.path).join("table.dat"))
            .expect("cannot read table.dat");
        let dat = casacure::parse_table_dat(&buf)
            .unwrap_or_else(|e| panic!("{name}: table.dat failed to parse: {e}"));

        let file = casacure::StandardStManFile::open(
            fixtures_dir.join(&table.path),
            0,
            dat.header.big_endian,
        )
        .unwrap_or_else(|e| panic!("{name}: data file failed to open: {e}"));
        assert_eq!(
            file.header.big_endian, table.big_endian,
            "{name}: data-file endianness"
        );
        // The fixture table has one scalar row; 10 columns in one index.
        assert_eq!(file.header.nr_index, 1, "{name}: expect one SSMIndex");
        assert_eq!(file.indices[0].last_row, vec![0], "{name}: one bucket");
        assert_eq!(file.indices[0].rows_per_bucket, 32, "{name}: rows/bucket");
        assert_eq!(file.indices[0].nr_columns, 10, "{name}: columns/index");

        let dm = &dat.column_set.data_managers[0];
        let spec = match &dm.blob {
            casacure::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("{name}: expected StandardStMan spec"),
        };
        for (col_idx, (col_name, expected)) in typed_values().iter().enumerate() {
            let desc = dat
                .desc
                .column(col_name)
                .unwrap_or_else(|| panic!("{name}.{col_name}: missing descriptor"));
            let value = file
                .read_scalar_cell(spec, col_idx, desc, 0)
                .unwrap_or_else(|e| panic!("{name}.{col_name}: read failed: {e}"));
            assert_eq!(
                value, *expected,
                "{name}.{col_name}: casacore-written value mismatch"
            );
        }
    }
}

#[test]
fn fixture_column_sets_parse() {
    let Some(manifest) = load_manifest() else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    for (name, table) in &manifest.tables {
        if name != "typed" {
            continue; // offset table / index-map constants are typed-specific
        }
        let buf = std::fs::read(fixtures_dir.join(&table.path).join("table.dat"))
            .expect("cannot read table.dat");
        let dat = casacure::parse_table_dat(&buf)
            .unwrap_or_else(|e| panic!("{name}: table.dat failed to parse: {e}"));

        let cs = &dat.column_set;
        assert_eq!(cs.version, 2, "{name}: unexpected ColumnSet version");
        assert_eq!(cs.nrow, table.nrows, "{name}: ColumnSet row count");
        assert_eq!(cs.storage_option, None, "{name}: v2 has no storage option");
        // One StandardStMan data manager owning every column.
        assert_eq!(cs.seq_count, 1, "{name}: wrong data-manager count");
        assert_eq!(cs.data_managers.len(), 1, "{name}: wrong DM list");
        let dm = &cs.data_managers[0];
        assert_eq!(dm.type_name, "StandardStMan", "{name}: wrong DM type");
        assert_eq!(dm.sequence_nr, 0, "{name}: wrong DM sequence");

        // Every column binds to data manager 0.
        assert_eq!(cs.columns.len(), table.columns.len());
        for info in &cs.columns {
            assert_eq!(info.data_manager_seq, 0, "{name}: wrong DM binding");
            assert_eq!(info.shape, None, "{name}: scalar columns only");
        }

        let ssm = match &dm.blob {
            casacure::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("{name}: expected a StandardStMan spec blob"),
        };
        assert_eq!(ssm.data_manager_name, "StandardStMan");
        assert_eq!(
            ssm.column_offset, TYPED_COLUMN_OFFSETS,
            "{name}: wrong column offset table"
        );
        assert_eq!(
            ssm.col_index_map,
            vec![0u32; TYPED_COLUMN_OFFSETS.len()],
            "{name}: wrong column index map"
        );
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
            assert!(
                matches!(
                    desc.data_manager_type.as_str(),
                    "StandardStMan" | "IncrementalStMan" | "TiledColumnStMan"
                ),
                "{name}.{col_name}: unsupported data manager {}",
                desc.data_manager_type
            );
            match &desc.kind {
                casacure::ColumnKind::Scalar(_) => {}
                casacure::ColumnKind::Array => {
                    // Fixed-shape array: descriptor carries the CASA-order
                    // (reversed logical) shape and the FixedShape option.
                    assert!(
                        desc.shape.is_some(),
                        "{name}.{col_name}: array column has no shape"
                    );
                    assert_eq!(desc.ndim, 2, "{name}.{col_name}: wrong ndim");
                    assert_eq!(desc.options & 4, 4, "{name}.{col_name}: not fixed shape");
                    if col_name == "ARR" {
                        assert_eq!(
                            desc.shape
                                .as_deref()
                                .map(|s| s.iter().map(|&d| d as i32).collect::<Vec<_>>()),
                            Some(vec![3, 2]),
                            "{name}.{col_name}: wrong CASA-order shape"
                        );
                    }
                }
                casacure::ColumnKind::Record => {
                    panic!("{name}.{col_name}: unexpected record column")
                }
            }
        }
    }
}

/// Reads the real casacore-written `array.tab`: the fixed-shape (2x3
/// complex) `ARR` column via its `table.f0i` array index file, and the
/// scalar `IDX` column from the data buckets.
#[test]
fn fixture_array_column_read() {
    use casacure::record::{ArrayData, RecordValue};
    let Some(manifest) = load_manifest() else {
        return;
    };
    let Some(array) = manifest.tables.get("array") else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    let dir = fixtures_dir.join(&array.path);
    let dat_bytes = std::fs::read(dir.join("table.dat")).expect("cannot read table.dat");
    let dat = casacure::parse_table_dat(&dat_bytes)
        .unwrap_or_else(|e| panic!("array: table.dat failed to parse: {e}"));

    let file = casacure::StandardStManFile::open(&dir, 0, dat.header.big_endian)
        .unwrap_or_else(|e| panic!("array: data file failed to open: {e}"));
    // ARR uses the array index file; IDX is in the buckets.
    assert!(file.f0i().is_some(), "array.tab should have a table.f0i");
    assert_eq!(
        file.indices[0].last_row,
        vec![1],
        "array: one bucket, 2 rows"
    );

    let dm = &dat.column_set.data_managers[0];
    let spec = match &dm.blob {
        casacure::DataManagerBlob::StandardStMan(s) => s,
        _ => panic!("array: expected StandardStMan spec"),
    };
    // The ARR binding carries the fixed CASA-order shape.
    assert_eq!(
        dat.column_set.columns[0].shape.as_deref(),
        Some(&[3i64, 2][..]),
        "array: ARR binding shape"
    );
    assert_eq!(spec.column_offset, vec![0, 256], "array: column offsets");

    for (row, base) in [(0u64, 0i32), (1, 10)] {
        let cell = casacure::read_array_cell(&file, spec, 0, &dat.desc.columns[0], row)
            .unwrap_or_else(|e| panic!("array: ARR row {row} read failed: {e}"));
        match cell {
            RecordValue::Array(arr) => {
                assert_eq!(arr.shape, vec![2, 3], "array: logical shape row {row}");
                match &arr.data {
                    ArrayData::Complex(vals) => {
                        let expect: Vec<(f32, f32)> =
                            (1..=6).map(|k| (base as f32, k as f32)).collect();
                        assert_eq!(&vals[..], &expect[..], "array: values row {row}");
                    }
                    other => panic!("array: expected complex data, got {other:?}"),
                }
            }
            other => panic!("array: expected array value, got {other:?}"),
        }
    }
    // Scalar IDX column reads from the buckets as usual.
    let idx = dat.desc.column("IDX").unwrap();
    assert_eq!(
        file.read_scalar_cell(spec, 1, idx, 0).unwrap(),
        RecordValue::Int(0)
    );
    assert_eq!(
        file.read_scalar_cell(spec, 1, idx, 1).unwrap(),
        RecordValue::Int(1)
    );
}

/// Reads the real casacore-written `longstr.tab`: variable strings longer
/// than 8 chars come back through the SSM string buckets (`table.f0`).
#[test]
fn fixture_long_strings_read() {
    use casacure::record::RecordValue;
    let Some(manifest) = load_manifest() else {
        return;
    };
    let Some(t) = manifest.tables.get("longstr") else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    let dir = fixtures_dir.join(&t.path);
    let dat_bytes = std::fs::read(dir.join("table.dat")).expect("cannot read table.dat");
    let dat = casacure::parse_table_dat(&dat_bytes)
        .unwrap_or_else(|e| panic!("longstr: table.dat failed to parse: {e}"));
    let file = casacure::StandardStManFile::open(&dir, 0, dat.header.big_endian)
        .unwrap_or_else(|e| panic!("longstr: data file failed to open: {e}"));
    // Data + index + one string bucket.
    assert_eq!(file.header.nr_buckets, 3, "longstr: three buckets");
    assert_eq!(file.header.last_string_bucket, 2, "longstr: string bucket");

    let dm = &dat.column_set.data_managers[0];
    let spec = match &dm.blob {
        casacure::DataManagerBlob::StandardStMan(s) => s,
        _ => panic!("longstr: expected StandardStMan spec"),
    };
    let txt = dat.desc.column("TXT").unwrap();
    let txt_values = t.values.get("TXT").cloned().unwrap_or_default();
    for (row, expected) in txt_values.iter().enumerate() {
        let got = file
            .read_scalar_cell(spec, 0, txt, row as u64)
            .unwrap_or_else(|e| panic!("longstr TXT row {row}: {e}"));
        assert_eq!(
            got,
            RecordValue::String(expected.as_str().unwrap().to_string()),
            "longstr TXT row {row}"
        );
    }
    let idx = dat.desc.column("IDX").unwrap();
    assert_eq!(
        file.read_scalar_cell(spec, 1, idx, 1).unwrap(),
        RecordValue::Int(1)
    );
}
/// Reads the real casacore-written `ism.tab`: IncrementalStMan (Direct)
/// TIME/ANT1 columns come back through the ISM interval index; VAL comes
/// from the StandardStMan file (separate data manager, table.f1).
#[test]
fn fixture_ism_read() {
    use casacure::record::RecordValue;
    let Some(manifest) = load_manifest() else {
        return;
    };
    let Some(t) = manifest.tables.get("ism") else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    let dir = fixtures_dir.join(&t.path);
    let dat_bytes = std::fs::read(dir.join("table.dat")).expect("cannot read table.dat");
    let dat = casacure::parse_table_dat(&dat_bytes)
        .unwrap_or_else(|e| panic!("ism: table.dat failed to parse: {e}"));

    let ism = casacure::IsmFile::open(&dir, 0, dat.header.big_endian)
        .unwrap_or_else(|e| panic!("ism: ISM file failed to open: {e}"));
    assert_eq!(
        ism.index.rows,
        vec![0, 6],
        "ism: one bucket covering 6 rows"
    );
    assert_eq!(ism.index.bucket_numbers, vec![0]);

    for (row, expected) in t.values["TIME"].iter().enumerate() {
        let got = ism
            .read_scalar_cell(0, dat.desc.column("TIME").unwrap(), row as u64)
            .unwrap_or_else(|e| panic!("ism TIME row {row}: {e}"));
        assert_eq!(
            got,
            RecordValue::Double(expected.as_f64().unwrap()),
            "ism TIME row {row}"
        );
    }
    let ant1_expected: Vec<i64> = t.values["ANT1"]
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    for (row, expected) in ant1_expected.iter().enumerate() {
        let got = ism
            .read_scalar_cell(1, dat.desc.column("ANT1").unwrap(), row as u64)
            .unwrap_or_else(|e| panic!("ism ANT1 row {row}: {e}"));
        assert_eq!(
            got,
            RecordValue::Int(*expected as i32),
            "ism ANT1 row {row}"
        );
    }

    // VAL lives in the StandardStMan data manager (table.f1, seq 1).
    let ssm = casacure::StandardStManFile::open(&dir, 1, dat.header.big_endian)
        .unwrap_or_else(|e| panic!("ism: SSM file failed to open: {e}"));
    let dm = dat
        .column_set
        .data_managers
        .iter()
        .find(|dm| dm.sequence_nr == 1)
        .unwrap();
    let spec = match &dm.blob {
        casacure::DataManagerBlob::StandardStMan(s) => s,
        _ => panic!("ism: expected StandardStMan spec"),
    };
    for row in 0..t.nrows {
        let got = ssm
            .read_scalar_cell(spec, 0, dat.desc.column("VAL").unwrap(), row)
            .unwrap_or_else(|e| panic!("ism VAL row {row}: {e}"));
        assert_eq!(
            got,
            RecordValue::Double(t.values["VAL"][row as usize].as_f64().unwrap()),
            "ism VAL row {row}"
        );
    }
}
/// Reads the real casacore-written `tsm.tab`: the fixed-shape (2x3 dcomplex)
/// DATA column comes back through the TiledColumnStMan tile file
/// (`table.f0_TSM0`).
#[test]
fn fixture_tsm_read() {
    use casacure::record::{ArrayData, RecordValue};
    let Some(manifest) = load_manifest() else {
        return;
    };
    let Some(t) = manifest.tables.get("tsm") else {
        return;
    };
    let fixtures_dir = manifest_path().parent().unwrap().to_path_buf();
    let dir = fixtures_dir.join(&t.path);
    let dat_bytes = std::fs::read(dir.join("table.dat")).expect("cannot read table.dat");
    let dat = casacure::parse_table_dat(&dat_bytes)
        .unwrap_or_else(|e| panic!("tsm: table.dat failed to parse: {e}"));

    let tsm = casacure::TsmFile::open(&dir, 0, dat.header.big_endian)
        .unwrap_or_else(|e| panic!("tsm: data file failed to open: {e}"));
    // TiledColumnStMan hypercolumn: cell [3,2] (CASA order) + rows.
    assert_eq!(tsm.header.hypercolumn_name, "TiledData_GROUP");
    assert_eq!(tsm.header.nrrow, 3, "tsm: row count");
    let cube = &tsm.header.cubes[0];
    assert_eq!(
        cube.cube_shape,
        vec![3, 2, 3],
        "tsm: cube shape (cell + rows)"
    );
    assert_eq!(cube.tile_shape[..2], [3, 2], "tsm: per-cell tile shape");
    assert_eq!(
        tsm.header.data_types,
        vec![casacure::record::DataType::DComplex]
    );

    let data_desc = dat.desc.column("DATA").unwrap();
    for row in 0..3u64 {
        let cell = tsm
            .read_cell(data_desc, row)
            .unwrap_or_else(|e| panic!("tsm DATA row {row}: {e}"));
        match cell {
            RecordValue::Array(a) => {
                assert_eq!(a.shape, vec![2, 3], "tsm: logical shape row {row}");
                match &a.data {
                    ArrayData::DComplex(vals) => {
                        let expect: Vec<(f64, f64)> =
                            (1..=6).map(|k| (row as f64, k as f64)).collect();
                        assert_eq!(&vals[..], &expect[..], "tsm: values row {row}");
                    }
                    other => panic!("tsm: expected dcomplex, got {other:?}"),
                }
            }
            other => panic!("tsm: expected array value, got {other:?}"),
        }
    }
    // IDX lives in the StandardStMan DM (table.f1).
    let ssm = casacure::StandardStManFile::open(&dir, 1, dat.header.big_endian).unwrap();
    let dm1 = &dat.column_set.data_managers[1];
    let spec = match &dm1.blob {
        casacure::DataManagerBlob::StandardStMan(s) => s,
        _ => panic!("tsm: expected StandardStMan spec"),
    };
    for row in 0..3u64 {
        assert_eq!(
            ssm.read_scalar_cell(spec, 0, dat.desc.column("IDX").unwrap(), row)
                .unwrap(),
            RecordValue::Int(row as i32)
        );
    }
}
