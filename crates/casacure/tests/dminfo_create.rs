//! `dminfo` on the creation paths (issue #20): records land columns in the
//! requested storage managers with the requested tile shapes — at creation,
//! across row growth, through a cell-splitting tile (dask-ms asks for
//! those), and through the deep-copy conversion — and they fail loudly on a
//! manager or SPEC field creation cannot honour.

#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use casacure::record::{
    parse_json_record, ArrayData, ArrayValue, DataType, RecordValue, TableRecord,
};
use casacure::tabledesc::{apply_dminfo, DmInfoError};
use casacure::{
    apply_dminfo_in_place, create_table, get_dminfo, ColumnDesc, ColumnKind, Table, TableDesc,
    WritableTable,
};

fn empty_record() -> TableRecord {
    TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    }
}

/// A fixed-shape array column with CASA (stored) shape `casa_shape`.
fn array_col(
    name: &str,
    dt: DataType,
    casa_shape: Vec<i64>,
    group: &str,
    dm_type: &str,
) -> ColumnDesc {
    ColumnDesc {
        name: name.into(),
        comment: String::new(),
        data_type: dt,
        data_manager_type: dm_type.into(),
        data_manager_group: group.into(),
        options: 4, // FixedShape
        ndim: casa_shape.len() as i32,
        shape: Some(casa_shape),
        max_length: 0,
        keywords: empty_record(),
        kind: ColumnKind::Array,
        tile_shape: None,
    }
}

fn desc_of(columns: Vec<ColumnDesc>) -> TableDesc {
    TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: empty_record(),
        private_keywords: empty_record(),
        columns,
    }
}

fn apply(dm_json: &str, desc: &mut TableDesc) -> Result<(), DmInfoError> {
    let rec = parse_json_record(dm_json).unwrap();
    apply_dminfo(&rec, &mut desc.columns)
}

/// Cell values with the LOGICAL shape (the CASA shape reversed), as the
/// reader returns them and the writers expect.
fn logical(casa: &[i64]) -> Vec<u32> {
    casa.iter().rev().map(|&d| d as u32).collect()
}

fn bools(rows: u64, casa_cell: &[i64]) -> Vec<RecordValue> {
    let n: usize = casa_cell.iter().product::<i64>() as usize;
    (0..rows)
        .map(|r| {
            let v: Vec<bool> = (0..n).map(|i| (i + r as usize).is_multiple_of(3)).collect();
            RecordValue::Array(ArrayValue {
                shape: logical(casa_cell),
                data: ArrayData::Bool(v),
            })
        })
        .collect()
}

fn complexes(rows: u64, casa_cell: &[i64]) -> Vec<RecordValue> {
    let n: usize = casa_cell.iter().product::<i64>() as usize;
    (0..rows)
        .map(|r| {
            let v: Vec<(f32, f32)> = (0..n).map(|i| (i as f32 + r as f32, -(i as f32))).collect();
            RecordValue::Array(ArrayValue {
                shape: logical(casa_cell),
                data: ArrayData::Complex(v),
            })
        })
        .collect()
}

fn values_of(t: &Table, name: &str) -> Vec<RecordValue> {
    let idx = t
        .dat
        .desc
        .columns
        .iter()
        .position(|c| c.name == name)
        .unwrap();
    let n = t.nrows();
    t.getcol(idx, 0, n).unwrap()
}

fn assert_array_eq(name: &str, got: &[RecordValue], want: &[RecordValue]) {
    assert_eq!(got.len(), want.len(), "{name}: row count");
    for (r, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        match (g, w) {
            (RecordValue::Array(a), RecordValue::Array(b)) => {
                assert_eq!(a.shape, b.shape, "{name}[{r}] shape");
                match (&a.data, &b.data) {
                    (ArrayData::Bool(x), ArrayData::Bool(y)) => assert_eq!(x, y, "{name}[{r}]"),
                    (ArrayData::Complex(x), ArrayData::Complex(y)) => {
                        assert_eq!(x, y, "{name}[{r}]")
                    }
                    other => panic!("{name}[{r}]: unexpected data {other:?}"),
                }
            }
            other => panic!("{name}[{r}]: unexpected value {other:?}"),
        }
    }
}

// ---------------------------------------------------------------- apply

#[test]
fn apply_dminfo_flat_record_covers_every_column() {
    let mut desc = desc_of(vec![
        array_col(
            "DATA",
            DataType::Complex,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
        array_col(
            "FLAG",
            DataType::Bool,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
    ]);
    apply(
        r#"{"TYPE": "TiledColumnStMan", "NAME": "TiledData",
            "SPEC": {"DEFAULTTILESHAPE": [6, 4, 5]}}"#,
        &mut desc,
    )
    .unwrap();
    for cd in &desc.columns {
        assert_eq!(cd.data_manager_type, "TiledColumnStMan");
        assert_eq!(cd.data_manager_group, "TiledData");
        // DEFAULTTILESHAPE [6,4,5] in logical order is [4,6,5] in CASA order
        assert_eq!(cd.tile_shape.as_deref(), Some(&[4, 6, 5][..]));
    }
}

#[test]
fn apply_dminfo_mapping_targets_named_columns() {
    let mut desc = desc_of(vec![
        array_col(
            "DATA",
            DataType::Complex,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
        array_col(
            "FLAG",
            DataType::Bool,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
    ]);
    apply(
        r#"{"*1": {"TYPE": "TiledColumnStMan", "NAME": "tiled",
                   "SPEC": {"DEFAULTTILESHAPE": [6, 4, 5]}, "COLUMNS": ["DATA"]},
            "*2": {"TYPE": "IncrementalStMan", "COLUMNS": ["FLAG"]}}"#,
        &mut desc,
    )
    .unwrap();
    assert_eq!(desc.columns[0].data_manager_type, "TiledColumnStMan");
    assert_eq!(desc.columns[0].data_manager_group, "tiled");
    // DEFAULTTILESHAPE [6,4,5] in logical order is [4,6,5] in CASA order
    assert_eq!(desc.columns[0].tile_shape.as_deref(), Some(&[4, 6, 5][..]));
    assert_eq!(desc.columns[1].data_manager_type, "IncrementalStMan");
    // Its group falls back to the record's key (the manager's dminfo name).
    assert_eq!(desc.columns[1].data_manager_group, "*2");
    assert_eq!(desc.columns[1].tile_shape, None);
}

#[test]
fn apply_dminfo_group_entry_refines_declared_group() {
    // python-casacore's `makedminfo(tabdesc, dmgroup_spec)` input shape: a
    // record keyed by group name with neither TYPE nor COLUMNS refines the
    // columns that already declare the group.
    let mut desc = desc_of(vec![
        array_col("UVW", DataType::Double, vec![3], "UVW", "TiledColumnStMan"),
        array_col(
            "MODEL_DATA",
            DataType::Complex,
            vec![4, 16],
            "DataGroup",
            "TiledColumnStMan",
        ),
        array_col(
            "FLAG",
            DataType::Bool,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
    ]);
    apply(
        r#"{"UVW": {"DEFAULTTILESHAPE": [3, 8192]},
            "DataGroup": {"DEFAULTTILESHAPE": [16, 4, 32]}}"#,
        &mut desc,
    )
    .unwrap();
    // UVW is 1D shape [3], so logical [3, 8192] (rows) is already CASA [3, 8192]
    assert_eq!(desc.columns[0].tile_shape.as_deref(), Some(&[3, 8192][..]));
    assert_eq!(desc.columns[0].data_manager_group, "UVW");
    // MODEL_DATA shape [4, 16] in CASA becomes [16, 4] in logical,
    // so DEFAULTTILESHAPE [16, 4, 32] logical becomes [4, 16, 32] CASA
    assert_eq!(
        desc.columns[1].tile_shape.as_deref(),
        Some(&[4, 16, 32][..])
    );
    // The untargeted column keeps its manager and gains no tile shape.
    assert_eq!(desc.columns[2].tile_shape, None);
    assert_eq!(desc.columns[2].data_manager_group, "StandardStMan");
}

#[test]
fn apply_dminfo_fails_loudly() {
    let mut desc = desc_of(vec![array_col(
        "DATA",
        DataType::Complex,
        vec![4, 6],
        "StandardStMan",
        "StandardStMan",
    )]);
    // An unsupported manager.
    let e = apply(r#"{"TYPE": "TiledDataStMan", "NAME": "t"}"#, &mut desc).unwrap_err();
    assert!(matches!(e, DmInfoError::UnsupportedType { .. }), "{e}");
    // A SPEC field creation cannot honour.
    let e = apply(
        r#"{"TYPE": "TiledShapeStMan", "NAME": "t",
            "SPEC": {"HYPERCUBES": {"*1": {}}}}"#,
        &mut desc,
    )
    .unwrap_err();
    assert!(
        matches!(&e, DmInfoError::BadSpecField { field, .. } if field == "HYPERCUBES"),
        "{e}"
    );
    // A tile shape on a non-tiled manager.
    let e = apply(
        r#"{"TYPE": "StandardStMan", "NAME": "s",
            "SPEC": {"DEFAULTTILESHAPE": [4, 6, 5]}}"#,
        &mut desc,
    )
    .unwrap_err();
    assert!(
        matches!(e, DmInfoError::BadSpecField { ref field, .. } if field == "DEFAULTTILESHAPE"),
        "{e}"
    );
    // A malformed tile shape.
    let e = apply(
        r#"{"TYPE": "TiledShapeStMan", "NAME": "t",
            "SPEC": {"DEFAULTTILESHAPE": [0, 4]}}"#,
        &mut desc,
    )
    .unwrap_err();
    assert!(matches!(e, DmInfoError::BadTileShape { .. }), "{e}");
    // A column the description does not have.
    let e = apply(
        r#"{"*1": {"TYPE": "TiledShapeStMan", "COLUMNS": ["NOSUCH"]}}"#,
        &mut desc,
    )
    .unwrap_err();
    assert!(matches!(e, DmInfoError::UnknownColumn { .. }), "{e}");
    // Cache-sizing SPEC fields are accepted and ignored.
    apply(
        r#"{"TYPE": "TiledShapeStMan", "NAME": "t",
            "SPEC": {"MaxCacheSize": 0, "DEFAULTTILESHAPE": [6, 4, 5]}}"#,
        &mut desc,
    )
    .unwrap();
    // DEFAULTTILESHAPE [6,4,5] in logical order is [4,6,5] in CASA order
    assert_eq!(desc.columns[0].tile_shape.as_deref(), Some(&[4, 6, 5][..]));
}

// ------------------------------------------------------------- creation

#[test]
fn created_columns_land_in_the_requested_managers() {
    let dir = testdir::TestDir::new("dminfo-create-tiled".to_string());
    let mut desc = desc_of(vec![
        array_col(
            "DATA",
            DataType::Complex,
            vec![4, 6],
            "tiled",
            "TiledColumnStMan",
        ),
        array_col(
            "FLAG",
            DataType::Bool,
            vec![4, 6],
            "tiledf",
            "TiledColumnStMan",
        ),
        array_col(
            "TIME",
            DataType::Double,
            vec![],
            "StandardStMan",
            "StandardStMan",
        ),
    ]);
    // TIME is a scalar in reality; make it one (shape [] would be odd).
    desc.columns[2].shape = None;
    desc.columns[2].ndim = -1;
    desc.columns[2].kind = ColumnKind::Scalar(RecordValue::Double(0.0));
    apply(
        r#"{"*1": {"TYPE": "TiledColumnStMan", "NAME": "tiled",
                   "SPEC": {"DEFAULTTILESHAPE": [6, 4, 5]}, "COLUMNS": ["DATA"]},
            "*2": {"TYPE": "TiledColumnStMan", "NAME": "tiledf",
                   "SPEC": {"DEFAULTTILESHAPE": [6, 4, 7]}, "COLUMNS": ["FLAG"]}}"#,
        &mut desc,
    )
    .unwrap();

    let nrow = 23; // neither 5 nor 7 divides it
    let values = vec![
        complexes(nrow, &[4, 6]),
        bools(nrow, &[4, 6]),
        (0..nrow).map(|r| RecordValue::Double(r as f64)).collect(),
    ];
    create_table(&dir, &desc, &values).unwrap();

    let t = Table::open(&dir, true).unwrap();
    assert_array_eq("DATA", &values_of(&t, "DATA"), &values[0]);
    assert_array_eq("FLAG", &values_of(&t, "FLAG"), &values[1]);

    // getdminfo reports the requested managers with their tile shapes.
    let info = get_dminfo(&dir, &t.dat).unwrap();
    let tiled: Vec<_> = info
        .values()
        .filter(|dm| dm.type_name == "TiledColumnStMan")
        .collect();
    assert_eq!(tiled.len(), 2, "both columns tiled: {info:?}");
    let by_name: std::collections::HashMap<&String, &casacure::DmInfo> =
        info.values().map(|dm| (&dm.name, dm)).collect();
    let data = by_name.get(&"tiled".to_string()).unwrap();
    assert_eq!(data.columns, vec!["DATA".to_string()]);
    let casacure::DmSpec::TiledColumnStMan { tile_shapes, .. } = &data.spec else {
        panic!("expected a TSM spec, got {:?}", data.spec)
    };
    assert_eq!(tile_shapes[0], vec![4, 6, 5]);
    let flag = by_name.get(&"tiledf".to_string()).unwrap();
    let casacure::DmSpec::TiledColumnStMan { tile_shapes, .. } = &flag.spec else {
        panic!("expected a TSM spec, got {:?}", flag.spec)
    };
    assert_eq!(tile_shapes[0], vec![4, 6, 7]);
}

#[test]
fn cell_splitting_tiles_round_trip() {
    // dask-ms's `_fit_tile_shape` caps cell dims at 4, so a wide row
    // (32 channels here) yields a tile whose cell part splits the cell
    // ([4, 4, 64] over cells of [4, 32]). Real casacore writes such
    // layouts; so must the dminfo path.
    let dir = testdir::TestDir::new("dminfo-create-straddle".to_string());
    let mut desc = desc_of(vec![
        array_col(
            "DATA",
            DataType::Complex,
            vec![4, 32],
            "tiled",
            "TiledColumnStMan",
        ),
        array_col(
            "FLAG",
            DataType::Bool,
            vec![4, 32],
            "tiledf",
            "TiledColumnStMan",
        ),
    ]);
    apply(
        r#"{"*1": {"TYPE": "TiledColumnStMan", "NAME": "tiled",
                   "SPEC": {"DEFAULTTILESHAPE": [4, 4, 64]}, "COLUMNS": ["DATA"]},
            "*2": {"TYPE": "TiledColumnStMan", "NAME": "tiledf",
                   "SPEC": {"DEFAULTTILESHAPE": [4, 4, 64]}, "COLUMNS": ["FLAG"]}}"#,
        &mut desc,
    )
    .unwrap();
    let nrow = 100;
    let values = vec![complexes(nrow, &[4, 32]), bools(nrow, &[4, 32])];
    create_table(&dir, &desc, &values).unwrap();

    let t = Table::open(&dir, true).unwrap();
    assert_array_eq("DATA", &values_of(&t, "DATA"), &values[0]);
    assert_array_eq("FLAG", &values_of(&t, "FLAG"), &values[1]);

    // The raw bit stream of the tiled Bool column packs the same bools.
    let idx = t
        .dat
        .desc
        .columns
        .iter()
        .position(|c| c.name == "FLAG")
        .unwrap();
    let mut packed = Vec::new();
    t.getcol_raw_bits(
        idx,
        0,
        nrow,
        &mut |row: usize, bytes: &[u8], skip, nelem| {
            assert_eq!(skip, 0, "multi-tile cells gather from bit 0");
            assert_eq!(nelem, 4 * 32);
            packed.push((row, bytes.to_vec()));
            Ok(())
        },
    )
    .unwrap();
    for (row, bytes) in packed {
        let RecordValue::Array(a) = &values[1][row] else {
            panic!("expected an array")
        };
        let ArrayData::Bool(want) = &a.data else {
            panic!("expected bools")
        };
        let bits: Vec<bool> = bytes
            .iter()
            .flat_map(|b| (0..8).map(move |i| (b >> i) & 1 == 1))
            .take(want.len())
            .collect();
        assert_eq!(bits, *want, "row {row}");
    }
}

#[test]
fn growth_preserves_the_requested_tile() {
    // Created empty (the dask-ms shape: default_ms(nrow=0), then rows are
    // added) and grown — the header must keep the requested tile shape.
    let dir = testdir::TestDir::new("dminfo-create-grow".to_string());
    let mut desc = desc_of(vec![array_col(
        "FLAG",
        DataType::Bool,
        vec![4, 6],
        "tiledf",
        "TiledColumnStMan",
    )]);
    apply(
        r#"{"*1": {"TYPE": "TiledColumnStMan", "NAME": "tiledf",
                   "SPEC": {"DEFAULTTILESHAPE": [6, 4, 5]}, "COLUMNS": ["FLAG"]}}"#,
        &mut desc,
    )
    .unwrap();
    let mut wt = WritableTable::create(&dir, desc);
    wt.addrows(23);
    let flags = bools(23, &[4, 6]);
    for (r, v) in flags.iter().enumerate() {
        wt.putcell(0, r as u64, v.clone()).unwrap();
    }
    wt.flush().unwrap();
    drop(wt);

    let t = Table::open(&dir, true).unwrap();
    assert_array_eq("FLAG", &values_of(&t, "FLAG"), &flags);
    // The stored header carries the requested tile shape.
    let info = get_dminfo(&dir, &t.dat).unwrap();
    let flag = info.values().find(|dm| dm.name == "tiledf").unwrap();
    assert_eq!(flag.type_name, "TiledColumnStMan");
    let tsm = casacure::TsmFile::open(&dir, flag.seqnr, t.dat.header.big_endian).unwrap();
    assert_eq!(
        tsm.header.cubes[0].tile_shape,
        vec![4, 6, 5],
        "growth kept the requested tile"
    );
}

// ------------------------------------------------------------- copy

#[test]
fn deep_copy_conversion_lands_in_the_requested_managers() {
    let src = testdir::TestDir::new("dminfo-copy-src".to_string());
    let dst = testdir::TestDir::new("dminfo-copy-dst".to_string());
    std::fs::create_dir_all(&dst).unwrap();
    let desc = desc_of(vec![
        array_col(
            "DATA",
            DataType::Complex,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
        array_col(
            "FLAG",
            DataType::Bool,
            vec![4, 6],
            "StandardStMan",
            "StandardStMan",
        ),
    ]);
    let nrow = 11;
    let values = vec![complexes(nrow, &[4, 6]), bools(nrow, &[4, 6])];
    create_table(&src, &desc, &values).unwrap();

    // A byte-level copy (what `tablecopy` does), then the conversion.
    for entry in std::fs::read_dir(&src).unwrap().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') || name.to_string_lossy().ends_with(".lock") {
            continue;
        }
        let to = dst.join(&name);
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&to).unwrap();
        } else {
            std::fs::copy(entry.path(), &to).unwrap();
        }
    }
    let rec = parse_json_record(
        r#"{"*1": {"TYPE": "TiledColumnStMan", "NAME": "tiled",
                   "SPEC": {"DEFAULTTILESHAPE": [4, 6, 3]}, "COLUMNS": ["DATA"]},
            "*2": {"TYPE": "TiledColumnStMan", "NAME": "tiledf",
                   "SPEC": {"DEFAULTTILESHAPE": [4, 6, 4]}, "COLUMNS": ["FLAG"]}}"#,
    )
    .unwrap();
    apply_dminfo_in_place(&dst, &rec).unwrap();

    let t = Table::open(&dst, true).unwrap();
    assert_array_eq("DATA", &values_of(&t, "DATA"), &values[0]);
    assert_array_eq("FLAG", &values_of(&t, "FLAG"), &values[1]);
    let info = get_dminfo(&dst, &t.dat).unwrap();
    let tiled: Vec<_> = info
        .values()
        .filter(|dm| dm.type_name == "TiledColumnStMan")
        .collect();
    assert_eq!(tiled.len(), 2, "both columns converted: {info:?}");
}

// ---------------------------------------------------------------- issue #21: addcols + DEFAULTTILESHAPE transpose

#[test]
fn issue_21_defaulttileshape_axis_order_is_reversed() {
    //! Issue #21: When applying dminfo with DEFAULTTILESHAPE, the input comes
    //! in logical (C-order) dimensions from the user, but the column shape is
    //! stored in CASA (Fortran-reversed) order. The DEFAULTTILESHAPE must be
    //! reversed to match the cell shape for proper validation and use.
    //!
    //! The dask-ms add-columns pattern hits this: user provides logical-order
    //! shapes, which must be converted to CASA order before applying to columns.

    // Column has CASA (stored) cell shape [4, 6] (channels reversed to chans x correlations)
    let mut desc = desc_of(vec![array_col(
        "DATA",
        DataType::Complex,
        vec![4, 6],
        "TiledData",
        "TiledShapeStMan",
    )]);

    // User provides DEFAULTTILESHAPE in logical (C-order) dimensions:
    // logical shape [6, 4] (chans, corr) + 5 rows per tile = [6, 4, 5]
    //
    // This should be reversed internally to CASA order [4, 6, 5] to match the
    // column's [4, 6] cell shape.
    let user_tile_shape_logical = vec![6i64, 4, 5]; // user input: logical order
    let expected_tile_shape_casa = vec![4i64, 6, 5]; // what we should store: CASA order

    // Build a dminfo record with DEFAULTTILESHAPE in logical order (user input)
    let mut spec = empty_record();
    spec.set(
        "DEFAULTTILESHAPE",
        RecordValue::Array(ArrayValue {
            shape: vec![3u32],
            data: ArrayData::Int64(user_tile_shape_logical.clone()),
        }),
    );

    apply(
        r#"{"TYPE": "TiledShapeStMan", "NAME": "TiledData",
            "SPEC": {"DEFAULTTILESHAPE": [6, 4, 5]}}"#,
        &mut desc,
    )
    .unwrap();

    // After applying dminfo, the tile_shape on the column should be in CASA order
    // [4, 6, 5], matching the reversed cell shape [4, 6].
    assert_eq!(
        desc.columns[0].tile_shape.as_deref(),
        Some(&expected_tile_shape_casa[..]),
        "DEFAULTTILESHAPE should be reversed from logical [6,4,5] to CASA [4,6,5]"
    );
}
