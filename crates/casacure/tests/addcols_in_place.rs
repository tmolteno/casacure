//! `addcol` + `flush` on an existing table extends it in place.
//!
//! Adding a column used to send the flush down the whole-table rewrite
//! (`materialize_all` + `create_table`): every cell of every column was read
//! into memory and rewritten, so adding one `[nchan]` weight column to a
//! 15 GB MeerKAT MS cost 19 s and a 32 GB peak where python-casacore needs
//! 0.2 s and nothing, and a process dying mid-rewrite left a half-written MS.
//! The flush now writes the new column's data manager(s) beside the old ones
//! and replaces `table.dat` last; the existing columns' files are not
//! touched.  These tests pin that: byte-identical (same inode, same mtime)
//! files for every existing data manager, values that round-trip, no
//! whole-table rewrite, and a correct fallback for layouts the in-place path
//! does not handle.

#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;

use casacure::record::{ArrayData, ArrayValue, DataType, RecordValue, TableRecord};
use casacure::{
    get_dminfo, parse_table_dat, ColumnDesc, ColumnKind, Table, TableDesc, WritableTable,
};

const NROWS: u64 = 700;
const NCHAN: i64 = 12;

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
        tile_shape: None,
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
        tile_shape: None,
    }
}

/// The column killMS adds as IMAGING_WEIGHT (its `getcoldesc`-style dict).
fn imaging_weight() -> ColumnDesc {
    array(
        "IMAGING_WEIGHT",
        DataType::Float,
        vec![NCHAN],
        "TiledShapeStMan",
        "imagingweight",
    )
}

/// The existing UVW column's descriptor under another name (killMS's UVWDT:
/// `desc = getcoldesc("UVW"); desc["name"] = "UVWDT"`), so it asks for the
/// same StandardStMan data manager UVW lives in.
fn uvwdt() -> ColumnDesc {
    let mut cd = ms_like_desc().columns[UVW].clone();
    cd.name = "UVWDT".into();
    cd
}

/// A cut-down MS main table: TIME on IncrementalStMan, ANTENNA2 + FLAG_ROW +
/// UVW sharing one StandardStMan (as in a real MS), DATA (complex), FLAG and
/// WEIGHT on their own tiled managers.
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
                vec![4, NCHAN],
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
            array(
                "DATA",
                DataType::Complex,
                vec![4, NCHAN],
                "TiledColumnStMan",
                "TiledData",
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
const DATA: usize = 6;

fn arr(shape: Vec<u32>, data: ArrayData) -> RecordValue {
    RecordValue::Array(ArrayValue { shape, data })
}

fn flag_cell(r: u64) -> RecordValue {
    arr(
        vec![NCHAN as u32, 4],
        ArrayData::Bool(
            (0..4 * NCHAN as u64)
                .map(|k| (r + k).is_multiple_of(3))
                .collect(),
        ),
    )
}

fn weight_cell(r: u64) -> RecordValue {
    arr(vec![4], ArrayData::Float(vec![r as f32 + 0.25; 4]))
}

fn uvw_cell(r: u64) -> RecordValue {
    arr(vec![3], ArrayData::Double(vec![r as f64, 0.5, -1.0]))
}

fn data_cell(r: u64) -> RecordValue {
    arr(
        vec![NCHAN as u32, 4],
        ArrayData::Complex(
            (0..4 * NCHAN as u64)
                .map(|k| ((r * 100 + k) as f32, -(k as f32)))
                .collect(),
        ),
    )
}

/// The `[nchan]` IMAGING_WEIGHT cell of row `r`.
fn iw_cell(r: u64) -> RecordValue {
    arr(
        vec![NCHAN as u32],
        ArrayData::Float((0..NCHAN).map(|k| r as f32 * 0.5 + k as f32).collect()),
    )
}

fn iw_default() -> RecordValue {
    arr(
        vec![NCHAN as u32],
        ArrayData::Float(vec![0.0; NCHAN as usize]),
    )
}

fn temp_dir(tag: &str) -> testdir::TestDir {
    testdir::TestDir::new(format!(
        "casacure-addcols-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

/// An `NROWS`-row MS-like table with every column written, on disk.
fn build_table(tag: &str) -> testdir::TestDir {
    let dir = temp_dir(tag);
    let mut wt = WritableTable::create(&dir, ms_like_desc());
    wt.addrows(NROWS);
    for r in 0..NROWS {
        wt.putcell(TIME, r, RecordValue::Double((r / 7) as f64))
            .unwrap();
        wt.putcell(ANTENNA2, r, RecordValue::Int((r % 63) as i32))
            .unwrap();
        wt.putcell(FLAG_ROW, r, RecordValue::Bool(r.is_multiple_of(5)))
            .unwrap();
        wt.putcell(UVW, r, uvw_cell(r)).unwrap();
        wt.putcell(FLAG, r, flag_cell(r)).unwrap();
        wt.putcell(WEIGHT, r, weight_cell(r)).unwrap();
        wt.putcell(DATA, r, data_cell(r)).unwrap();
    }
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 1, "the create is the only whole write");
    dir
}

/// (bytes, inode, mtime ns) of every `table.f*` file in `dir`.
type Files = BTreeMap<String, (Vec<u8>, u64, i64, i64)>;

fn snapshot(dir: &std::path::Path) -> Files {
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let name = e.file_name().into_string().unwrap();
        if name.starts_with("table.f") {
            let m = e.metadata().unwrap();
            out.insert(
                name,
                (
                    std::fs::read(e.path()).unwrap(),
                    m.ino(),
                    m.mtime(),
                    m.mtime_nsec(),
                ),
            );
        }
    }
    out
}

/// Every file in `before` except those in `replaced` is still there,
/// byte-for-byte, with the same inode and mtime.
fn assert_untouched(before: &Files, dir: &std::path::Path, replaced: &[&str]) {
    let after = snapshot(dir);
    for (name, f) in before {
        if replaced.contains(&name.as_str()) {
            continue;
        }
        let g = after
            .get(name)
            .unwrap_or_else(|| panic!("{name} disappeared"));
        assert_eq!(f.0, g.0, "{name} bytes changed");
        assert_eq!(f.1, g.1, "{name} was replaced (inode changed)");
        assert_eq!((f.2, f.3), (g.2, g.3), "{name} mtime changed");
    }
}

fn assert_existing_columns_intact(dir: &std::path::Path) {
    let t = Table::open(dir, false).unwrap();
    assert_eq!(t.nrows(), NROWS);
    for r in 0..NROWS {
        assert_eq!(
            t.getcell(TIME, r).unwrap(),
            RecordValue::Double((r / 7) as f64)
        );
        assert_eq!(
            t.getcell(ANTENNA2, r).unwrap(),
            RecordValue::Int((r % 63) as i32)
        );
        assert_eq!(
            t.getcell(FLAG_ROW, r).unwrap(),
            RecordValue::Bool(r.is_multiple_of(5))
        );
        assert_eq!(t.getcell(UVW, r).unwrap(), uvw_cell(r), "UVW {r}");
        assert_eq!(t.getcell(FLAG, r).unwrap(), flag_cell(r), "FLAG {r}");
        assert_eq!(t.getcell(WEIGHT, r).unwrap(), weight_cell(r), "WEIGHT {r}");
        assert_eq!(t.getcell(DATA, r).unwrap(), data_cell(r), "DATA {r}");
    }
}

fn colnames(dir: &std::path::Path) -> Vec<String> {
    let dat = parse_table_dat(&std::fs::read(dir.join("table.dat")).unwrap()).unwrap();
    dat.desc.columns.iter().map(|c| c.name.clone()).collect()
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

const ALL_FILES_OF_OTHER_DMS: [&str; 0] = [];

#[test]
fn killms_imaging_weight_is_added_without_touching_existing_files() {
    let dir = build_table("iw");
    let before = snapshot(&dir);
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(imaging_weight());
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0, "add-column flush rewrote the table");
    drop(wt);

    assert_untouched(&before, &dir, &ALL_FILES_OF_OTHER_DMS);
    assert!(colnames(&dir).contains(&"IMAGING_WEIGHT".to_string()));
    assert_existing_columns_intact(&dir);
    let t = Table::open(&dir, false).unwrap();
    let iw = t.dat.desc.columns.len() - 1;
    for r in [0, 1, NROWS / 2, NROWS - 1] {
        assert_eq!(t.getcell(iw, r).unwrap(), iw_default(), "default row {r}");
    }
    // The new column is a manager of its own, named by its group.
    let dm = get_dminfo(&dir, &t.dat).unwrap();
    let hit: Vec<_> = dm
        .values()
        .filter(|d| d.columns.iter().any(|c| c == "IMAGING_WEIGHT"))
        .collect();
    assert_eq!(hit.len(), 1);
    assert_eq!(hit[0].type_name, "TiledShapeStMan");
    assert_eq!(hit[0].name, "imagingweight");
    assert_eq!(hit[0].columns, names(&["IMAGING_WEIGHT"]));
}

#[test]
fn addcol_then_chunked_putcol_then_flush_round_trips() {
    let dir = build_table("chunks");
    let before = snapshot(&dir);
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(imaging_weight());
    let iw = ms_like_desc().columns.len();
    for start in (0..NROWS).step_by(250) {
        let n = 250.min(NROWS - start);
        let vals: Vec<RecordValue> = (start..start + n).map(iw_cell).collect();
        wt.putcol(iw, start, &vals).unwrap();
    }
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0);
    drop(wt);
    assert_untouched(&before, &dir, &ALL_FILES_OF_OTHER_DMS);
    assert_existing_columns_intact(&dir);
    let t = Table::open(&dir, false).unwrap();
    for r in 0..NROWS {
        assert_eq!(t.getcell(iw, r).unwrap(), iw_cell(r), "IMAGING_WEIGHT {r}");
    }
}

#[test]
fn scalar_standardstman_column_in_its_own_manager() {
    let dir = build_table("scalar");
    let before = snapshot(&dir);
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(scalar(
        "EXTRA",
        DataType::Double,
        RecordValue::Double(0.0),
        "StandardStMan",
        "ExtraGroup",
    ));
    let c = ms_like_desc().columns.len();
    let vals: Vec<RecordValue> = (0..NROWS)
        .map(|r| RecordValue::Double(r as f64 / 3.0))
        .collect();
    wt.putcol(c, 0, &vals).unwrap();
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0);
    drop(wt);
    assert_untouched(&before, &dir, &ALL_FILES_OF_OTHER_DMS);
    assert_existing_columns_intact(&dir);
    let t = Table::open(&dir, false).unwrap();
    for r in 0..NROWS {
        assert_eq!(
            t.getcell(c, r).unwrap(),
            RecordValue::Double(r as f64 / 3.0)
        );
    }
}

#[test]
fn column_copied_from_uvw_joins_uvws_manager_and_only_that_manager_is_rebuilt() {
    let dir = build_table("uvwdt");
    let before = snapshot(&dir);
    let dat = parse_table_dat(&std::fs::read(dir.join("table.dat")).unwrap()).unwrap();
    let uvw_seq = dat.column_set.columns[UVW].data_manager_seq;
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(uvwdt());
    let c = ms_like_desc().columns.len();
    let vals: Vec<RecordValue> = (0..NROWS)
        .map(|r| arr(vec![3], ArrayData::Double(vec![r as f64 * 2.0, 1.0, 2.0])))
        .collect();
    wt.putcol(c, 0, &vals).unwrap();
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0);
    drop(wt);

    // Only the files of UVW's StandardStMan (data + array index) may change:
    // the old files stay untouched where they are (the rebuilt manager gets a
    // fresh sequence number) or are replaced.
    let after = snapshot(&dir);
    let own = [format!("table.f{uvw_seq}"), format!("table.f{uvw_seq}i")];
    for (name, f) in &before {
        if own.contains(name) {
            continue;
        }
        let g = after
            .get(name)
            .unwrap_or_else(|| panic!("{name} disappeared"));
        assert_eq!(
            (&f.0, f.1, f.2, f.3),
            (&g.0, g.1, g.2, g.3),
            "{name} was touched"
        );
    }
    assert_existing_columns_intact(&dir);
    let t = Table::open(&dir, false).unwrap();
    let dm = get_dminfo(&dir, &t.dat).unwrap();
    let shared: Vec<_> = dm
        .values()
        .filter(|d| d.columns.iter().any(|x| x == "UVWDT"))
        .collect();
    assert_eq!(shared.len(), 1);
    assert!(
        shared[0].columns.iter().any(|x| x == "UVW"),
        "UVWDT joined UVW's manager"
    );
    for r in 0..NROWS {
        assert_eq!(
            t.getcell(c, r).unwrap(),
            arr(vec![3], ArrayData::Double(vec![r as f64 * 2.0, 1.0, 2.0]))
        );
    }
}

#[test]
fn column_added_and_never_written_reads_its_default() {
    let dir = build_table("default");
    let before = snapshot(&dir);
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(scalar(
        "COUNT",
        DataType::Int,
        RecordValue::Int(-7),
        "StandardStMan",
        "Counts",
    ));
    wt.addcol(array(
        "FLAG2",
        DataType::Bool,
        vec![4, NCHAN],
        "TiledShapeStMan",
        "TiledFlag2",
    ));
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0);
    drop(wt);
    assert_untouched(&before, &dir, &ALL_FILES_OF_OTHER_DMS);
    let t = Table::open(&dir, false).unwrap();
    let c = ms_like_desc().columns.len();
    for r in [0, 17, NROWS - 1] {
        assert_eq!(t.getcell(c, r).unwrap(), RecordValue::Int(-7));
        assert_eq!(
            t.getcell(c + 1, r).unwrap(),
            arr(
                vec![NCHAN as u32, 4],
                ArrayData::Bool(vec![false; 4 * NCHAN as usize])
            )
        );
    }
}

#[test]
fn several_addcols_before_one_flush() {
    let dir = build_table("several");
    let before = snapshot(&dir);
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    let c = ms_like_desc().columns.len();
    wt.addcol(imaging_weight());
    wt.addcol(uvwdt());
    wt.addcol(scalar(
        "A",
        DataType::Int,
        RecordValue::Int(0),
        "StandardStMan",
        "Grp",
    ));
    wt.addcol(scalar(
        "B",
        DataType::Double,
        RecordValue::Double(0.0),
        "StandardStMan",
        "Grp",
    ));
    let iwv: Vec<RecordValue> = (0..NROWS).map(iw_cell).collect();
    wt.putcol(c, 0, &iwv).unwrap();
    let av: Vec<RecordValue> = (0..NROWS).map(|r| RecordValue::Int(r as i32)).collect();
    wt.putcol(c + 2, 0, &av).unwrap();
    let bv: Vec<RecordValue> = (0..NROWS)
        .map(|r| RecordValue::Double(-(r as f64)))
        .collect();
    wt.putcol(c + 3, 0, &bv).unwrap();
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0);
    drop(wt);
    let own: Vec<String> = {
        let dat = parse_table_dat(&std::fs::read(dir.join("table.dat")).unwrap()).unwrap();
        let s = dat.column_set.columns[UVW].data_manager_seq;
        // (the manager that held UVW before the flush)
        let _ = s;
        Vec::new()
    };
    let _ = own;
    assert_existing_columns_intact(&dir);
    assert_eq!(
        colnames(&dir)[c..],
        names(&["IMAGING_WEIGHT", "UVWDT", "A", "B"])
    );
    let t = Table::open(&dir, false).unwrap();
    for r in 0..NROWS {
        assert_eq!(t.getcell(c, r).unwrap(), iw_cell(r));
        assert_eq!(t.getcell(c + 2, r).unwrap(), RecordValue::Int(r as i32));
        assert_eq!(
            t.getcell(c + 3, r).unwrap(),
            RecordValue::Double(-(r as f64))
        );
    }
    // A and B share one new manager, IMAGING_WEIGHT has its own.
    let dm = get_dminfo(&dir, &t.dat).unwrap();
    assert!(dm.values().any(|d| d.columns == names(&["A", "B"])));
    let _ = before;
}

#[test]
fn addcols_flush_then_addcols_flush_again() {
    let dir = build_table("twice");
    let before = snapshot(&dir);
    let c = ms_like_desc().columns.len();
    {
        let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
        wt.addcol(imaging_weight());
        let v: Vec<RecordValue> = (0..NROWS).map(iw_cell).collect();
        wt.putcol(c, 0, &v).unwrap();
        wt.flush().unwrap();
        assert_eq!(wt.full_rewrites(), 0);
        // The same handle keeps working: a second column, then a write of
        // the first column's rows.
        wt.addcol(scalar(
            "SECOND",
            DataType::Int,
            RecordValue::Int(3),
            "StandardStMan",
            "Second",
        ));
        wt.putcell(c, 5, iw_cell(9999)).unwrap();
        wt.flush().unwrap();
        assert_eq!(wt.full_rewrites(), 0);
    }
    {
        // And a fresh handle, a third time.
        let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
        wt.addcol(array(
            "THIRD",
            DataType::Float,
            vec![2],
            "TiledShapeStMan",
            "Third",
        ));
        wt.flush().unwrap();
        assert_eq!(wt.full_rewrites(), 0);
    }
    assert_untouched(&before, &dir, &ALL_FILES_OF_OTHER_DMS);
    assert_existing_columns_intact(&dir);
    let t = Table::open(&dir, false).unwrap();
    assert_eq!(
        colnames(&dir)[c..],
        names(&["IMAGING_WEIGHT", "SECOND", "THIRD"])
    );
    for r in 0..NROWS {
        let want = if r == 5 { iw_cell(9999) } else { iw_cell(r) };
        assert_eq!(t.getcell(c, r).unwrap(), want, "IMAGING_WEIGHT {r}");
        assert_eq!(t.getcell(c + 1, r).unwrap(), RecordValue::Int(3));
    }
    // Manager sequence numbers and names are distinct.
    let dm = get_dminfo(&dir, &t.dat).unwrap();
    let mut seen = std::collections::HashSet::new();
    for d in dm.values() {
        assert!(
            seen.insert(d.name.clone()),
            "duplicate manager name {}",
            d.name
        );
    }
}

#[test]
fn addcol_with_writes_to_existing_columns_and_new_rows_in_the_same_flush() {
    let dir = build_table("mixed");
    let c = ms_like_desc().columns.len();
    {
        let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
        wt.addrows(50);
        wt.addcol(imaging_weight());
        wt.putcell(WEIGHT, 3, weight_cell(777)).unwrap();
        wt.putcell(ANTENNA2, 4, RecordValue::Int(-4)).unwrap();
        for r in 0..NROWS + 50 {
            wt.putcell(c, r, iw_cell(r)).unwrap();
        }
        wt.flush().unwrap();
        assert_eq!(wt.full_rewrites(), 0);
    }
    let t = Table::open(&dir, false).unwrap();
    assert_eq!(t.nrows(), NROWS + 50);
    for r in 0..NROWS {
        let w = if r == 3 {
            weight_cell(777)
        } else {
            weight_cell(r)
        };
        assert_eq!(t.getcell(WEIGHT, r).unwrap(), w);
        let a = if r == 4 { -4 } else { (r % 63) as i32 };
        assert_eq!(t.getcell(ANTENNA2, r).unwrap(), RecordValue::Int(a));
        assert_eq!(t.getcell(DATA, r).unwrap(), data_cell(r));
    }
    for r in 0..NROWS + 50 {
        assert_eq!(t.getcell(c, r).unwrap(), iw_cell(r));
    }
}

#[test]
fn column_keywords_and_table_keywords_survive_an_add() {
    let dir = build_table("kw");
    let c = ms_like_desc().columns.len();
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    let mut cd = imaging_weight();
    cd.comment = "weights for imaging".into();
    wt.addcol(cd);
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 0);
    drop(wt);
    let dat = parse_table_dat(&std::fs::read(dir.join("table.dat")).unwrap()).unwrap();
    assert_eq!(dat.desc.columns[c].comment, "weights for imaging");
    assert_eq!(dat.desc.columns[c].shape, Some(vec![NCHAN]));
}

// --- layouts the in-place path does not handle fall back, correctly ---

#[test]
fn column_joining_an_incremental_manager_falls_back_to_the_full_rewrite() {
    let dir = build_table("ismjoin");
    let c = ms_like_desc().columns.len();
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(scalar(
        "TIME2",
        DataType::Double,
        RecordValue::Double(0.0),
        "IncrementalStMan",
        "ISMData",
    ));
    let v: Vec<RecordValue> = (0..NROWS).map(|r| RecordValue::Double(r as f64)).collect();
    wt.putcol(c, 0, &v).unwrap();
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 1, "this layout is rewritten whole");
    drop(wt);
    assert_existing_columns_intact(&dir);
    let t = Table::open(&dir, false).unwrap();
    for r in 0..NROWS {
        assert_eq!(t.getcell(c, r).unwrap(), RecordValue::Double(r as f64));
    }
}

#[test]
fn removecol_plus_addcol_falls_back_to_the_full_rewrite() {
    let dir = build_table("readd");
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    // Drop the last column (DATA) and add another one: the column count is
    // unchanged but the on-disk columns are no longer a prefix of the
    // schema's data, so the table is regenerated.
    wt.removecol(DATA);
    wt.addcol(imaging_weight());
    let v: Vec<RecordValue> = (0..NROWS).map(iw_cell).collect();
    wt.putcol(DATA, 0, &v).unwrap();
    wt.flush().unwrap();
    assert_eq!(wt.full_rewrites(), 1);
    drop(wt);
    let t = Table::open(&dir, false).unwrap();
    assert_eq!(colnames(&dir)[DATA..], names(&["IMAGING_WEIGHT"]));
    for r in 0..NROWS {
        assert_eq!(t.getcell(DATA, r).unwrap(), iw_cell(r));
        assert_eq!(t.getcell(UVW, r).unwrap(), uvw_cell(r));
        assert_eq!(t.getcell(WEIGHT, r).unwrap(), weight_cell(r));
        assert_eq!(t.getcell(FLAG, r).unwrap(), flag_cell(r));
    }
}

// --- crash safety ---

/// New files are written first and `table.dat` last: when writing the new
/// column's files fails the old `table.dat` is untouched and the table opens
/// and reads exactly as before.
#[test]
fn a_failed_add_leaves_the_old_table_readable() {
    let dir = build_table("crash");
    let dat_before = std::fs::read(dir.join("table.dat")).unwrap();
    let dat = parse_table_dat(&dat_before).unwrap();
    // The new manager takes the next sequence number: block its header file
    // with a directory so the atomic write of it fails.
    let next_seq = dat.column_set.seq_count;
    let before = snapshot(&dir);
    std::fs::create_dir(dir.join(format!("table.f{next_seq}"))).unwrap();
    let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
    wt.addcol(imaging_weight());
    assert!(wt.flush().is_err(), "the blocked write must fail the flush");
    drop(wt);
    assert!(
        std::fs::read(dir.join("table.dat")).unwrap() == dat_before,
        "table.dat was replaced although writing the new column's files failed"
    );
    std::fs::remove_dir(dir.join(format!("table.f{next_seq}"))).unwrap();
    assert_untouched(&before, &dir, &[]);
    assert_existing_columns_intact(&dir);
    assert_eq!(colnames(&dir).len(), ms_like_desc().columns.len());
}
