#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use casacure::images::CoordinateSystem;
use casacure::record::{ArrayData, ArrayValue, RecordValue, TableRecord};
use casacure::tabledesc::TableDesc;
use casacure::WritableTable;

fn roundtrip(label: &str, rec: TableRecord) {
    let dir = testdir::TestDir::new(format!("recbisect-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&*dir).unwrap();
    let desc = TableDesc::from_desc_json(r#"{"X": {"valueType": "double", "option": 0}}"#).unwrap();
    let mut wt = WritableTable::create(&dir, desc);
    wt.putkeyword("K", RecordValue::Record(rec));
    wt.flush().unwrap();
    drop(wt);
    let t = casacure::Table::open(&dir, true).unwrap();
    match t.dat.desc.keywords.get("K") {
        Some(RecordValue::Record(back)) => {
            println!("{label}: OK ({} fields)", back.desc.fields.len());
        }
        other => println!("{label}: FAIL {other:?}"),
    }
}

#[test]
fn bisect_default_coords_record() {
    let csys = CoordinateSystem::default_for(&[2, 1, 8, 8]);
    let full = csys.raw_record();
    let names: Vec<String> = full.desc.fields.iter().map(|f| f.name.clone()).collect();
    println!("fields: {names:?}");
    roundtrip("full", full.clone());
    // Field-by-field: keep only one top-level entry at a time.
    for name in &names {
        let mut rec = TableRecord::default();
        let pos = full
            .desc
            .fields
            .iter()
            .position(|f| &f.name == name)
            .unwrap();
        rec.set(name, full.values[pos].clone());
        roundtrip(&name.replace(|c: char| !c.is_alphanumeric(), ""), rec);
    }
}

#[test]
fn bisect_growing() {
    let csys = CoordinateSystem::default_for(&[2, 1, 8, 8]);
    let full = csys.raw_record();
    // Grow to the full record one field at a time; find the breaking size.
    for keep in 1..=full.desc.fields.len() {
        let mut rec = TableRecord::default();
        for (i, f) in full.desc.fields.iter().take(keep).enumerate() {
            rec.set(&f.name, full.values[i].clone());
        }
        let dir = testdir::TestDir::new(format!("recgrow-{keep}-{}", std::process::id()));
        std::fs::create_dir_all(&*dir).unwrap();
        let desc =
            TableDesc::from_desc_json(r#"{"X": {"valueType": "double", "option": 0}}"#).unwrap();
        let mut wt = WritableTable::create(&dir, desc);
        wt.putkeyword("K", RecordValue::Record(rec));
        wt.flush().unwrap();
        drop(wt);
        let t = casacure::Table::open(&dir, true).unwrap();
        let ok = t.dat.desc.keywords.get("K").is_some();
        println!("keep {keep}: {}", if ok { "OK" } else { "FAIL" });
    }
}

#[test]
fn bisect_top_level_combinations() {
    let csys = CoordinateSystem::default_for(&[2, 1, 8, 8]);
    let full = csys.raw_record();
    let get = |name: &str| -> RecordValue {
        let pos = full
            .desc
            .fields
            .iter()
            .position(|f| f.name == name)
            .unwrap();
        full.values[pos].clone()
    };
    let mut r2 = TableRecord::default();
    r2.set("direction0", get("direction0"));
    r2.set("pixelmap0", get("pixelmap0"));
    roundtrip("dir-plus-pixelmap", r2);
    let mut r3 = TableRecord::default();
    r3.set("pixelmap0", get("pixelmap0"));
    r3.set("stokes1", get("stokes1"));
    roundtrip("pixelmap-then-stokes", r3);
    let mut r4 = TableRecord::default();
    r4.set("direction0", get("direction0"));
    r4.set("stokes1", get("stokes1"));
    roundtrip("dir-then-stokes", r4);
    let mut r5 = TableRecord::default();
    r5.set("stokes1", get("stokes1"));
    r5.set("spectral2", get("spectral2"));
    roundtrip("stokes-then-spectral", r5);
    let mut r6 = TableRecord::default();
    r6.set("direction0", get("direction0"));
    roundtrip("dir-only", r6);
}

#[test]
fn bisect_synthesised_spectral() {
    // The spectral record alone, and with fields removed one by one.
    let csys = CoordinateSystem::default_for(&[2, 1, 8, 8]);
    let full = csys.raw_record();
    let pos = full
        .desc
        .fields
        .iter()
        .position(|f| f.name == "spectral2")
        .unwrap();
    let spectral = match &full.values[pos] {
        RecordValue::Record(r) => r.clone(),
        _ => panic!("spectral2 not a record"),
    };
    let names: Vec<String> = spectral
        .desc
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect();
    println!("spectral fields: {names:?}");
    roundtrip("spectral-full", spectral.clone());
    for name in &names {
        let mut rec = TableRecord::default();
        let p = spectral
            .desc
            .fields
            .iter()
            .position(|f| &f.name == name)
            .unwrap();
        rec.set(name, spectral.values[p].clone());
        roundtrip(
            &format!("spectral-only-{name}").replace(|c: char| !c.is_alphanumeric(), ""),
            rec,
        );
    }
    let _ = ArrayData::Double(vec![]);
    let _ = ArrayValue {
        shape: vec![],
        data: ArrayData::Double(vec![]),
    };
}

#[test]
fn dump_raw_record_alignment() {
    let csys = CoordinateSystem::default_for(&[2, 1, 8, 8]);
    let rec = csys.raw_record();
    for (i, f) in rec.desc.fields.iter().enumerate() {
        let vt = match &rec.values.get(i) {
            Some(RecordValue::Record(_)) => "Record",
            Some(RecordValue::Array(_)) => "Array",
            Some(RecordValue::String(_)) => "String",
            Some(RecordValue::Double(_)) => "Double",
            Some(RecordValue::Int(_)) => "Int",
            _ => "?",
        };
        println!("{i}: desc={} value={vt}", f.name);
    }
}
