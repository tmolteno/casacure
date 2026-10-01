//! Multi-threaded tests for casacure's documented thread-safety contracts.
//!
//! `Table` is documented `Send + Sync` — "data files are mapped read-only
//! into the handle (a stable snapshot for the handle's lifetime), so a
//! `Table` is safe to hold across threads (dask-ms serializes access on its
//! side)".  These tests pin that contract under real thread interleaving:
//! many readers on one shared snapshot, the process-wide lock-file registry
//! racing concurrent attaches down to one instance, and — the sharpest edge —
//! readers holding a snapshot while another thread flushes writes into the
//! same table directory (the dask-ms read-while-write overlap the
//! `py.detach` bindings make reachable from Python).

#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use casacure::lockfile::{self, LockOptions};
use casacure::record::{ArrayData, ArrayValue, RecordValue};
use casacure::tabledesc::TableDesc;
use casacure::{lock_sync_nrrow, Table, WritableTable};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread::{self, JoinHandle};

fn temp_dir(tag: &str) -> testdir::TestDir {
    let dir = testdir::TestDir::new(format!(
        "casacure-threads-{tag}-{}-{:?}",
        std::process::id(),
        thread::current().id()
    ));
    std::fs::create_dir_all(&*dir).unwrap();
    dir
}

fn ssm_double_desc() -> TableDesc {
    let json = r#"{"C": {"valueType": "double", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "S", "option": 0}}"#;
    TableDesc::from_desc_json(json).unwrap()
}

fn ssm_string_desc() -> TableDesc {
    let json = r#"{"S": {"valueType": "string", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "S", "option": 0}}"#;
    TableDesc::from_desc_json(json).unwrap()
}

fn tsm_float_desc() -> TableDesc {
    // A fixed-shape 3x2 float array column tiled by TiledColumnStMan (the
    // DATA-column layout of a measurement set).
    let json = r#"{"DATA": {"valueType": "float", "dataManagerType": "TiledColumnStMan",
        "dataManagerGroup": "TiledData", "ndim": 2, "shape": [3, 2], "option": 0}}"#;
    TableDesc::from_desc_json(json).unwrap()
}

/// The pinned thread contracts: the read handle is shareable (`Send +
/// Sync`), the write handle is movable to a worker thread (`Send`).  If a
/// refactor breaks either, this fails at compile time.
#[test]
fn table_types_honour_their_thread_contracts() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<Table>();
    send::<WritableTable>();
}

/// Many threads reading through one shared `Arc<Table>` (the dask threaded
/// executor pattern) must always see the snapshot's values, with the page
/// cache dropped between passes so reads re-fault the mapping instead of
/// serving resident pages.
#[test]
fn concurrent_readers_share_one_snapshot() {
    const NROW: u64 = 4096;
    const READERS: usize = 8;
    let dir = temp_dir("readers");
    let desc = ssm_double_desc();
    let values = vec![(0..NROW).map(|r| RecordValue::Double(r as f64)).collect()];
    Table::create_with_lock(&dir, &desc, &values, LockOptions::no_locking()).unwrap();

    let table = Arc::new(Table::open(&dir, true).unwrap());
    let start = Arc::new(Barrier::new(READERS));
    let errors = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<JoinHandle<()>> = (0..READERS)
        .map(|tid| {
            let table = Arc::clone(&table);
            let start = Arc::clone(&start);
            let errors = Arc::clone(&errors);
            thread::spawn(move || {
                start.wait();
                for i in 0..50u64 {
                    let first = (tid as u64 * 613 + i * 251) % (NROW - 256);
                    match table.getcol(0, first, 256) {
                        Ok(vals) => {
                            for (j, v) in vals.iter().enumerate() {
                                let want = RecordValue::Double((first + j as u64) as f64);
                                if *v != want {
                                    errors.lock().unwrap().push(format!(
                                        "t{tid} pass {i}: row {} = {v:?}, want {want:?}",
                                        first + j as u64
                                    ));
                                    return;
                                }
                            }
                        }
                        Err(e) => {
                            errors.lock().unwrap().push(format!("t{tid} pass {i}: {e}"));
                            return;
                        }
                    }
                    // Re-fault the mapping on later passes (the python bulk
                    // read drops pages the same way).
                    if i % 8 == 7 {
                        table.drop_data_file_pages();
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "{errors:?}");
}

/// Concurrent `attach` calls for one directory must all land on the SAME
/// shared lock-file instance (`Arc::ptr_eq`), and the registry-backed
/// `lock_sync_nrrow` must keep answering while attaches race it — the fd a
/// process's fcntl locks live on must never be duplicated.
#[test]
fn concurrent_attaches_share_one_lock_file() {
    const THREADS: usize = 16;
    let dir = temp_dir("attach");
    let desc = ssm_double_desc();
    let values = vec![vec![RecordValue::Double(1.5), RecordValue::Double(2.5)]];
    Table::create_with_lock(&dir, &desc, &values, LockOptions::locking_default()).unwrap();
    // The sync record first appears on a writable flush (`create_with_lock`
    // only zeroes the request area), so put one on disk before racing.
    let (_read, mut wt) =
        WritableTable::open_for_update_with_lock(&dir, LockOptions::locking_default()).unwrap();
    wt.putcell(0, 0, RecordValue::Double(9.5)).unwrap();
    wt.flush().unwrap();

    let eff = LockOptions::locking_default().effective();
    let start = Arc::new(Barrier::new(THREADS));
    let instances = Arc::new(Mutex::new(Vec::new()));
    let errors = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<JoinHandle<()>> = (0..THREADS)
        .map(|_| {
            let dir = dir.to_path_buf();
            let start = Arc::clone(&start);
            let instances = Arc::clone(&instances);
            let errors = Arc::clone(&errors);
            thread::spawn(move || {
                start.wait();
                for _ in 0..64 {
                    match lockfile::attach(&dir, &eff, false) {
                        Ok(Some(lf)) => instances.lock().unwrap().push(lf),
                        Ok(None) => {
                            errors.lock().unwrap().push("attach returned None".into());
                            return;
                        }
                        Err(e) => {
                            errors.lock().unwrap().push(format!("attach: {e}"));
                            return;
                        }
                    }
                    if lock_sync_nrrow(&dir) != Some(2) {
                        errors.lock().unwrap().push(format!(
                            "lock_sync_nrrow lost the sync record: got {:?}",
                            lock_sync_nrrow(&dir)
                        ));
                        return;
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "{errors:?}");
    let instances = instances.lock().unwrap();
    let first = &instances[0];
    for (i, lf) in instances.iter().enumerate() {
        assert!(
            Arc::ptr_eq(first, lf),
            "attach #{i} produced a second lock-file instance for one directory"
        );
    }
}

/// The snapshot-stress driver.  Readers loop full-column reads of a
/// snapshot opened BEFORE any write while ONE writer thread (dask-ms
/// serializes same-table writes) runs open-for-update → putcell → flush
/// cycles that rewrite the table's data files.
///
/// What a reader may observe depends on the flush path, and both modes
/// assert the real contract:
///
/// * `strict` (whole-file rewrites): every flush replaces the file's inode
///   atomically (`datafile::write_atomic`), so a live mapping keeps the
///   captured inode — readers must see ONLY the original values, forever.
/// * patch-path columns (byte patches at stable offsets in the same inode,
///   casacore's behaviour for an unlocked reader): each cell is either its
///   original or its rewritten value — never another row's value, never
///   misaligned bytes, never an error.
fn stress_readers_vs_writer(
    dir: &Path,
    desc: &TableDesc,
    originals: Vec<RecordValue>,
    rewritten: impl Fn(u64) -> RecordValue + Send + Sync + 'static,
    strict: bool,
) {
    const READERS: usize = 4;
    const WRITES: u32 = 200;
    let nrow = originals.len() as u64;
    let values = vec![originals.clone()];
    Table::create_with_lock(dir, desc, &values, LockOptions::no_locking()).unwrap();

    let snapshot = Arc::new(Table::open(dir, true).unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(Mutex::new(Vec::new()));
    let rewritten = Arc::new(rewritten);

    let readers: Vec<JoinHandle<()>> = (0..READERS)
        .map(|tid| {
            let snapshot = Arc::clone(&snapshot);
            let stop = Arc::clone(&stop);
            let reads = Arc::clone(&reads);
            let errors = Arc::clone(&errors);
            let originals = originals.clone();
            let rewritten = Arc::clone(&rewritten);
            thread::spawn(move || {
                let legal: Vec<RecordValue> = (0..nrow).map(|r| rewritten(r)).collect();
                while !stop.load(Ordering::Relaxed) {
                    match snapshot.getcol(0, 0, nrow) {
                        Ok(vals) => {
                            for (r, v) in vals.iter().enumerate() {
                                let ok = if strict {
                                    *v == originals[r]
                                } else {
                                    *v == originals[r] || *v == legal[r]
                                };
                                if !ok {
                                    errors.lock().unwrap().push(format!(
                                        "t{tid}: row {r} corrupted: got {v:?}, \
                                         original {:?}, rewritten {:?}",
                                        originals[r], legal[r]
                                    ));
                                    return;
                                }
                            }
                            reads.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            errors
                                .lock()
                                .unwrap()
                                .push(format!("t{tid} read failed: {e}"));
                            return;
                        }
                    }
                    snapshot.drop_data_file_pages();
                }
            })
        })
        .collect();

    // The writer: each cycle a fresh open-for-update, one new cell, flush
    // (the per-chunk dask-ms write pattern).
    for it in 0..WRITES {
        let (_read, mut wt) = WritableTable::open_for_update(dir).unwrap();
        let row = (u64::from(it) * 7) % nrow;
        wt.putcell(0, row, rewritten(row)).unwrap();
        wt.flush().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for h in readers {
        h.join().unwrap();
    }
    let errors = errors.lock().unwrap();
    assert!(
        errors.is_empty(),
        "reader snapshot was violated: {:?}",
        &errors[..errors.len().min(10)]
    );
    assert!(
        reads.load(Ordering::Relaxed) > 0,
        "readers never completed a pass"
    );

    // The writer's overlay is fully visible to a fresh open: written rows
    // carry the rewritten value, untouched rows the original.
    let mut expect = originals;
    for it in 0..WRITES {
        let row = (u64::from(it) * 7) % nrow;
        expect[row as usize] = rewritten(row);
    }
    let after = Table::open(dir, true).unwrap();
    let got = after.getcol(0, 0, nrow).unwrap();
    for (r, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
        assert_eq!(g, e, "fresh open row {r} after concurrent flushes");
    }
}

/// StandardStMan double scalars: flushes byte-patch the buckets in place,
/// so a same-process unlocked reader may see a cell's new value (as in
/// casacore) but never a torn or misaligned one.
#[test]
fn reader_snapshot_survives_concurrent_ssm_scalar_flushes() {
    let dir = temp_dir("ssm-scalar");
    let nrow = 512;
    let originals: Vec<RecordValue> = (0..nrow).map(|r| RecordValue::Double(r as f64)).collect();
    stress_readers_vs_writer(
        &dir,
        &ssm_double_desc(),
        originals,
        |row| RecordValue::Double(1_000_000.0 + row as f64),
        false,
    );
}

/// StandardStMan strings: variable-size cells force a whole-column rebuild
/// on every flush — an atomic inode replacement, so a reader's snapshot must
/// stay EXACTLY the values it captured.
#[test]
fn reader_snapshot_survives_concurrent_string_rebuilds() {
    let dir = temp_dir("ssm-string");
    let nrow = 256;
    let originals: Vec<RecordValue> = (0..nrow)
        .map(|r| RecordValue::String(format!("original-row-{r}")))
        .collect();
    stress_readers_vs_writer(
        &dir,
        &ssm_string_desc(),
        originals,
        |row| RecordValue::String(format!("rewritten-{row}-with-a-longer-value")),
        true,
    );
}

/// TiledColumnStMan float arrays (the MS DATA layout): flushes byte-patch
/// tile runs in place under the flush gate, so a reader observes each cell
/// wholly before or wholly after a patch — never a cell torn mid-array by
/// a page boundary (or a multi-tile cell patched piece by piece).
#[test]
fn reader_snapshot_survives_concurrent_tsm_flushes() {
    let dir = temp_dir("tsm-array");
    let nrow = 512;
    let cell = |row: i32, salt: i32| {
        RecordValue::Array(ArrayValue {
            shape: vec![3, 2],
            data: ArrayData::Float((0..6).map(|k| (row * 100 + k * 7 + salt) as f32).collect()),
        })
    };
    let originals: Vec<RecordValue> = (0..nrow).map(|r| cell(r, 0)).collect();
    stress_readers_vs_writer(
        &dir,
        &tsm_float_desc(),
        originals,
        move |row| cell(row as i32, 97),
        false,
    );
}
