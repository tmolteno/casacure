//! Cross-process fcntl locking is a POSIX facility; Windows attaches
//! nothing yet (see `src/lockfile.rs`).
#![cfg(unix)]

//! Cross-process tests for casacore's locking protocol. fcntl record locks
//! never conflict within one process, so every contention scenario here
//! runs a child copy of this test binary (the `locking_child` entry) under
//! an env-selected role.

use casacure::lockfile::{attach, LockMode, LockOptions, TableSyncData};
use casacure::record::RecordValue;
use casacure::tabledesc::TableDesc;
use casacure::{Table, WritableTable};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "casacure-locking-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn scalar_desc() -> TableDesc {
    let json = r#"{"C": {"valueType": "double", "dataManagerType": "StandardStMan",
        "dataManagerGroup": "S", "option": 0}}"#;
    TableDesc::from_desc_json(json).unwrap()
}

fn create_table(dir: &Path, nrow: u64) {
    let desc = scalar_desc();
    let values = vec![vec![RecordValue::Double(1.5); nrow as usize]];
    Table::create_with_lock(dir, &desc, &values, LockOptions::no_locking()).unwrap();
}

fn wait_for(path: &Path, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

/// Child entry: runs one role selected by `CASACURE_LOCK_CHILD`. In the
/// normal suite run the env is unset and this test is a fast no-op.
#[test]
fn locking_child() {
    let Ok(role) = std::env::var("CASACURE_LOCK_CHILD") else {
        return;
    };
    let dir = PathBuf::from(std::env::var("CASACURE_LOCK_DIR").unwrap());
    let ready = dir.join(format!("ready-{role}"));
    let stop = dir.join(format!("stop-{role}"));
    match role.as_str() {
        // Hold a permanent write lock until told to stop.
        "hold-write" => {
            let mut t = Table::open_with_lock(
                &dir,
                false,
                LockOptions {
                    mode: LockMode::PermanentLockingWait,
                    ..LockOptions::locking_default()
                },
            )
            .unwrap();
            assert!(t.has_lock(true));
            std::fs::write(&ready, b"").unwrap();
            while !stop.exists() {
                std::thread::sleep(Duration::from_millis(20));
            }
            t.unlock();
        }
        // Open for update (permanent write lock), grow the table, flush —
        // the sync record must become visible to other processes — and
        // hold the lock until told to stop.
        "grow-and-hold" => {
            let (read, mut wt) = WritableTable::open_for_update_with_lock(
                &dir,
                LockOptions {
                    mode: LockMode::PermanentLockingWait,
                    ..LockOptions::locking_default()
                },
            )
            .unwrap();
            let base = read.nrows();
            wt.addrows(2);
            for r in base..base + 2 {
                wt.putcell(0, r, RecordValue::Double(r as f64)).unwrap();
            }
            wt.flush().unwrap();
            // The (read, wt) pair's snapshot is stale after a flush by
            // contract — the sync record is what other processes see.
            let fresh = Table::open(&dir, true).unwrap();
            assert_eq!(fresh.nrows(), base + 2);
            std::fs::write(&ready, b"").unwrap();
            while !stop.exists() {
                std::thread::sleep(Duration::from_millis(20));
            }
            wt.unlock().unwrap();
        }
        other => panic!("unknown child role {other}"),
    }
    std::fs::remove_file(&ready).ok();
}

fn spawn_child(role: &str, dir: &Path) -> std::process::Child {
    use std::process::Stdio;
    std::process::Command::new(std::env::current_exe().unwrap())
        .args(["locking_child", "--exact", "--nocapture"])
        .env("CASACURE_LOCK_CHILD", role)
        .env("CASACURE_LOCK_DIR", dir)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap()
}

/// Writes the stop marker even when the parent fails, so a child never
/// outlives its test polling for it.
struct StopOnDrop<'a>(&'a Path);
impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        let _ = std::fs::write(self.0, b"");
    }
}

/// A permanent write lock held by another process excludes this process's
/// writable open (one-attempt `permanent` fails; a read open still works),
/// and everything clears when the holder leaves.
#[test]
fn cross_process_permanent_write_exclusion() {
    let dir = temp_dir("excl");
    create_table(&dir, 2);
    let mut child = spawn_child("hold-write", &dir);
    let stop = dir.join("stop-hold-write");
    let _guard = StopOnDrop(&stop);
    assert!(
        wait_for(&dir.join("ready-hold-write"), Duration::from_secs(10)),
        "child never signalled ready"
    );
    // One-attempt writable open fails with casacore's message...
    let err = Table::open_with_lock(
        &dir,
        false,
        LockOptions {
            mode: LockMode::PermanentLocking,
            ..LockOptions::locking_default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("Permanent lock on table"), "{err}");
    // ...and so does a readonly open: its read lock conflicts with the
    // held write lock.
    let ro = Table::open_with_lock(
        &dir,
        true,
        LockOptions {
            mode: LockMode::PermanentLocking,
            ..LockOptions::locking_default()
        },
    );
    assert!(
        ro.is_err(),
        "readonly open must conflict with the write lock"
    );
    // A reader that takes no byte-0 lock (`autonoread`) still sees the
    // table as in use via the byte-1 lock.
    let probe = Table::open_with_lock(
        &dir,
        true,
        LockOptions {
            mode: LockMode::AutoNoReadLocking,
            ..LockOptions::locking_default()
        },
    )
    .unwrap();
    assert!(probe.is_multi_used().unwrap());
    drop(probe);

    std::fs::write(&stop, b"").unwrap();
    child.wait().unwrap();
    // Lock release on close: the table is no longer in use and a writable
    // permanent open succeeds.
    let after = Table::open_with_lock(
        &dir,
        false,
        LockOptions {
            mode: LockMode::PermanentLocking,
            ..LockOptions::locking_default()
        },
    )
    .unwrap();
    assert!(!after.is_multi_used().unwrap());
}

/// A writer holding the lock (with a flushed grow) blocks a reader's
/// `lock()`; once the writer releases, the reader's fresh acquire resyncs
/// its row count from the sync record.
#[test]
fn cross_process_reader_resyncs_after_writer() {
    let dir = temp_dir("resync");
    create_table(&dir, 2);
    // The parent holds a user-mode handle: no lock at rest.
    let mut reader = Table::open_with_lock(
        &dir,
        true,
        LockOptions {
            mode: LockMode::UserLocking,
            ..LockOptions::locking_default()
        },
    )
    .unwrap();
    assert_eq!(reader.nrows(), 2);
    assert!(!reader.has_lock(false));

    let mut child = spawn_child("grow-and-hold", &dir);
    let stop = dir.join("stop-grow-and-hold");
    let _guard = StopOnDrop(&stop);
    assert!(wait_for(
        &dir.join("ready-grow-and-hold"),
        Duration::from_secs(10)
    ));

    // The writer holds the lock: a one-attempt read lock gives up.
    let err = reader.lock(false, 2).unwrap_err();
    assert!(err.to_string().contains("gave up"), "{err}");
    // The snapshot is stale (2 rows) while the writer holds the lock.

    std::fs::write(&stop, b"").unwrap();
    child.wait().unwrap();
    // Blocking acquire after release, and the resync sees 4 rows.
    reader.lock(false, 0).unwrap();
    assert_eq!(reader.nrows(), 4, "lock() must resync from the sync record");
    reader.unlock();
}

/// A table created by [`Table::create_with_lock`] carries a lock file whose
/// sync record matches the row count, so another process opening it sees
/// the right rows without ever reading `table.dat`'s (possibly stale)
/// header.
#[test]
fn created_table_writes_a_sync_record() {
    let dir = temp_dir("syncrec");
    create_table(&dir, 3);
    let lock_bytes = std::fs::read(dir.join("table.lock")).unwrap();
    assert!(
        lock_bytes.len() >= casacure::lockfile::SIZE_REQ_ID,
        "created table.lock = {} bytes",
        lock_bytes.len()
    );
    // Parse via a fresh readonly open: the sync-record preference is
    // exercised by every open, so just check the file is well-formed and
    // the open sees all rows.
    let t = Table::open(&dir, true).unwrap();
    assert_eq!(t.nrows(), 3);
}

/// A `user`-mode handle holds nothing at rest: the open's read lock is
/// released immediately, and only explicit `lock` calls hold one.
#[test]
fn user_mode_open_releases_the_open_lock() {
    let dir = temp_dir("usermode");
    create_table(&dir, 2);
    let mut t = Table::open_with_lock(
        &dir,
        true,
        LockOptions {
            mode: LockMode::UserLocking,
            ..LockOptions::locking_default()
        },
    )
    .unwrap();
    assert_eq!(t.nrows(), 2);
    assert!(!t.has_lock(false));
    assert!(!t.has_lock(true));

    t.lock(true, 0).unwrap();
    assert!(t.has_lock(true));
    assert!(t.has_lock(false)); // a write lock covers reads
                                // A second request is satisfied without nesting.
    t.lock(false, 0).unwrap();
    assert!(t.has_lock(true));

    t.unlock();
    assert!(!t.has_lock(false));
    assert!(!t.has_lock(true));
}

/// A writable flush stores the sync record with the authoritative row
/// count and column count while PRESERVING the change counters — the
/// contract that lets a casacore reader resync rows cheaply instead of
/// re-reading the whole table.
#[test]
fn writable_flush_stores_the_sync_record_and_preserves_counters() {
    let dir = temp_dir("syncrecord");
    create_table(&dir, 2);
    // Seed a distinctive record (as a previous casacore session would have
    // left it).
    let opts = LockOptions::locking_default().effective();
    let seed = TableSyncData {
        nrrow: 2,
        nrcolumn: 1,
        modify_counter: 7,
        table_change_counter: 5,
        dm_counters: vec![2, 3],
    };
    attach(&dir, &opts, false)
        .unwrap()
        .unwrap()
        .lock()
        .unwrap()
        .put_info(&seed)
        .unwrap();

    let (_read, mut wt) =
        WritableTable::open_for_update_with_lock(&dir, LockOptions::locking_default()).unwrap();
    wt.addrows(2);
    wt.putcell(0, 2, RecordValue::Double(2.0)).unwrap();
    wt.putcell(0, 3, RecordValue::Double(3.0)).unwrap();
    wt.flush().unwrap();

    let record = attach(&dir, &opts, false)
        .unwrap()
        .unwrap()
        .lock()
        .unwrap()
        .get_info()
        .unwrap()
        .expect("flush must write the sync record");
    assert_eq!(record.nrrow, 4);
    assert_eq!(record.nrcolumn, 1);
    assert_eq!(record.modify_counter, 8, "modify counter increments");
    assert_eq!(record.table_change_counter, 5, "table counter preserved");
    assert_eq!(record.dm_counters, vec![2, 3], "DM counters preserved");
    // And a fresh reader sees the new rows.
    assert_eq!(Table::open(&dir, true).unwrap().nrows(), 4);
}

/// Regression: a rewrite through a handle without a lock file attached
/// (the TaQL UPDATE/DELETE path) used to leave the lock file's sync record
/// at the stale row count, so the next open trusted it and failed reads
/// with "row N not covered by any indexed bucket". Every flush must
/// refresh the record — downwards too.
#[test]
fn nolocking_rewrite_keeps_the_sync_record_fresh() {
    let dir = temp_dir("shrink");
    create_table(&dir, 6);
    // Make the record exist (a real session's flush would have).
    let (read, mut wt) =
        WritableTable::open_for_update_with_lock(&dir, LockOptions::locking_default()).unwrap();
    wt.putcell(0, 0, RecordValue::Double(0.0)).unwrap();
    wt.flush().unwrap();
    assert_eq!(read.nrows(), 6);
    drop(read);
    drop(wt);

    // The shrink: a fresh (no-locking) handle rewrites the table with
    // fewer rows — the full-rewrite path.
    let desc = scalar_desc();
    let mut wt = WritableTable::create(&dir, desc);
    wt.addrows(3);
    for r in 0..3 {
        wt.putcell(0, r, RecordValue::Double(r as f64)).unwrap();
    }
    wt.flush().unwrap();

    let t = Table::open(&dir, true).unwrap();
    assert_eq!(t.nrows(), 3, "stale sync record must not shadow table.dat");
    let ids = t.getcol(0, 0, 3).unwrap();
    assert_eq!(ids.len(), 3);
}

/// A read snapshot and a writable backing of the same directory coexist in
/// one process without self-deadlocking (POSIX record locks do not
/// conflict within a process), and a flush under the held write lock
/// neither blocks nor loses the lock.
#[test]
fn read_and_write_handles_coexist_in_one_process() {
    let dir = temp_dir("coexist");
    create_table(&dir, 2);
    let (read, mut wt) =
        WritableTable::open_for_update_with_lock(&dir, LockOptions::locking_default()).unwrap();
    assert_eq!(read.nrows(), 2);

    // Hold the write lock explicitly across flushes (the dask-ms
    // _writelock_runner pattern).
    wt.lock(true, 0).unwrap();
    wt.addrows(1);
    wt.putcell(0, 2, RecordValue::Double(2.0)).unwrap();
    wt.flush().unwrap();
    assert!(wt.has_lock(true), "an explicit hold survives a flush");
    wt.unlock().unwrap();
    assert!(!wt.has_lock(true));

    // Fresh reader: the growth is visible through the sync record.
    assert_eq!(Table::open(&dir, true).unwrap().nrows(), 3);
}
