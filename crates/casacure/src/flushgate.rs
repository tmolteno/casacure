//! The per-table-directory flush gate: excludes readers from a table's
//! files while a same-process writer mutates them in place.
//!
//! Read handles map a table's data files for the handle's lifetime (see
//! [`crate::datafile::Buffer`]), and whole-file rewrites swap in a fresh
//! inode atomically — but the *patch* flush paths (StandardStMan bucket
//! writes, TiledColumnStMan tile runs, array-file record rewrites) change
//! bytes of the mapped inode itself, because their cost must track the
//! written chunk, not the column (`tests/flush_cost.rs`).  A reader
//! decoding a cell while such a write is in flight observes a half-written
//! cell — in particular a cell straddling a page boundary tears, its pages
//! faulting to either side of the write.
//!
//! So every patch's write phase holds the directory's gate for writing,
//! and every read that decodes cells from the mappings (the `Table` read
//! methods, and opening a table directory) holds it for reading.  A reader
//! then observes each cell wholly before or wholly after any one patch —
//! casacore's unlocked-reader semantics, minus the tearing.  The gate is
//! strictly in-process (cross-process overlap remains the fcntl locking
//! protocol's job) and never wraps code that opens a table, so it cannot
//! deadlock against itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// One gate per table directory, alive for the process (the strong
/// reference is deliberate: a table deleted and re-created at the same
/// path must still exclude readers of the old files).
static GATES: OnceLock<Mutex<HashMap<PathBuf, Arc<RwLock<()>>>>> = OnceLock::new();

/// The flush gate of a table directory.  Hold `.read()` while decoding
/// cells from the directory's mapped files, `.write()` while patching
/// them in place.
pub(crate) fn gate(dir: &Path) -> Arc<RwLock<()>> {
    // Canonicalised, as the lock-file registry keys itself: two handles of
    // one table must always land on one gate, whatever path form each used.
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let reg = GATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut reg = reg.lock().unwrap();
    if let Some(g) = reg.get(&key) {
        return Arc::clone(g);
    }
    let g = Arc::new(RwLock::new(()));
    reg.insert(key, Arc::clone(&g));
    g
}
