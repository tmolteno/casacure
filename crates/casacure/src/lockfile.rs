//! casacore's table locking protocol, byte- and behaviour-compatible with
//! `casacore/casa/IO/LockFile` and `casacore/tables/Tables/TableLock*`.
//!
//! A CASA table is coordinated between processes through fcntl record locks
//! on `<tabledir>/table.lock`:
//!
//! - **byte 0** (1 byte): the lock itself — a shared read lock (`F_RDLCK`)
//!   for readers, an exclusive write lock (`F_WRLCK`) for writers.
//! - **byte 1** (1 byte): the "in use" lock, taken shared by every process
//!   that has the table open; trying a write lock on it (`is_multi_used`)
//!   tells whether another process holds the table open.
//! - **bytes 1-2** (2 bytes): the same region widened while a *permanent*
//!   lock is held (the third byte marks permanence).
//!
//! The file body holds a 260-byte request list (`u32` count + up to 32
//! `(pid, hostId)` pairs) used as the courtesy flag that makes an
//! `AutoLocking` holder release early, followed by the length-prefixed
//! *info* — an AipsIO stream whose root object is the table's `sync`
//! record (authoritative `nrrow`, column count and change counters).
//!
//! POSIX closes a process's *entire* fcntl lock set on a file whenever any
//! fd to that file is closed, so every `table.lock` in this process is
//! opened once and shared: [`attach`] returns an `Arc<Mutex<LockFile>>`
//! whose fd stays alive for as long as any handle references it. Nothing
//! else may `std::fs`-open `table.lock` while a lock is held.
//!
//! Windows attaches nothing for now (the stubs at the bottom): every lock
//! request succeeds, while the sync record stays byte-compatible.

use crate::aipsio::{Reader, Writer};
#[cfg(windows)]
use std::io;
#[cfg(windows)]
use std::path::Path;
use std::sync::{Arc, Mutex};

/// `LockFile::SIZEREQID`: the fixed-size request list heading the file.
pub const SIZE_REQ_ID: usize = (1 + 2 * 32) * 4;

/// `TableLock::LockOption`, in casacore's numeric order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    PermanentLocking,
    PermanentLockingWait,
    AutoLocking,
    UserLocking,
    AutoNoReadLocking,
    UserNoReadLocking,
    NoLocking,
    DefaultLocking,
}

impl LockMode {
    /// The canonical python-casacore option word.
    pub fn as_str(self) -> &'static str {
        match self {
            LockMode::PermanentLocking => "permanent",
            LockMode::PermanentLockingWait => "permanentwait",
            LockMode::AutoLocking => "auto",
            LockMode::UserLocking => "user",
            LockMode::AutoNoReadLocking => "autonoread",
            LockMode::UserNoReadLocking => "usernoread",
            LockMode::NoLocking => "nolock",
            LockMode::DefaultLocking => "default",
        }
    }

    pub fn parse(s: &str) -> Option<LockMode> {
        let mode = match s {
            "default" => LockMode::DefaultLocking,
            "auto" => LockMode::AutoLocking,
            "autonoread" => LockMode::AutoNoReadLocking,
            "user" => LockMode::UserLocking,
            "usernoread" => LockMode::UserNoReadLocking,
            "permanent" => LockMode::PermanentLocking,
            "permanentwait" => LockMode::PermanentLockingWait,
            "nolock" => LockMode::NoLocking,
            _ => return None,
        };
        Some(mode)
    }

    fn is_permanent(self) -> bool {
        matches!(
            self,
            LockMode::PermanentLocking | LockMode::PermanentLockingWait
        )
    }
}

/// `TableLock`: the locking mode plus its inspection interval (seconds) and
/// maximum wait, with casacore's defaults (interval 5 s, wait indefinitely)
/// and normalizations (`TableLock::init`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockOptions {
    pub mode: LockMode,
    pub interval: u32,
    pub maxwait: u32,
}

impl LockOptions {
    /// `LockOptions::default()`: `DefaultLocking` with casacore's defaults.
    pub fn locking_default() -> LockOptions {
        LockOptions {
            mode: LockMode::DefaultLocking,
            interval: 5,
            maxwait: 0,
        }
    }

    pub fn no_locking() -> LockOptions {
        LockOptions {
            mode: LockMode::NoLocking,
            interval: 5,
            maxwait: 0,
        }
    }

    /// `TableLock::init`: resolve `DefaultLocking` -> `AutoLocking`, fold the
    /// `*NoReadLocking` variants into their base mode, and drop read locking
    /// for `NoLocking`.
    pub fn effective(self) -> EffectiveLockOptions {
        let mut o = self;
        let mut read_locking = true;
        match o.mode {
            LockMode::DefaultLocking => o.mode = LockMode::AutoLocking,
            LockMode::AutoNoReadLocking => {
                o.mode = LockMode::AutoLocking;
                read_locking = false;
            }
            LockMode::UserNoReadLocking => {
                o.mode = LockMode::UserLocking;
                read_locking = false;
            }
            LockMode::NoLocking => read_locking = false,
            _ => {}
        }
        EffectiveLockOptions {
            mode: o.mode,
            interval: self.interval,
            maxwait: self.maxwait,
            read_locking,
        }
    }
}

/// A `LockOptions` with the `DefaultLocking` / `*NoReadLocking` cases
/// resolved (`TableLock` after `init()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveLockOptions {
    pub mode: LockMode,
    pub interval: u32,
    pub maxwait: u32,
    pub read_locking: bool,
}

impl EffectiveLockOptions {
    pub fn is_permanent(&self) -> bool {
        self.mode.is_permanent()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    Read,
    Write,
}

/// The table `sync` record carried in the lock file's info area
/// (`TableSyncData`): authoritative row count, column count and the change
/// counters casacore compares to decide how much to resync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSyncData {
    pub nrrow: u64,
    /// `-1` in the short form (counters omitted; casacore then treats the
    /// table as changed in every respect).
    pub nrcolumn: i32,
    pub modify_counter: u32,
    pub table_change_counter: u32,
    pub dm_counters: Vec<u32>,
}

impl TableSyncData {
    /// Parse the info bytes (an AipsIO stream whose root object is `sync`).
    pub fn parse(info: &[u8]) -> Result<TableSyncData, String> {
        let mut r = Reader::new(info);
        let (version, _) = r
            .read_object(true, "sync")
            .map_err(|e| format!("table.lock sync record: {e}"))?;
        let nrrow = match version {
            1 => u64::from(r.read_u32().map_err(|e| e.to_string())?),
            2 => r.read_u64().map_err(|e| e.to_string())?,
            v => return Err(format!("unsupported table.lock sync version {v}")),
        };
        let nrcolumn = r.read_i32().map_err(|e| e.to_string())?;
        let modify_counter = r.read_u32().map_err(|e| e.to_string())?;
        let (table_change_counter, dm_counters) = if nrcolumn >= 0 {
            let tcc = r.read_u32().map_err(|e| e.to_string())?;
            // Nested Block object: [len]["Block"][ver=1][u32 nelem][...]
            let (bver, _) = r
                .read_object(false, "Block")
                .map_err(|e| format!("table.lock sync Block: {e}"))?;
            if bver != 1 {
                return Err(format!("unsupported table.lock Block version {bver}"));
            }
            let nelem = r.read_u32().map_err(|e| e.to_string())? as usize;
            let mut counters = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                counters.push(r.read_u32().map_err(|e| e.to_string())?);
            }
            (tcc, counters)
        } else {
            (0, Vec::new())
        };
        Ok(TableSyncData {
            nrrow,
            nrcolumn,
            modify_counter,
            table_change_counter,
            dm_counters,
        })
    }

    /// Serialize as the AipsIO stream putInfo stores (root `sync` object;
    /// the huge-nrrow v2 form above `u32::MAX` like casacore's
    /// `TableSyncData::write`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        let version = if self.nrrow > u64::from(u32::MAX) {
            2
        } else {
            1
        };
        w.put_root_object_start("sync", version);
        if version == 1 {
            w.put_u32(self.nrrow as u32);
        } else {
            w.put_u64(self.nrrow);
        }
        w.put_i32(self.nrcolumn);
        w.put_u32(self.modify_counter);
        if self.nrcolumn >= 0 {
            w.put_u32(self.table_change_counter);
            w.put_object_start("Block", 1);
            w.put_u32(self.dm_counters.len() as u32);
            for c in &self.dm_counters {
                w.put_u32(*c);
            }
            w.put_object_end();
        }
        w.put_object_end();
        w.into_bytes()
    }
}

pub type SharedLockFile = Arc<Mutex<LockFile>>;

// The fd-backed implementation: POSIX fcntl record locks (Linux, macOS).
// Windows currently attaches nothing (see the stubs at the bottom): every
// lock request succeeds, while the sync record stays byte-compatible.
#[cfg(unix)]
mod posix {
    use super::{EffectiveLockOptions, LockMode, LockType, TableSyncData, SIZE_REQ_ID};
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::{Arc, Mutex, OnceLock, Weak};
    use std::time::{Duration, Instant};

    /// `FileLocker`: an fcntl record lock over `[start, start+len)` of an fd.
    #[derive(Debug)]
    pub struct FileLocker {
        fd: std::os::unix::io::RawFd,
        start: i64,
        len: i64,
    }

    /// One-time note that the filesystem refuses record locks (casacore's
    /// broken-NFS workaround: `ENOLCK` counts as success).
    fn enolck_note() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static NOTED: AtomicBool = AtomicBool::new(false);
        if !NOTED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "LockFile: locks not supported on this filesystem (ENOLCK); proceeding without"
            );
        }
    }

    impl FileLocker {
        pub fn new(fd: std::os::unix::io::RawFd, start: i64, len: i64) -> FileLocker {
            FileLocker { fd, start, len }
        }

        /// One fcntl attempt. `Ok(true)` = acquired. `Ok(false)` = held by
        /// someone else (`EAGAIN`/`EACCES`). `Err` = hard error (incl.
        /// `EBADF` for a write lock on a read-only fd).
        fn try_flock(&self, l_type: libc::c_short) -> io::Result<bool> {
            let mut fl: libc::flock = unsafe { std::mem::zeroed() };
            fl.l_type = l_type;
            fl.l_whence = libc::SEEK_SET as libc::c_short;
            fl.l_start = self.start;
            fl.l_len = self.len;
            let rc = unsafe { libc::fcntl(self.fd, libc::F_SETLK, &fl) };
            if rc == -1 {
                let err = io::Error::last_os_error();
                match err.raw_os_error() {
                    Some(libc::EAGAIN) | Some(libc::EACCES) => return Ok(false),
                    Some(libc::ENOLCK) => {
                        enolck_note();
                        return Ok(true);
                    }
                    _ => return Err(err),
                }
            }
            Ok(true)
        }

        /// `FileLocker::acquire`: `nattempts == 0` blocks (`F_SETLKW`); any
        /// other value is that many non-blocking `F_SETLK` tries one second
        /// apart. `Ok(false)` = gave up.
        pub fn acquire(&self, typ: LockType, nattempts: u32) -> io::Result<bool> {
            let l_type = match typ {
                LockType::Read => libc::F_RDLCK as libc::c_short,
                LockType::Write => libc::F_WRLCK as libc::c_short,
            };
            if nattempts == 0 {
                let mut fl: libc::flock = unsafe { std::mem::zeroed() };
                fl.l_type = l_type;
                fl.l_whence = libc::SEEK_SET as libc::c_short;
                fl.l_start = self.start;
                fl.l_len = self.len;
                let rc = unsafe { libc::fcntl(self.fd, libc::F_SETLKW, &fl) };
                if rc == -1 {
                    let err = io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::ENOLCK) {
                        enolck_note();
                        return Ok(true);
                    }
                    return Err(err);
                }
                return Ok(true);
            }
            for i in 0..nattempts {
                match self.try_flock(l_type)? {
                    true => return Ok(true),
                    false => {
                        if i + 1 < nattempts {
                            std::thread::sleep(Duration::from_secs(1));
                        }
                    }
                }
            }
            Ok(false)
        }

        pub fn release(&self) -> io::Result<()> {
            self.try_flock(libc::F_UNLCK as libc::c_short).map(|_| ())
        }

        /// Whether an exclusive lock could be taken right now (used both for
        /// the byte-1 "in use" probe and for read-lock holders to detect a
        /// concurrent writer). Leaves no lock behind.
        pub fn can_write_lock(&self) -> io::Result<bool> {
            if !self.try_flock(libc::F_WRLCK as libc::c_short)? {
                return Ok(false);
            }
            self.release()?;
            Ok(true)
        }
    }

    /// Shared per-process state of one open `table.lock`.
    #[derive(Debug)]
    pub struct LockFile {
        path: PathBuf,
        file: File,
        /// Opened read-only (or missing): no write locks can be taken; lock
        /// requests degrade to success-without-locking like casacore's
        /// no-locking `LockFile` mode.
        pub read_only_fd: bool,
        /// The lock file did not exist at attach: every request succeeds
        /// without touching the filesystem (casacore `mustExist=False`).
        pub missing: bool,
        /// The "in use" / permanent marker locker (byte 1; bytes 1-2 for
        /// permanent locking), held shared for the handle's lifetime.
        use_locker: FileLocker,
        /// The read/write locker (byte 0).
        main_locker: FileLocker,
        interval: u32,
        inspect_count: u32,
        last_inspect: Option<Instant>,
        /// The main lock currently acquired (process-wide; mirrors what this
        /// process last acquired through any handle of this file).
        pub held: Option<LockType>,
    }

    impl LockFile {
        /// Open (or create) `<dir>/table.lock` and take the shared "in use"
        /// lock, mirroring the `LockFile` constructor (`create`, `mustExist`,
        /// `permLocking` map onto the use-locker width). Returns `Ok(None)`
        /// for `NoLocking`.
        pub fn attach(
            dir: &Path,
            options: &EffectiveLockOptions,
            create: bool,
        ) -> io::Result<Option<LockFile>> {
            if options.mode == LockMode::NoLocking {
                return Ok(None);
            }
            let path = dir.join("table.lock");
            let file = if create {
                // A create may run before the table directory itself exists
                // (the writable handle is built, then the first flush writes
                // the files).
                let _ = std::fs::create_dir_all(dir);
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&path)?
            } else {
                match OpenOptions::new().read(true).write(true).open(&path) {
                    Ok(f) => f,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {
                        // casacore `mustExist=False`: every lock request
                        // succeeds without doing actual locking.
                        return Ok(Some(LockFile {
                            path,
                            file: no_underlying_file()?,
                            read_only_fd: true,
                            missing: true,
                            use_locker: FileLocker::new(-1, 1, 1),
                            main_locker: FileLocker::new(-1, 0, 1),
                            interval: options.interval,
                            inspect_count: 0,
                            last_inspect: None,
                            held: None,
                        }));
                    }
                    Err(e2) if rw_open_failed_kind(&e2) => {
                        // Cannot open read/write (read-only medium): readonly
                        // fd — read locks still work, write locks degrade.
                        OpenOptions::new().read(true).open(&path)?
                    }
                    Err(e) => return Err(e),
                }
            };
            let fd = file.as_raw_fd();
            // Write locks need a writable fd (fcntl F_WRLCK fails with EBADF
            // otherwise); probe rather than trust the open mode.
            let read_only_fd = !can_write(&file);
            let use_locker = FileLocker::new(fd, 1, if options.is_permanent() { 2 } else { 1 });
            let main_locker = FileLocker::new(fd, 0, 1);
            // The shared "in use" lock, as `LockFile`'s constructor takes.
            let _ = use_locker.acquire(LockType::Read, 1);
            let mut lf = LockFile {
                path,
                file,
                read_only_fd,
                missing: false,
                use_locker,
                main_locker,
                interval: options.interval,
                inspect_count: 0,
                last_inspect: None,
                held: None,
            };
            if create {
                lf.init_empty_file()?;
            }
            Ok(Some(lf))
        }

        pub fn path(&self) -> &Path {
            &self.path
        }

        /// `LockFile::isMultiUsed`: another process has the table open.
        pub fn is_multi_used(&self) -> io::Result<bool> {
            if self.missing || self.read_only_fd {
                // Without a writable fd the probe itself cannot run; report
                // the honest "not observable" answer casacore's no-locking
                // mode gives.
                return Ok(false);
            }
            Ok(!self.use_locker.can_write_lock()?)
        }

        /// `LockFile::acquire` (+ `addReqId` while retrying). `nattempts ==
        /// 0` blocks. `Ok(false)` = gave up after `nattempts`.
        ///
        /// A write lock already held by this process satisfies any request
        /// without touching fcntl — re-acquiring a read lock over it would
        /// *downgrade* the region (`FileLocker::acquire`'s write-lock probe).
        pub fn acquire(&mut self, typ: LockType, nattempts: u32) -> io::Result<bool> {
            if self.missing || (typ == LockType::Write && self.read_only_fd) {
                self.held = Some(typ);
                return Ok(true);
            }
            if self.held == Some(LockType::Write) {
                return Ok(true);
            }
            if self.held == Some(typ) {
                return Ok(true);
            }
            let mut succ = self.main_locker.acquire(typ, 1)?;
            if !succ && nattempts != 1 {
                self.add_req_id();
                succ = self.main_locker.acquire(typ, nattempts)?;
                self.remove_req_id();
            }
            if succ {
                self.held = Some(typ);
            }
            Ok(succ)
        }

        /// Release a *read* hold. No-op while a write lock is held — it
        /// covers the read case, and unlocking the byte would drop the
        /// writer's exclusion for the whole process.
        pub fn release_read(&mut self) -> io::Result<()> {
            if self.held != Some(LockType::Read) {
                return Ok(());
            }
            if !self.missing {
                self.main_locker.release()?;
            }
            self.held = None;
            Ok(())
        }

        /// `LockFile::release`: drop the write lock, first storing the sync
        /// info when given (the `TableLockData::release` callback). No-op
        /// when no write lock is held.
        pub fn release_write(&mut self, info: Option<&TableSyncData>) -> io::Result<()> {
            if self.held != Some(LockType::Write) {
                return Ok(());
            }
            if let Some(data) = info {
                if !self.missing && !self.read_only_fd {
                    self.put_info(data)?;
                }
            }
            if !self.missing {
                self.main_locker.release()?;
            }
            self.held = None;
            Ok(())
        }

        /// `TableLockData::autoRelease` + `LockFile::inspect`: throttled
        /// check (every 25th call and at most once per interval) for a
        /// waiting process in the request list.
        pub fn inspect_has_waiter(&mut self, always: bool) -> io::Result<bool> {
            if !always {
                if self.interval > 0 && self.inspect_count < 25 {
                    self.inspect_count += 1;
                    return Ok(false);
                }
                self.inspect_count = 0;
                if self.interval > 0 {
                    if let Some(t) = self.last_inspect {
                        if t.elapsed() < Duration::from_secs(u64::from(self.interval)) {
                            return Ok(false);
                        }
                    }
                }
            }
            let nr = self.nr_req_id()?;
            self.last_inspect = Some(Instant::now());
            Ok(nr > 0)
        }

        /// `LockFile::getInfo`: the info length at `SIZE_REQ_ID`, then the
        /// stream. Missing/short info (a just-created or foreign file) ->
        /// `None`.
        pub fn get_info(&self) -> Result<Option<TableSyncData>, String> {
            if self.missing {
                return Ok(None);
            }
            let mut len_buf = [0u8; 4];
            pread_exact(&self.file, SIZE_REQ_ID as u64, &mut len_buf).map_err(|e| e.to_string())?;
            let len = u32::from_be_bytes(len_buf) as usize;
            if len == 0 || len > 64 * 1024 * 1024 {
                return Ok(None);
            }
            let mut info = vec![0u8; len];
            if pread_exact(&self.file, (SIZE_REQ_ID + 4) as u64, &mut info).is_err() {
                return Ok(None);
            }
            TableSyncData::parse(&info).map(Some)
        }

        /// `LockFile::putInfo`: length-prefixed info at `SIZE_REQ_ID` +
        /// fsync. A no-op for a missing lock file (copied tables operate
        /// lock-free, exactly like the retired `patch_lock_nrrow` no-op
        /// contract) and for a read-only fd.
        pub fn put_info(&self, data: &TableSyncData) -> io::Result<()> {
            if self.missing || self.read_only_fd {
                return Ok(());
            }
            let bytes = data.to_bytes();
            let mut buf = Vec::with_capacity(4 + bytes.len());
            buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(&bytes);
            pwrite_all(&self.file, SIZE_REQ_ID as u64, &buf)?;
            self.file.sync_all().ok();
            Ok(())
        }

        /// A fresh lock file: empty request list, empty info (casacore's
        /// constructor writes the 260 zero bytes; the first released sync
        /// fills the info).
        fn init_empty_file(&mut self) -> io::Result<()> {
            let len = self.file.metadata()?.len();
            if len < SIZE_REQ_ID as u64 {
                let zeros = vec![0u8; SIZE_REQ_ID];
                pwrite_all(&self.file, 0, &zeros)?;
                self.file.sync_all().ok();
            }
            Ok(())
        }

        fn nr_req_id(&self) -> io::Result<u32> {
            let mut buf = [0u8; 4];
            pread_exact(&self.file, 0, &mut buf)?;
            Ok(u32::from_be_bytes(buf))
        }

        /// `LockFile::addReqId`: append `(pid, 0)` to the request list.
        fn add_req_id(&mut self) {
            if self.missing || self.read_only_fd {
                return;
            }
            let count = self.nr_req_id().unwrap_or(0);
            if usize::try_from(count).map(|c| c >= 32).unwrap_or(true) {
                return;
            }
            let mut entry = Vec::with_capacity(8);
            entry.extend_from_slice(&(process::id() as i32).to_be_bytes());
            entry.extend_from_slice(&0i32.to_be_bytes());
            let off = (4 + 8 * count) as u64;
            let _ = pwrite_all(&self.file, off, &entry);
            let _ = pwrite_all(&self.file, 0, &(count + 1).to_be_bytes());
        }

        /// `LockFile::removeReqId`: drop our entry and earlier stale pids.
        fn remove_req_id(&mut self) {
            if self.missing || self.read_only_fd {
                return;
            }
            let count = match self.nr_req_id() {
                Ok(c) if c > 0 && c <= 32 => c as usize,
                _ => return,
            };
            let mut buf = vec![0u8; 4 + 8 * count];
            if pread_exact(&self.file, 0, &mut buf).is_err() {
                return;
            }
            let mut kept: Vec<[u8; 8]> = Vec::with_capacity(count);
            for i in 0..count {
                let e = &buf[4 + 8 * i..4 + 8 * (i + 1)];
                let pid = i32::from_be_bytes(e[0..4].try_into().unwrap());
                let stale = process_dead(pid);
                if !stale && pid != process::id() as i32 {
                    kept.push(e.try_into().unwrap());
                }
            }
            let mut out = Vec::with_capacity(4 + 8 * kept.len());
            out.extend_from_slice(&(kept.len() as u32).to_be_bytes());
            for e in kept {
                out.extend_from_slice(&e);
            }
            let _ = pwrite_all(&self.file, 0, &out);
        }
    }

    fn can_write(f: &File) -> bool {
        // The only portable way to know the fd is writable is to try: probe
        // an fcntl write lock over a byte we do not otherwise use (the tail
        // of the request list area is file data, so probe far beyond any
        // real file — fcntl locks work past EOF). Byte at offset 1<<20.
        matches!(
            FileLocker::new(f.as_raw_fd(), 1 << 20, 1).can_write_lock(),
            Ok(true)
        )
    }

    fn rw_open_failed_kind(e: &io::Error) -> bool {
        e.kind() == io::ErrorKind::PermissionDenied
            || e.raw_os_error() == Some(libc::EROFS)
            || e.raw_os_error() == Some(libc::EACCES)
    }

    fn no_underlying_file() -> io::Result<File> {
        // A placeholder for the missing-file case; never used for I/O
        // because every path checks `missing` first.
        File::open("/dev/null")
    }

    fn process_dead(pid: i32) -> bool {
        if pid <= 0 {
            return true;
        }
        let rc = unsafe { libc::kill(pid, 0) };
        rc == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }

    fn pread_exact(f: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut off = offset;
        let mut done = 0usize;
        while done < buf.len() {
            let n = unsafe {
                libc::pread(
                    f.as_raw_fd(),
                    buf[done..].as_mut_ptr().cast(),
                    buf.len() - done,
                    off as libc::off_t,
                )
            };
            if n < 0 {
                return Err(io::Error::last_os_error());
            }
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "pread"));
            }
            done += n as usize;
            off += n as u64;
        }
        Ok(())
    }

    fn pwrite_all(f: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
        let mut off = offset;
        let mut done = 0usize;
        while done < buf.len() {
            let n = unsafe {
                libc::pwrite(
                    f.as_raw_fd(),
                    buf[done..].as_ptr().cast(),
                    buf.len() - done,
                    off as libc::off_t,
                )
            };
            if n < 0 {
                return Err(io::Error::last_os_error());
            }
            done += n as usize;
            off += n as u64;
        }
        Ok(())
    }

    /// One `table.lock` per path per process, shared by every handle: the
    /// fd must stay open while *any* lock is held, and POSIX drops the
    /// process's locks the moment any fd to the file is closed. Handles
    /// therefore share one fd through this registry; the fd (and its locks)
    /// live until the last referencing handle drops.
    static LOCK_REGISTRY: OnceLock<
        Mutex<std::collections::HashMap<PathBuf, Weak<Mutex<LockFile>>>>,
    > = OnceLock::new();

    pub type SharedLockFile = Arc<Mutex<LockFile>>;

    /// The process's shared instance for a directory's `table.lock`, when
    /// one exists. Callers about to open `table.lock` transiently must
    /// check here first: POSIX drops a process's record locks when *any* fd
    /// to the file closes, so with a live instance every read must go
    /// through its fd.
    pub(crate) fn lookup(dir: &Path) -> Option<SharedLockFile> {
        let path = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let reg = LOCK_REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let reg = reg.lock().unwrap();
        reg.get(&path).and_then(|weak| weak.upgrade())
    }

    /// Open (or create) the directory's `table.lock`, sharing an existing
    /// instance with other handles of this process. `Ok(None)` = no locking
    /// (`NoLocking`).
    pub fn attach(
        dir: &Path,
        options: &EffectiveLockOptions,
        create: bool,
    ) -> io::Result<Option<SharedLockFile>> {
        if options.mode == LockMode::NoLocking {
            return Ok(None);
        }
        let path = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let reg = LOCK_REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let mut reg = reg.lock().unwrap();
        if let Some(weak) = reg.get(&path) {
            if let Some(arc) = weak.upgrade() {
                return Ok(Some(arc));
            }
        }
        let Some(lf) = LockFile::attach(dir, options, create)? else {
            return Ok(None);
        };
        let arc: SharedLockFile = Arc::new(Mutex::new(lf));
        reg.insert(path, Arc::downgrade(&arc));
        Ok(Some(arc))
    }
}

#[cfg(unix)]
pub(crate) use posix::lookup;
#[cfg(unix)]
pub use posix::{attach, LockFile};

/// Windows: casacore-compatible locking is not implemented yet, so every
/// handle attaches nothing and all lock requests succeed (the no-locking
/// mode); the sync record stays byte-compatible via the portable code
/// above.
#[cfg(windows)]
#[derive(Debug)]
pub struct LockFile {
    pub read_only_fd: bool,
    pub missing: bool,
    pub held: Option<LockType>,
}

#[cfg(windows)]
impl LockFile {
    /// Never runs (attach returns `None`): exists so caller code compiles.
    pub fn acquire(&mut self, typ: LockType, _nattempts: u32) -> io::Result<bool> {
        self.held = Some(typ);
        Ok(true)
    }

    /// Never runs (attach returns `None`).
    pub fn release_read(&mut self) -> io::Result<()> {
        self.held = None;
        Ok(())
    }

    /// Never runs (attach returns `None`).
    pub fn release_write(&mut self, _info: Option<&TableSyncData>) -> io::Result<()> {
        self.held = None;
        Ok(())
    }

    /// Never runs (attach returns `None`).
    pub fn get_info(&self) -> Result<Option<TableSyncData>, String> {
        Ok(None)
    }

    /// Never runs (attach returns `None`).
    pub fn put_info(&self, _data: &TableSyncData) -> io::Result<()> {
        Ok(())
    }

    /// Never runs (attach returns `None`).
    pub fn is_multi_used(&self) -> io::Result<bool> {
        Ok(false)
    }

    /// Never runs (attach returns `None`).
    pub fn inspect_has_waiter(&mut self, _always: bool) -> io::Result<bool> {
        Ok(false)
    }
}

#[cfg(windows)]
pub fn attach(
    _dir: &Path,
    _options: &EffectiveLockOptions,
    _create: bool,
) -> io::Result<Option<SharedLockFile>> {
    Ok(None)
}

#[cfg(windows)]
pub(crate) fn lookup(_dir: &Path) -> Option<SharedLockFile> {
    None
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
    #[cfg(unix)]
    use std::path::PathBuf;

    #[cfg(unix)]
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "casacure-lockfile-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sync_record_round_trips_v1_and_v2() {
        let small = TableSyncData {
            nrrow: 276,
            nrcolumn: 24,
            modify_counter: 20,
            table_change_counter: 2,
            dm_counters: vec![3; 10],
        };
        let bytes = small.to_bytes();
        assert_eq!(TableSyncData::parse(&bytes).unwrap(), small);
        // Root layout: [magic][objlen][namelen=4]["sync"][version=1].
        assert_eq!(&bytes[8..12], &[0, 0, 0, 4]);
        assert_eq!(&bytes[12..16], b"sync");
        assert_eq!(&bytes[16..20], &[0, 0, 0, 1]);

        let huge = TableSyncData {
            nrrow: 5_000_000_000,
            nrcolumn: 3,
            modify_counter: 1,
            table_change_counter: 1,
            dm_counters: vec![1, 2],
        };
        let bytes = huge.to_bytes();
        let parsed = TableSyncData::parse(&bytes).unwrap();
        assert_eq!(parsed, huge);
        assert_eq!(parsed.nrrow, 5_000_000_000);
    }

    #[test]
    fn short_form_sync_parses_with_defaults() {
        let mut w = Writer::new();
        w.put_root_object_start("sync", 1);
        w.put_u32(7);
        w.put_i32(-1);
        w.put_u32(4);
        w.put_object_end();
        let parsed = TableSyncData::parse(&w.into_bytes()).unwrap();
        assert_eq!(parsed.nrrow, 7);
        assert_eq!(parsed.nrcolumn, -1);
        assert_eq!(parsed.modify_counter, 4);
        assert_eq!(parsed.table_change_counter, 0);
        assert!(parsed.dm_counters.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn create_then_attach_shares_one_instance() {
        let dir = temp_dir("shared");
        let opts = LockOptions::locking_default().effective();
        let a = attach(&dir, &opts, true).unwrap().unwrap();
        let b = attach(&dir, &opts, true).unwrap().unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert!(a.lock().unwrap().path().ends_with("table.lock"));
        assert!(!a.lock().unwrap().missing);
    }

    #[cfg(unix)]
    #[test]
    fn missing_lock_file_degrades_to_no_locking() {
        let dir = temp_dir("missing");
        let opts = LockOptions::locking_default().effective();
        let lf = attach(&dir, &opts, false).unwrap().unwrap();
        let mut lf = lf.lock().unwrap();
        assert!(lf.missing);
        // Every request succeeds; nothing is written.
        assert!(lf.acquire(LockType::Write, 1).unwrap());
        assert!(lf.get_info().unwrap().is_none());
        lf.release_write(None).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn info_round_trips_through_the_file() {
        let dir = temp_dir("info");
        let opts = LockOptions::locking_default().effective();
        let lf = attach(&dir, &opts, true).unwrap().unwrap();
        let lf = lf.lock().unwrap();
        let data = TableSyncData {
            nrrow: 42,
            nrcolumn: 2,
            modify_counter: 3,
            table_change_counter: 1,
            dm_counters: vec![1, 1],
        };
        lf.put_info(&data).unwrap();
        assert_eq!(lf.get_info().unwrap().unwrap(), data);
    }
}
