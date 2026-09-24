//! Backing store for on-disk table data files.
//!
//! A data file is either an owned byte buffer (small tables, in-memory
//! fixtures, tests) or a memory map (large on-disk tables). Mapping lets a
//! read handle touch only the pages of the rows it actually reads, so
//! chunked dask-ms reads scale memory with the chunk size instead of loading
//! the whole file into RAM per open.

use std::ops::Deref;

use thiserror::Error;

/// An I/O failure together with the file it happened on: `<path>: <cause>`.
///
/// casacore names the file in these failures — `RegularFileIO: error in open or
/// create of file <path>: <cause>` — while casacure used to report only the
/// cause, so a failure surfacing through dask-ms (`ndarray_putcol` ->
/// `table.flush()`) named neither the table nor the block at fault.  Every open
/// in the storage managers knows its own file name, so they report it here and
/// every caller inherits it.
#[derive(Debug, Error)]
#[error("{path}: {source}")]
pub struct FileIoError {
    /// The file the operation failed on.
    pub path: std::path::PathBuf,
    /// What the filesystem said.
    #[source]
    pub source: std::io::Error,
}

impl FileIoError {
    /// Wrap `source`, naming the file it failed on.
    pub fn new(path: impl Into<std::path::PathBuf>, source: std::io::Error) -> Self {
        Self {
            path: path.into(),
            source,
        }
    }
}

/// A data file's bytes.
#[derive(Debug)]
pub enum Buffer {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Buffer {
    /// Back a data file from an open file handle: memory-map it when it has
    /// content (empty files stay owned), keeping the mapping as a stable
    /// snapshot for the lifetime of the handle.
    pub fn from_file(file: std::fs::File) -> std::io::Result<Buffer> {
        let len = file.metadata()?.len();
        if len == 0 {
            return Ok(Buffer::Owned(Vec::new()));
        }
        // SAFETY: the mapping is read-only and outlives no other reference;
        // casacure keeps one opened file per handle, and a handle is an open
        // snapshot (documented: a concurrent flush rewrites the file, so a
        // stale handle reads its own captured state).
        let map = unsafe { memmap2::Mmap::map(&file) }?;
        // Streaming access (dask-ms chunked full-column scans): tell the
        // kernel the mapping is read sequentially so it frees pages it has
        // finished with, keeping a full pass resident at ~the working set
        // instead of the whole file — the analogue of casacore's bounded
        // LRU storage-manager cache. Best-effort; a kernel without the hint
        // is unaffected. Trade-off (documented in MEMORY.md): pages freed
        // after their first pass are re-faulted if re-read, so repeated
        // random single-cell access on a huge file loses that page cache.
        #[cfg(unix)]
        {
            let _ = map.advise(memmap2::Advice::Sequential);
        }
        Ok(Buffer::Mapped(map))
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            Buffer::Owned(v) => v,
            Buffer::Mapped(m) => m,
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Drop the mapping's pages from the page cache (`madvise(MADV_DONTNEED)`).
    /// The cell data has already been copied into the caller's buffer, so a
    /// later read simply re-faults the pages from disk. This keeps a long
    /// streaming scan (dask-ms chunked full-column reads) resident at ~the
    /// current chunk instead of the whole file — the analogue of casacore's
    /// bounded LRU storage-manager cache. Best-effort and unix-only; a
    /// mapping whose pages are re-read later pays page faults again, so
    /// callers only invoke this after bulk reads that have consumed the data.
    pub fn drop_pages(&self) {
        #[cfg(unix)]
        if let Buffer::Mapped(map) = self {
            // SAFETY: `MADV_DONTNEED` only discards clean file-backed pages
            // that are re-faulted from disk on the next read; the cells were
            // already copied into the caller's buffer, so nothing is lost.
            let _ = unsafe { map.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        }
    }
}

impl From<Vec<u8>> for Buffer {
    fn from(v: Vec<u8>) -> Self {
        Buffer::Owned(v)
    }
}

impl Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
