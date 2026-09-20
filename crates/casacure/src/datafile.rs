//! Backing store for on-disk table data files.
//!
//! A data file is either an owned byte buffer (small tables, in-memory
//! fixtures, tests) or a memory map (large on-disk tables). Mapping lets a
//! read handle touch only the pages of the rows it actually reads, so
//! chunked dask-ms reads scale memory with the chunk size instead of loading
//! the whole file into RAM per open.

use std::ops::Deref;

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
