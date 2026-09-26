//! Growing a table's files in place: appending rows without re-encoding
//! the rows already on disk.
//!
//! dask-ms writes a new table in row chunks, and each chunk either appends
//! its rows (`addrows` then `putcol`) or lands in rows added up front (one
//! `addrows(nrow)`).  Either way the writer flushes a table that is larger
//! than its files.  Regenerating every file from the buffered cells at each
//! such flush made memory grow with the table and time quadratic in it.
//! Instead, every storage manager here extends its files with default rows:
//!
//! - **Tiled (TSM):** tile `t` stays at `t * bucket_size`, so the tile file
//!   is zero-extended (the default of every type) and the header rewritten.
//! - **StandardStMan:** new default buckets are appended after the existing
//!   ones.  A partly filled last bucket is topped up first.  The index is
//!   rewritten to name the new buckets, re-using its bucket chain, and
//!   fixed-shape array columns get default records appended to the array
//!   file.
//! - **IncrementalStMan:** the last bucket is re-encoded with its new rows,
//!   further buckets are appended, and the trailing index is rewritten.
//!
//! The rows then written are patched in by the in-place flush.  Each
//! function has a `dry_run` mode that checks, before anything is written,
//! that the files have a layout it can grow.  It answers `Ok(false)` when
//! they do not, so the caller can keep the whole-table rewrite.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::aipsio::Reader;
use crate::record::{DataType, RecordValue};
use crate::tabledesc::{ColumnDesc, ColumnKind};

/// `Ok(true)`: grown (or growable, in a dry run); `Ok(false)`: this layout
/// cannot be grown in place; `Err`: an I/O failure.
pub(crate) type GrowResult = Result<bool, String>;

/// Buffered writes are issued in pieces of about this size, so growing by a
/// whole table (one `addrows(nrow)`) never holds more than this in memory.
const WRITE_BATCH: usize = 8 << 20;

fn io_err(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

fn write_at(file: &mut std::fs::File, path: &Path, off: u64, bytes: &[u8]) -> Result<(), String> {
    file.seek(SeekFrom::Start(off)).map_err(io_err(path))?;
    file.write_all(bytes).map_err(io_err(path))
}

fn read_at(file: &mut std::fs::File, path: &Path, off: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(off)).map_err(io_err(path))?;
    file.read_exact(&mut buf).map_err(io_err(path))?;
    Ok(buf)
}

fn open_rw(path: &Path) -> Result<std::fs::File, String> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(io_err(path))
}

fn i64_bytes(v: i64, big_endian: bool) -> [u8; 8] {
    if big_endian {
        v.to_be_bytes()
    } else {
        v.to_le_bytes()
    }
}

// ---------------------------------------------------------------- table.dat

/// `(offset, width)` of the two row counts in `table.dat`: the `Table`
/// header's and the ColumnSet's.
fn table_dat_nrow_fields(bytes: &[u8]) -> Option<[(usize, usize); 2]> {
    let mut r = Reader::new(bytes);
    let obj = r.read_object_start(true).ok()?;
    if obj.type_name != "Table" {
        return None;
    }
    let hdr = r.position();
    let hdr_width = match obj.version {
        2 => 4,
        3 => 8,
        _ => return None,
    };
    r.skip(hdr_width).ok()?;
    r.read_u32().ok()?; // endian format
    r.read_string().ok()?; // table kind
    crate::tabledesc::parse_table_desc(&mut r).ok()?;
    let cs = r.position();
    let v = r.read_i32().ok()?;
    let field = if v < 0 {
        (cs + 4, if v.unsigned_abs() <= 2 { 4 } else { 8 })
    } else {
        (cs, 4) // ancient v1: the leading Int is the row count itself
    };
    Some([(hdr, hdr_width), field])
}

fn fits(nrow: u64, width: usize) -> bool {
    width == 8 || nrow <= u64::from(u32::MAX)
}

/// Whether both row counts of `dir/table.dat` can hold `nrow` in place.
pub(crate) fn table_dat_nrow_fits(dir: &Path, nrow: u64) -> bool {
    std::fs::read(dir.join("table.dat"))
        .ok()
        .and_then(|b| table_dat_nrow_fields(&b))
        .is_some_and(|f| f.iter().all(|&(_, w)| fits(nrow, w)))
}

/// Rewrite both row counts of `dir/table.dat` in place (the rest of the
/// header, every column's descriptor included, stays byte-identical).
pub(crate) fn patch_table_dat_nrow(dir: &Path, nrow: u64) -> Result<(), String> {
    let path = dir.join("table.dat");
    let mut bytes = std::fs::read(&path).map_err(io_err(&path))?;
    let fields = table_dat_nrow_fields(&bytes)
        .ok_or_else(|| format!("{}: cannot locate the row count", path.display()))?;
    for (off, width) in fields {
        if !fits(nrow, width) {
            return Err(format!("{}: {nrow} rows do not fit", path.display()));
        }
        bytes[off..off + width].copy_from_slice(&nrow.to_be_bytes()[8 - width..]);
    }
    std::fs::write(&path, bytes).map_err(io_err(&path))
}

/// Rewrite the row count of every `sync` record in `dir/table.lock`
/// (casacore takes the row count from there in preference to `table.dat`),
/// if the table has a lock file.
pub(crate) fn patch_lock_nrrow(dir: &Path, nrow: u64) -> Result<(), String> {
    let path = dir.join("table.lock");
    // With a live shared instance, patch through its fd (a transient open +
    // close would drop this process's fcntl locks on the file) and keep the
    // record's other fields by rewriting the parsed info.
    if let Some(lf) = crate::lockfile::lookup(dir) {
        let lf = lf.lock().unwrap();
        if lf.missing {
            return Ok(());
        }
        if let Some(data) = lf
            .get_info()
            .map_err(|e| path.display().to_string() + ": " + &e)?
        {
            let mut data = data;
            data.nrrow = nrow;
            lf.put_info(&data)
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        return Ok(());
    }
    let Ok(mut bytes) = std::fs::read(&path) else {
        return Ok(());
    };
    let mut changed = false;
    let mut i = 4;
    while i + 8 <= bytes.len() {
        if &bytes[i..i + 4] != b"sync" || bytes[i - 4..i] != [0, 0, 0, 4] {
            i += 1;
            continue;
        }
        let ver = u32::from_be_bytes(bytes[i + 4..i + 8].try_into().unwrap());
        let (off, width) = match ver {
            1 => (i + 8, 4),
            2 => (i + 8, 8),
            _ => {
                i += 1;
                continue;
            }
        };
        if off + width <= bytes.len() {
            if !fits(nrow, width) {
                return Err(format!("{}: {nrow} rows do not fit", path.display()));
            }
            bytes[off..off + width].copy_from_slice(&nrow.to_be_bytes()[8 - width..]);
            changed = true;
        }
        i = off + width;
    }
    if changed {
        std::fs::write(&path, bytes).map_err(io_err(&path))?;
    }
    Ok(())
}

// ---------------------------------------------------------------- TSM

/// Grow the tiled column `cd` (data manager `seq`) from `old` to `new` rows.
///
/// A table created empty has no rows to keep, so its header is simply
/// regenerated for the cell shape: the descriptor's shape, else the shape
/// of the cells being written (`pending_shape`, CASA order).
#[allow(clippy::too_many_arguments)]
pub(crate) fn tsm_grow(
    dir: &Path,
    seq: u32,
    big_endian: bool,
    stman_type: &str,
    cd: &ColumnDesc,
    old: u64,
    new: u64,
    pending_shape: Option<Vec<i64>>,
    dry_run: bool,
) -> GrowResult {
    let Ok(tsm) = crate::tsm::TsmFile::open(dir, seq, big_endian) else {
        return Ok(false);
    };
    let h = tsm.header.clone();
    drop(tsm);
    if h.nrrow != old || h.root_type != stman_type || h.data_types.len() != 1 {
        return Ok(false);
    }
    let dtype = h.data_types[0];
    if dtype != cd.data_type {
        return Ok(false);
    }
    let real = usize::from(stman_type == "TiledShapeStMan");
    let on_disk: Option<Vec<i64>> = h
        .cubes
        .get(real)
        .filter(|c| !c.cube_shape.is_empty())
        .map(|c| c.cube_shape[..c.cube_shape.len() - 1].to_vec());
    let usable = |s: &Vec<i64>| !s.is_empty() && s.iter().all(|&d| d > 0);
    let cell_shape = if old > 0 {
        match on_disk {
            Some(s) if crate::tsm::is_casacure_layout(&h, dtype, &s) => s,
            _ => return Ok(false),
        }
    } else {
        match [cd.shape.clone(), pending_shape, on_disk]
            .into_iter()
            .flatten()
            .find(usable)
        {
            Some(s) => s,
            None => return Ok(false),
        }
    };
    if dry_run {
        return Ok(true);
    }
    let (hdr, tile_len, file_seq) = crate::tsm::tsm_grown_header(
        stman_type,
        big_endian,
        seq,
        &h.hypercolumn_name,
        dtype,
        &cell_shape,
        new,
    )
    .map_err(|e| e.to_string())?;
    let tile_path = dir.join(format!("table.f{seq}_TSM{file_seq}"));
    let tile = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&tile_path)
        .map_err(io_err(&tile_path))?;
    if old == 0 {
        // Nothing to keep; the zeroed file is every cell's default.
        tile.set_len(0).map_err(io_err(&tile_path))?;
    }
    let have = tile.metadata().map_err(io_err(&tile_path))?.len();
    if have < tile_len {
        // Sparse zero-extension: no bytes are written for the new rows.
        tile.set_len(tile_len).map_err(io_err(&tile_path))?;
    }
    drop(tile);
    let path = dir.join(format!("table.f{seq}"));
    std::fs::write(&path, hdr).map_err(io_err(&path))?;
    Ok(true)
}

// ---------------------------------------------------------------- SSM

/// A new row's cell in a StandardStMan bucket.
enum SsmDefault {
    /// A bit-packed Bool scalar.
    Bit(bool),
    /// A fixed-size scalar cell.
    Bytes(Vec<u8>),
    /// A fixed-shape array: a record appended to the array file per row,
    /// referenced by an 8-byte offset cell.
    Record(Vec<u8>),
    /// A variable-shape array: an undefined cell (offset 0), as casacore
    /// leaves a row it adds.
    Null,
    /// A Direct Bool array: this many zero bits, inline.
    ZeroBits(u64),
}

impl SsmDefault {
    fn bits(&self) -> u64 {
        match self {
            SsmDefault::Bit(_) => 1,
            SsmDefault::Bytes(b) => 8 * b.len() as u64,
            SsmDefault::Record(_) | SsmDefault::Null => 8 * u64::from(crate::ssm::ARRAY_REF_SIZE),
            SsmDefault::ZeroBits(n) => *n,
        }
    }
}

/// Every array-file (`table.f{seq}i`) write of one flush: appended records,
/// plus in-place rewrites of existing records.
pub(crate) struct ArrayFile {
    path: std::path::PathBuf,
    big_endian: bool,
    /// Records carry a reference count (file version > 0).
    pub(crate) refcount: bool,
    exists: bool,
    /// Where the next appended record goes (the file length).
    end: u64,
    /// Appended bytes not yet written, starting at `pending_at`.
    append: Vec<u8>,
    pending_at: u64,
    /// In-place record rewrites: `(offset, bytes)`.
    rewrites: Vec<(u64, Vec<u8>)>,
}

impl ArrayFile {
    pub(crate) fn open(dir: &Path, seq: u32, big_endian: bool) -> Result<ArrayFile, String> {
        let path = dir.join(format!("table.f{seq}i"));
        let (exists, version, end) = match std::fs::File::open(&path) {
            Ok(mut f) => {
                let len = f.metadata().map_err(io_err(&path))?.len();
                let mut v = [0u8; 4];
                if len >= 16 {
                    f.read_exact(&mut v).map_err(io_err(&path))?;
                }
                let version = if big_endian {
                    u32::from_be_bytes(v)
                } else {
                    u32::from_le_bytes(v)
                };
                (true, version, len.max(16))
            }
            Err(_) => (false, 0, 16),
        };
        Ok(ArrayFile {
            path,
            big_endian,
            refcount: version > 0,
            exists,
            end,
            append: Vec::new(),
            pending_at: end,
            rewrites: Vec::new(),
        })
    }

    /// Queue `record` (`[ndim][dims][data]`) at the end; returns its offset.
    pub(crate) fn push(&mut self, record: &[u8]) -> Result<i64, String> {
        let off = self.end;
        if self.refcount {
            let one: u32 = 1;
            self.append.extend_from_slice(&if self.big_endian {
                one.to_be_bytes()
            } else {
                one.to_le_bytes()
            });
        }
        self.append.extend_from_slice(record);
        self.end = self.pending_at + self.append.len() as u64;
        if self.append.len() >= WRITE_BATCH {
            self.write_appended()?;
        }
        Ok(off as i64)
    }

    /// Queue an in-place rewrite of the record at `off` (same size).
    pub(crate) fn rewrite(&mut self, off: u64, record: Vec<u8>) {
        let at = off + if self.refcount { 4 } else { 0 };
        self.rewrites.push((at, record));
    }

    fn file(&mut self) -> Result<std::fs::File, String> {
        if !self.exists {
            // StManArrayFile header, completed by `finish`.
            std::fs::write(&self.path, [0u8; 16]).map_err(io_err(&self.path))?;
            self.exists = true;
        }
        open_rw(&self.path)
    }

    fn write_appended(&mut self) -> Result<(), String> {
        if self.append.is_empty() {
            return Ok(());
        }
        let mut f = self.file()?;
        let path = self.path.clone();
        write_at(&mut f, &path, self.pending_at, &self.append)?;
        self.pending_at += self.append.len() as u64;
        self.append.clear();
        Ok(())
    }

    /// Write everything queued and the header's file length.
    pub(crate) fn finish(mut self) -> Result<(), String> {
        if self.append.is_empty() && self.rewrites.is_empty() && self.exists {
            return Ok(());
        }
        self.write_appended()?;
        let mut f = self.file()?;
        let path = self.path.clone();
        for (off, bytes) in std::mem::take(&mut self.rewrites) {
            write_at(&mut f, &path, off, &bytes)?;
        }
        // [u32 version][Int64 length]: only the length changes.
        write_at(
            &mut f,
            &path,
            4,
            &i64_bytes(self.end as i64, self.big_endian),
        )
    }
}

/// The index-bucket chain of a StandardStMan file, in order.
fn ssm_index_chain(f: &crate::ssm::StandardStManFile) -> Vec<u32> {
    let h = &f.header;
    let mut chain = Vec::new();
    if h.first_index_bucket < 0 || h.nr_index_buckets == 0 {
        return chain;
    }
    let mut b = h.first_index_bucket as u32;
    for _ in 0..h.nr_index_buckets {
        chain.push(b);
        if h.index_bucket_offset > 0 {
            break; // a single-bucket index
        }
        let next = f
            .bucket_bytes(b)
            .ok()
            .and_then(|bytes| bytes.get(4..8))
            .map(|n| i32::from_be_bytes(n.try_into().unwrap()));
        match next {
            Some(n) if n >= 0 => b = n as u32,
            _ => break,
        }
    }
    chain
}

/// Place `def` as the cell of the row at `intra` in a bucket buffer.
fn put_ssm_default(
    bucket: &mut [u8],
    offset: u32,
    intra: u64,
    def: &SsmDefault,
    array_file: Option<&mut ArrayFile>,
    big_endian: bool,
) -> Result<(), String> {
    let base = offset as usize;
    match def {
        SsmDefault::Bit(v) => {
            let byte = base + (intra / 8) as usize;
            let mask = 1u8 << (intra % 8);
            if *v {
                bucket[byte] |= mask;
            } else {
                bucket[byte] &= !mask;
            }
        }
        SsmDefault::Bytes(cell) => {
            let at = base + intra as usize * cell.len();
            bucket[at..at + cell.len()].copy_from_slice(cell);
        }
        SsmDefault::Record(rec) => {
            let file = array_file.ok_or("an array column without an array file")?;
            let off = file.push(rec)?;
            let at = base + intra as usize * 8;
            bucket[at..at + 8].copy_from_slice(&i64_bytes(off, big_endian));
        }
        SsmDefault::Null => {
            let at = base + intra as usize * 8;
            bucket[at..at + 8].fill(0);
        }
        SsmDefault::ZeroBits(n) => {
            let first = offset as u64 * 8 + intra * n;
            for bit in first..first + n {
                bucket[(bit / 8) as usize] &= !(1u8 << (bit % 8));
            }
        }
    }
    Ok(())
}

/// Grow the StandardStMan data manager `seq` (columns `cds`, in the order
/// the manager holds them) from `old` to `new` rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssm_grow(
    dir: &Path,
    seq: u32,
    big_endian: bool,
    spec: &crate::columnset::StandardStMan,
    cds: &[&ColumnDesc],
    old: u64,
    new: u64,
    dry_run: bool,
) -> GrowResult {
    use crate::ssm::{encode_array_record, encode_scalar_cell, StandardStManFile, DATA_START};

    let Ok(f) = StandardStManFile::open(dir, seq, big_endian) else {
        return Ok(false);
    };
    let h = f.header.clone();
    if !(h.version == 2 || h.version == 3) || h.nr_index != 1 || f.indices.len() != 1 {
        return Ok(false);
    }
    if spec.column_offset.len() != cds.len() || spec.col_index_map.iter().any(|&i| i != 0) {
        return Ok(false);
    }
    let index = f.indices[0].clone();
    let rpb = u64::from(index.rows_per_bucket);
    if rpb == 0 || index.last_row.len() != index.bucket_number.len() {
        return Ok(false);
    }
    if index.last_row.last().map_or(0, |&l| l + 1) != old {
        return Ok(false);
    }
    let bs = h.bucket_size as usize;
    let mut defs = Vec::with_capacity(cds.len());
    for cd in cds {
        let def = match &cd.kind {
            ColumnKind::Scalar(v) => {
                let Ok(cell) = encode_scalar_cell(big_endian, cd, v) else {
                    return Ok(false); // e.g. a long default string
                };
                if cd.data_type == DataType::Bool {
                    SsmDefault::Bit(cell.first().is_some_and(|&b| b != 0))
                } else {
                    SsmDefault::Bytes(cell)
                }
            }
            ColumnKind::Array if crate::ssm::is_direct_array(cd) => {
                if !f.direct_cells_inline(spec, defs.len(), cd) {
                    return Ok(false); // the old casacure layout: rewrite whole
                }
                let bits = crate::ssm::direct_cell_bits(cd);
                if cd.data_type == DataType::Bool {
                    SsmDefault::ZeroBits(bits)
                } else {
                    SsmDefault::Bytes(vec![0u8; (bits / 8) as usize])
                }
            }
            ColumnKind::Array if cd.data_type != DataType::String => {
                let fixed = cd
                    .shape
                    .as_ref()
                    .is_some_and(|s| !s.is_empty() && s.iter().all(|&d| d > 0));
                if !fixed {
                    SsmDefault::Null
                } else {
                    let Some(RecordValue::Array(arr)) = crate::table::default_cell_value(cd) else {
                        return Ok(false);
                    };
                    let Ok(rec) = encode_array_record(big_endian, cd.data_type, &arr) else {
                        return Ok(false);
                    };
                    SsmDefault::Record(rec)
                }
            }
            // Record cells and string arrays live in string buckets.
            _ => return Ok(false),
        };
        let region = (rpb * def.bits()).div_ceil(8);
        let offset = u64::from(spec.column_offset[defs.len()]);
        if offset + region > bs as u64 {
            return Ok(false);
        }
        defs.push(def);
    }
    if dry_run {
        return Ok(true);
    }
    let chain = ssm_index_chain(&f);
    drop(f);

    let path = dir.join(format!("table.f{seq}"));
    let mut file = open_rw(&path)?;
    let mut array_file = if defs.iter().any(|d| matches!(d, SsmDefault::Record(_))) {
        Some(ArrayFile::open(dir, seq, big_endian)?)
    } else {
        None
    };
    let at = |b: u32| (DATA_START + b as usize * bs) as u64;
    let mut last_row = index.last_row;
    let mut numbers = index.bucket_number;
    let mut row = old;

    // Top up a partly filled last bucket.
    if let (Some(&last), Some(&number)) = (last_row.last(), numbers.last()) {
        let n = last_row.len();
        let start = if n >= 2 { last_row[n - 2] + 1 } else { 0 };
        let end = (start + rpb).min(new);
        if last + 1 < end {
            let mut bucket = read_at(&mut file, &path, at(number), bs)?;
            for r in row..end {
                for (c, def) in defs.iter().enumerate() {
                    put_ssm_default(
                        &mut bucket,
                        spec.column_offset[c],
                        r - start,
                        def,
                        array_file.as_mut(),
                        big_endian,
                    )?;
                }
            }
            write_at(&mut file, &path, at(number), &bucket)?;
            *last_row.last_mut().unwrap() = end - 1;
            row = end;
        }
    }

    // Append default buckets: one template, with fresh array records.
    let mut next = h.nr_buckets;
    if row < new {
        let mut template = vec![0u8; bs];
        for (c, def) in defs.iter().enumerate() {
            if matches!(
                def,
                SsmDefault::Bit(_) | SsmDefault::Bytes(_) | SsmDefault::ZeroBits(_)
            ) {
                for r in 0..rpb {
                    put_ssm_default(
                        &mut template,
                        spec.column_offset[c],
                        r,
                        def,
                        None,
                        big_endian,
                    )?;
                }
            }
        }
        let mut batch: Vec<u8> = Vec::new();
        let mut batch_first = next;
        while row < new {
            let n = rpb.min(new - row);
            let mut bucket = template.clone();
            for (c, def) in defs.iter().enumerate() {
                if matches!(def, SsmDefault::Record(_)) {
                    for r in 0..n {
                        put_ssm_default(
                            &mut bucket,
                            spec.column_offset[c],
                            r,
                            def,
                            array_file.as_mut(),
                            big_endian,
                        )?;
                    }
                }
            }
            batch.extend_from_slice(&bucket);
            last_row.push(row + n - 1);
            numbers.push(next);
            next += 1;
            row += n;
            if batch.len() >= WRITE_BATCH {
                write_at(&mut file, &path, at(batch_first), &batch)?;
                batch.clear();
                batch_first = next;
            }
        }
        write_at(&mut file, &path, at(batch_first), &batch)?;
    }

    // The index, in its chain (extended at the end when it outgrows it).
    let stream = crate::ssm::encode_ssm_index(
        big_endian,
        index.rows_per_bucket,
        cds.len(),
        &last_row,
        &numbers,
    );
    let needed = crate::ssm::index_bucket_count(stream.len(), h.bucket_size);
    let mut chain = chain;
    while chain.len() < needed {
        chain.push(next);
        next += 1;
    }
    chain.truncate(needed);
    for (i, piece) in stream.chunks(bs - 8).enumerate() {
        let mut b = vec![0u8; bs];
        let link = chain.get(i + 1).map_or(-1, |&n| n as i32);
        b[0..4].copy_from_slice(&(-1i32).to_be_bytes());
        b[4..8].copy_from_slice(&link.to_be_bytes());
        b[8..8 + piece.len()].copy_from_slice(piece);
        write_at(&mut file, &path, at(chain[i]), &b)?;
    }
    let header = crate::ssm::StandardStManHeader {
        nr_buckets: next,
        nr_index_buckets: needed as u32,
        first_index_bucket: chain.first().map_or(-1, |&b| b as i32),
        index_bucket_offset: if needed <= 1 { 8 } else { 0 },
        index_length: stream.len() as u32,
        nr_index: 1,
        ..h
    };
    let mut hdr = crate::ssm::encode_ssm_header(&header);
    if hdr.len() > DATA_START {
        return Err(format!("{}: header too long", path.display()));
    }
    hdr.resize(DATA_START, 0);
    write_at(&mut file, &path, 0, &hdr)?;
    drop(file);
    if let Some(a) = array_file {
        a.finish()?;
    }
    Ok(true)
}

// ---------------------------------------------------------------- ISM

/// Grow the IncrementalStMan data manager `seq` (columns `cds`, in the
/// order the manager holds them) from `old` to `new` rows and write the
/// cells in `pending` (per column: `(row, encoded cell)`, ascending rows).
/// Only the buckets holding pending rows, a topped-up last bucket and the
/// new buckets are re-encoded; the rest of the file is untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ism_update(
    dir: &Path,
    seq: u32,
    big_endian: bool,
    cds: &[&ColumnDesc],
    old: u64,
    new: u64,
    pending: &[Vec<(u64, Vec<u8>)>],
    dry_run: bool,
) -> GrowResult {
    use crate::ism::{
        encode_ism_bucket, encode_ism_header, encode_ism_index, ism_cell_size, ism_rows_per_bucket,
        IsmFile, WriteIsmColumn, DATA_START,
    };

    if cds
        .iter()
        .any(|cd| cd.data_type == DataType::String || !matches!(cd.kind, ColumnKind::Scalar(_)))
    {
        return Ok(false);
    }
    let Ok(f) = IsmFile::open(dir, seq, big_endian) else {
        return Ok(false);
    };
    let h = f.header.clone();
    if !(h.version == 4 || h.version == 5) {
        return Ok(false);
    }
    let rows = &f.index.rows;
    let nused = f.index.bucket_numbers.len();
    if rows.len() != nused + 1 || rows[0] != 0 || rows[nused] != old {
        return Ok(false);
    }
    let bs = h.bucket_size as usize;
    let sizes: Vec<u32> = cds.iter().map(|cd| ism_cell_size(cd)).collect();
    let shape: Vec<WriteIsmColumn<'_>> = sizes
        .iter()
        .map(|&cell_size| WriteIsmColumn {
            cell_size,
            bytes: &[],
        })
        .collect();
    let cap = ism_rows_per_bucket(bs, &shape) as u64;
    let mut defaults = Vec::with_capacity(cds.len());
    for cd in cds {
        let ColumnKind::Scalar(v) = &cd.kind else {
            return Ok(false);
        };
        let Ok(cell) = crate::ssm::encode_scalar_cell(big_endian, cd, v) else {
            return Ok(false);
        };
        if cell.len() != ism_cell_size(cd) as usize {
            return Ok(false);
        }
        defaults.push(cell);
    }

    // Rows [start, end) of every column: `disk_end` rows from the file, the
    // rest defaults, then the pending cells overlaid.
    let gather = |f: Option<&IsmFile>, start: u64, disk_end: u64, end: u64| {
        let mut cols: Vec<Vec<u8>> = Vec::with_capacity(cds.len());
        for (c, cd) in cds.iter().enumerate() {
            let size = sizes[c] as usize;
            let mut buf = Vec::with_capacity((end - start) as usize * size);
            if let Some(f) = f {
                if disk_end > start {
                    f.for_each_cell_raw(c, cd, start, disk_end - start, |cell| {
                        buf.extend_from_slice(cell);
                        Ok::<(), crate::ism::IsmError>(())
                    })
                    .map_err(|e| e.to_string())?;
                }
            }
            while buf.len() < (end - start) as usize * size {
                buf.extend_from_slice(&defaults[c]);
            }
            if let Some(p) = pending.get(c) {
                let lo = p.partition_point(|(r, _)| *r < start);
                for (r, cell) in p[lo..].iter().take_while(|(r, _)| *r < end) {
                    if cell.len() != size {
                        return Err(format!("ISM cell of {} bytes, want {size}", cell.len()));
                    }
                    let at = (r - start) as usize * size;
                    buf[at..at + size].copy_from_slice(cell);
                }
            }
            cols.push(buf);
        }
        Ok::<_, String>(cols)
    };
    let encode = |cols: &[Vec<u8>], n: u64| {
        let w: Vec<WriteIsmColumn<'_>> = cols
            .iter()
            .zip(&sizes)
            .map(|(bytes, &cell_size)| WriteIsmColumn { cell_size, bytes })
            .collect();
        encode_ism_bucket(big_endian, bs, n, &w)
    };

    // Existing buckets to re-encode: those holding pending rows, and the
    // last one when it has room for new rows.
    let mut affected: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for p in pending {
        for (r, _) in p {
            if *r < old {
                affected.insert(rows.partition_point(|&b| b <= *r) - 1);
            }
        }
    }
    let top_up = nused > 0 && new > old && old - rows[nused - 1] < cap;
    if top_up {
        affected.insert(nused - 1);
    }
    let mut new_rows = rows.clone();
    let mut encoded: Vec<(u32, Vec<u8>)> = Vec::with_capacity(affected.len());
    for &i in &affected {
        let start = rows[i];
        let disk_end = rows[i + 1];
        let end = if top_up && i == nused - 1 {
            (start + cap).min(new)
        } else {
            disk_end
        };
        let cols = gather(Some(&f), start, disk_end, end)?;
        let Some(bucket) = encode(&cols, end - start) else {
            return Ok(false); // a casacore bucket too full to re-encode
        };
        new_rows[i + 1] = end;
        encoded.push((f.index.bucket_numbers[i], bucket));
    }
    if dry_run {
        return Ok(true);
    }
    let mut numbers = f.index.bucket_numbers.clone();
    drop(f);

    let path = dir.join(format!("table.f{seq}"));
    let mut file = open_rw(&path)?;
    let at = |b: u32| (DATA_START + b as usize * bs) as u64;
    for (number, bucket) in &encoded {
        write_at(&mut file, &path, at(*number), bucket)?;
    }
    let mut row = *new_rows.last().unwrap();
    let mut next = h.nbucket;
    while row < new {
        let end = (row + cap).min(new);
        let cols = gather(None, row, row, end)?;
        let bucket = encode(&cols, end - row)
            .ok_or_else(|| format!("{}: a new ISM bucket overflowed", path.display()))?;
        write_at(&mut file, &path, at(next), &bucket)?;
        numbers.push(next);
        new_rows.push(end);
        next += 1;
        row = end;
    }
    let index = encode_ism_index(big_endian, &new_rows, &numbers);
    write_at(&mut file, &path, at(next), &index)?;
    file.set_len(at(next) + index.len() as u64)
        .map_err(io_err(&path))?;
    let mut hdr = encode_ism_header(&crate::ism::IsmHeader { nbucket: next, ..h });
    if hdr.len() > DATA_START {
        return Err(format!("{}: header too long", path.display()));
    }
    hdr.resize(DATA_START, 0);
    write_at(&mut file, &path, 0, &hdr)?;
    Ok(true)
}
