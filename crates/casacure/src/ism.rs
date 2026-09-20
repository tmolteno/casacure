//! Reader for the IncrementalStMan data file (`table.f0`)
//! (`casacore/tables/DataMan/ISMBase.cc`, `ISMIndex.cc`,
//! `ISMBucket.cc`, `ISMColumn.cc`).
//!
//! The data file is an AipsIO stream in the table's *data-file* endianness
//! (big or little). Layout:
//!
//! - A root `"IncrementalStMan"` v4/v5 object: bucket size and counts
//!   (`nbucket`, persistent cache, unique nr, free-bucket list).
//! - Buckets from fixed offset 512: data bucket `b` at `512 + b*bucket_size`.
//! - The `"ISMIndex"` object immediately after the last bucket: the
//!   bucket-boundary rows (`nused+1` ascending row numbers) and the bucket
//!   number per range.
//! - Every bucket holds `[u32 indexOffset]` (high bits encode the row-number
//!   width) then the raw data area, then a per-column index: for each column
//!   `[u32 nr][nr × rowNr][nr × data offset]`. Each entry is an *interval*:
//!   entry `i` covers rows `[rowNr[i], rowNr[i+1])` (or to the last row of
//!   the bucket) and stores one value at `data[offset]` — the incremental
//!   compression (a repeated value across consecutive rows is stored once).
//!   Rows whose value never repeats just get per-row intervals.
//!
//! `option 1` (Direct) on the column descriptor is a schema hint; the on
//! disk layout does not depend on it.

use crate::aipsio::{AipsIoError, Reader};
use crate::record::{DataType, RecordValue};
use crate::tabledesc::ColumnDesc;
use thiserror::Error;

/// Bytes between the start of the data file and the first bucket (same as
/// the StandardStMan bucket cache start offset).
pub const DATA_START: usize = 512;

/// Errors from reading an IncrementalStMan data file.
#[derive(Debug, Error)]
pub enum IsmError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("unexpected object type {found:?}, expected {expected:?}")]
    UnexpectedType { expected: String, found: String },
    #[error("data file endian flag {flag} does not match table flag {expected}")]
    EndianMismatch { flag: bool, expected: bool },
    #[error("row {row} not covered by any bucket")]
    RowOutOfRange { row: u64 },
    #[error("bucket {0} is out of range (file has {1})")]
    InvalidBucket(u32, u32),
    #[error("column {column} of the IncrementalStMan has no data for row {row}")]
    ColumnEmpty { column: usize, row: u64 },
    #[error(transparent)]
    Ssm(#[from] crate::ssm::SsmError),
    #[error("IncrementalStMan strings are not supported yet")]
    StringUnsupported,
    #[error("column {0} is not a scalar IncrementalStMan column")]
    NotScalar(String),
}

/// The `IncrementalStMan` header at the start of the data file
/// (`ISMBase::readIndex` / `writeIndex`).
#[derive(Debug, Clone, PartialEq)]
pub struct IsmHeader {
    pub version: u32,
    /// Data-file byte order.
    pub big_endian: bool,
    /// Size of one bucket in bytes.
    pub bucket_size: u32,
    /// Number of data buckets (the index object follows them).
    pub nbucket: u32,
    pub pers_cache_size: u32,
    pub uniq_nr: u32,
    pub n_free_bucket: u32,
    pub first_free_bucket: i32,
}

/// The row-to-bucket index at the end of the file (`ISMIndex`).
#[derive(Debug, Clone, PartialEq)]
pub struct IsmIndex {
    /// `nused + 1` ascending row boundaries: bucket `i` holds rows
    /// `rows[i] .. rows[i+1] - 1`.
    pub rows: Vec<u64>,
    /// Data bucket holding each row range.
    pub bucket_numbers: Vec<u32>,
}

impl IsmIndex {
    /// Bucket for `row`, plus the intra-bucket row and bucket row count
    /// (`ISMIndex::getBucketNr`).
    pub fn bucket_for(&self, row: u64) -> Option<(u32, u64, u64)> {
        // First boundary >= row; if not exact, the previous one brackets it.
        let lb = self.rows.partition_point(|&r| r < row);
        if lb == self.rows.len() {
            return None; // past the last boundary
        }
        // If rows[lb] == row the interval starts at lb, otherwise at lb - 1.
        let index = if self.rows[lb] == row {
            lb
        } else {
            lb.checked_sub(1)?
        };
        let start = self.rows[index];
        // rows has nused+1 entries, so index+1 always exists for a valid row.
        let next = *self.rows.get(index + 1)?;
        Some((*self.bucket_numbers.get(index)?, row - start, next - start))
    }
}

/// A parsed IncrementalStMan data file.
#[derive(Debug)]
pub struct IsmFile {
    pub header: IsmHeader,
    pub index: IsmIndex,
    data: crate::datafile::Buffer,
}

/// Per-column per-bucket interval index.
#[derive(Debug)]
struct ColumnIndex {
    /// Ascending intra-bucket row numbers where each interval starts.
    rows: Vec<u64>,
    /// Byte offset into the bucket's data area per interval.
    offsets: Vec<u32>,
}

impl IsmFile {
    /// Open `<table_dir>/table.f{seq}` (memory-mapped so chunked reads only
    /// touch their pages) and parse it.
    pub fn open(
        table_dir: impl AsRef<std::path::Path>,
        seq_nr: u32,
        table_big_endian: bool,
    ) -> Result<IsmFile, IsmError> {
        let file = std::fs::File::open(table_dir.as_ref().join(format!("table.f{seq_nr}")))?;
        let data = crate::datafile::Buffer::from_file(file)?;
        let (header, index) = Self::parse_meta(&data, table_big_endian)?;
        Ok(IsmFile {
            header,
            index,
            data,
        })
    }

    /// Parse an IncrementalStMan data file.
    pub fn parse(data: &[u8], table_big_endian: bool) -> Result<IsmFile, IsmError> {
        let (header, index) = Self::parse_meta(data, table_big_endian)?;
        Ok(IsmFile {
            header,
            index,
            data: crate::datafile::Buffer::from(data.to_vec()),
        })
    }

    /// Parse the header and bucket index of an IncrementalStMan data file
    /// (no data copy; the caller keeps the backing bytes).
    fn parse_meta(data: &[u8], table_big_endian: bool) -> Result<(IsmHeader, IsmIndex), IsmError> {
        let mut r = reader(data, table_big_endian);
        let obj = r.read_object_start(true)?;
        if obj.type_name != "IncrementalStMan" {
            return Err(IsmError::UnexpectedType {
                expected: "IncrementalStMan".into(),
                found: obj.type_name,
            });
        }
        let version = obj.version;
        let stored_endian = if version >= 5 { r.read_bool()? } else { true };
        if version >= 5 && stored_endian != table_big_endian {
            return Err(IsmError::EndianMismatch {
                flag: stored_endian,
                expected: table_big_endian,
            });
        }
        let header = IsmHeader {
            version,
            big_endian: table_big_endian,
            bucket_size: r.read_u32()?,
            nbucket: r.read_u32()?,
            pers_cache_size: r.read_u32()?,
            uniq_nr: r.read_u32()?,
            n_free_bucket: r.read_u32()?,
            first_free_bucket: r.read_i32()?,
        };
        let index = read_index_bytes(data, &header, table_big_endian)?;
        Ok((header, index))
    }

    fn bucket_bytes(&self, bucket: u32) -> Result<&[u8], IsmError> {
        let size = self.header.bucket_size as usize;
        let start = DATA_START + bucket as usize * size;
        self.data
            .get(start..start + size)
            .ok_or(IsmError::InvalidBucket(bucket, self.header.nbucket))
    }

    fn column_index(&self, bucket: u32, column: usize) -> Result<ColumnIndex, IsmError> {
        let b = self.bucket_bytes(bucket)?;
        let ioff = read_u32_e(b, 0, self.header.big_endian);
        let index_off = (ioff & 0x0fff_ffff) as usize;
        // The high nibble is the row-number width; 0 means 32-bit rows.
        let use32 = ioff & 0xf000_0000 == 0;
        let mut pos = index_off;
        for c in 0..=column {
            if c == column {
                let nr = read_u32_e(b, pos, self.header.big_endian) as usize;
                pos += 4;
                let mut rows = Vec::with_capacity(nr);
                for _ in 0..nr {
                    let v = if use32 {
                        u64::from(read_u32_e(b, pos, self.header.big_endian))
                    } else {
                        read_u64_e(b, pos, self.header.big_endian)
                    };
                    pos += if use32 { 4 } else { 8 };
                    rows.push(v);
                }
                let mut offsets = Vec::with_capacity(nr);
                for _ in 0..nr {
                    offsets.push(read_u32_e(b, pos, self.header.big_endian));
                    pos += 4;
                }
                return Ok(ColumnIndex { rows, offsets });
            }
            // Skip this column's entries.
            let nr = read_u32_e(b, pos, self.header.big_endian) as usize;
            pos += 4;
            pos += nr * if use32 { 4 } else { 8 };
            pos += nr * 4;
        }
        Err(IsmError::ColumnEmpty { column, row: 0 })
    }

    /// Read the scalar value for `column` (index within this data manager)
    /// at `row`. The value is the interval entry covering `row`.
    pub fn read_scalar_cell(
        &self,
        column: usize,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<RecordValue, IsmError> {
        if desc.data_type == DataType::String {
            return Err(IsmError::StringUnsupported);
        }
        let (bucket, intra_row, _bucket_rows) = self
            .index
            .bucket_for(row)
            .ok_or(IsmError::RowOutOfRange { row })?;
        let ci = self.column_index(bucket, column)?;
        // Interval containing the intra-bucket row.
        let lb = ci.rows.partition_point(|&r| r < intra_row);
        let inx = if lb < ci.rows.len() && ci.rows[lb] == intra_row {
            lb
        } else {
            lb.checked_sub(1)
                .ok_or(IsmError::ColumnEmpty { column, row })?
        };
        let offset = ci.offsets[inx] as usize;
        let b = self.bucket_bytes(bucket)?;
        let data_off = 4usize; // data area starts after the index-offset u32
        let cell = b
            .get(data_off + offset..data_off + offset + 32)
            .ok_or(IsmError::ColumnEmpty { column, row })?;
        let want = crate::ssm::scalar_cell_size(desc) as usize;
        let cell = &cell[..want];
        Ok(crate::ssm::decode_scalar(
            cell,
            desc,
            self.header.big_endian,
        )?)
    }
}

fn reader(data: &[u8], big_endian: bool) -> Reader<'_> {
    if big_endian {
        Reader::new(data)
    } else {
        Reader::new_le(data)
    }
}

fn read_u32_e(b: &[u8], off: usize, big_endian: bool) -> u32 {
    let x = [b[off], b[off + 1], b[off + 2], b[off + 3]];
    if big_endian {
        u32::from_be_bytes(x)
    } else {
        u32::from_le_bytes(x)
    }
}

fn read_u64_e(b: &[u8], off: usize, big_endian: bool) -> u64 {
    let x = [
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ];
    if big_endian {
        u64::from_be_bytes(x)
    } else {
        u64::from_le_bytes(x)
    }
}

/// Read the `ISMIndex` object following the last bucket (`ISMIndex::get`).
fn read_index_bytes(
    data: &[u8],
    header: &IsmHeader,
    big_endian: bool,
) -> Result<IsmIndex, IsmError> {
    let off = DATA_START + header.nbucket as usize * header.bucket_size as usize;
    let mut r = reader(&data[off..], big_endian);
    let (version, _) = r.read_object(true, "ISMIndex")?;
    let nused = r.read_u32()? as usize;
    // rows block: nused+1 ascending boundaries; u32 rows in v1, u64 in v2.
    let block = r.read_object_start(false)?;
    if block.type_name != "Block" {
        return Err(IsmError::UnexpectedType {
            expected: "Block".into(),
            found: block.type_name,
        });
    }
    let n = r.read_u32()? as usize;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        rows.push(if version > 1 {
            r.read_u64()?
        } else {
            u64::from(r.read_u32()?)
        });
    }
    // bucketNr block.
    let block = r.read_object_start(false)?;
    if block.type_name != "Block" {
        return Err(IsmError::UnexpectedType {
            expected: "Block".into(),
            found: block.type_name,
        });
    }
    let n = r.read_u32()? as usize;
    let mut bucket_numbers = Vec::with_capacity(n);
    for _ in 0..n {
        bucket_numbers.push(r.read_u32()?);
    }
    bucket_numbers.truncate(nused);
    Ok(IsmIndex {
        rows,
        bucket_numbers,
    })
}

/// Per-column raw cell bytes handed to `write_ism_file`.
#[derive(Debug, Clone, Copy)]
pub struct WriteIsmColumn<'a> {
    /// Bytes per row cell (fixed element size; ISM scalar columns only).
    pub cell_size: u32,
    /// Concatenated cell bytes: `n_rows * cell_size` bytes.
    pub bytes: &'a [u8],
}

/// Serialize a complete IncrementalStMan data file — header, data buckets
/// with interval compression, and the `ISMIndex` at the end — matching what
/// casacore writes for scalar columns. Consecutive rows with identical bytes
/// in a column share one stored value (an interval entry), exactly as
/// `ISMBucket::addData` does.
pub fn write_ism_file(big_endian: bool, n_rows: u64, cols: &[WriteIsmColumn<'_>]) -> Vec<u8> {
    let ncols = cols.len();
    // Match casacore's default ISM bucket size; rows per bucket constrained
    // so data + per-column index fit.
    let bucket_size = 32768usize;
    let sum_sizes: usize = cols.iter().map(|c| c.cell_size as usize).sum();
    let per_row = sum_sizes + 8 * ncols;
    let rows_per_bucket = ((bucket_size.saturating_sub(4 * ncols)) / per_row.max(1)).max(1);
    let nr_buckets = n_rows.div_ceil(rows_per_bucket as u64) as usize;

    let mut file = Vec::with_capacity(512 + nr_buckets * bucket_size + 256);

    // Header.
    let mut hw = crate::aipsio::Writer::new();
    if !big_endian {
        hw = crate::aipsio::Writer::new_le();
    }
    if big_endian {
        hw.put_root_object_start("IncrementalStMan", 4);
    } else {
        hw.put_root_object_start("IncrementalStMan", 5);
        hw.put_bool(false);
    }
    hw.put_u32(bucket_size as u32);
    hw.put_u32(nr_buckets as u32);
    hw.put_u32(0); // pers cache size
    hw.put_u32(0); // uniqnr
    hw.put_u32(0); // free buckets
    hw.put_i32(-1); // first free bucket
    hw.put_object_end();
    file.extend_from_slice(&hw.into_bytes());
    file.resize(512, 0);

    // Buckets.
    for b in 0..nr_buckets {
        let start_row = b as u64 * rows_per_bucket as u64;
        let end_row = (start_row + rows_per_bucket as u64).min(n_rows);
        let mut data: Vec<u8> = Vec::new();
        // Per column: (intrabucket start row, data offset) of each interval.
        let mut starts: Vec<Vec<u32>> = vec![Vec::new(); ncols];
        let mut offsets: Vec<Vec<u32>> = vec![Vec::new(); ncols];
        let mut prev: Vec<Option<u32>> = vec![None; ncols]; // prev row idx for eq test
        for row in start_row..end_row {
            for (c, col) in cols.iter().enumerate() {
                let cell = &col.bytes[(row * col.cell_size as u64) as usize
                    ..((row + 1) * col.cell_size as u64) as usize];
                let same_as_prev = prev[c].is_some_and(|pr| {
                    let prev_cell = &col.bytes[(pr as u64 * col.cell_size as u64) as usize
                        ..((pr as u64 + 1) * col.cell_size as u64) as usize];
                    prev_cell == cell
                });
                if !same_as_prev {
                    starts[c].push((row - start_row) as u32);
                    offsets[c].push(data.len() as u32);
                    data.extend_from_slice(cell);
                }
                prev[c] = Some((row - start_row) as u32);
            }
        }
        // indexOffset = dataLeng + 4; 32-bit rows (high bit clear).
        let index_offset = (data.len() + 4) as u32;
        let mut bucket = vec![0u8; bucket_size];
        put_u32_e(&mut bucket, 0, index_offset, big_endian);
        bucket[4..4 + data.len()].copy_from_slice(&data);
        let mut pos = 4usize + data.len();
        for c in 0..ncols {
            put_u32_e(&mut bucket, pos, starts[c].len() as u32, big_endian);
            pos += 4;
            for &s in &starts[c] {
                put_u32_e(&mut bucket, pos, s, big_endian);
                pos += 4;
            }
            for &o in &offsets[c] {
                put_u32_e(&mut bucket, pos, o, big_endian);
                pos += 4;
            }
        }
        file.extend_from_slice(&bucket);
    }

    // ISMIndex at the end.
    let mut iw = crate::aipsio::Writer::new();
    if !big_endian {
        iw = crate::aipsio::Writer::new_le();
    }
    iw.put_root_object_start("ISMIndex", 1);
    iw.put_u32(nr_buckets as u32); // nused
    iw.put_object_start("Block", 1);
    iw.put_u32((nr_buckets + 1) as u32);
    for b in 0..=nr_buckets {
        let boundary = (b as u64 * rows_per_bucket as u64).min(n_rows) as u32;
        iw.put_u32(boundary);
    }
    iw.put_object_end();
    iw.put_object_start("Block", 1);
    iw.put_u32(nr_buckets as u32);
    for b in 0..nr_buckets {
        iw.put_u32(b as u32);
    }
    iw.put_object_end();
    iw.put_object_end();
    file.extend_from_slice(&iw.into_bytes());
    file
}

/// Serialize the IncrementalStMan spec blob stored in `table.dat`
/// (`ISMBase::flush`): a root `"ISM"` v3 AipsIO stream with the data-manager
/// name. The blob is always canonical big endian.
pub fn write_ism_blob(name: &str) -> Vec<u8> {
    let mut w = crate::aipsio::Writer::new();
    w.put_root_object_start("ISM", 3);
    w.put_string(name);
    w.put_object_end();
    w.into_bytes()
}

fn put_u32_e(buf: &mut [u8], off: usize, v: u32, big_endian: bool) {
    let b = if big_endian {
        v.to_be_bytes()
    } else {
        v.to_le_bytes()
    };
    buf[off..off + 4].copy_from_slice(&b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aipsio::Writer;
    use crate::record::RecordValue;
    use crate::tabledesc::{ColumnDesc, ColumnKind};

    fn scalar_desc(name: &str, dt: DataType) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "IncrementalStMan".into(),
            data_manager_group: "IncrementalStMan".into(),
            options: 1,
            ndim: -1,
            shape: None,
            max_length: 0,
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind: ColumnKind::Scalar(RecordValue::Int(0)),
        }
    }

    /// Build a little-endian ISM file: one bucket with a single int column,
    /// rows 0..3 compressed into one interval holding the value 42.
    fn one_interval_file() -> Vec<u8> {
        let bs = 64u32;
        let mut file = vec![0u8; DATA_START + bs as usize];
        let mut w = Writer::new_le();
        w.put_root_object_start("IncrementalStMan", 5);
        w.put_bool(false); // little endian
        w.put_u32(bs);
        w.put_u32(1); // nbucket
        w.put_u32(0); // pers cache
        w.put_u32(0); // uniqnr
        w.put_u32(0); // nfree
        w.put_i32(-1); // first free
        w.put_object_end();
        let hdr = w.into_bytes();
        file[..hdr.len()].copy_from_slice(&hdr);
        let b = DATA_START;
        // indexOffset: data is 8 bytes -> index at 12 (data 4..12).
        file[b..b + 4].copy_from_slice(&12u32.to_le_bytes());
        file[b + 4..b + 8].copy_from_slice(&42i32.to_le_bytes());
        // column 0 index: nr=1, rownrs=[0], offsets=[0].
        file[b + 12..b + 16].copy_from_slice(&1u32.to_le_bytes());
        file[b + 16..b + 20].copy_from_slice(&0u32.to_le_bytes());
        file[b + 20..b + 24].copy_from_slice(&0u32.to_le_bytes());
        // ISMIndex after the last bucket.
        let mut w = Writer::new_le();
        w.put_root_object_start("ISMIndex", 1);
        w.put_u32(1); // nused
        w.put_object_start("Block", 1);
        w.put_u32(2);
        w.put_u32(0);
        w.put_u32(3);
        w.put_object_end();
        w.put_object_start("Block", 1);
        w.put_u32(1);
        w.put_u32(0);
        w.put_object_end();
        w.put_object_end();
        file.extend_from_slice(&w.into_bytes());
        file
    }

    #[test]
    fn reads_interval_compressed_values() {
        let f = IsmFile::parse(&one_interval_file(), false).unwrap();
        assert_eq!(f.header.bucket_size, 64);
        assert_eq!(f.header.nbucket, 1);
        assert_eq!(f.index.rows, vec![0, 3]);
        assert_eq!(f.index.bucket_numbers, vec![0]);
        let d = scalar_desc("X", DataType::Int);
        // All rows in the interval [0..2] share the single stored value.
        for row in 0..3 {
            assert_eq!(
                f.read_scalar_cell(0, &d, row).unwrap(),
                RecordValue::Int(42),
                "row {row}"
            );
        }
        assert!(matches!(
            f.read_scalar_cell(0, &d, 3),
            Err(IsmError::RowOutOfRange { .. })
        ));
    }

    #[test]
    fn reads_distinct_rows_across_intervals() {
        // Rows 0..2 -> 0, 2..5 -> 1, 5..6 -> 2 (three intervals), two columns
        // (Int at data offsets 0/4, Double at 8/16/24/32...).
        let bs = 128u32;
        let mut file = vec![0u8; DATA_START + bs as usize];
        let mut w = Writer::new_le();
        w.put_root_object_start("IncrementalStMan", 5);
        w.put_bool(false);
        w.put_u32(bs);
        w.put_u32(1);
        w.put_u32(0);
        w.put_u32(0);
        w.put_u32(0);
        w.put_i32(-1);
        w.put_object_end();
        let hdr = w.into_bytes();
        file[..hdr.len()].copy_from_slice(&hdr);
        let b = DATA_START;
        // Data: col0 ints {0@0, 10@4, 20@8}; col1 doubles {0.5@12, 1.5@20, 2.5@28}.
        // dataLeng 36 -> index at 40.
        file[b..b + 4].copy_from_slice(&40u32.to_le_bytes());
        file[b + 4..b + 8].copy_from_slice(&0i32.to_le_bytes());
        file[b + 8..b + 12].copy_from_slice(&10i32.to_le_bytes());
        file[b + 12..b + 16].copy_from_slice(&20i32.to_le_bytes());
        file[b + 16..b + 24].copy_from_slice(&0.5f64.to_le_bytes());
        file[b + 24..b + 32].copy_from_slice(&1.5f64.to_le_bytes());
        file[b + 32..b + 40].copy_from_slice(&2.5f64.to_le_bytes());
        // col0: nr=3 rownrs [0,2,4] offsets [0,4,8]; col1: nr=3 rownrs [0,2,4] offsets [12,20,28].
        let mut pos = 40usize;
        for col in 0..2usize {
            for v in [3u32, 0, 2, 4] {
                file[b + pos..b + pos + 4].copy_from_slice(&v.to_le_bytes());
                pos += 4;
            }
            let offsets: Vec<u32> = if col == 0 {
                vec![0, 4, 8]
            } else {
                vec![12, 20, 28]
            };
            for o in offsets {
                file[b + pos..b + pos + 4].copy_from_slice(&o.to_le_bytes());
                pos += 4;
            }
        }
        let mut w = Writer::new_le();
        w.put_root_object_start("ISMIndex", 1);
        w.put_u32(1);
        w.put_object_start("Block", 1);
        w.put_u32(2);
        w.put_u32(0);
        w.put_u32(6);
        w.put_object_end();
        w.put_object_start("Block", 1);
        w.put_u32(1);
        w.put_u32(0);
        w.put_object_end();
        w.put_object_end();
        file.extend_from_slice(&w.into_bytes());

        let f = IsmFile::parse(&file, false).unwrap();
        let int_col = scalar_desc("I", DataType::Int);
        let dbl_col = scalar_desc("D", DataType::Double);
        let ints = [0, 0, 10, 10, 20, 20];
        let dbls = [0.5, 0.5, 1.5, 1.5, 2.5, 2.5];
        for row in 0..6 {
            assert_eq!(
                f.read_scalar_cell(0, &int_col, row).unwrap(),
                RecordValue::Int(ints[row as usize]),
                "int row {row}"
            );
            assert_eq!(
                f.read_scalar_cell(1, &dbl_col, row).unwrap(),
                RecordValue::Double(dbls[row as usize]),
                "double row {row}"
            );
        }
    }

    /// Round-trip a `write_ism_file` output through `IsmFile::parse` in
    /// either byte order. `icomp` repeats values so the writer must use
    /// interval compression; `jcol` changes every row.
    fn writer_round_trip(big: bool) {
        let n: u64 = 9;
        let icomp = [5i32, 5, 5, 7, 7, 2, 2, 9, 9];
        let jcol = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let icol = scalar_desc("I", DataType::Int);
        let dcol = scalar_desc("J", DataType::Double);
        let mut ibytes = Vec::new();
        let mut jbytes = Vec::new();
        for i in 0..n as usize {
            ibytes.extend_from_slice(
                &crate::ssm::encode_scalar_cell(big, &icol, &RecordValue::Int(icomp[i])).unwrap(),
            );
            jbytes.extend_from_slice(
                &crate::ssm::encode_scalar_cell(big, &dcol, &RecordValue::Double(jcol[i])).unwrap(),
            );
        }
        let cols = [
            WriteIsmColumn {
                cell_size: 4,
                bytes: &ibytes,
            },
            WriteIsmColumn {
                cell_size: 8,
                bytes: &jbytes,
            },
        ];
        let file = write_ism_file(big, n, &cols);
        let f = IsmFile::parse(&file, big).unwrap();
        assert_eq!(f.header.bucket_size, 32768);
        assert_eq!(f.header.nbucket, 1);
        assert_eq!(f.index.rows, vec![0, 9]);
        for i in 0..n {
            assert_eq!(
                f.read_scalar_cell(0, &icol, i).unwrap(),
                RecordValue::Int(icomp[i as usize]),
                "col0 row {i}"
            );
            assert_eq!(
                f.read_scalar_cell(1, &dcol, i).unwrap(),
                RecordValue::Double(jcol[i as usize]),
                "col1 row {i}"
            );
        }
    }

    #[test]
    fn writer_round_trips_both_endians() {
        writer_round_trip(true);
        writer_round_trip(false);
    }

    #[test]
    fn writer_spans_buckets() {
        // One Int column: rows_per_bucket = (32768 - 4)/12 = 2730, so 2800
        // rows force a second ISM bucket; check around the boundary.
        let n: u64 = 2800;
        let icol = scalar_desc("I", DataType::Int);
        let vals: Vec<i32> = (0..n as i32).map(|i| i % 7 - 3).collect();
        let mut bytes = Vec::with_capacity(n as usize * 4);
        for &v in &vals {
            bytes.extend_from_slice(
                &crate::ssm::encode_scalar_cell(false, &icol, &RecordValue::Int(v)).unwrap(),
            );
        }
        let cols = [WriteIsmColumn {
            cell_size: 4,
            bytes: &bytes,
        }];
        let file = write_ism_file(false, n, &cols);
        let f = IsmFile::parse(&file, false).unwrap();
        assert!(
            f.header.nbucket > 1,
            "must span buckets: nbucket={}",
            f.header.nbucket
        );
        assert_eq!(f.index.rows.first(), Some(&0));
        assert_eq!(f.index.rows.last(), Some(&2800));
        for (i, &v) in vals.iter().enumerate().skip(2700).take(150) {
            assert_eq!(
                f.read_scalar_cell(0, &icol, i as u64).unwrap(),
                RecordValue::Int(v),
                "row {i}"
            );
        }
    }

    #[test]
    fn writer_ism_blob_parses_as_spec() {
        let blob = write_ism_blob("IncrementalStMan");
        let mut r = crate::aipsio::Reader::new(&blob);
        let obj = r.read_object_start(true).unwrap();
        assert_eq!(obj.type_name, "ISM");
        assert_eq!(obj.version, 3);
        assert_eq!(r.read_string().unwrap(), "IncrementalStMan");
    }
}
