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

/// The bucket size casacure writes (casacore's default).
pub const ISM_BUCKET_SIZE: usize = 32768;

/// Errors from reading an IncrementalStMan data file.
#[derive(Debug, Error)]
pub enum IsmError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    FileIo(#[from] crate::datafile::FileIoError),
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
    #[error("IncrementalStMan array column {0} is not supported yet")]
    NotScalar(String),
    #[error("IncrementalStMan array column {0} has no array index file (table.f{{seq}}i)")]
    MissingArrayFile(String),
    #[error("IncrementalStMan string cell of column {column} at row {row} is corrupt (cell bytes {cell})")]
    CorruptString { column: usize, row: u64, cell: u32 },
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
    /// Optional `table.f{seq}i` array index file: an ISM indirect array
    /// column's bucket cell is an 8-byte offset into this `StManArrayFile`
    /// (the same record format StandardStMan uses, issue #48).
    f0i: Option<crate::datafile::Buffer>,
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
        let path = table_dir.as_ref().join(format!("table.f{seq_nr}"));
        let file = std::fs::File::open(&path)
            .map_err(|e| IsmError::FileIo(crate::datafile::FileIoError::new(&path, e)))?;
        let data = crate::datafile::Buffer::from_file(file)
            .map_err(|e| IsmError::FileIo(crate::datafile::FileIoError::new(&path, e)))?;
        let (header, index) = Self::parse_meta(&data, table_big_endian)?;
        let f0i = Self::open_array_file(table_dir.as_ref(), seq_nr)?;
        Ok(IsmFile {
            header,
            index,
            data,
            f0i,
        })
    }

    /// The optional array index file (`table.f{seq}i`) only exists for
    /// tables with ISM indirect array columns.
    fn open_array_file(
        table_dir: &std::path::Path,
        seq_nr: u32,
    ) -> Result<Option<crate::datafile::Buffer>, IsmError> {
        let path = table_dir.join(format!("table.f{seq_nr}i"));
        if !path.is_file() {
            return Ok(None);
        }
        let file = std::fs::File::open(&path)
            .map_err(|e| IsmError::FileIo(crate::datafile::FileIoError::new(&path, e)))?;
        crate::datafile::Buffer::from_file(file)
            .map(Some)
            .map_err(|e| IsmError::FileIo(crate::datafile::FileIoError::new(&path, e)))
    }

    /// Parse an IncrementalStMan data file.
    pub fn parse(data: &[u8], table_big_endian: bool) -> Result<IsmFile, IsmError> {
        let (header, index) = Self::parse_meta(data, table_big_endian)?;
        Ok(IsmFile {
            header,
            index,
            data: crate::datafile::Buffer::from(data.to_vec()),
            f0i: None,
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

    /// Drop this file's mapped data pages (used after a bulk read has copied
    /// the cells out, to keep streaming scans resident at ~the working set).
    pub fn drop_data_pages(&self) {
        self.data.drop_pages();
        if let Some(f0i) = &self.f0i {
            f0i.drop_pages();
        }
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

    /// Read the cell of `column` (index within this data manager) at `row`:
    /// a scalar decodes inline, a string from its u32-prefixed inline cell,
    /// an array through its 8-byte reference into the array index file
    /// (issue #48). A row the column has no interval for is
    /// [`IsmError::ColumnEmpty`], exactly as for a scalar.
    pub fn read_cell(
        &self,
        column: usize,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<RecordValue, IsmError> {
        match (&desc.kind, desc.data_type) {
            (crate::tabledesc::ColumnKind::Array, DataType::String) => {
                // An array of strings stores its elements in the string
                // buckets of the index file; no fixture covers it yet, so
                // the #47 named error stays until one does.
                Err(IsmError::NotScalar(desc.name.clone()))
            }
            (crate::tabledesc::ColumnKind::Array, _) => {
                let mut out = None;
                self.for_each_cell_at(column, row, 1, |b, off, r| {
                    out = Some(self.decode_array_at(b, off, r, column, desc)?);
                    Ok::<(), IsmError>(())
                })?;
                out.ok_or(IsmError::RowOutOfRange { row })
            }
            (_, DataType::String) => Ok(RecordValue::String(self.read_string_cell(column, row)?)),
            _ => self.read_scalar_cell(column, desc, row),
        }
    }

    /// Read the scalar value for `column` (index within this data manager)
    /// at `row`. The value is the interval entry covering `row`.
    pub fn read_scalar_cell(
        &self,
        column: usize,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<RecordValue, IsmError> {
        let mut out = None;
        self.for_each_cell_raw(column, desc, row, 1, |cell| {
            out = Some(crate::ssm::decode_scalar(
                cell,
                desc,
                self.header.big_endian,
            )?);
            Ok::<(), IsmError>(())
        })?;
        out.ok_or(IsmError::RowOutOfRange { row })
    }

    /// Read one scalar string: the inline cell is `[u32 cell_bytes =
    /// 4 + nchar][chars]` in the data file's byte order, packed without
    /// alignment (verified against casacore-written tables, issue #48).
    pub fn read_string_cell(&self, column: usize, row: u64) -> Result<String, IsmError> {
        let mut out = None;
        self.for_each_cell_at(column, row, 1, |b, off, r| {
            out = Some(self.decode_string_at(b, off, r, column)?);
            Ok::<(), IsmError>(())
        })?;
        out.ok_or(IsmError::RowOutOfRange { row })
    }

    fn decode_string_at(
        &self,
        b: &[u8],
        off: usize,
        row: u64,
        column: usize,
    ) -> Result<String, IsmError> {
        let head = b
            .get(off..off + 4)
            .ok_or(IsmError::ColumnEmpty { column, row })?;
        let cell = if self.header.big_endian {
            u32::from_be_bytes(head.try_into().unwrap())
        } else {
            u32::from_le_bytes(head.try_into().unwrap())
        };
        let len = cell
            .checked_sub(4)
            .ok_or(IsmError::CorruptString { column, row, cell })?;
        let bytes = b
            .get(off + 4..off + 4 + len as usize)
            .ok_or(IsmError::CorruptString { column, row, cell })?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Decode one indirect array cell: the 8-byte reference is null (0)
    /// for a row that was never written — casacore's "no array", which
    /// reads as the column default exactly as a StandardStMan undefined
    /// cell does (#45) — otherwise it points at a `StIndArray` record in
    /// the array index file (`[u32 refcount][u32 ndim][u32 CASA dims...]
    /// [elements]`, the StandardStMan format).
    fn decode_array_at(
        &self,
        b: &[u8],
        off: usize,
        row: u64,
        column: usize,
        desc: &ColumnDesc,
    ) -> Result<RecordValue, IsmError> {
        let cell = b
            .get(off..off + 8)
            .ok_or(IsmError::ColumnEmpty { column, row })?;
        let offset = if self.header.big_endian {
            i64::from_be_bytes(cell.try_into().unwrap())
        } else {
            i64::from_le_bytes(cell.try_into().unwrap())
        };
        if offset == 0 {
            return crate::table::default_cell_value(desc)
                .ok_or_else(|| IsmError::NotScalar(desc.name.clone()));
        }
        let f0i = self
            .f0i
            .as_deref()
            .ok_or_else(|| IsmError::MissingArrayFile(desc.name.clone()))?;
        let (logical, nelem, data) =
            crate::ssm::st_ind_array_region(f0i, self.header.big_endian, offset, desc.data_type)?;
        Ok(crate::ssm::decode_array_elements(
            self.header.big_endian,
            desc.data_type,
            logical,
            nelem,
            data,
        )?)
    }

    /// Whether the array cell of `column` at `row` holds an array:
    /// casacore's `isShapeDefined` — the 8-byte reference is non-null.
    /// Scalar and string columns are defined in every row; a row with no
    /// interval at all (added but never written) is not.
    pub fn is_cell_defined(
        &self,
        column: usize,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<bool, IsmError> {
        if !matches!(desc.kind, crate::tabledesc::ColumnKind::Array) {
            return Ok(true);
        }
        let mut out = false;
        self.for_each_cell_at(column, row, 1, |b, off, _| {
            let cell = b
                .get(off..off + 8)
                .ok_or(IsmError::ColumnEmpty { column, row })?;
            let offset = if self.header.big_endian {
                i64::from_be_bytes(cell.try_into().unwrap())
            } else {
                i64::from_le_bytes(cell.try_into().unwrap())
            };
            out = offset != 0;
            Ok::<(), IsmError>(())
        })?;
        Ok(out)
    }

    /// Visit the decoded cells of `column` for rows `start .. start+nrow`
    /// (the `getcol` path): one linear walk of each bucket's interval
    /// index, decoding per the column kind.
    pub fn for_each_decoded<E: From<IsmError>>(
        &self,
        column: usize,
        desc: &ColumnDesc,
        start: u64,
        nrow: u64,
        mut visit: impl FnMut(RecordValue) -> Result<(), E>,
    ) -> Result<(), E> {
        match (&desc.kind, desc.data_type) {
            (crate::tabledesc::ColumnKind::Array, DataType::String) => {
                Err(IsmError::NotScalar(desc.name.clone()).into())
            }
            (crate::tabledesc::ColumnKind::Array, _) => {
                self.for_each_cell_at(column, start, nrow, |b, off, r| {
                    visit(self.decode_array_at(b, off, r, column, desc)?)
                })
            }
            (_, DataType::String) => self.for_each_cell_at(column, start, nrow, |b, off, r| {
                visit(RecordValue::String(
                    self.decode_string_at(b, off, r, column)?,
                ))
            }),
            _ => {
                let big_endian = self.header.big_endian;
                self.for_each_cell_raw(column, desc, start, nrow, |cell| {
                    visit(
                        crate::ssm::decode_scalar(cell, desc, big_endian)
                            .map_err(IsmError::from)?,
                    )
                })
            }
        }
    }

    /// Walk the cells of `column` for rows `start .. start+nrow`, handing
    /// the visitor the bucket bytes and the data-area offset of each row's
    /// cell. The cell's extent depends on the column kind (fixed-size
    /// scalar, u32-prefixed string, 8-byte array reference), so unlike
    /// [`IsmFile::for_each_cell_raw`] the visitor bounds-checks its own
    /// slice. Each bucket's interval index is parsed once and walked
    /// forward with the rows, so a range read is linear in the rows plus
    /// the intervals it spans.
    fn for_each_cell_at<'a, E: From<IsmError>>(
        &'a self,
        column: usize,
        start: u64,
        nrow: u64,
        mut visit: impl FnMut(&'a [u8], usize, u64) -> Result<(), E>,
    ) -> Result<(), E> {
        let data_off = 4usize; // data area starts after the index-offset u32
        let end = start.saturating_add(nrow);
        let mut row = start;
        while row < end {
            let (bucket, intra_row, bucket_rows) = self
                .index
                .bucket_for(row)
                .ok_or(IsmError::RowOutOfRange { row })?;
            if intra_row >= bucket_rows {
                return Err(IsmError::RowOutOfRange { row }.into());
            }
            let ci = self.column_index(bucket, column)?;
            let b = self.bucket_bytes(bucket)?;
            // Interval containing the first intra-bucket row.
            let lb = ci.rows.partition_point(|&r| r < intra_row);
            let mut inx = if lb < ci.rows.len() && ci.rows[lb] == intra_row {
                lb
            } else {
                lb.checked_sub(1)
                    .ok_or(IsmError::ColumnEmpty { column, row })?
            };
            let stop = (bucket_rows - intra_row).min(end - row);
            for k in 0..stop {
                let ir = intra_row + k;
                while inx + 1 < ci.rows.len() && ci.rows[inx + 1] <= ir {
                    inx += 1;
                }
                let offset = data_off + ci.offsets[inx] as usize;
                visit(b, offset, row + k)?;
            }
            row += stop;
        }
        Ok(())
    }

    /// Visit the raw stored bytes (data-file byte order, [`ism_cell_size`]
    /// long) of `column` for rows `start .. start + nrow`, borrowed from
    /// the mapped file. Scalar columns only (the raw bulk-fill path).
    ///
    /// Each bucket's interval index is parsed once and walked forward with
    /// the rows, so a range read is linear in the rows plus the intervals
    /// it spans. (Resolving every row on its own re-parses the whole bucket
    /// index per row: quadratic for a column that changes every row, like
    /// an MS `ANTENNA1`.)
    pub fn for_each_cell_raw<'a, E: From<IsmError>>(
        &'a self,
        column: usize,
        desc: &ColumnDesc,
        start: u64,
        nrow: u64,
        mut visit: impl FnMut(&'a [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        if desc.data_type == DataType::String {
            return Err(IsmError::NotScalar(desc.name.clone()).into());
        }
        if matches!(desc.kind, crate::tabledesc::ColumnKind::Array) {
            // An array cell is not one scalar at the stored offset.
            return Err(IsmError::NotScalar(desc.name.clone()).into());
        }
        let want = ism_cell_size(desc) as usize;
        self.for_each_cell_at(column, start, nrow, |b, offset, row| {
            let cell = b
                .get(offset..offset + want)
                .ok_or(IsmError::ColumnEmpty { column, row })?;
            visit(cell)
        })
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

/// Bytes stored per row cell of a scalar column in an ISM bucket.
///
/// Unlike StandardStMan — which bit-packs Bool rows into a shared byte, so
/// `scalar_cell_size` reports 0 — an IncrementalStMan bucket stores whole
/// cells back to back, and casacore writes one byte per Bool cell (three rows
/// are `01 00 01`). Decoding with the SSM size would slice the (empty) cell
/// down to zero bytes and fail with "buffer too short: need 1 bytes".
pub fn ism_cell_size(desc: &ColumnDesc) -> u32 {
    match desc.data_type {
        DataType::Bool => 1,
        _ => crate::ssm::scalar_cell_size(desc),
    }
}

/// Per-column encoded cells handed to `write_ism_file`. Cells vary in size
/// (strings carry their u32 length prefix; arrays an 8-byte reference), so
/// the column is a list of per-row byte strings rather than one flat
/// `n_rows * cell_size` buffer.
#[derive(Debug, Clone, Copy)]
pub struct WriteIsmColumn<'a> {
    /// The encoded cell bytes, one per row.
    pub cells: &'a [Vec<u8>],
}

/// Encode one scalar string cell as ISM stores it inline in the bucket
/// data area: `[u32 cell_bytes = 4 + nchar][chars]` in the data file's
/// byte order, packed without alignment (issue #48).
pub fn encode_ism_string_cell(big_endian: bool, s: &str) -> Vec<u8> {
    let mut cell = Vec::with_capacity(4 + s.len());
    let n = (4 + s.len()) as u32;
    cell.extend_from_slice(&if big_endian {
        n.to_be_bytes()
    } else {
        n.to_le_bytes()
    });
    cell.extend_from_slice(s.as_bytes());
    cell
}

/// Assemble an array index file (`table.f{seq}i`) from `StIndArray` record
/// payloads and their reference counts. casacore's ISM opens its array file
/// with `StManArrayFile(fileName() + 'i', opt, 1, ...)` — version **1**,
/// which prefixes every record with a 4-byte reference count — so the
/// header is `[u32 version = 1][Int64 file length][u32 spare]` in the data
/// file's byte order and each record is `[u32 refcount][ndim][CASA dims]
/// [elements]` (the refcount is casacore's shared-record bookkeeping:
/// adjacent rows holding equal arrays share one record). Readers of both
/// versions exist on both sides, so files written either way read
/// everywhere.
pub fn build_array_index_file(big_endian: bool, records: &[(Vec<u8>, u32)]) -> Vec<u8> {
    let mut f0i = vec![0u8; 16];
    let len = (16 + records.iter().map(|(r, _)| 4 + r.len()).sum::<usize>()) as i64;
    f0i[0..4].copy_from_slice(&if big_endian {
        1u32.to_be_bytes()
    } else {
        1u32.to_le_bytes()
    });
    f0i[4..12].copy_from_slice(&if big_endian {
        len.to_be_bytes()
    } else {
        len.to_le_bytes()
    });
    for (r, refcount) in records {
        f0i.extend_from_slice(&if big_endian {
            refcount.to_be_bytes()
        } else {
            refcount.to_le_bytes()
        });
        f0i.extend_from_slice(r);
    }
    f0i
}

/// Serialize a complete IncrementalStMan data file — header, data buckets
/// with interval compression, and the `ISMIndex` at the end — matching what
/// casacore writes for scalar columns. Consecutive rows with identical bytes
/// in a column share one stored value (an interval entry), exactly as
/// `ISMBucket::addData` does.
pub fn write_ism_file(big_endian: bool, n_rows: u64, cols: &[WriteIsmColumn<'_>]) -> Vec<u8> {
    write_ism_file_with(big_endian, n_rows, cols, 1, 0)
}

/// One ISM bucket under construction: the data area bytes and, per column,
/// the interval index (start rows and data offsets) plus the value bytes
/// covering the next row (interval compression state).
struct BucketState {
    data: Vec<u8>,
    /// Per column: (interval start rows, data offsets).
    index: Vec<(Vec<u64>, Vec<u32>)>,
    /// Bytes of the value covering the next row, per column.
    last: Vec<Option<Vec<u8>>>,
}

/// [`write_ism_file`] with the header's persistent-cache and unique-number
/// fields spelled out. casacore's ISM defaults `persCacheSize` to 1 (its
/// spec constructor, `ISMBase.cc`); `uniqnr` is 1 per indirect-array column
/// the group holds (opening its `StManArrayFile` bumps the manager's
/// counter), 0 for scalar-only groups — verified byte-for-byte against
/// casacore-written tables.
pub fn write_ism_file_with(
    big_endian: bool,
    n_rows: u64,
    cols: &[WriteIsmColumn<'_>],
    pers_cache_size: u32,
    uniq_nr: u32,
) -> Vec<u8> {
    // Match casacore's default ISM bucket size.
    let bucket_size = ISM_BUCKET_SIZE;

    // Fill buckets the way a clean casacore column write fills them
    // (ISMBucket::canAddData): a column's cell starts a new interval
    // whenever its bytes differ from the previous row's, each new interval
    // costing its bytes plus 12 of index, with a bucket holding rows until
    // `dataLeng + cell + indexLeng + 4 + 8 > bucketSize` — the same
    // capacity casacore's 64-bit-rownr builds account (single-column tables
    // come out byte-identical to casacore's). Boundaries are row-atomic: a
    // row that would straddle a full bucket starts the next one, with every
    // column re-adding its interval there — intervals never span buckets,
    // so a read of any column at any row finds it. (casacore allows
    // mid-row boundaries and duplicates the straddling values; its own
    // boundaries moreover depend on the write session's history — putcol
    // and putcell of the same data produce different files — so byte parity
    // beyond the single-column layout is not defined; see DIFFERENCES.md.)
    let ncols = cols.len();
    let mut buckets: Vec<BucketState> = Vec::new();
    let mut cur = BucketState {
        data: Vec::new(),
        index: vec![(Vec::new(), Vec::new()); ncols],
        last: vec![None; ncols],
    };
    // The index length tracks the encoded form: the indexOffset u32, one
    // u32 per column, and u32 row + u32 offset + u64 rownr per interval
    // (ISMBucket::canAddData's accounting).
    let mut cur_index_len = 4 + 4 * ncols;
    let mut new_bucket = |cur: &mut BucketState, index_len: &mut usize| {
        buckets.push(std::mem::replace(
            cur,
            BucketState {
                data: Vec::new(),
                index: vec![(Vec::new(), Vec::new()); ncols],
                last: vec![None; ncols],
            },
        ));
        *index_len = 4 + 4 * ncols;
    };
    for r in 0..n_rows as usize {
        // Would every new interval of this row fit the current bucket,
        // checked in order with casacore's exact canAddData accounting
        // (`dataLeng + cell + indexLeng + 4 + 8 <= bucketSize`, against the
        // index length before each add's own 12 bytes)?
        let mut data_len = cur.data.len();
        let mut index_len = cur_index_len;
        let mut fits = true;
        for (c, col) in cols.iter().enumerate() {
            if cur.last[c].as_deref() != Some(col.cells[r].as_slice()) {
                if data_len + col.cells[r].len() + index_len + 4 + 8 > bucket_size {
                    fits = false;
                    break;
                }
                data_len += col.cells[r].len();
                index_len += 12;
            }
        }
        if !fits {
            new_bucket(&mut cur, &mut cur_index_len);
        }
        for (c, col) in cols.iter().enumerate() {
            let cell = &col.cells[r];
            if cur.last[c].as_deref() == Some(cell.as_slice()) {
                continue; // the open interval covers this row too
            }
            cur.index[c].0.push(r as u64);
            cur.index[c].1.push(cur.data.len() as u32);
            cur.data.extend_from_slice(cell);
            cur.last[c] = Some(cell.clone());
            cur_index_len += 12;
        }
    }
    buckets.push(cur);
    let nr_buckets = buckets.len();

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
    hw.put_u32(pers_cache_size);
    hw.put_u32(uniq_nr);
    hw.put_u32(0); // free buckets
    hw.put_i32(-1); // first free bucket
    hw.put_object_end();
    file.extend_from_slice(&hw.into_bytes());
    file.resize(512, 0);

    // Buckets.
    let mut boundaries: Vec<u64> = Vec::with_capacity(nr_buckets + 1);
    let mut numbers: Vec<u32> = Vec::with_capacity(nr_buckets);
    for (i, b) in buckets.into_iter().enumerate() {
        // A bucket's start is the first row any of its intervals covers.
        let start = b
            .index
            .iter()
            .flat_map(|(rows, _)| rows.iter().copied())
            .min()
            .unwrap_or(0);
        boundaries.push(start);
        numbers.push(i as u32);
        file.extend_from_slice(&encode_bucket_state(big_endian, bucket_size, start, b));
    }
    boundaries.push(n_rows);

    // ISMIndex at the end.
    file.extend_from_slice(&encode_ism_index(big_endian, &boundaries, &numbers));
    file
}

/// Encode one filled bucket state into `bucket_size` bytes: the indexOffset
/// u32, the data area, then the per-column interval index with **bucket-
/// relative** row numbers (32-bit).
fn encode_bucket_state(
    big_endian: bool,
    bucket_size: usize,
    start_row: u64,
    state: BucketState,
) -> Vec<u8> {
    let mut bucket = vec![0u8; bucket_size];
    let index_len: usize = state
        .index
        .iter()
        .map(|(rows, _)| 4 + 4 * rows.len() + 4 * rows.len())
        .sum();
    let index_offset = (state.data.len() + 4) as u32;
    debug_assert!(4 + state.data.len() + index_len <= bucket_size);
    put_u32_e(&mut bucket, 0, index_offset, big_endian);
    bucket[4..4 + state.data.len()].copy_from_slice(&state.data);
    let mut pos = 4usize + state.data.len();
    for (rows, offsets) in &state.index {
        put_u32_e(&mut bucket, pos, rows.len() as u32, big_endian);
        pos += 4;
        for &s in rows {
            put_u32_e(&mut bucket, pos, (s - start_row) as u32, big_endian);
            pos += 4;
        }
        for &o in offsets {
            put_u32_e(&mut bucket, pos, o, big_endian);
            pos += 4;
        }
    }
    bucket
}

/// The rows an IncrementalStMan bucket of `bucket_size` bytes can always
/// hold, whatever the values: every row a new interval in every column
/// (`[u32 indexOffset]` + the cells + per column `[u32 nr]` and a u32 row
/// and u32 offset per interval). Variable-size cells are bounded by their
/// longest row.
pub fn ism_rows_per_bucket(bucket_size: usize, cols: &[WriteIsmColumn<'_>]) -> usize {
    let ncols = cols.len();
    let sum_sizes: usize = cols
        .iter()
        .map(|c| c.cells.iter().map(Vec::len).max().unwrap_or(0))
        .sum();
    let per_row = sum_sizes + 8 * ncols;
    (bucket_size.saturating_sub(4 + 4 * ncols) / per_row.max(1)).max(1)
}

/// Encode one IncrementalStMan bucket of `nrows` rows: `cols[c].cells`
/// holds exactly `nrows` encoded cells of column `c`.  Consecutive rows
/// with identical bytes share one stored value (an interval entry), exactly
/// as `ISMBucket::addData` does.  `None` when the encoding does not fit.
pub fn encode_ism_bucket(
    big_endian: bool,
    bucket_size: usize,
    nrows: u64,
    cols: &[WriteIsmColumn<'_>],
) -> Option<Vec<u8>> {
    let ncols = cols.len();
    let mut data: Vec<u8> = Vec::new();
    // Per column: (intrabucket start row, data offset) of each interval.
    let mut starts: Vec<Vec<u32>> = vec![Vec::new(); ncols];
    let mut offsets: Vec<Vec<u32>> = vec![Vec::new(); ncols];
    for (c, col) in cols.iter().enumerate() {
        let mut prev: Option<&[u8]> = None;
        for (row, cell) in col.cells.iter().enumerate() {
            if prev != Some(cell.as_slice()) {
                starts[c].push(row as u32);
                offsets[c].push(0);
            }
            prev = Some(cell);
        }
    }
    // Lay the data out row-major across the columns (the order casacore's
    // addData appends it), then fill in each interval's offset.
    let mut next = vec![0usize; ncols];
    for row in 0..nrows as usize {
        for (c, col) in cols.iter().enumerate() {
            if next[c] < starts[c].len() && starts[c][next[c]] as usize == row {
                offsets[c][next[c]] = data.len() as u32;
                data.extend_from_slice(&col.cells[row]);
                next[c] += 1;
            }
        }
    }
    let index_len: usize = starts.iter().map(|s| 4 + 8 * s.len()).sum();
    if 4 + data.len() + index_len > bucket_size {
        return None;
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
    Some(bucket)
}

/// Serialize the `ISMIndex` object: `boundaries` holds the `nused + 1`
/// ascending bucket-boundary rows, `numbers` the bucket of each range.
/// Version 1 (u32 rows) unless a row needs 64 bits.
pub fn encode_ism_index(big_endian: bool, boundaries: &[u64], numbers: &[u32]) -> Vec<u8> {
    let wide = boundaries.last().is_some_and(|&r| r > u64::from(u32::MAX));
    let mut iw = if big_endian {
        crate::aipsio::Writer::new()
    } else {
        crate::aipsio::Writer::new_le()
    };
    iw.put_root_object_start("ISMIndex", if wide { 2 } else { 1 });
    iw.put_u32(numbers.len() as u32); // nused
    iw.put_object_start("Block", 1);
    iw.put_u32(boundaries.len() as u32);
    for &b in boundaries {
        if wide {
            iw.put_u64(b);
        } else {
            iw.put_u32(b as u32);
        }
    }
    iw.put_object_end();
    iw.put_object_start("Block", 1);
    iw.put_u32(numbers.len() as u32);
    for &n in numbers {
        iw.put_u32(n);
    }
    iw.put_object_end();
    iw.put_object_end();
    iw.into_bytes()
}

/// Serialize the IncrementalStMan data-file header (padded to
/// [`DATA_START`] by the caller): v4 is big endian with no flag, v5 carries
/// the endian flag.
pub fn encode_ism_header(h: &IsmHeader) -> Vec<u8> {
    let mut hw = if h.big_endian {
        crate::aipsio::Writer::new()
    } else {
        crate::aipsio::Writer::new_le()
    };
    hw.put_root_object_start("IncrementalStMan", h.version);
    if h.version >= 5 {
        hw.put_bool(h.big_endian);
    }
    hw.put_u32(h.bucket_size);
    hw.put_u32(h.nbucket);
    hw.put_u32(h.pers_cache_size);
    hw.put_u32(h.uniq_nr);
    hw.put_u32(h.n_free_bucket);
    hw.put_i32(h.first_free_bucket);
    hw.put_object_end();
    hw.into_bytes()
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

/// Decode the data-manager name an IncrementalStMan spec blob carries
/// (`ISMBase::flush` writes the manager name as the payload of a root
/// `"ISM"` object and `ISMBase::open64` reads it back, so casacore's
/// `ColumnSet::dataManagerInfo` reports it as the `NAME` of the
/// `getdminfo()` entry). `None` for bytes that are not an `"ISM"` blob —
/// an unknown manager's raw spec, which stays named by its type.
pub fn parse_ism_name(blob: &[u8]) -> Option<String> {
    let mut r = crate::aipsio::Reader::new(blob);
    let obj = r.read_object_start(true).ok()?;
    if obj.type_name != "ISM" {
        return None;
    }
    r.read_string().ok()
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
                record_type: crate::record::RECORD_VARIABLE,
                values: Vec::new(),
            },
            kind: ColumnKind::Scalar(RecordValue::Int(0)),
            tile_shape: None,
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
    fn rejects_only_string_arrays() {
        // A string-array column keeps the #47 named error: its elements live
        // in the string buckets of the index file, which no fixture covers
        // yet. Numeric arrays and scalar strings read (issues #47/#48).
        let f = IsmFile::parse(&one_interval_file(), false).unwrap();
        let mut arr = scalar_desc("A", DataType::Float);
        arr.ndim = 1;
        arr.shape = Some(vec![4]);
        arr.kind = ColumnKind::Array;
        // The bucket cell is an 8-byte array reference; without an array
        // index file the read names the missing file.
        assert!(matches!(
            f.read_cell(0, &arr, 0),
            Err(IsmError::MissingArrayFile(name)) if name == "A"
        ));
        assert!(matches!(
            f.is_cell_defined(0, &arr, 0),
            Ok(true) // the synthetic cell holds a non-null reference
        ));
        let mut sa = scalar_desc("SA", DataType::String);
        sa.ndim = 1;
        sa.kind = ColumnKind::Array;
        assert!(matches!(
            f.read_cell(0, &sa, 0),
            Err(IsmError::NotScalar(name)) if name == "SA"
        ));
    }

    #[test]
    fn reads_inline_string_cells() {
        // Scalar strings are inline `[u32 cell_bytes = 4 + nchar][chars]`,
        // interval-compressed like scalars (issue #48). Hand-build a bucket:
        // rows 0..2 "aa", 2..4 "b", 4..5 "" — three intervals, one column.
        let bs = 128u32;
        // Cells: "aa" -> u32 6 + 2 chars, "b" -> u32 5 + 1 char, "" -> u32 4.
        let cells: [&[u8]; 3] = [
            &[6, 0, 0, 0, b'a', b'a'],
            &[5, 0, 0, 0, b'b'],
            &[4, 0, 0, 0],
        ];
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
        // Data area at 4..4+15; the index follows at 19.
        file[b..b + 4].copy_from_slice(&19u32.to_le_bytes());
        let mut pos = b + 4;
        for cell in cells {
            file[pos..pos + cell.len()].copy_from_slice(cell);
            pos += cell.len();
        }
        // column 0 index: nr=3, rownrs=[0,2,4], offsets=[0,6,11].
        file[pos..pos + 4].copy_from_slice(&3u32.to_le_bytes());
        pos += 4;
        for r in [0u32, 2, 4] {
            file[pos..pos + 4].copy_from_slice(&r.to_le_bytes());
            pos += 4;
        }
        for o in [0u32, 6, 11] {
            file[pos..pos + 4].copy_from_slice(&o.to_le_bytes());
            pos += 4;
        }
        let mut w = Writer::new_le();
        w.put_root_object_start("ISMIndex", 1);
        w.put_u32(1);
        w.put_object_start("Block", 1);
        w.put_u32(2);
        w.put_u32(0);
        w.put_u32(5);
        w.put_object_end();
        w.put_object_start("Block", 1);
        w.put_u32(1);
        w.put_u32(0);
        w.put_object_end();
        w.put_object_end();
        file.extend_from_slice(&w.into_bytes());

        let f = IsmFile::parse(&file, false).unwrap();
        let desc = scalar_desc("S", DataType::String);
        let want = ["aa", "aa", "b", "b", ""];
        for (row, expect) in want.iter().enumerate() {
            assert_eq!(
                f.read_cell(0, &desc, row as u64).unwrap(),
                RecordValue::String(expect.to_string()),
                "row {row}"
            );
        }
        // Whole-column and windowed walks agree with the per-cell reads.
        let mut all = Vec::new();
        f.for_each_decoded(0, &desc, 0, 5, |v| {
            all.push(v);
            Ok::<(), IsmError>(())
        })
        .unwrap();
        assert_eq!(all.len(), 5);
        assert_eq!(
            all[4],
            RecordValue::String(String::new()),
            "windowed walk row 4"
        );
        assert!(matches!(
            f.read_cell(0, &desc, 5),
            Err(IsmError::RowOutOfRange { row: 5 })
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
        let icells: Vec<Vec<u8>> = (0..n as usize)
            .map(|i| {
                crate::ssm::encode_scalar_cell(big, &icol, &RecordValue::Int(icomp[i])).unwrap()
            })
            .collect();
        let jcells: Vec<Vec<u8>> = (0..n as usize)
            .map(|i| {
                crate::ssm::encode_scalar_cell(big, &dcol, &RecordValue::Double(jcol[i])).unwrap()
            })
            .collect();
        let cols = [
            WriteIsmColumn { cells: &icells },
            WriteIsmColumn { cells: &jcells },
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
        let cells: Vec<Vec<u8>> = vals
            .iter()
            .map(|&v| crate::ssm::encode_scalar_cell(false, &icol, &RecordValue::Int(v)).unwrap())
            .collect();
        let cols = [WriteIsmColumn { cells: &cells }];
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

    #[test]
    fn parses_the_manager_name_out_of_an_ism_blob() {
        // The name the writer put in is the name casacore reports back.
        assert_eq!(
            parse_ism_name(&write_ism_blob("IndexingGroup")).as_deref(),
            Some("IndexingGroup")
        );
        // Bytes that are not an ISM blob yield no name (they keep the
        // type-name fallback).
        assert_eq!(parse_ism_name(b"\xde\xad\xbe\xef"), None);
        assert_eq!(parse_ism_name(&[]), None);
    }
}
