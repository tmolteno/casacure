//! Reader for the StandardStMan data file (`table.f0`) of a CASA table
//! (`casacore/tables/DataMan/SSMBase.cc`, `SSMIndex.cc`, `SSMColumn.cc`).
//!
//! The data file is an AipsIO stream in the table's *data-file* endianness
//! (big or little). Layout:
//!
//! - A root `"StandardStMan"` v2/v3 object: the storage-manager header
//!   (bucket size, bucket counts, index location; v3 writes an explicit
//!   endianness flag).
//! - Buckets starting at fixed file offset 512: data bucket `b` lives at
//!   `512 + b*bucket_size`. Index buckets live in the same region.
//! - The index is a chain of index buckets (each carrying `[checkNr][nextBucket]`
//!   in its first 8 bytes); together they hold a contiguous AipsIO stream
//!   with one framed root `"SSMIndex"` object per column group.
//! - Each `SSMIndex` maps row ranges to data buckets: for the used buckets,
//!   `lastRow[i]` (ascending) and `bucketNumber[i]`. Row `r` lives in bucket
//!   `bucketNumber[i]` with `lastRow[i-1] < r <= lastRow[i]`.
//! - Within a data bucket, every column owned by the index has a contiguous
//!   region of `rowsPerBucket * cell_size` bytes starting at the column's
//!   `columnOffset`; row `r`'s cell is at `+ (r - start_row) * cell_size`
//!   (bit-packed Bool, inline short strings, etc. — see `read_scalar_cell`).

use crate::aipsio::{AipsIoError, Reader};
use crate::record::{DataType, RecordValue};
use crate::tabledesc::{ColumnDesc, ColumnKind};
use thiserror::Error;

/// Bytes between the start of the data file and the first bucket
/// (`BucketCache` start offset in `SSMBase::makeCache`).
pub const DATA_START: usize = 512;

/// Errors from reading a StandardStMan data file.
#[derive(Debug, Error)]
pub enum SsmError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("unexpected object type {found:?}, expected {expected:?}")]
    UnexpectedType { expected: String, found: String },
    #[error("data file endian flag {flag} does not match table flag {expected}")]
    EndianMismatch { flag: bool, expected: bool },
    #[error("index says bucket {0} but the file has no such bucket")]
    InvalidBucket(u32),
    #[error("index {index} not found (file holds {count})")]
    IndexMissing { index: usize, count: usize },
    #[error("row {row} not covered by any indexed bucket")]
    RowOutOfRange { row: u64 },
    #[error("cell at offset {offset} (len {len}) falls outside bucket {bucket}")]
    CellOutOfRange { bucket: u32, offset: u64, len: u64 },
    #[error(
        "string data stored in string buckets (bucket nr {bucket}, len {len}) is not supported yet"
    )]
    StringBucketUnsupported { bucket: i32, len: i32 },
    #[error("array columns are not supported yet (column {0})")]
    ArrayColumn(String),
    #[error("column {0} has no StandardStMan data (data-manager type {1})")]
    NotStandardStMan(String, String),
    #[error("array column {0} needs the {1} array index file, which is missing")]
    MissingArrayFile(String, String),
    #[error("row {row} of {column} has no array (empty reference)")]
    EmptyArray { row: u64, column: String },
    #[error("array reference {offset} falls outside the array index file (len {len})")]
    BadArrayRef { offset: i64, len: usize },
    #[error("array elements of type {0:?} are not supported in the array index file yet")]
    UnsupportedArrayType(DataType),
    #[error("bad string reference: bucket {bucket}, offset {offset}, length {length}")]
    BadStringRef {
        bucket: i32,
        offset: i32,
        length: u32,
    },
}

/// The `StandardStMan` header at the start of the data file
/// (`SSMBase::readHeader` / `writeIndex`).
#[derive(Debug, Clone, PartialEq)]
pub struct StandardStManHeader {
    pub version: u32,
    /// Data-file byte order (`asBigEndian`).
    pub big_endian: bool,
    /// Size of one bucket in bytes.
    pub bucket_size: u32,
    /// Number of data buckets present.
    pub nr_buckets: u32,
    pub pers_cache_size: u32,
    pub n_free_bucket: u32,
    pub first_free_bucket: i32,
    /// Index buckets in the bucket chain.
    pub nr_index_buckets: u32,
    pub first_index_bucket: i32,
    /// Offset of the index inside its first bucket when it fits in one
    /// bucket; 0 when the index spans several buckets.
    pub index_bucket_offset: i32,
    pub last_string_bucket: i32,
    /// Total bytes of the assembled index stream.
    pub index_length: u32,
    /// Number of `SSMIndex` objects in the index stream.
    pub nr_index: u32,
}

/// One column-group index (`casacore/tables/DataMan/SSMIndex.cc`).
#[derive(Debug, Clone, PartialEq)]
pub struct SsmIndex {
    pub rows_per_bucket: u32,
    pub nr_columns: i32,
    /// Last row number contained in each used bucket (ascending).
    pub last_row: Vec<u64>,
    /// Data bucket holding each used bucket's rows.
    pub bucket_number: Vec<u32>,
}

/// A row range mapped to a data bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedBucket {
    pub number: u32,
    pub start_row: u64,
    pub end_row: u64,
}

impl SsmIndex {
    /// The bucket holding `row`, if any: the first used bucket whose
    /// `last_row` is >= `row` (`binarySearchBrackets` in `getIndex`/`find`).
    pub fn find(&self, row: u64) -> Option<IndexedBucket> {
        let i = self.last_row.partition_point(|&last| last < row);
        if i >= self.last_row.len() {
            return None;
        }
        let end_row = self.last_row[i];
        let start_row = if i == 0 { 0 } else { self.last_row[i - 1] + 1 };
        self.bucket_number.get(i).map(|&number| IndexedBucket {
            number,
            start_row,
            end_row,
        })
    }
}

/// A parsed StandardStMan data file: header, decoded index, and the raw
/// bucket region for direct cell access.
#[derive(Debug)]
pub struct StandardStManFile {
    pub header: StandardStManHeader,
    pub indices: Vec<SsmIndex>,
    data: Vec<u8>,
    /// Optional `table.f*{seq}i` array index file (StandardStMan arrays).
    f0i: Option<Vec<u8>>,
}

impl StandardStManFile {
    /// Read `<table_dir>/table.f{seq}` and parse it.
    pub fn open(
        table_dir: impl AsRef<std::path::Path>,
        seq_nr: u32,
        table_big_endian: bool,
    ) -> Result<StandardStManFile, SsmError> {
        let dir = table_dir.as_ref();
        let data = std::fs::read(dir.join(format!("table.f{seq_nr}")))?;
        let mut file = StandardStManFile::parse(&data, table_big_endian)?;
        // The array index file (`table.f{seq}i`) only exists for tables with
        // StandardStMan array columns.
        let f0i_path = dir.join(format!("table.f{seq_nr}i"));
        if f0i_path.is_file() {
            file.f0i = Some(std::fs::read(f0i_path)?);
        }
        Ok(file)
    }

    /// Parse a StandardStMan data file. `table_big_endian` is the data-file
    /// endianness flag from the `table.dat` header; for header version >= 3
    /// the stored flag must match it.
    pub fn parse(data: &[u8], table_big_endian: bool) -> Result<StandardStManFile, SsmError> {
        let mut r = reader(data, table_big_endian);
        let obj = r.read_object_start(true)?;
        if obj.type_name != "StandardStMan" {
            return Err(SsmError::UnexpectedType {
                expected: "StandardStMan".into(),
                found: obj.type_name,
            });
        }
        let version = obj.version;
        let stored_endian = if version >= 3 { r.read_bool()? } else { true };
        if version >= 3 && stored_endian != table_big_endian {
            return Err(SsmError::EndianMismatch {
                flag: stored_endian,
                expected: table_big_endian,
            });
        }
        let header = StandardStManHeader {
            version,
            big_endian: table_big_endian,
            bucket_size: r.read_u32()?,
            nr_buckets: r.read_u32()?,
            pers_cache_size: r.read_u32()?,
            n_free_bucket: r.read_u32()?,
            first_free_bucket: r.read_i32()?,
            nr_index_buckets: r.read_u32()?,
            first_index_bucket: r.read_i32()?,
            index_bucket_offset: r.read_i32()?,
            last_string_bucket: r.read_i32()?,
            index_length: r.read_u32()?,
            nr_index: r.read_u32()?,
        };
        let indices = read_index(data, &header, table_big_endian)?;
        Ok(StandardStManFile {
            header,
            indices,
            data: data.to_vec(),
            f0i: None,
        })
    }

    /// The optional array index file (`table.f0i`) contents, if present.
    pub fn f0i(&self) -> Option<&[u8]> {
        self.f0i.as_deref()
    }

    /// Raw bytes of data bucket `number`.
    pub fn bucket_bytes(&self, number: u32) -> Result<&[u8], SsmError> {
        let size = self.header.bucket_size as usize;
        let start =
            DATA_START
                .checked_add(number as usize * size)
                .ok_or(AipsIoError::Truncated {
                    needed: size,
                    offset: DATA_START,
                    len: self.data.len(),
                })?;
        self.data
            .get(start..start + size)
            .map(|b| &b[..size])
            .ok_or(SsmError::InvalidBucket(number))
    }

    /// Bytes of one row's cell in the given column group's index.
    ///
    /// `index_nr` is the index for the column (column `col_index_map[col]`),
    /// `column_offset` its start within the bucket, `cell_size` the bytes
    /// per row.
    pub fn cell_bytes(
        &self,
        index_nr: usize,
        column_offset: u32,
        row: u64,
        cell_size: u32,
    ) -> Result<&[u8], SsmError> {
        let index = self.indices.get(index_nr).ok_or(SsmError::IndexMissing {
            index: index_nr,
            count: self.indices.len(),
        })?;
        let bucket = index.find(row).ok_or(SsmError::RowOutOfRange { row })?;
        self.cell_bytes_in_bucket(column_offset, row, cell_size, bucket)
    }

    fn cell_bytes_in_bucket(
        &self,
        column_offset: u32,
        row: u64,
        cell_size: u32,
        bucket: IndexedBucket,
    ) -> Result<&[u8], SsmError> {
        let offset = u64::from(column_offset) + (row - bucket.start_row) * u64::from(cell_size);
        let base =
            (DATA_START as u64) + u64::from(bucket.number) * u64::from(self.header.bucket_size);
        let start = base + offset;
        let len = u64::from(cell_size);
        let start_us = usize::try_from(start).map_err(|_| SsmError::CellOutOfRange {
            bucket: bucket.number,
            offset,
            len,
        })?;
        let end = start_us
            .checked_add(len as usize)
            .ok_or(SsmError::CellOutOfRange {
                bucket: bucket.number,
                offset,
                len,
            })?;
        self.data
            .get(start_us..end)
            .ok_or(SsmError::CellOutOfRange {
                bucket: bucket.number,
                offset,
                len,
            })
    }

    /// Decode one scalar cell for `column` (table column index `col_idx`)
    /// at `row`, using the SSM spec offsets from `table.dat`.
    pub fn read_scalar_cell(
        &self,
        spec: &crate::columnset::StandardStMan,
        col_idx: usize,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<RecordValue, SsmError> {
        if desc.data_manager_type != "StandardStMan" {
            return Err(SsmError::NotStandardStMan(
                desc.name.clone(),
                desc.data_manager_type.clone(),
            ));
        }
        if matches!(desc.kind, ColumnKind::Array) {
            return Err(SsmError::ArrayColumn(desc.name.clone()));
        }
        let index_nr = spec
            .col_index_map
            .get(col_idx)
            .copied()
            .ok_or(SsmError::IndexMissing {
                index: col_idx,
                count: spec.col_index_map.len(),
            })? as usize;
        let column_offset = spec
            .column_offset
            .get(col_idx)
            .ok_or(SsmError::IndexMissing {
                index: col_idx,
                count: spec.column_offset.len(),
            })?;
        let cell = self.cell_bytes(index_nr, *column_offset, row, scalar_cell_size(desc))?;
        if desc.data_type == DataType::String && desc.max_length <= 0 {
            // Variable-length string: the cell is a 3-Int reference; strings
            // longer than 8 chars live in the string buckets.
            let big = self.header.big_endian;
            let len = read_i32_at(cell, 8, big)?;
            if len > 8 {
                let bucket = read_i32_at(cell, 0, big)?;
                let offset = read_i32_at(cell, 4, big)?;
                let s = self.read_long_string(bucket, offset, len as u32)?;
                return Ok(RecordValue::String(s));
            }
        }
        decode_scalar(cell, desc, self.header.big_endian)
    }

    /// Read a string stored in the SSM string buckets
    /// (`SSMStringHandler::get`/`getData`). String buckets carry a 16-byte
    /// **big-endian canonical** header `[unused][usedLength][nDeleted][nextBucket]`
    /// (independent of the data-file endianness) with the string data from
    /// byte 16; strings spanning buckets chain via `nextBucket`.
    fn read_long_string(&self, bucket: i32, offset: i32, length: u32) -> Result<String, SsmError> {
        if bucket < 0 || offset < 0 {
            return Err(SsmError::BadStringRef {
                bucket,
                offset,
                length,
            });
        }
        let mut out = Vec::with_capacity(length as usize);
        let mut b = bucket;
        let mut off = offset as i64;
        let mut remaining = i64::from(length);
        while remaining > 0 {
            let bucket_bytes = self.bucket_bytes(b as u32)?;
            let used = i64::from(i32::from_be_bytes(bucket_bytes[4..8].try_into().unwrap()));
            let next = i32::from_be_bytes(bucket_bytes[12..16].try_into().unwrap());
            let avail = used - off;
            if avail <= 0 {
                return Err(SsmError::BadStringRef {
                    bucket: b,
                    offset: off as i32,
                    length: remaining as u32,
                });
            }
            let take = remaining.min(avail) as usize;
            let start = 16 + off as usize;
            let end = start + take;
            if end > bucket_bytes.len() {
                return Err(SsmError::BadStringRef {
                    bucket: b,
                    offset: off as i32,
                    length,
                });
            }
            out.extend_from_slice(&bucket_bytes[start..end]);
            remaining -= take as i64;
            if remaining > 0 {
                if next < 0 {
                    return Err(SsmError::BadStringRef {
                        bucket: b,
                        offset: off as i32,
                        length,
                    });
                }
                b = next;
                off = 0;
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

/// Bytes per row of an array column's cell in the bucket: always an
/// `Int64` reference into the array index file (`table.f0i`).
pub const ARRAY_REF_SIZE: u32 = 8;

/// Decode one array cell for `column` (table column index `col_idx`) at
/// `row`, returning the logical (row-major) shape and element values.
///
/// The bucket cell holds an `Int64` byte offset into the array index file;
/// there the record is `[ndim][CASA-order dims][element data]` (with a
/// reference count in front when the file version > 0) — see
/// `SSMIndColumn::getShape`/`StIndArray`, `StManArrayFile::getShape`.
pub fn read_array_cell(
    file: &StandardStManFile,
    spec: &crate::columnset::StandardStMan,
    col_idx: usize,
    desc: &ColumnDesc,
    row: u64,
) -> Result<RecordValue, SsmError> {
    use crate::record::{ArrayData, ArrayValue};
    if !matches!(desc.kind, ColumnKind::Array) {
        return Err(SsmError::ArrayColumn(desc.name.clone()));
    }
    let index_nr = spec
        .col_index_map
        .get(col_idx)
        .copied()
        .ok_or(SsmError::IndexMissing {
            index: col_idx,
            count: spec.col_index_map.len(),
        })? as usize;
    let column_offset = spec
        .column_offset
        .get(col_idx)
        .ok_or(SsmError::IndexMissing {
            index: col_idx,
            count: spec.column_offset.len(),
        })?;
    // The reference cell is an Int64 in the data-file byte order.
    let cell = file.cell_bytes(index_nr, *column_offset, row, ARRAY_REF_SIZE)?;
    let offset = if file.header.big_endian {
        i64::from_be_bytes(cell[0..8].try_into().unwrap())
    } else {
        i64::from_le_bytes(cell[0..8].try_into().unwrap())
    };
    if offset == 0 {
        return Err(SsmError::EmptyArray {
            row,
            column: desc.name.clone(),
        });
    }
    let f0i = file
        .f0i
        .as_deref()
        .ok_or_else(|| SsmError::MissingArrayFile(desc.name.clone(), "table.f0i".into()))?;
    let off = usize::try_from(offset).map_err(|_| SsmError::BadArrayRef {
        offset,
        len: f0i.len(),
    })?;
    if off >= f0i.len() {
        return Err(SsmError::BadArrayRef {
            offset,
            len: f0i.len(),
        });
    }
    let version = u32_from(f0i, 0, file.header.big_endian);
    let mut p = off;
    if version > 0 {
        let _ref_count = u32_at(f0i, p, file.header.big_endian)?;
        p += 4;
    }
    let ndim = u32_at(f0i, p, file.header.big_endian)? as usize;
    p += 4;
    let mut casa_dims: Vec<u32> = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let d = u32_at(f0i, p, file.header.big_endian)?;
        p += 4;
        casa_dims.push(d);
    }
    let logical: Vec<u32> = casa_dims.iter().rev().copied().collect();
    let nelem: usize = casa_dims.iter().map(|&d| d as usize).product();
    let elem = desc.data_type;
    // StManArrayFile: Bool elements are bit-packed; others store one
    // element per `array_elem_size` bytes.
    let region_size = if elem == DataType::Bool {
        nelem.div_ceil(8)
    } else {
        nelem
            .checked_mul(array_elem_size(elem))
            .ok_or(SsmError::BadArrayRef {
                offset,
                len: f0i.len(),
            })?
    };
    let data_start = p;
    let data_end = data_start
        .checked_add(region_size)
        .ok_or(SsmError::BadArrayRef {
            offset,
            len: f0i.len(),
        })?;
    if data_end > f0i.len() {
        return Err(SsmError::BadArrayRef {
            offset,
            len: f0i.len(),
        });
    }
    let data = &f0i[data_start..data_end];
    let mut r = reader(data, file.header.big_endian);
    let array_data = match elem {
        DataType::Bool => {
            let nbytes = nelem.div_ceil(8);
            let mut packed = Vec::with_capacity(nbytes);
            for _ in 0..nbytes {
                packed.push(r.read_u8()?);
            }
            ArrayData::Bool(
                (0..nelem)
                    .map(|i| packed[i / 8] >> (i % 8) & 1 != 0)
                    .collect(),
            )
        }
        DataType::Char | DataType::UChar => {
            ArrayData::UChar((0..nelem).map(|_| r.read_u8()).collect::<Result<_, _>>()?)
        }
        DataType::Short => {
            ArrayData::Short((0..nelem).map(|_| r.read_i16()).collect::<Result<_, _>>()?)
        }
        DataType::UShort => {
            ArrayData::UShort((0..nelem).map(|_| r.read_u16()).collect::<Result<_, _>>()?)
        }
        DataType::Int => {
            ArrayData::Int((0..nelem).map(|_| r.read_i32()).collect::<Result<_, _>>()?)
        }
        DataType::UInt => {
            ArrayData::UInt((0..nelem).map(|_| r.read_u32()).collect::<Result<_, _>>()?)
        }
        DataType::Int64 => {
            ArrayData::Int64((0..nelem).map(|_| r.read_i64()).collect::<Result<_, _>>()?)
        }
        DataType::Float => {
            ArrayData::Float((0..nelem).map(|_| r.read_f32()).collect::<Result<_, _>>()?)
        }
        DataType::Double => {
            ArrayData::Double((0..nelem).map(|_| r.read_f64()).collect::<Result<_, _>>()?)
        }
        DataType::Complex => ArrayData::Complex({
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                v.push((r.read_f32()?, r.read_f32()?));
            }
            v
        }),
        DataType::DComplex => ArrayData::DComplex({
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                v.push((r.read_f64()?, r.read_f64()?));
            }
            v
        }),
        dt => return Err(SsmError::UnsupportedArrayType(dt)),
    };
    Ok(RecordValue::Array(ArrayValue {
        shape: logical,
        data: array_data,
    }))
}

fn u32_from(data: &[u8], off: usize, big_endian: bool) -> u32 {
    let b = data[off..off + 4].try_into().unwrap();
    if big_endian {
        u32::from_be_bytes(b)
    } else {
        u32::from_le_bytes(b)
    }
}

fn u32_at(data: &[u8], off: usize, big_endian: bool) -> Result<u32, SsmError> {
    let b = data.get(off..off + 4).ok_or(SsmError::CellOutOfRange {
        bucket: 0,
        offset: off as u64,
        len: 4,
    })?;
    Ok(if big_endian {
        u32::from_be_bytes(b.try_into().unwrap())
    } else {
        u32::from_le_bytes(b.try_into().unwrap())
    })
}

/// Bytes of one array element in the index file (Bool is bit-packed only
/// across a row's whole array, so its size is 1 here per `copyArrayBool`).
fn array_elem_size(dt: DataType) -> usize {
    match dt {
        DataType::Bool => 1,
        DataType::Char | DataType::UChar => 1,
        DataType::Short | DataType::UShort => 2,
        DataType::Int | DataType::UInt | DataType::Float => 4,
        DataType::Int64 | DataType::Double | DataType::Complex => 8,
        DataType::DComplex => 16,
        _ => 8,
    }
}

/// Bytes per row of a scalar column in the bucket (external size).
pub fn scalar_cell_size(desc: &ColumnDesc) -> u32 {
    match desc.data_type {
        DataType::String => {
            if desc.max_length > 0 {
                desc.max_length as u32
            } else {
                // A variable string cell holds a 3-Int (bucket, offset, len)
                // reference; short strings are stored inline over it.
                12
            }
        }
        DataType::Bool => 1, // bit-packed per element; a scalar is one bit in a byte
        DataType::UChar | DataType::Char => 1,
        DataType::Short | DataType::UShort => 2,
        DataType::Int | DataType::UInt | DataType::Float => 4,
        DataType::Double | DataType::Int64 | DataType::Complex => 8,
        DataType::DComplex => 16,
        _ => 0,
    }
}

fn reader(data: &[u8], big_endian: bool) -> Reader<'_> {
    if big_endian {
        Reader::new(data)
    } else {
        Reader::new_le(data)
    }
}

fn read_i32_at(data: &[u8], offset: usize, big_endian: bool) -> Result<i32, SsmError> {
    let b = data
        .get(offset..offset + 4)
        .ok_or(SsmError::CellOutOfRange {
            bucket: 0,
            offset: offset as u64,
            len: 4,
        })?;
    Ok(if big_endian {
        i32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } else {
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    })
}

/// Build the contiguous index stream from the index-bucket chain and decode
/// the `SSMIndex` objects (`SSMBase::readIndexBuckets`).
fn read_index(
    data: &[u8],
    header: &StandardStManHeader,
    big_endian: bool,
) -> Result<Vec<SsmIndex>, SsmError> {
    if header.nr_index == 0 {
        return Ok(Vec::new());
    }
    if header.first_index_bucket < 0 {
        return Err(SsmError::InvalidBucket(0));
    }
    let bucket_size = header.bucket_size as usize;
    let idx_bucket_size = bucket_size - 8;
    let mut stream = Vec::with_capacity(header.index_length as usize);
    let mut a_nr = header.index_length as usize;
    // First index bucket and its offset (0 = index spread over buckets).
    let mut bucket_nr = header.first_index_bucket as u32;
    let offset = if header.index_bucket_offset > 0 {
        Some(header.index_bucket_offset as usize)
    } else {
        None
    };
    for _ in 0..header.nr_index_buckets {
        let bp = bucket_start(bucket_size, bucket_nr);
        let bucket = data
            .get(bp..bp + bucket_size)
            .ok_or(SsmError::InvalidBucket(bucket_nr))?;
        // [checkNr][nextBucket] at the start of each index bucket.
        let next = read_i32_at(bucket, 4, big_endian)?;
        let take = if let Some(off) = offset {
            let n = a_nr.min(bucket_size.saturating_sub(off));
            stream.extend_from_slice(&bucket[off..off + n]);
            n
        } else {
            let n = a_nr.min(idx_bucket_size);
            stream.extend_from_slice(&bucket[8..8 + n]);
            n
        };
        a_nr = a_nr.saturating_sub(take);
        if next < 0 {
            break;
        }
        bucket_nr = next as u32;
    }

    let mut r = reader(&stream, big_endian);
    let mut indices = Vec::with_capacity(header.nr_index as usize);
    for _ in 0..header.nr_index {
        indices.push(read_index_object(&mut r)?);
    }
    Ok(indices)
}

fn bucket_start(bucket_size: usize, bucket_nr: u32) -> usize {
    DATA_START + bucket_nr as usize * bucket_size
}

/// Decode one framed root `"SSMIndex"` object (`SSMIndex::get`).
fn read_index_object(r: &mut Reader<'_>) -> Result<SsmIndex, SsmError> {
    let obj = r.read_object_start(true)?;
    if obj.type_name != "SSMIndex" {
        return Err(SsmError::UnexpectedType {
            expected: "SSMIndex".into(),
            found: obj.type_name,
        });
    }
    let _n_used = r.read_u32()?;
    let rows_per_bucket = r.read_u32()?;
    let nr_columns = r.read_i32()?;
    // Free-space map: framed "SimpleOrderedMap" (default, size, incr, pairs).
    let map = r.read_object_start(false)?;
    if map.type_name != "SimpleOrderedMap" {
        return Err(SsmError::UnexpectedType {
            expected: "SimpleOrderedMap".into(),
            found: map.type_name,
        });
    }
    let _default = r.read_i32()?;
    let n_free = r.read_u32()?;
    let _incr = r.read_u32()?;
    for _ in 0..n_free {
        let _key = r.read_i32()?;
        let _value = r.read_i32()?;
    }
    let last_row = read_u32_or_u64_block(r, obj.version)?;
    let bucket_number = read_u32_block(r)?;
    Ok(SsmIndex {
        rows_per_bucket,
        nr_columns,
        last_row,
        bucket_number,
    })
}

/// `Block<uInt>` (SSMIndex v1) or `Block<rownr_t>` (v2) of last rows.
fn read_u32_or_u64_block(r: &mut Reader<'_>, version: u32) -> Result<Vec<u64>, SsmError> {
    let block = r.read_object_start(false)?;
    if block.type_name != "Block" {
        return Err(SsmError::UnexpectedType {
            expected: "Block".into(),
            found: block.type_name,
        });
    }
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(if version == 1 {
            u64::from(r.read_u32()?)
        } else {
            r.read_u64()?
        });
    }
    Ok(out)
}

/// `Block<uInt>` of data-bucket numbers.
fn read_u32_block(r: &mut Reader<'_>) -> Result<Vec<u32>, SsmError> {
    let block = r.read_object_start(false)?;
    if block.type_name != "Block" {
        return Err(SsmError::UnexpectedType {
            expected: "Block".into(),
            found: block.type_name,
        });
    }
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.read_u32()?);
    }
    Ok(out)
}

/// Decode a scalar cell value for the column.
pub(crate) fn decode_scalar(
    cell: &[u8],
    desc: &ColumnDesc,
    big_endian: bool,
) -> Result<RecordValue, SsmError> {
    let mut r = reader(cell, big_endian);
    Ok(match desc.data_type {
        DataType::Bool => RecordValue::Bool(r.read_bool()?),
        DataType::UChar => RecordValue::UChar(r.read_u8()?),
        DataType::Char => RecordValue::UChar(r.read_u8()?),
        DataType::Short => RecordValue::Short(r.read_i16()?),
        DataType::UShort => RecordValue::UShort(r.read_u16()?),
        DataType::Int => RecordValue::Int(r.read_i32()?),
        DataType::UInt => RecordValue::UInt(r.read_u32()?),
        DataType::Int64 => RecordValue::Int64(r.read_i64()?),
        DataType::Float => RecordValue::Float(r.read_f32()?),
        DataType::Double => RecordValue::Double(r.read_f64()?),
        DataType::Complex => RecordValue::Complex(r.read_f32()?, r.read_f32()?),
        DataType::DComplex => RecordValue::DComplex(r.read_f64()?, r.read_f64()?),
        DataType::String => {
            if desc.max_length > 0 {
                // Fixed-length: `maxlen` chars, padded with NULs; the string
                // runs to the first NUL (or the full cell for maxlen chars).
                let s = cell
                    .iter()
                    .position(|&b| b == 0)
                    .map(|i| cell[..i].to_vec())
                    .unwrap_or_else(|| cell.to_vec());
                RecordValue::String(String::from_utf8_lossy(&s).into_owned())
            } else {
                // Variable-length: 3-Int reference; the length is the last
                // Int. Strings of 8 chars or less are kept inline over the
                // first bytes of the cell.
                let mut len_r = reader(&cell[8..12], big_endian);
                let len = len_r.read_i32()?;
                if len > 8 {
                    let mut ref_r = reader(&cell[..12], big_endian);
                    let bucket = ref_r.read_i32()?;
                    return Err(SsmError::StringBucketUnsupported { bucket, len });
                }
                let s = cell[..len.max(0) as usize].to_vec();
                RecordValue::String(String::from_utf8_lossy(&s).into_owned())
            }
        }
        dt => return Err(SsmError::ArrayColumn(format!("{:?} ({})", dt, desc.name))),
    })
}

/// Per-column raw cell bytes handed to `write_standard_stman_file`.
pub struct WriteColumn<'a> {
    /// Bytes per row cell in the bucket (external size).
    pub cell_size: u32,
    /// Bits per row cell in the bucket (1 for bit-packed Booleans, else
    /// `8 * cell_size`).
    pub cell_bits: u32,
    /// Concatenated cell bytes: `n_rows * cell_size` bytes.
    pub bytes: &'a [u8],
}

/// Layout of a StandardStMan bucket tile for `rows_per_bucket` rows:
/// each column owns `(rows_per_bucket * cell_bits + 7) / 8` bytes laid out
/// contiguously from offset 0 (the casacore best-fit packing), and the
/// bucket is exactly the tile size.
pub struct StandardStManLayout {
    pub rows_per_bucket: u32,
    pub bucket_size: u32,
    /// Byte offset of each column's region in every data bucket.
    pub column_offset: Vec<u32>,
}

/// Compute the bucket tile for the given per-column bit sizes.
pub fn layout(rows_per_bucket: u32, cell_bits: &[u32]) -> StandardStManLayout {
    let mut offset = 0u32;
    let mut column_offset = Vec::with_capacity(cell_bits.len());
    for &bits in cell_bits {
        column_offset.push(offset);
        let region = (u64::from(rows_per_bucket) * u64::from(bits)).div_ceil(8);
        offset += region as u32;
    }
    StandardStManLayout {
        rows_per_bucket,
        bucket_size: offset.max(1),
        column_offset,
    }
}

/// Serialize the `SSMIndex` stream for the layout (used to size the index
/// bucket chain before writing; the data-file writer calls it too).
pub fn build_index_stream(
    big_endian: bool,
    n_rows: u64,
    layout: &StandardStManLayout,
    nr_columns: usize,
) -> Vec<u8> {
    let mut iw = crate::aipsio::Writer::new();
    if !big_endian {
        iw = crate::aipsio::Writer::new_le();
    }
    write_index_stream(&mut iw, n_rows, layout, nr_columns);
    iw.into_bytes()
}

/// Number of index buckets a stream of `index_len` bytes needs, given the
/// bucket size (mirror of the chain allocation in write/read).
pub fn index_bucket_count(index_len: usize, bucket_size: u32) -> usize {
    if index_len == 0 {
        return 0;
    }
    let capacity = bucket_size as usize - 8;
    index_len.div_ceil(capacity)
}

/// Serialize a complete StandardStMan data file — header, data buckets, the
/// index bucket chain, and any SSM string buckets — matching what casacore
/// writes for the tile layout from `layout`. `big_endian` is the data-file
/// byte order (from the `table.dat` header); little-endian files get the v3
/// header with the explicit endian flag, big-endian files the v2 header.
/// `string_buckets` holds pre-formatted string-bucket contents (16-byte
/// big-endian header + data); they are appended after the index buckets and
/// counted in `nr_buckets`, with the last one reported in
/// `last_string_bucket`.
pub fn write_standard_stman_file(
    big_endian: bool,
    n_rows: u64,
    cols: &[WriteColumn<'_>],
    layout: &StandardStManLayout,
    string_buckets: &[Vec<u8>],
) -> Vec<u8> {
    let n_bucket_rows = layout.rows_per_bucket as u64;
    let nr_data_buckets = n_rows.div_ceil(n_bucket_rows) as u32;
    let bucket_size = layout.bucket_size as usize;

    // SSMIndex stream (endianness = data file).
    let index_stream = build_index_stream(big_endian, n_rows, layout, cols.len());

    // Index bucket(s).
    let a_clen = 8usize;
    let idx_capacity = bucket_size - a_clen;
    let mut index_buckets: Vec<Vec<u8>> = Vec::new();
    let mut chunks: Vec<&[u8]> = Vec::new();
    if !index_stream.is_empty() {
        let mut rest: &[u8] = &index_stream;
        while !rest.is_empty() {
            let take = rest.len().min(idx_capacity);
            chunks.push(&rest[..take]);
            rest = &rest[take..];
        }
    }
    for (i, chunk) in chunks.iter().enumerate() {
        let mut b = Vec::with_capacity(bucket_size);
        let check = -1i32;
        let next = if i + 1 < chunks.len() {
            (nr_data_buckets + i as u32 + 1) as i32
        } else {
            -1i32
        };
        if big_endian {
            b.extend_from_slice(&check.to_be_bytes());
            b.extend_from_slice(&next.to_be_bytes());
        } else {
            b.extend_from_slice(&check.to_le_bytes());
            b.extend_from_slice(&next.to_le_bytes());
        }
        b.extend_from_slice(chunk);
        b.resize(bucket_size, 0);
        index_buckets.push(b);
    }
    let single_bucket = chunks.len() <= 1;
    let idx_bucket_offset = if single_bucket { 8i32 } else { 0i32 };
    let first_string_bucket = nr_data_buckets + index_buckets.len() as u32;
    let last_string_bucket = if string_buckets.is_empty() {
        -1
    } else {
        (first_string_bucket + string_buckets.len() as u32 - 1) as i32
    };

    // Data buckets: copy each column's per-bucket row range into its region.
    let mut file = Vec::with_capacity(
        DATA_START
            + (nr_data_buckets as usize + index_buckets.len() + string_buckets.len()) * bucket_size,
    );
    // Header.
    let mut hw = crate::aipsio::Writer::new();
    if !big_endian {
        hw = crate::aipsio::Writer::new_le();
    }
    if big_endian {
        hw.put_root_object_start("StandardStMan", 2);
    } else {
        hw.put_root_object_start("StandardStMan", 3);
        hw.put_bool(false); // little endian
    }
    hw.put_u32(bucket_size as u32);
    hw.put_u32(nr_data_buckets + index_buckets.len() as u32 + string_buckets.len() as u32);
    hw.put_u32(0); // persistent cache size
    hw.put_u32(0); // free buckets
    hw.put_i32(-1); // first free bucket
    hw.put_u32(index_buckets.len() as u32); // nr index buckets
    hw.put_i32(if chunks.is_empty() {
        -1
    } else {
        nr_data_buckets as i32
    }); // first index bucket
    hw.put_i32(idx_bucket_offset);
    hw.put_i32(last_string_bucket);
    hw.put_u32(index_stream.len() as u32);
    hw.put_u32(1); // nr indices
    hw.put_object_end();
    file.extend_from_slice(&hw.into_bytes());
    file.resize(DATA_START, 0);

    for b in 0..nr_data_buckets {
        let mut bucket = vec![0u8; bucket_size];
        let start_row = u64::from(b) * n_bucket_rows;
        let end_row = (start_row + n_bucket_rows).min(n_rows);
        for (c, col) in cols.iter().enumerate() {
            let cell_size = col.cell_size as usize;
            let region_start = (start_row * cell_size as u64) as usize;
            let region_end = (end_row * cell_size as u64) as usize;
            let src = &col.bytes[region_start..region_end];
            let off = layout.column_offset[c] as usize;
            bucket[off..off + src.len()].copy_from_slice(src);
        }
        file.extend_from_slice(&bucket);
    }
    for b in index_buckets {
        file.extend_from_slice(&b);
    }
    for sb in string_buckets {
        file.extend_from_slice(sb);
    }
    file
}
/// Serialize the single `SSMIndex` object for the layout.
fn write_index_stream(
    iw: &mut crate::aipsio::Writer,
    n_rows: u64,
    layout: &StandardStManLayout,
    nr_columns: usize,
) {
    let rpb = layout.rows_per_bucket as u64;
    let nr_buckets = n_rows.div_ceil(rpb) as usize;
    iw.put_root_object_start("SSMIndex", 1);
    iw.put_u32(nr_buckets as u32); // itsNUsed
    iw.put_u32(layout.rows_per_bucket);
    iw.put_i32(nr_columns as i32);
    // Empty SimpleOrderedMap free-space.
    iw.put_object_start("SimpleOrderedMap", 1);
    iw.put_i32(0); // old default value
    iw.put_u32(0); // size
    iw.put_u32(1); // old increment
    iw.put_object_end();
    // lastRow: ascending last row per bucket.
    iw.put_object_start("Block", 1);
    iw.put_u32(nr_buckets as u32);
    for b in 0..nr_buckets as u64 {
        iw.put_u32((((b + 1) * rpb).min(n_rows) - 1) as u32);
    }
    iw.put_object_end();
    // bucketNumber: data bucket numbers 0..n-1.
    iw.put_object_start("Block", 1);
    iw.put_u32(nr_buckets as u32);
    for b in 0..nr_buckets as u32 {
        iw.put_u32(b);
    }
    iw.put_object_end();
    iw.put_object_end();
}

/// Encode one scalar cell for `desc`/`value` into the exact
/// `scalar_cell_size(desc)` bytes stored in a bucket (mirror of
/// `decode_scalar`). Variable-length strings up to 8 chars are stored
/// inline; longer ones need the (unsupported) string buckets.
pub fn encode_scalar_cell(
    big_endian: bool,
    desc: &ColumnDesc,
    value: &RecordValue,
) -> Result<Vec<u8>, SsmError> {
    use crate::aipsio::Writer;
    fn wr(big_endian: bool) -> Writer {
        if big_endian {
            Writer::new()
        } else {
            Writer::new_le()
        }
    }
    let mut w = wr(big_endian);
    match desc.data_type {
        DataType::Bool => {
            let b = matches!(value, RecordValue::Bool(true))
                || matches!(value, RecordValue::UChar(u) if *u != 0);
            let mut cell = vec![0u8; scalar_cell_size(desc) as usize];
            cell[0] = b as u8 & 1;
            Ok(cell)
        }
        DataType::String if desc.max_length > 0 => {
            let maxlen = desc.max_length as usize;
            let mut cell = vec![0u8; maxlen];
            let s = match value {
                RecordValue::String(s) => s.as_bytes(),
                _ => b"",
            };
            let n = s.len().min(maxlen);
            cell[..n].copy_from_slice(&s[..n]);
            Ok(cell)
        }
        DataType::String => {
            let s = match value {
                RecordValue::String(s) => s.as_bytes(),
                _ => b"",
            };
            if s.len() > 8 {
                return Err(SsmError::StringBucketUnsupported {
                    bucket: 0,
                    len: s.len() as i32,
                });
            }
            let mut cell = vec![0u8; 12];
            cell[..s.len()].copy_from_slice(s);
            put_scalar_ints(&mut cell[8..12], s.len() as i32, big_endian);
            Ok(cell)
        }
        _ => {
            crate::record::write_scalar_value(&mut w, desc.data_type, value);
            let cell = w.into_bytes();
            let want = scalar_cell_size(desc) as usize;
            debug_assert_eq!(cell.len(), want, "cell size for {:?}", desc.data_type);
            let mut out = vec![0u8; want];
            out[..cell.len().min(want)].copy_from_slice(&cell[..cell.len().min(want)]);
            Ok(out)
        }
    }
}

fn put_scalar_ints(dst: &mut [u8], v: i32, big_endian: bool) {
    if big_endian {
        dst.copy_from_slice(&v.to_be_bytes());
    } else {
        dst.copy_from_slice(&v.to_le_bytes());
    }
}

/// Encode one array value into its `table.f0i` record: `[ndim][CASA-order
/// dims][element data]` (byte-identical to `StManArrayFile::putShape` +
/// the element writes, for the table's data-file endianness).
pub fn encode_array_record(
    big_endian: bool,
    elem: DataType,
    value: &crate::record::ArrayValue,
) -> Result<Vec<u8>, SsmError> {
    use crate::aipsio::Writer;
    use crate::record::ArrayData;
    if elem == DataType::String {
        return Err(SsmError::UnsupportedArrayType(DataType::String));
    }
    let mut w = if big_endian {
        Writer::new()
    } else {
        Writer::new_le()
    };
    w.put_u32(value.shape.len() as u32);
    for d in value.shape.iter().rev() {
        w.put_i32(*d as i32); // CASA dim order = reversed logical
    }
    let mut body = w.into_bytes();
    let mut push = |bytes: &[u8]| body.extend_from_slice(bytes);
    match &value.data {
        ArrayData::Bool(bits) => {
            let nbytes = bits.len().div_ceil(8);
            let mut packed = vec![0u8; nbytes];
            for (i, b) in bits.iter().enumerate() {
                if *b {
                    packed[i / 8] |= 1 << (i % 8);
                }
            }
            push(&packed);
        }
        ArrayData::UChar(v) => push(&v.to_vec()),
        ArrayData::Short(v) => {
            let mut b = vec![0u8; v.len() * 2];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_be_bytes()
                } else {
                    x.to_le_bytes()
                };
                b[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::UShort(v) => {
            let mut b = vec![0u8; v.len() * 2];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_be_bytes()
                } else {
                    x.to_le_bytes()
                };
                b[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::Int(v) => {
            let mut b = vec![0u8; v.len() * 4];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_be_bytes()
                } else {
                    x.to_le_bytes()
                };
                b[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::UInt(v) => {
            let mut b = vec![0u8; v.len() * 4];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_be_bytes()
                } else {
                    x.to_le_bytes()
                };
                b[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::Int64(v) => {
            let mut b = vec![0u8; v.len() * 8];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_be_bytes()
                } else {
                    x.to_le_bytes()
                };
                b[i * 8..i * 8 + 8].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::Float(v) => {
            let mut b = vec![0u8; v.len() * 4];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_bits().to_be_bytes()
                } else {
                    x.to_bits().to_le_bytes()
                };
                b[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::Double(v) => {
            let mut b = vec![0u8; v.len() * 8];
            for (i, x) in v.iter().enumerate() {
                let bytes = if big_endian {
                    x.to_bits().to_be_bytes()
                } else {
                    x.to_bits().to_le_bytes()
                };
                b[i * 8..i * 8 + 8].copy_from_slice(&bytes);
            }
            push(&b);
        }
        ArrayData::Complex(v) => {
            let mut b = vec![0u8; v.len() * 8];
            for (i, (re, im)) in v.iter().enumerate() {
                let reb = if big_endian {
                    re.to_bits().to_be_bytes()
                } else {
                    re.to_bits().to_le_bytes()
                };
                let imb = if big_endian {
                    im.to_bits().to_be_bytes()
                } else {
                    im.to_bits().to_le_bytes()
                };
                b[i * 8..i * 8 + 4].copy_from_slice(&reb);
                b[i * 8 + 4..i * 8 + 8].copy_from_slice(&imb);
            }
            push(&b);
        }
        ArrayData::DComplex(v) => {
            let mut b = vec![0u8; v.len() * 16];
            for (i, (re, im)) in v.iter().enumerate() {
                let reb = if big_endian {
                    re.to_bits().to_be_bytes()
                } else {
                    re.to_bits().to_le_bytes()
                };
                let imb = if big_endian {
                    im.to_bits().to_be_bytes()
                } else {
                    im.to_bits().to_le_bytes()
                };
                b[i * 16..i * 16 + 8].copy_from_slice(&reb);
                b[i * 16 + 8..i * 16 + 16].copy_from_slice(&imb);
            }
            push(&b);
        }
        ArrayData::String(_) => unreachable!("string arrays rejected above"),
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aipsio::MAGIC;

    /// Minimal BigEndian file-builder helpers for unit tests.
    struct Be(Vec<u8>);
    impl Be {
        fn u32(&mut self, v: u32) {
            self.0.extend_from_slice(&v.to_be_bytes());
        }
        fn framed(&mut self, type_name: &str, version: u32, payload: &[u8]) {
            self.u32((4 + 4 + type_name.len() + 4 + payload.len()) as u32);
            self.u32(type_name.len() as u32);
            self.0.extend_from_slice(type_name.as_bytes());
            self.u32(version);
            self.0.extend_from_slice(payload);
        }
        fn root(&mut self, type_name: &str, version: u32, payload: &[u8]) {
            self.0.extend_from_slice(&MAGIC.to_be_bytes());
            self.framed(type_name, version, payload);
        }
        fn block_u32(&mut self, values: &[u32]) {
            let mut payload = Vec::new();
            payload.extend_from_slice(&(values.len() as u32).to_be_bytes());
            for v in values {
                payload.extend_from_slice(&v.to_be_bytes());
            }
            self.framed("Block", 1, &payload);
        }
    }

    fn scalar_desc(name: &str, data_type: DataType, max_len: i32) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: max_len,
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind: ColumnKind::Scalar(crate::record::RecordValue::Int(0)),
        }
    }

    fn spec(offsets: &[u32]) -> crate::columnset::StandardStMan {
        crate::columnset::StandardStMan {
            data_manager_name: "StandardStMan".into(),
            column_offset: offsets.to_vec(),
            col_index_map: vec![0; offsets.len()],
        }
    }

    /// Build a file with one index and one data bucket per `data_buckets`
    /// entry (bucket numbers 0..n), plus a trailing index bucket.
    fn build_file(bucket_size: u32, data_buckets: &[&[u8]], index_payload: &[u8]) -> Vec<u8> {
        let mut file = Vec::new();
        // Header: v2 (big endian, no flag).
        let mut payload = Vec::new();
        payload.extend_from_slice(&bucket_size.to_be_bytes());
        payload.extend_from_slice(&(data_buckets.len() as u32 + 1).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes()); // pers cache
        payload.extend_from_slice(&0u32.to_be_bytes()); // free buckets
        payload.extend_from_slice(&(-1i32).to_be_bytes()); // first free
        payload.extend_from_slice(&1u32.to_be_bytes()); // nr idx buckets
        payload.extend_from_slice(&(data_buckets.len() as u32).to_be_bytes()); // first idx bucket
        payload.extend_from_slice(&8i32.to_be_bytes()); // idx bucket offset
        payload.extend_from_slice(&(-1i32).to_be_bytes()); // last string bucket
        payload.extend_from_slice(&(index_payload.len() as u32).to_be_bytes()); // index length
        payload.extend_from_slice(&1u32.to_be_bytes()); // nr indices
        let mut header = Be(Vec::new());
        header.root("StandardStMan", 2, &payload);
        file.extend_from_slice(&header.0);
        // Pad to DATA_START.
        file.resize(DATA_START, 0);
        // Data buckets (numbers 0..nused).
        for b in data_buckets {
            file.extend_from_slice(b);
            file.resize(file.len() + bucket_size as usize - b.len(), 0);
        }
        // Index bucket (number nused): [checkNr=-1][next=-1][index stream].
        let idx_bucket_nr = data_buckets.len() as u32;
        let mut ib = Vec::new();
        ib.extend_from_slice(&(-1i32).to_be_bytes());
        ib.extend_from_slice(&(-1i32).to_be_bytes());
        ib.extend_from_slice(index_payload);
        file.extend_from_slice(&ib);
        file.resize(file.len() + bucket_size as usize - ib.len(), 0);
        assert_eq!(
            file.len(),
            DATA_START + (idx_bucket_nr as usize + 1) * bucket_size as usize
        );
        file
    }

    fn index_payload(last_rows: &[u64], bucket_numbers: &[u32]) -> Vec<u8> {
        // Single SSMIndex, v1.
        let mut payload = Vec::new();
        payload.extend_from_slice(&(last_rows.len() as u32).to_be_bytes()); // nused
        payload.extend_from_slice(&32u32.to_be_bytes()); // rows per bucket
        payload.extend_from_slice(&1i32.to_be_bytes()); // nr columns
                                                        // SimpleOrderedMap (empty): framed object with default, size, incr.
        let mut map_payload = Vec::new();
        map_payload.extend_from_slice(&0i32.to_be_bytes());
        map_payload.extend_from_slice(&0u32.to_be_bytes());
        map_payload.extend_from_slice(&1u32.to_be_bytes());
        let mut map = Be(Vec::new());
        map.framed("SimpleOrderedMap", 1, &map_payload);
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&payload);
        payload2.extend_from_slice(&map.0);
        // last_row Block (u32 vals) + bucket_number Block.
        let mut lr = Be(Vec::new());
        lr.block_u32(&last_rows.iter().map(|&x| x as u32).collect::<Vec<_>>());
        let mut bn = Be(Vec::new());
        bn.block_u32(bucket_numbers);
        payload2.extend_from_slice(&lr.0);
        payload2.extend_from_slice(&bn.0);
        let mut obj = Be(Vec::new());
        obj.root("SSMIndex", 1, &payload2);
        obj.0
    }

    fn parse(file: &[u8]) -> StandardStManFile {
        StandardStManFile::parse(file, true).unwrap()
    }

    #[test]
    fn parses_header_and_index() {
        let file = build_file(256, &[&[]], &index_payload(&[0], &[0]));
        let f = parse(&file);
        assert_eq!(f.header.version, 2);
        assert!(f.header.big_endian);
        assert_eq!(f.header.bucket_size, 256);
        assert_eq!(f.header.nr_buckets, 2);
        assert_eq!(f.header.nr_index, 1);
        assert_eq!(f.indices.len(), 1);
        assert_eq!(f.indices[0].rows_per_bucket, 32);
        assert_eq!(f.indices[0].last_row, vec![0]);
        assert_eq!(f.indices[0].bucket_number, vec![0]);
    }

    #[test]
    fn finds_rows_across_buckets() {
        let idx = SsmIndex {
            rows_per_bucket: 32,
            nr_columns: 1,
            last_row: vec![31, 63],
            bucket_number: vec![4, 9],
        };
        assert_eq!(
            idx.find(0),
            Some(IndexedBucket {
                number: 4,
                start_row: 0,
                end_row: 31
            })
        );
        assert_eq!(
            idx.find(30),
            Some(IndexedBucket {
                number: 4,
                start_row: 0,
                end_row: 31
            })
        );
        assert_eq!(
            idx.find(40),
            Some(IndexedBucket {
                number: 9,
                start_row: 32,
                end_row: 63
            })
        );
        assert_eq!(idx.find(64), None);
        assert_eq!(
            SsmIndex {
                rows_per_bucket: 32,
                nr_columns: 1,
                last_row: vec![],
                bucket_number: vec![]
            }
            .find(0),
            None
        );
    }

    #[test]
    fn reads_numeric_scalar_cells() {
        // One data bucket holding: bool(1B)@0, int(4B)@8, double(8B)@16.
        let mut data_bucket = [0u8; 64];
        data_bucket[0] = 1; // bool true
        data_bucket[8..12].copy_from_slice(&12345i32.to_be_bytes());
        data_bucket[16..24].copy_from_slice(&1.25f64.to_be_bytes());
        let file = build_file(256, &[&data_bucket], &index_payload(&[0], &[0]));
        let f = parse(&file);
        let s = spec(&[0, 8, 16]);
        let b = f
            .read_scalar_cell(&s, 0, &scalar_desc("B", DataType::Bool, 0), 0)
            .unwrap();
        assert_eq!(b, RecordValue::Bool(true));
        let i = f
            .read_scalar_cell(&s, 1, &scalar_desc("I", DataType::Int, 0), 0)
            .unwrap();
        assert_eq!(i, RecordValue::Int(12345));
        let d = f
            .read_scalar_cell(&s, 2, &scalar_desc("D", DataType::Double, 0), 0)
            .unwrap();
        assert_eq!(d, RecordValue::Double(1.25));
    }

    #[test]
    fn reads_inline_short_string() {
        // Variable string cell: 12-byte reference area with the payload
        // memcpy'd over the front; total length in the trailing Int.
        let mut data_bucket = [0u8; 64];
        // "hello" over the first bytes, len 5 as the 3rd Int.
        data_bucket[0..5].copy_from_slice(b"hello");
        data_bucket[8..12].copy_from_slice(&5i32.to_be_bytes());
        let file = build_file(256, &[&data_bucket], &index_payload(&[0], &[0]));
        let f = parse(&file);
        let s = spec(&[0]);
        let v = f
            .read_scalar_cell(&s, 0, &scalar_desc("S", DataType::String, 0), 0)
            .unwrap();
        assert_eq!(v, RecordValue::String("hello".into()));
    }

    #[test]
    fn reads_fixed_length_string() {
        let mut data_bucket = [0u8; 64];
        data_bucket[0..6].copy_from_slice(b"hi\0\0\0\0");
        let file = build_file(256, &[&data_bucket], &index_payload(&[0], &[0]));
        let f = parse(&file);
        let s = spec(&[0]);
        let v = f
            .read_scalar_cell(&s, 0, &scalar_desc("S", DataType::String, 6), 0)
            .unwrap();
        assert_eq!(v, RecordValue::String("hi".into()));
    }

    #[test]
    fn reads_long_string_from_string_bucket() {
        // A cell reference to a string bucket is resolved: build a bucket 2
        // holding a long string, referenced as [bucket 2][offset 16][…].
        // String-bucket header is big-endian canonical:
        // [unused][usedLength][nDeleted][nextBucket].
        let content = b"a string that is definitely longer than eight characters";
        let mut sb = vec![0u8; 256];
        sb[4..8].copy_from_slice(&(content.len() as u32).to_be_bytes()); // used
        sb[8..12].copy_from_slice(&((256 - 16 - content.len()) as u32).to_be_bytes()); // nDeleted
        sb[12..16].copy_from_slice(&(-1i32).to_be_bytes()); // next = -1
        sb[16..16 + content.len()].copy_from_slice(content);
        // Bucket 0: cell ref [bucket 2][offset 16][len]; bucket 1: index;
        // bucket 2: the string bucket.
        let mut data_bucket = [0u8; 256];
        data_bucket[0..4].copy_from_slice(&2i32.to_be_bytes());
        data_bucket[4..8].copy_from_slice(&0i32.to_be_bytes()); // data-area offset 0
        data_bucket[8..12].copy_from_slice(&(content.len() as i32).to_be_bytes());
        let mut file = build_file(256, &[&data_bucket], &index_payload(&[0], &[0]));
        file.extend_from_slice(&sb);
        let f = parse(&file);
        let s = spec(&[0]);
        assert_eq!(
            f.read_scalar_cell(&s, 0, &scalar_desc("S", DataType::String, 0), 0)
                .unwrap(),
            RecordValue::String(String::from_utf8(content.to_vec()).unwrap())
        );
    }

    #[test]
    fn rejects_row_out_of_range() {
        let file = build_file(256, &[&[0u8; 64]], &index_payload(&[31], &[0]));
        let f = parse(&file);
        let s = spec(&[0]);
        assert!(matches!(
            f.read_scalar_cell(&s, 0, &scalar_desc("I", DataType::Int, 0), 32),
            Err(SsmError::RowOutOfRange { row: 32 })
        ));
    }

    #[test]
    fn rejects_array_columns() {
        let file = build_file(256, &[&[0u8; 64]], &index_payload(&[0], &[0]));
        let f = parse(&file);
        let s = spec(&[0]);
        let mut desc = scalar_desc("A", DataType::Int, 0);
        desc.kind = ColumnKind::Array;
        assert!(matches!(
            f.read_scalar_cell(&s, 0, &desc, 0),
            Err(SsmError::ArrayColumn(_))
        ));
    }

    #[test]
    fn reads_index_spread_across_buckets() {
        // An SSMIndex whose stream (127 bytes) exceeds one index bucket's
        // data area (bucket_size - 8) is chained across index buckets with
        // the next-bucket pointer in bytes 4..8; index_bucket_offset = 0.
        let mut b0 = [0u8; 64];
        b0[0..4].copy_from_slice(&11i32.to_be_bytes()); // cell of row 0
        let mut b1 = [0u8; 64];
        // Rows 32..63 live in bucket 1; row 40's cell is at (40-32)*4 = 32.
        b1[32..36].copy_from_slice(&22i32.to_be_bytes());
        let full = index_payload(&[31, 63], &[0, 1]);
        // Split the index across buckets of 64 - 8 usable bytes.
        let mut chunks: Vec<&[u8]> = Vec::new();
        let mut rest: &[u8] = &full;
        while !rest.is_empty() {
            let take = rest.len().min(64 - 8);
            chunks.push(&rest[..take]);
            rest = &rest[take..];
        }
        assert!(chunks.len() > 1, "test index must span several buckets");

        let bucket_size = 64u32;
        let n_index = chunks.len() as u32;
        let first_idx = 2u32; // after the two data buckets
        let mut payload = Vec::new();
        payload.extend_from_slice(&bucket_size.to_be_bytes());
        payload.extend_from_slice(&(2 + n_index).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes()); // pers cache
        payload.extend_from_slice(&0u32.to_be_bytes()); // free buckets
        payload.extend_from_slice(&(-1i32).to_be_bytes());
        payload.extend_from_slice(&n_index.to_be_bytes()); // nr idx buckets
        payload.extend_from_slice(&first_idx.to_be_bytes());
        payload.extend_from_slice(&0i32.to_be_bytes()); // idx offset 0 -> chain
        payload.extend_from_slice(&(-1i32).to_be_bytes());
        payload.extend_from_slice(&(full.len() as u32).to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes()); // nr indices
        let mut header = Be(Vec::new());
        header.root("StandardStMan", 2, &payload);
        let mut file = header.0;
        file.resize(DATA_START, 0);
        for b in [&b0[..], &b1[..]] {
            file.extend_from_slice(b);
            file.resize(file.len() + 64 - b.len(), 0);
        }
        for (i, chunk) in chunks.iter().enumerate() {
            let mut b = Vec::new();
            b.extend_from_slice(&(-1i32).to_be_bytes()); // check nr
            let next = if i + 1 < chunks.len() {
                (first_idx + (i + 1) as u32) as i32
            } else {
                -1
            };
            b.extend_from_slice(&next.to_be_bytes());
            b.extend_from_slice(chunk);
            file.extend_from_slice(&b);
            file.resize(file.len() + 64 - b.len(), 0);
        }

        let f = parse(&file);
        assert_eq!(f.indices[0].last_row, vec![31, 63]);
        assert_eq!(f.indices[0].bucket_number, vec![0, 1]);
        let s = spec(&[0]);
        let desc = scalar_desc("I", DataType::Int, 0);
        assert_eq!(
            f.read_scalar_cell(&s, 0, &desc, 0).unwrap(),
            RecordValue::Int(11)
        );
        assert_eq!(
            f.read_scalar_cell(&s, 0, &desc, 40).unwrap(),
            RecordValue::Int(22)
        );
    }

    #[test]
    fn rejects_endian_flag_mismatch() {
        // Header v3 with flag = false, parsed as a big-endian table.
        let mut payload = Vec::new();
        payload.push(0); // bool flag false
        payload.extend_from_slice(&64u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&(-1i32).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&(-1i32).to_be_bytes());
        payload.extend_from_slice(&0i32.to_be_bytes());
        payload.extend_from_slice(&(-1i32).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        let mut header = Be(Vec::new());
        header.root("StandardStMan", 3, &payload);
        let mut file = header.0;
        file.resize(DATA_START, 0);
        assert!(matches!(
            StandardStManFile::parse(&file, true),
            Err(SsmError::EndianMismatch {
                flag: false,
                expected: true
            })
        ));
    }
}
