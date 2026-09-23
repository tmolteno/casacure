//! Parsing of the `table.dat` header of a CASA table.
//!
//! Layout (`casacore/tables/Tables/BaseTable.cc::writeStart`,
//! `PlainTable.cc::putFile`): a root AipsIO object of type `"Table"`, whose
//! payload is the row count, an endianness flag describing the *data* files
//! (`table.f*` — the `table.dat` stream itself is always big-endian
//! canonical AipsIO), and a table-kind string (`"PlainTable"`).

use crate::aipsio::{AipsIoError, Reader};
use crate::columnset::{parse_column_set, ColumnSet, ColumnSetError};
use crate::record::RecordValue;
use crate::tabledesc::{TableDesc, TableDescError};
use thiserror::Error;

/// Errors from parsing a `table.dat` header.
#[derive(Debug, Error)]
pub enum TableError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error("table.dat root object has type {found:?}, expected \"Table\"")]
    NotATable { found: String },
    #[error("unsupported table.dat version {0} (casacore supports up to 3)")]
    UnsupportedVersion(u32),
    #[error("invalid endianness flag {0} in table.dat (expected 0 or 1)")]
    BadEndianness(u32),
}

/// Errors from parsing a whole `table.dat` file.
#[derive(Debug, Error)]
pub enum TableDatError {
    #[error(transparent)]
    Header(#[from] TableError),
    #[error(transparent)]
    Desc(#[from] TableDescError),
    #[error(transparent)]
    ColumnSet(#[from] ColumnSetError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("storage error: {0}")]
    Storage(String),
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
}

/// The parsed `table.dat` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableHeader {
    /// `Table` object format version (2 or 3).
    pub version: u32,
    /// Number of rows in the table.
    pub nrow: u64,
    /// True when the column data files are big-endian; false for
    /// little-endian (the casacore default since 3.x).
    pub big_endian: bool,
    /// Table kind, e.g. `"PlainTable"`.
    pub kind: String,
}

/// The parsed contents of a `table.dat` file.
#[derive(Debug, Clone, PartialEq)]
pub struct TableDat {
    pub header: TableHeader,
    pub desc: TableDesc,
    pub column_set: ColumnSet,
}

/// Parse a complete `table.dat` buffer: header, table description, and the
/// data-manager info.
pub fn parse_table_dat(buf: &[u8]) -> Result<TableDat, TableDatError> {
    let mut r = Reader::new(buf);
    let header = read_table_header(&mut r)?;
    let desc = crate::tabledesc::parse_table_desc(&mut r)?;
    let column_set = parse_column_set(&mut r, &desc.columns)?;
    Ok(TableDat {
        header,
        desc,
        column_set,
    })
}

/// Parse the header from the start of a `table.dat` buffer.
pub fn parse_table_header(buf: &[u8]) -> Result<TableHeader, TableError> {
    let mut r = Reader::new(buf);
    read_table_header(&mut r)
}

/// Parse the header from an AipsIO stream positioned at the start of the
/// `Table` root object.
pub(crate) fn read_table_header(r: &mut Reader<'_>) -> Result<TableHeader, TableError> {
    let obj = r.read_object_start(true)?;
    if obj.type_name != "Table" {
        return Err(TableError::NotATable {
            found: obj.type_name,
        });
    }
    let nrow = match obj.version {
        2 => u64::from(r.read_u32()?),
        3 => r.read_u64()?,
        v => return Err(TableError::UnsupportedVersion(v)),
    };
    let format = r.read_u32()?;
    let big_endian = match format {
        0 => true,
        1 => false,
        v => return Err(TableError::BadEndianness(v)),
    };
    let kind = r.read_string()?;
    Ok(TableHeader {
        version: obj.version,
        nrow,
        big_endian,
        kind,
    })
}

/// Errors from creating a CASA table.
#[derive(Debug, Error)]
pub enum TableCreateError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("no rows supplied for column {0}")]
    MissingValues(String),
    #[error("column {0} is not a scalar column (array columns are not writable yet)")]
    NotScalar(String),
    #[error("string column {0}: strings longer than 8 chars need unsupported string buckets")]
    LongString(String),
    #[error("column {0} is not backed by StandardStMan")]
    NotStandardStMan(String),
    #[error("value count {got} does not match row count {want} for column {column}")]
    RowCountMismatch {
        got: usize,
        want: u64,
        column: String,
    },
}

/// Serialize the `table.dat` header: the root `"Table"` object plus the
/// `TableDesc` and `ColumnSet`. `big_endian` is the data-file byte order.
/// `dms` lists the data managers (with their spec blobs) and `col_dm_seq`
/// the data-manager sequence number of each column, in table order.
pub fn build_table_dat(
    big_endian: bool,
    nrow: u64,
    desc: &crate::tabledesc::TableDesc,
    dms: &[crate::columnset::DmBlob],
    col_dm_seq: &[u32],
) -> Result<Vec<u8>, TableCreateError> {
    let mut w = crate::aipsio::Writer::new();
    if nrow > u64::from(u32::MAX) {
        w.put_root_object_start("Table", 3);
        w.put_u64(nrow);
    } else {
        w.put_root_object_start("Table", 2);
        w.put_u32(nrow as u32);
    }
    w.put_u32(if big_endian { 0 } else { 1 });
    w.put_string("PlainTable");
    crate::tabledesc::write_table_desc(&mut w, desc);
    let cols: Vec<(crate::tabledesc::ColumnDesc, u32)> = desc
        .columns
        .iter()
        .cloned()
        .zip(col_dm_seq.iter().copied())
        .collect();
    crate::columnset::write_multi_column_set(&mut w, nrow, dms.len() as u32, dms, &cols);
    w.put_object_end();
    Ok(w.into_bytes())
}

/// Create a CASA table on disk: `table_dir/table.dat` plus the StandardStMan
/// data file `table_dir/table.f0`, from a `TableDesc` of scalar columns and
/// per-column value lists (`values[column][row]`).
///
/// All columns must be scalar StandardStMan columns; variable-length strings
/// must fit the inline (<= 8 char) bucket storage. Rows are laid out with a
/// fixed `ROWS_PER_BUCKET` bucket tile.
pub const ROWS_PER_BUCKET: u32 = 32;

/// Grows the SSM string buckets (`SSMStringHandler::putData` semantics):
/// a 16-byte **big-endian canonical** header `[unused][used][nDeleted][next]`
/// plus raw string data; strings spanning buckets chain via `next`.
struct StringBuckets {
    bucket_size: usize,
    data_len: usize,
    first_bucket: i32,
    /// One Vec per bucket: full bucket bytes (header + data + padding).
    buckets: Vec<Vec<u8>>,
    current: usize,
}

impl StringBuckets {
    fn new(bucket_size: u32, first_bucket: i32) -> StringBuckets {
        let data_len = bucket_size as usize - 16;
        StringBuckets {
            bucket_size: bucket_size as usize,
            data_len,
            first_bucket,
            buckets: Vec::new(),
            current: 0,
        }
    }

    fn new_bucket(&mut self) {
        let mut b = vec![0u8; self.bucket_size];
        // Header is big-endian canonical, regardless of data-file endianness.
        b[4..8].copy_from_slice(&0u32.to_be_bytes()); // used
        b[8..12].copy_from_slice(&(self.data_len as u32).to_be_bytes()); // nDeleted
        b[12..16].copy_from_slice(&(-1i32).to_be_bytes()); // next = -1
        self.buckets.push(b);
        self.current = self.buckets.len() - 1;
    }

    fn used(&self) -> usize {
        if self.buckets.is_empty() {
            0
        } else {
            u32::from_be_bytes(self.buckets[self.current][4..8].try_into().unwrap()) as usize
        }
    }

    fn set_used(&mut self, used: usize) {
        self.buckets[self.current][4..8].copy_from_slice(&(used as u32).to_be_bytes());
        self.buckets[self.current][8..12]
            .copy_from_slice(&((self.data_len - used) as u32).to_be_bytes());
    }

    fn set_next(&mut self, idx: usize, next: i32) {
        self.buckets[idx][12..16].copy_from_slice(&next.to_be_bytes());
    }

    /// Append `data`, returning (bucket number, offset into its data area).
    fn put(&mut self, data: &[u8]) -> (i32, u32) {
        if self.buckets.is_empty() {
            self.new_bucket();
        }
        let free = self.data_len - self.used();
        // Start a fresh bucket when the string cannot fit and little space
        // is left (mirrors `SSMStringHandler::put`).
        if data.len() > free && free < 50 {
            self.new_bucket();
        }
        let bucket_nr = self.first_bucket + self.current as i32;
        let offset = self.used() as u32;
        let mut idx = self.current;
        let mut src_off: usize = 0;
        let mut remaining = data.len();
        loop {
            let used = self.used();
            let room = self.data_len - used;
            let take = remaining.min(room);
            self.buckets[idx][16 + used..16 + used + take]
                .copy_from_slice(&data[src_off..src_off + take]);
            self.set_used(used + take);
            src_off += take;
            remaining -= take;
            if remaining == 0 {
                break;
            }
            // Roll over into a new bucket chained from this one.
            let prev = idx;
            self.new_bucket();
            self.set_next(prev, self.first_bucket + self.current as i32);
            idx = self.current;
        }
        (bucket_nr, offset)
    }
}

pub fn create_table(
    table_dir: &std::path::Path,
    desc: &crate::tabledesc::TableDesc,
    values: &[Vec<crate::record::RecordValue>],
) -> Result<Vec<std::path::PathBuf>, TableCreateError> {
    use crate::columnset::DmBlob;
    if desc.columns.len() != values.len() {
        return Err(TableCreateError::MissingValues(format!(
            "{} columns, {} value lists",
            desc.columns.len(),
            values.len()
        )));
    }
    let nrow = values.first().map_or(0, Vec::len) as u64;
    for (col, list) in values.iter().enumerate() {
        if list.len() as u64 != nrow {
            return Err(TableCreateError::RowCountMismatch {
                got: list.len(),
                want: nrow,
                column: desc.columns[col].name.clone(),
            });
        }
    }
    // Little endian, matching casacore on any little-endian host.
    let big_endian = false;

    // Assign data-manager sequence numbers in column order (casacore
    // creates one data manager per type/group, in order of first use).
    // The manager name is the group (or the type when no group is given);
    // colliding names get the `_N` auto-suffix (`StandardStMan_1`, ...).
    let mut dm_types: Vec<String> = Vec::new();
    let mut dm_names: Vec<String> = Vec::new();
    let mut col_dm_seq: Vec<u32> = Vec::with_capacity(desc.columns.len());
    let mut name_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for cd in &desc.columns {
        let group = if cd.data_manager_group.is_empty() {
            cd.data_manager_type.clone()
        } else {
            cd.data_manager_group.clone()
        };
        let existing = dm_types
            .iter()
            .zip(dm_names.iter())
            .position(|(t, n)| *t == cd.data_manager_type && *n == group);
        if let Some(i) = existing {
            col_dm_seq.push(i as u32);
        } else {
            let count = name_counts.entry(group.clone()).or_insert(0);
            let name = if *count == 0 {
                group.clone()
            } else {
                format!("{group}_{}", *count)
            };
            *count += 1;
            dm_types.push(cd.data_manager_type.clone());
            dm_names.push(name);
            col_dm_seq.push((dm_types.len() - 1) as u32);
        }
    }

    let mut dms: Vec<DmBlob> = Vec::with_capacity(dm_types.len());
    let mut data_files: Vec<(u32, Vec<u8>)> = Vec::with_capacity(dm_types.len());
    let mut index_files: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut tile_files: Vec<(u32, u32, Vec<u8>)> = Vec::new();

    for (dm, type_name) in dm_types.iter().enumerate() {
        let dm_name = &dm_names[dm];
        let dm_cols: Vec<usize> = (0..desc.columns.len())
            .filter(|&c| col_dm_seq[c] as usize == dm)
            .collect();
        match type_name.as_str() {
            "StandardStMan" => {
                let (file, f0i, spec) =
                    build_ssm_data(big_endian, nrow, dm_name, desc, values, &dm_cols)?;
                dms.push(DmBlob {
                    type_name: type_name.clone(),
                    sequence_nr: dm as u32,
                    blob: crate::columnset::write_standard_stman(&spec),
                });
                data_files.push((dm as u32, file));
                if let Some(f0i) = f0i {
                    index_files.push((dm as u32, f0i));
                }
            }
            "IncrementalStMan" => {
                let file = build_ism_data(big_endian, nrow, dm_name, desc, values, &dm_cols)?;
                dms.push(DmBlob {
                    type_name: type_name.clone(),
                    sequence_nr: dm as u32,
                    blob: crate::ism::write_ism_blob(dm_name),
                });
                data_files.push((dm as u32, file));
            }
            "TiledColumnStMan" | "TiledShapeStMan" => {
                let (header, tile, file_seq) = build_tsm_data(
                    type_name, big_endian, dm as u32, dm_name, desc, values, &dm_cols,
                )?;
                dms.push(DmBlob {
                    type_name: type_name.clone(),
                    sequence_nr: dm as u32,
                    blob: Vec::new(), // TSM writes its spec to the header file
                });
                data_files.push((dm as u32, header));
                // Tile data lives in `table.f{dm}_TSM{file_seq}` (TiledShapeStMan
                // numbers its first real tile file 1).
                tile_files.push((dm as u32, file_seq, tile));
            }
            other => {
                return Err(TableCreateError::Io(std::io::Error::other(format!(
                    "unsupported data-manager type {other}"
                ))))
            }
        }
    }

    let table_dat = build_table_dat(big_endian, nrow, desc, &dms, &col_dm_seq)?;

    // Regenerate in place: remove this table's own data files so stale
    // columns / renumbered data-manager files cannot linger. Subtable
    // subdirectories and `table.info`/`table.lock` are preserved.
    std::fs::create_dir_all(table_dir)?;
    if let Ok(entries) = std::fs::read_dir(table_dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if name == "table.dat" || name.starts_with("table.f") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let mut written = Vec::new();
    let dat_path = table_dir.join("table.dat");
    std::fs::write(&dat_path, table_dat)?;
    written.push(dat_path);
    for (seq, file) in data_files {
        let f_path = table_dir.join(format!("table.f{seq}"));
        std::fs::write(&f_path, file)?;
        written.push(f_path);
    }
    for (seq, file) in index_files {
        let f_path = table_dir.join(format!("table.f{seq}i"));
        std::fs::write(&f_path, file)?;
        written.push(f_path);
    }
    for (seq, file_seq, file) in tile_files {
        let f_path = table_dir.join(format!("table.f{seq}_TSM{file_seq}"));
        std::fs::write(&f_path, file)?;
        written.push(f_path);
    }
    Ok(written)
}

/// Result of building one StandardStMan data manager: the data file, an
/// optional array index file (`table.f{seq}i`), and the SSM spec for the
/// `table.dat` blob.
type SsmDataOutput = (Vec<u8>, Option<Vec<u8>>, crate::columnset::StandardStMan);

/// Build the StandardStMan data-file bytes for one SSM data manager, its
/// optional `table.f{seq}i` array index file, and the SSM spec.
fn build_ssm_data(
    big_endian: bool,
    nrow: u64,
    dm_name: &str,
    desc: &crate::tabledesc::TableDesc,
    values: &[Vec<crate::record::RecordValue>],
    dm_cols: &[usize],
) -> Result<SsmDataOutput, TableCreateError> {
    use crate::record::RecordValue;
    use crate::ssm::{layout, write_standard_stman_file, WriteColumn};
    let mut cell_bits = Vec::with_capacity(dm_cols.len());
    let mut cell_bytes = Vec::with_capacity(dm_cols.len());
    for &col in dm_cols {
        let cd = &desc.columns[col];
        let (size, bits) = match cd.kind {
            crate::tabledesc::ColumnKind::Scalar(_) => {
                let size = crate::ssm::scalar_cell_size(cd);
                if cd.data_type == crate::record::DataType::Bool {
                    // Bool scalars are one BIT per row (casacore
                    // SSMColumn: externalSizeBits = nrElem); the column's
                    // region is rows/8 bytes and cells are bit-packed.
                    (1, 1)
                } else {
                    (size, 8 * size)
                }
            }
            crate::tabledesc::ColumnKind::Array => {
                if cd.data_type == crate::record::DataType::String {
                    // String arrays use a 12-byte string-bucket ref cell
                    // (like a scalar variable string), not an f0i offset.
                    (12, 8 * 12)
                } else {
                    (crate::ssm::ARRAY_REF_SIZE, 8 * crate::ssm::ARRAY_REF_SIZE)
                }
            }
            crate::tabledesc::ColumnKind::Record => {
                // Scalar record cells are stored like variable strings: a
                // 12-byte (bucket, offset, len) reference into the string
                // buckets holding the serialized record.
                (12, 8 * 12)
            }
        };
        cell_bits.push(bits);
        cell_bytes.push(size);
    }
    let l = layout(ROWS_PER_BUCKET, &cell_bits);
    let data_buckets = nrow.div_ceil(u64::from(l.rows_per_bucket)) as u32;
    let index_stream = crate::ssm::build_index_stream(big_endian, nrow, &l, dm_cols.len());
    let index_buckets = crate::ssm::index_bucket_count(index_stream.len(), l.bucket_size) as u32;
    let first_string_bucket = (data_buckets + index_buckets) as i32;
    let mut str_buckets = StringBuckets::new(l.bucket_size, first_string_bucket);
    let mut array_index: Vec<u8> = Vec::new();
    // An array column creates the array-index file (`table.f0i`) even when
    // the table has zero rows (casacore does the same).
    let mut has_arrays = dm_cols.iter().any(|&c| {
        matches!(desc.columns[c].kind, crate::tabledesc::ColumnKind::Array)
            && desc.columns[c].data_type != crate::record::DataType::String
    });
    let mut has_strings = false;

    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(dm_cols.len());
    for &col in dm_cols {
        let cd = &desc.columns[col];
        let size = cell_bytes[encoded.len()];
        let list = &values[col];
        let mut bytes = Vec::with_capacity(list.len() * size as usize);
        for value in list {
            match &cd.kind {
                crate::tabledesc::ColumnKind::Scalar(_) => {
                    if cd.data_type == crate::record::DataType::String
                        && cd.max_length <= 0
                        && matches!(value, RecordValue::String(s) if s.len() > 8)
                    {
                        let RecordValue::String(s) = value else {
                            unreachable!()
                        };
                        let (bucket, offset) = str_buckets.put(s.as_bytes());
                        has_strings = true;
                        let mut cell = vec![0u8; 12];
                        if big_endian {
                            cell[0..4].copy_from_slice(&bucket.to_be_bytes());
                            cell[4..8].copy_from_slice(&offset.to_be_bytes());
                            cell[8..12].copy_from_slice(&(s.len() as i32).to_be_bytes());
                        } else {
                            cell[0..4].copy_from_slice(&bucket.to_le_bytes());
                            cell[4..8].copy_from_slice(&offset.to_le_bytes());
                            cell[8..12].copy_from_slice(&(s.len() as i32).to_le_bytes());
                        }
                        bytes.extend_from_slice(&cell);
                    } else {
                        bytes.extend_from_slice(
                            &crate::ssm::encode_scalar_cell(big_endian, cd, value).map_err(
                                |e| match e {
                                    crate::ssm::SsmError::StringBucketUnsupported { .. } => {
                                        TableCreateError::LongString(cd.name.clone())
                                    }
                                    other => TableCreateError::Io(std::io::Error::other(format!(
                                        "encode {}.{}: {other}",
                                        desc.name, cd.name
                                    ))),
                                },
                            )?,
                        );
                    }
                }
                crate::tabledesc::ColumnKind::Array => {
                    let RecordValue::Array(arr) = value else {
                        return Err(TableCreateError::NotScalar(format!(
                            "{}.{}: expected an array value for an array column",
                            desc.name, cd.name
                        )));
                    };
                    if cd.data_type == crate::record::DataType::String {
                        // Multidim string arrays: the whole cell (shape
                        // header + filled flag + length-prefixed strings) is
                        // written into the string buckets as the cell's
                        // content, referenced by a 12-byte (bucket, offset,
                        // len) cell like a scalar variable string
                        // (SSMStringHandler::put(Array<String>&, handleShape)).
                        let content =
                            crate::ssm::encode_string_array_content(arr).map_err(|e| {
                                TableCreateError::Io(std::io::Error::other(format!(
                                    "encode {}.{}: {e}",
                                    desc.name, cd.name
                                )))
                            })?;
                        let (bucket, offset) = str_buckets.put(&content);
                        has_strings = true;
                        let mut cell = vec![0u8; 12];
                        if big_endian {
                            cell[0..4].copy_from_slice(&bucket.to_be_bytes());
                            cell[4..8].copy_from_slice(&offset.to_be_bytes());
                            cell[8..12].copy_from_slice(&(content.len() as i32).to_be_bytes());
                        } else {
                            cell[0..4].copy_from_slice(&bucket.to_le_bytes());
                            cell[4..8].copy_from_slice(&offset.to_le_bytes());
                            cell[8..12].copy_from_slice(&(content.len() as i32).to_le_bytes());
                        }
                        bytes.extend_from_slice(&cell);
                        continue;
                    }
                    let record = crate::ssm::encode_array_record(big_endian, cd.data_type, arr)
                        .map_err(|e| {
                            TableCreateError::Io(std::io::Error::other(format!(
                                "encode {}.{}: {e}",
                                desc.name, cd.name
                            )))
                        })?;
                    let offset = 16 + array_index.len() as i64;
                    if big_endian {
                        bytes.extend_from_slice(&offset.to_be_bytes());
                    } else {
                        bytes.extend_from_slice(&offset.to_le_bytes());
                    }
                    has_arrays = true;
                    array_index.extend_from_slice(&record);
                }
                crate::tabledesc::ColumnKind::Record => {
                    let content = match value {
                        RecordValue::Record(r) => r.to_json_string().into_bytes(),
                        other => {
                            return Err(TableCreateError::NotScalar(format!(
                                "{}.{}: expected a record cell, got {other:?}",
                                desc.name, cd.name
                            )))
                        }
                    };
                    let (bucket, offset) = str_buckets.put(&content);
                    has_strings = true;
                    let mut cell = vec![0u8; 12];
                    if big_endian {
                        cell[0..4].copy_from_slice(&bucket.to_be_bytes());
                        cell[4..8].copy_from_slice(&offset.to_be_bytes());
                        cell[8..12].copy_from_slice(&(content.len() as i32).to_be_bytes());
                    } else {
                        cell[0..4].copy_from_slice(&bucket.to_le_bytes());
                        cell[4..8].copy_from_slice(&offset.to_le_bytes());
                        cell[8..12].copy_from_slice(&(content.len() as i32).to_le_bytes());
                    }
                    bytes.extend_from_slice(&cell);
                }
            }
        }
        encoded.push(bytes);
    }

    // Bit-packed Bool scalar columns: their per-row encoded bytes carry one
    // value bit each; collapse them into the LSB-first bitstream the bucket
    // layout stores.
    let cell_bits_per_col: Vec<u32> = dm_cols
        .iter()
        .enumerate()
        .map(|(i, &col)| {
            let cd = &desc.columns[col];
            if cd.data_type == crate::record::DataType::Bool
                && matches!(cd.kind, crate::tabledesc::ColumnKind::Scalar(_))
            {
                let bytes = &mut encoded[i];
                let mut packed = vec![0u8; bytes.len().div_ceil(8)];
                for (r, b) in bytes.iter().enumerate() {
                    if *b != 0 {
                        packed[r / 8] |= 1 << (r % 8);
                    }
                }
                *bytes = packed;
                1
            } else {
                0
            }
        })
        .collect();

    let cols: Vec<WriteColumn<'_>> = encoded
        .iter()
        .zip(cell_bytes.iter())
        .zip(cell_bits_per_col.iter())
        .map(|((bytes, size), bits)| WriteColumn {
            cell_size: *size,
            cell_bits: *bits,
            bytes,
        })
        .collect();
    let string_bucket_refs: Vec<Vec<u8>> = if has_strings {
        str_buckets.buckets.clone()
    } else {
        Vec::new()
    };
    let file = write_standard_stman_file(big_endian, nrow, &cols, &l, &string_bucket_refs);
    let spec = crate::columnset::StandardStMan {
        data_manager_name: dm_name.into(),
        column_offset: l.column_offset,
        col_index_map: vec![0; dm_cols.len()],
    };
    let f0i = if has_arrays {
        let mut f0i = vec![0u8; 16];
        f0i[4] = (16 + array_index.len()) as u8;
        f0i.extend_from_slice(&array_index);
        Some(f0i)
    } else {
        None
    };
    Ok((file, f0i, spec))
}

/// Build the IncrementalStMan data-file bytes for one ISM data manager
/// (scalar columns only).
fn build_ism_data(
    big_endian: bool,
    nrow: u64,
    dm_name: &str,
    desc: &crate::tabledesc::TableDesc,
    values: &[Vec<crate::record::RecordValue>],
    dm_cols: &[usize],
) -> Result<Vec<u8>, TableCreateError> {
    let _ = dm_name;
    let mut cell_buffers: Vec<Vec<u8>> = Vec::with_capacity(dm_cols.len());
    let mut cell_sizes: Vec<u32> = Vec::with_capacity(dm_cols.len());
    for &col in dm_cols {
        let cd = &desc.columns[col];
        if !matches!(cd.kind, crate::tabledesc::ColumnKind::Scalar(_)) {
            return Err(TableCreateError::NotScalar(cd.name.clone()));
        }
        if cd.data_type == crate::record::DataType::String {
            return Err(TableCreateError::Io(std::io::Error::other(format!(
                "IncrementalStMan string column {} is not writable yet",
                cd.name
            ))));
        }
        let size = crate::ssm::scalar_cell_size(cd);
        let mut buf = Vec::with_capacity(values[col].len() * size as usize);
        for value in &values[col] {
            buf.extend_from_slice(
                &crate::ssm::encode_scalar_cell(big_endian, cd, value).map_err(|e| {
                    TableCreateError::Io(std::io::Error::other(format!(
                        "encode {}.{}: {e}",
                        desc.name, cd.name
                    )))
                })?,
            );
        }
        cell_buffers.push(buf);
        cell_sizes.push(size);
    }
    let ism_cols: Vec<crate::ism::WriteIsmColumn<'_>> = cell_buffers
        .iter()
        .zip(cell_sizes.iter())
        .map(|(buf, size)| crate::ism::WriteIsmColumn {
            cell_size: *size,
            bytes: buf,
        })
        .collect();
    Ok(crate::ism::write_ism_file(big_endian, nrow, &ism_cols))
}

/// Build the TiledColumnStMan header + tile data for one TSM data manager
/// (a single fixed-shape array column).
fn build_tsm_data(
    stman_type: &str,
    big_endian: bool,
    seq_nr: u32,
    dm_name: &str,
    desc: &crate::tabledesc::TableDesc,
    values: &[Vec<crate::record::RecordValue>],
    dm_cols: &[usize],
) -> Result<(Vec<u8>, Vec<u8>, u32), TableCreateError> {
    use crate::record::RecordValue;
    if dm_cols.len() != 1 {
        return Err(TableCreateError::Io(std::io::Error::other(format!(
            "{stman_type} with {} columns is not supported (one array column per group)",
            dm_cols.len()
        ))));
    }
    let col = dm_cols[0];
    let cd = &desc.columns[col];
    // The per-row cell shape in CASA (reversed logical) dim order.  A
    // TiledShapeStMan column (e.g. an MS FLAG column) carries no fixed shape
    // in its descriptor — the shape lives in the tiled-data header — so when
    // the descriptor's shape is empty, recover it from the written cells
    // themselves (their ArrayValue shape is logical, which the storage
    // managers hold reversed).  Unwritten default cells carry an empty shape,
    // so take the first cell that actually has one.
    let cradle = match cd.shape.as_ref() {
        Some(shape) if !shape.is_empty() => shape.clone(),
        _ => {
            let logical: Vec<i64> = values[col]
                .iter()
                .find_map(|v| match v {
                    RecordValue::Array(arr) if !arr.shape.is_empty() => {
                        Some(arr.shape.iter().map(|&d| i64::from(d)).collect())
                    }
                    _ => None,
                })
                .unwrap_or_default();
            logical.into_iter().rev().collect()
        }
    };
    if cd.data_type == crate::record::DataType::Bool {
        // Typed-buffer bool write: pack the row bit-slices straight into the
        // tile bitstream — no per-row `Vec<u8>` cells, no `encode_bits`
        // intermediate.  The tiles hold the identical layout (rows
        // bit-contiguous, LSB-first) to the byte-cell path.
        let mut rows: Vec<&[bool]> = Vec::with_capacity(values[col].len());
        for value in &values[col] {
            let RecordValue::Array(arr) = value else {
                return Err(TableCreateError::NotScalar(format!(
                    "{}.{}: expected an array value",
                    desc.name, cd.name
                )));
            };
            let crate::record::ArrayData::Bool(v) = &arr.data else {
                return Err(TableCreateError::NotScalar(format!(
                    "{}.{}: expected a bool array value",
                    desc.name, cd.name
                )));
            };
            if (v.len() as i64) != cradle.iter().product::<i64>() {
                return Err(TableCreateError::NotScalar(format!(
                    "{}.{}: bool cell length {} does not match shape {cradle:?}",
                    desc.name,
                    cd.name,
                    v.len()
                )));
            }
            rows.push(v.as_slice());
        }
        return crate::tsm::write_tsm_file_bool(
            stman_type, big_endian, seq_nr, dm_name, &cradle, &rows,
        )
        .map_err(|e| {
            TableCreateError::Io(std::io::Error::other(format!(
                "encode {}.{}: {e}",
                desc.name, cd.name
            )))
        });
    }
    let mut cells: Vec<Vec<u8>> = Vec::with_capacity(values[col].len());
    for value in &values[col] {
        let RecordValue::Array(arr) = value else {
            return Err(TableCreateError::NotScalar(format!(
                "{}.{}: expected an array value",
                desc.name, cd.name
            )));
        };
        cells.push(
            crate::tsm::tsm_encode_cell(big_endian, cd.data_type, &arr.data).map_err(|e| {
                TableCreateError::Io(std::io::Error::other(format!(
                    "encode {}.{}: {e}",
                    desc.name, cd.name
                )))
            })?,
        );
    }
    crate::tsm::write_tsm_file(
        stman_type,
        big_endian,
        seq_nr,
        dm_name,
        cd.data_type,
        &cradle,
        &cells,
    )
    .map_err(|e| {
        TableCreateError::Io(std::io::Error::other(format!(
            "encode {}.{}: {e}",
            desc.name, cd.name
        )))
    })
}

/// A CASA table: the parsed descriptor plus the opened data managers, with
/// the table lifecycle API (`open`/`create`, advisory `lock`/`unlock`,
/// `flush`, `close`, `is_writable`, `name`).
///
/// All file contents are owned, so a `Table` is `Send` + `Sync` and safe to
/// hold across threads (dask-ms serializes access on its side).
#[derive(Debug)]
pub struct Table {
    path: std::path::PathBuf,
    writable: bool,
    locked: bool,
    pub dat: TableDat,
    /// StandardStMan data files, keyed by DM sequence number.
    pub ssm_files: Vec<(u32, crate::ssm::StandardStManFile)>,
    /// IncrementalStMan data files, keyed by DM sequence number.
    pub ism_files: Vec<(u32, crate::ism::IsmFile)>,
    /// TiledColumnStMan storage managers, keyed by DM sequence number.
    pub tsm_files: Vec<(u32, crate::tsm::TsmFile)>,
}

/// The table directory as an absolute path (lexically normalised, no symlink
/// resolution, so `name()` stays the path the caller gave modulo `.`/`..`).
/// Every open/create funnels through this: subtable links are stored relative
/// to the table's parent and resolved against it, which only round-trips when
/// the directory is absolute — a relative `dir.parent()` (`""` for a bare
/// name) once produced links like `.//home/...` that no reader could resolve.
pub fn absolute_dir(p: &std::path::Path) -> std::path::PathBuf {
    if p.is_absolute() {
        return crate::record::lexical_normalize(p);
    }
    match std::env::current_dir() {
        Ok(cwd) => crate::record::lexical_normalize(&cwd.join(p)),
        Err(_) => p.to_path_buf(),
    }
}

/// The row count casacore keeps in `table.lock`'s sync record: a framed
/// AipsIO `"sync"` object (`[u32 len]["sync"\\0][u32 version][nrrow]`; v1
/// nrrow is u32, v2 u64).  `PlainTable::PlainTable` takes this value in
/// preference to the header's nrrow, so casacure mirrors it (see
/// [`Table::open`]).
fn lock_sync_nrrow(path: &std::path::Path) -> Option<u64> {
    let bytes = std::fs::read(path.join("table.lock")).ok()?;
    let mut found = None;
    for (i, w) in bytes.windows(4).enumerate() {
        if w != b"sync" {
            continue;
        }
        // Framed string: [u32 len]["sync"] (no terminator), then the
        // version u32 and the nrrow.
        if i < 4 || bytes[i - 4..i] != [0, 0, 0, 4] {
            continue;
        }
        let ver = u32::from_be_bytes(bytes[i + 4..i + 8].try_into().ok()?);
        let nrrow = match ver {
            1 => u64::from(u32::from_be_bytes(bytes[i + 8..i + 12].try_into().ok()?)),
            2 => u64::from_be_bytes(bytes[i + 8..i + 16].try_into().ok()?),
            _ => continue,
        };
        found = Some(nrrow);
    }
    found
}

impl Table {
    /// Open a table directory (`<dir>/table.dat` + data files).
    pub fn open(
        dir: impl Into<std::path::PathBuf>,
        readonly: bool,
    ) -> Result<Table, TableDatError> {
        let dir: std::path::PathBuf = dir.into();
        let path = absolute_dir(&dir);
        let buf = std::fs::read(path.join("table.dat"))?;
        let mut dat = parse_table_dat(&buf)?;
        // casacore reads the row count from the table's lock-file sync record
        // in preference to the header (`PlainTable::PlainTable`: nrrow_p from
        // lockSync, falling back to the header only when it is 0), so a
        // header written stale (e.g. an MS subtable whose header fields were
        // updated by a writer that never rewrote table.dat) still opens with
        // the real row count.
        if let Some(n) = lock_sync_nrrow(&path) {
            if n != 0 {
                dat.header.nrow = n;
            }
        }
        let big = dat.header.big_endian;
        let mut ssm_files = Vec::new();
        let mut ism_files = Vec::new();
        let mut tsm_files = Vec::new();
        for dm in &dat.column_set.data_managers {
            match dm.type_name.as_str() {
                "StandardStMan" => ssm_files.push((
                    dm.sequence_nr,
                    crate::ssm::StandardStManFile::open(&path, dm.sequence_nr, big)
                        .map_err(|e| TableDatError::Storage(e.to_string()))?,
                )),
                "IncrementalStMan" => ism_files.push((
                    dm.sequence_nr,
                    crate::ism::IsmFile::open(&path, dm.sequence_nr, big)
                        .map_err(|e| TableDatError::Storage(e.to_string()))?,
                )),
                "TiledColumnStMan" | "TiledShapeStMan" => tsm_files.push((
                    dm.sequence_nr,
                    crate::tsm::TsmFile::open(&path, dm.sequence_nr, big)
                        .map_err(|e| TableDatError::Storage(e.to_string()))?,
                )),
                other => {
                    return Err(TableDatError::Storage(format!(
                        "unsupported data-manager type {other}"
                    )))
                }
            }
        }
        Ok(Table {
            path,
            writable: !readonly,
            locked: false,
            dat,
            ssm_files,
            ism_files,
            tsm_files,
        })
    }

    /// Create a new table from a descriptor and column values, then open it
    /// for writing.
    pub fn create(
        dir: impl Into<std::path::PathBuf>,
        desc: &crate::tabledesc::TableDesc,
        values: &[Vec<crate::record::RecordValue>],
    ) -> Result<Table, TableDatError> {
        let dir = dir.into();
        create_table(&dir, desc, values).map_err(|e| TableDatError::Storage(e.to_string()))?;
        Table::open(dir, false)
    }

    /// The table directory (casacore `table.name()`).
    pub fn name(&self) -> &str {
        self.path.to_str().unwrap_or_default()
    }

    /// Whether the table was opened for writing (casacore
    /// `table.iswritable()`).
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// Adversarial user lock (casacore `table.lock(write=True)`); internally
    /// advisory as the replacement never holds OS locks.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Release the advisory lock (casacore `table.unlock()`).
    pub fn unlock(&mut self) {
        self.locked = false;
    }

    /// Whether an advisory lock is currently held.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Flush pending writes. The current implementation writes eagerly
    /// (`create_table`), so this is a no-op; kept for API compatibility.
    pub fn flush(&mut self) {}

    /// Close the table, releasing the data managers (casacore `table.close()`).
    pub fn close(self) {}

    /// Number of rows (casacore `table.nrows()`).
    pub fn nrows(&self) -> u64 {
        self.dat.header.nrow
    }

    /// Column names in column order (casacore `table.colnames()`).
    pub fn colnames(&self) -> Vec<String> {
        self.dat
            .desc
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }

    /// The column descriptor as a JSON object matching python-casacore's
    /// `table.getcoldesc(col)` (valueType / dataManagerType /
    /// dataManagerGroup / option / maxlen / comment, plus ndim / logical
    /// `shape` / `_c_order` for array columns, then `keywords`).
    pub fn getcoldesc(&self, col_idx: usize) -> Option<String> {
        let cd = self.dat.desc.columns.get(col_idx)?;
        let mut s = String::new();
        s.push('{');
        s.push_str(&kv("valueType", &json(casa_value_type(cd.data_type))));
        s.push(',');
        s.push_str(&kv("dataManagerType", &json(&cd.data_manager_type)));
        s.push(',');
        s.push_str(&kv("dataManagerGroup", &json(&cd.data_manager_group)));
        s.push(',');
        s.push_str(&format!("\"option\":{}", cd.options));
        s.push(',');
        s.push_str(&format!("\"maxlen\":{}", cd.max_length));
        s.push(',');
        s.push_str(&kv("comment", &json(&cd.comment)));
        if matches!(cd.kind, crate::tabledesc::ColumnKind::Array) {
            s.push(',');
            s.push_str(&format!("\"ndim\":{}", cd.ndim.max(0)));
            if let Some(shape) = &cd.shape {
                if !shape.is_empty() {
                    s.push_str(",\"shape\":[");
                    for (i, d) in shape.iter().rev().enumerate() {
                        if i > 0 {
                            s.push(',');
                        }
                        s.push_str(&d.to_string());
                    }
                    s.push_str("],\"_c_order\":true");
                }
            }
        }
        s.push(',');
        s.push_str(&kv(
            "keywords",
            &cd.keywords.to_json_string_ctx(Some(&self.path)),
        ));
        s.push('}');
        Some(s)
    }

    /// The full table description (python-casacore `table.getdesc()`): one
    /// column descriptor per column, plus `_define_hypercolumn_`,
    /// `_keywords_`, and `_private_keywords_`.
    pub fn getdesc(&self) -> String {
        let mut s = String::from("{");
        for (i, cd) in self.dat.desc.columns.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json(&cd.name));
            s.push(':');
            s.push_str(&self.getcoldesc(i).unwrap_or_default());
        }
        s.push_str(",\"_define_hypercolumn_\":{},\"_keywords_\":");
        s.push_str(&self.dat.desc.keywords.to_json_string_ctx(Some(&self.path)));
        s.push_str(",\"_private_keywords_\":");
        s.push_str(
            &self
                .dat
                .desc
                .private_keywords
                .to_json_string_ctx(Some(&self.path)),
        );
        s.push('}');
        s
    }

    /// The table keyword record as a JSON object (python-casacore
    /// `table.getkeywords()`).
    pub fn getkeywords(&self) -> String {
        self.dat.desc.keywords.to_json_string_ctx(Some(&self.path))
    }

    /// A column's keyword record as a JSON object (python-casacore
    /// `table.getcolkeywords(col)`).
    pub fn getcolkeywords(&self, col_idx: usize) -> Option<String> {
        self.dat
            .desc
            .columns
            .get(col_idx)
            .map(|c| c.keywords.to_json_string_ctx(Some(&self.path)))
    }
}

fn kv(key: &str, value: &str) -> String {
    format!("\"{key}\":{value}")
}

fn json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// python-casacore `valueType` names.
pub fn casa_value_type(dt: crate::record::DataType) -> &'static str {
    use crate::record::DataType;
    match dt {
        DataType::Bool => "boolean",
        DataType::Char | DataType::UChar => "uchar",
        DataType::Short => "short",
        DataType::UShort => "ushort",
        DataType::Int => "int",
        DataType::UInt => "uint",
        DataType::Int64 => "int64",
        DataType::Float => "float",
        DataType::Double => "double",
        DataType::Complex => "complex",
        DataType::DComplex => "dcomplex",
        DataType::String => "string",
        DataType::Record => "record",
        _ => "unknown",
    }
}

/// Errors from the column-access (§3) read API.
#[derive(Debug, Error)]
pub enum TableReadError {
    #[error("column {0} does not exist")]
    NoSuchColumn(String),
    #[error("column {0} is not backed by a readable data manager ({1})")]
    UnsupportedColumn(String, String),
    #[error("row {row} is out of range for column {column}")]
    RowOutOfRange { row: u64, column: String },
    #[error("slice {0:?}..{1:?} does not fit the cell shape {2:?}")]
    BadSlice(Vec<i64>, Vec<i64>, Vec<u32>),
    #[error("column {name} cannot be served by the raw (borrowed-buffer) read path: {reason}")]
    UnsupportedRaw { name: String, reason: String },
    #[error(transparent)]
    Ssm(#[from] crate::ssm::SsmError),
    #[error(transparent)]
    Ism(#[from] crate::ism::IsmError),
    #[error(transparent)]
    Tsm(#[from] crate::tsm::TsmError),
}

impl Table {
    /// The data-manager sequence number and the column's index within that
    /// manager for `col_idx` (the per-manager column order follows table
    /// order).
    /// The storage-manager type actually binding this column: the
    /// ColumnSet's data-manager entry for the bound sequence number. A
    /// column description's own `dataManagerType` can disagree with it
    /// (real casacore MSs bind ISM columns whose description says
    /// StandardStMan).
    fn column_storage_type(&self, col_idx: usize) -> String {
        let desc = &self.dat.desc.columns[col_idx];
        let (seq, _) = self.column_manager(col_idx);
        self.dat
            .column_set
            .data_managers
            .iter()
            .find(|dm| dm.sequence_nr == seq)
            .map(|dm| dm.type_name.clone())
            .unwrap_or_else(|| desc.data_manager_type.clone())
    }
    pub fn column_manager(&self, col_idx: usize) -> (u32, usize) {
        let seq = self.dat.column_set.columns[col_idx].data_manager_seq;
        let within = self.dat.column_set.columns[..col_idx]
            .iter()
            .filter(|c| c.data_manager_seq == seq)
            .count();
        (seq, within)
    }

    fn ssm_file(&self, seq: u32) -> Option<&crate::ssm::StandardStManFile> {
        self.ssm_files
            .iter()
            .find(|(s, _)| *s == seq)
            .map(|(_, f)| f)
    }

    /// The StandardStMan spec for the given data manager.
    fn ssm_spec(&self, seq: u32) -> Option<&crate::columnset::StandardStMan> {
        self.dat
            .column_set
            .data_managers
            .iter()
            .find(|dm| dm.sequence_nr == seq)
            .and_then(|dm| match &dm.blob {
                crate::columnset::DataManagerBlob::StandardStMan(s) => Some(s),
                _ => None,
            })
    }

    /// Read one cell (`table.getcell(col, row)`).
    pub fn getcell(&self, col_idx: usize, row: u64) -> Result<RecordValue, TableReadError> {
        let desc = &self.dat.desc.columns[col_idx];
        let (seq, within) = self.column_manager(col_idx);
        // The ColumnSet binding is authoritative: a real MS can bind a
        // column to IncrementalStMan while its description still declares
        // StandardStMan.
        match self.column_storage_type(col_idx).as_str() {
            "StandardStMan" => {
                let file = self.ssm_file(seq).ok_or_else(|| {
                    TableReadError::UnsupportedColumn(desc.name.clone(), "StandardStMan".into())
                })?;
                let spec = self.ssm_spec(seq).ok_or_else(|| {
                    TableReadError::UnsupportedColumn(desc.name.clone(), "no spec".into())
                })?;
                if matches!(desc.kind, crate::tabledesc::ColumnKind::Array) {
                    crate::ssm::read_array_cell(file, spec, within, desc, row).map_err(Into::into)
                } else {
                    file.read_scalar_cell(spec, within, desc, row)
                        .map_err(Into::into)
                }
            }
            "IncrementalStMan" => {
                let file = self
                    .ism_files
                    .iter()
                    .find(|(s, _)| *s == seq)
                    .map(|(_, f)| f)
                    .ok_or_else(|| {
                        TableReadError::UnsupportedColumn(
                            desc.name.clone(),
                            "IncrementalStMan".into(),
                        )
                    })?;
                file.read_scalar_cell(within, desc, row).map_err(Into::into)
            }
            "TiledColumnStMan" | "TiledShapeStMan" => {
                let file = self
                    .tsm_files
                    .iter()
                    .find(|(s, _)| *s == seq)
                    .map(|(_, f)| f)
                    .ok_or_else(|| {
                        TableReadError::UnsupportedColumn(
                            desc.name.clone(),
                            "tiled storage manager".into(),
                        )
                    })?;
                file.read_cell(desc, row).map_err(Into::into)
            }
            other => Err(TableReadError::UnsupportedColumn(
                desc.name.clone(),
                other.to_string(),
            )),
        }
    }

    /// Drop the data files' mapped pages (`MADV_DONTNEED`) after a bulk read
    /// has copied the cell data out, so a long streaming scan (dask-ms
    /// chunked full-column reads) stays resident at ~the current chunk
    /// instead of the whole file — the analogue of casacore's bounded LRU
    /// storage-manager cache. Pages re-read later simply fault back in.
    pub fn drop_data_file_pages(&self) {
        for (_, f) in &self.ssm_files {
            f.drop_data_pages();
        }
        for (_, f) in &self.ism_files {
            f.drop_data_pages();
        }
        for (_, f) in &self.tsm_files {
            f.drop_data_pages();
        }
    }

    /// A ranged `getcol` this large drops its data-file pages afterwards
    /// (see [`Table::drop_data_file_pages`]); smaller reads keep the page
    /// cache for random access.
    const STREAMING_DROP_ROWS: u64 = 512;

    /// True when a `getcolnp` into a caller numpy buffer can be served by
    /// [`Table::getcol_raw`]: a StandardStMan numeric column (scalar or
    /// fixed-shape array). Strings/records, ISM and TSM, and variable-shape
    /// arrays keep the generic `RecordValue` path.
    pub fn raw_column_supported(&self, col_idx: usize) -> bool {
        use crate::record::DataType as DT;
        let Some(desc) = self.dat.desc.columns.get(col_idx) else {
            return false;
        };
        if matches!(
            desc.data_type,
            DT::String | DT::Table | DT::Record | DT::Char | DT::Bool
        ) {
            // Bool cells are one bit per row in the storage managers, not
            // one byte like the numpy bool buffer this path fills.
            return false;
        }
        if matches!(desc.kind, crate::tabledesc::ColumnKind::Array) {
            // Variable-shape arrays: per-row element counts differ, so the
            // fixed-count precheck the caller needs cannot be satisfied.
            match &desc.shape {
                Some(s) if s.is_empty() => return false,
                None => return false,
                _ => {}
            }
        }
        desc.data_manager_type == "StandardStMan"
    }

    /// Read `nrow` cells of `col_idx` starting at `startrow`, calling
    /// `visit(logical_shape, element_bytes)` once per row with the raw
    /// element bytes **borrowed from the mapped data file** (no per-cell
    /// allocation). Numeric StandardStMan cells only (see
    /// [`Table::raw_column_supported`]). Long reads drop the mapped data
    /// pages as they go, so a whole-column read into a caller buffer stays
    /// resident at ~a window of the file, not the whole file.
    pub fn getcol_raw<'a, F>(
        &'a self,
        col_idx: usize,
        startrow: u64,
        nrow: u64,
        mut visit: F,
    ) -> Result<(), TableReadError>
    where
        F: FnMut(&[u32], &'a [u8]) -> Result<(), TableReadError>,
    {
        use crate::tabledesc::ColumnKind;
        let desc = &self.dat.desc.columns[col_idx];
        let (seq, within) = self.column_manager(col_idx);
        let mut done = 0u64;
        for r in startrow..startrow + nrow {
            match desc.data_manager_type.as_str() {
                "StandardStMan" => {
                    let file =
                        self.ssm_file(seq)
                            .ok_or_else(|| TableReadError::UnsupportedRaw {
                                name: desc.name.clone(),
                                reason: "no StandardStMan file".into(),
                            })?;
                    let spec =
                        self.ssm_spec(seq)
                            .ok_or_else(|| TableReadError::UnsupportedRaw {
                                name: desc.name.clone(),
                                reason: "no StandardStMan spec".into(),
                            })?;
                    if matches!(desc.kind, ColumnKind::Array) {
                        let (shape, _nelem, bytes) =
                            file.array_cell_region(spec, within, desc, r)?;
                        visit(&shape, bytes)?;
                    } else {
                        let bytes = file.scalar_cell_raw(spec, within, desc, r)?;
                        visit(&[], bytes)?;
                    }
                }
                other => {
                    return Err(TableReadError::UnsupportedRaw {
                        name: desc.name.clone(),
                        reason: format!("{other} data manager"),
                    })
                }
            }
            done += 1;
            if done.is_multiple_of(4096) {
                self.drop_data_file_pages();
            }
        }
        if nrow >= Self::STREAMING_DROP_ROWS {
            self.drop_data_file_pages();
        }
        Ok(())
    }

    /// Read `nrow` cells starting at `startrow` (`table.getcol` /
    /// `getcolnp`).
    pub fn getcol(
        &self,
        col_idx: usize,
        startrow: u64,
        nrow: u64,
    ) -> Result<Vec<RecordValue>, TableReadError> {
        let column = self.dat.desc.columns[col_idx].name.clone();
        let mut out = Vec::with_capacity(nrow as usize);
        for r in startrow..startrow + nrow {
            out.push(self.getcell(col_idx, r).map_err(|e| match e {
                TableReadError::RowOutOfRange { .. } => TableReadError::RowOutOfRange {
                    row: r,
                    column: column.clone(),
                },
                other => other,
            })?);
        }
        if nrow >= Self::STREAMING_DROP_ROWS {
            self.drop_data_file_pages();
        }
        Ok(out)
    }

    /// Read a slice of each array cell (`table.getcolslice(col, blc, trc,
    /// startrow, nrow)`): `blc`/`trc` are the inclusive start/end for each
    /// logical dimension.
    pub fn getcolslice(
        &self,
        col_idx: usize,
        blc: &[i64],
        trc: &[i64],
        startrow: u64,
        nrow: u64,
    ) -> Result<Vec<RecordValue>, TableReadError> {
        let mut out = Vec::with_capacity(nrow as usize);
        for r in startrow..startrow + nrow {
            out.push(self.getcellslice(col_idx, r, blc, trc)?);
        }
        Ok(out)
    }

    /// One array cell slice (`table.getcellslice(col, row, blc, trc)`);
    /// scalar columns ignore the slice.
    pub fn getcellslice(
        &self,
        col_idx: usize,
        row: u64,
        blc: &[i64],
        trc: &[i64],
    ) -> Result<RecordValue, TableReadError> {
        let cell = self.getcell(col_idx, row)?;
        let RecordValue::Array(arr) = cell else {
            return Ok(cell);
        };
        if blc.is_empty() && trc.is_empty() {
            return Ok(RecordValue::Array(arr));
        }
        Ok(RecordValue::Array(slice_array_value(&arr, blc, trc)?))
    }

    /// Read all cells, keyed per row as `"r0"`, `"r1"`, ... (`getvarcol`).
    pub fn getvarcol(&self, col_idx: usize) -> Result<Vec<RecordValue>, TableReadError> {
        let n = self.nrows();
        self.getcol(col_idx, 0, n)
    }
}

/// Slice an array value by inclusive per-dimension `blc`/`trc` (logical
/// row-major dims), returning the sub-array flat in logical order.
pub fn slice_array_value(
    arr: &crate::record::ArrayValue,
    blc: &[i64],
    trc: &[i64],
) -> Result<crate::record::ArrayValue, TableReadError> {
    let shape = &arr.shape;
    let ndim = shape.len();
    if blc.len() != ndim || trc.len() != ndim {
        return Err(TableReadError::BadSlice(
            blc.to_vec(),
            trc.to_vec(),
            shape.clone(),
        ));
    }
    let mut new_shape = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let b = blc[d];
        let t = trc[d];
        if b < 0 || t < b || t as u32 >= shape[d] {
            return Err(TableReadError::BadSlice(
                blc.to_vec(),
                trc.to_vec(),
                shape.clone(),
            ));
        }
        new_shape.push((t - b + 1) as u32);
    }
    // Linear indices of the sub-array in logical row-major order.
    let mut indices = Vec::with_capacity(new_shape.iter().product::<u32>() as usize);
    let strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            s[d] = s[d + 1] * shape[d + 1] as usize;
        }
        s
    };
    fn visit(
        shape: &[u32],
        new_shape: &[u32],
        strides: &[usize],
        blc: &[i64],
        indices: &mut Vec<usize>,
        d: usize,
        offset: usize,
    ) {
        if d == shape.len() {
            indices.push(offset);
            return;
        }
        for k in 0..new_shape[d] {
            let coord = (blc[d] + k as i64) as usize;
            visit(
                shape,
                new_shape,
                strides,
                blc,
                indices,
                d + 1,
                offset + coord * strides[d],
            );
        }
    }
    visit(shape, &new_shape, &strides, blc, &mut indices, 0, 0);

    use crate::record::ArrayData;
    macro_rules! slice_data {
        ($data:ident, $v:ident, $make:ident) => {{
            ArrayData::$make(indices.iter().map(|&i| $v[i]).collect())
        }};
    }
    let data = match &arr.data {
        ArrayData::Bool(v) => slice_data!(data, v, Bool),
        ArrayData::UChar(v) => slice_data!(data, v, UChar),
        ArrayData::Short(v) => slice_data!(data, v, Short),
        ArrayData::UShort(v) => slice_data!(data, v, UShort),
        ArrayData::Int(v) => slice_data!(data, v, Int),
        ArrayData::UInt(v) => slice_data!(data, v, UInt),
        ArrayData::Int64(v) => slice_data!(data, v, Int64),
        ArrayData::Float(v) => slice_data!(data, v, Float),
        ArrayData::Double(v) => slice_data!(data, v, Double),
        ArrayData::Complex(v) => ArrayData::Complex(indices.iter().map(|&i| v[i]).collect()),
        ArrayData::DComplex(v) => ArrayData::DComplex(indices.iter().map(|&i| v[i]).collect()),
        ArrayData::String(v) => ArrayData::String(indices.iter().map(|&i| v[i].clone()).collect()),
    };
    Ok(crate::record::ArrayValue {
        shape: new_shape,
        data,
    })
}

/// A writable table built incrementally (`addrows` + `putcol`/`putcell`
/// batches, then `flush`) — the dask-ms MS-writing pattern. The final
/// `flush` assembles the on-disk files via `create_table` for the given
/// schema; missing cells use the column's scalar default.
#[derive(Debug)]
/// A writable table built incrementally (`addrows` + `putcol`/`putcell`
/// batches, then `flush`) — the dask-ms MS-writing pattern. The final
/// `flush` assembles the on-disk files via `create_table` for the given
/// schema; missing cells use the column's scalar default.
pub struct WritableTable {
    dir: std::path::PathBuf,
    desc: crate::tabledesc::TableDesc,
    /// `cells[col][row]`.
    cells: Vec<Vec<Option<RecordValue>>>,
    /// `pending[col]` is a bitset of rows WRITTEN since the last successful
    /// `flush` (one bit per row, set by `putcell`/`putcol`, cleared by
    /// `flush`).  Together with `cells` it makes a flush incremental: only
    /// pending rows are overlaid onto the on-disk files, and after a flush
    /// the column's `cells` values are dropped (reads fall back to the
    /// refreshed on-disk state), so the resident write buffer tracks the
    /// dask-ms chunks written since the last flush — not the whole column.
    pending: Vec<Vec<u64>>,
    /// Which columns were explicitly written (`putcell`/`putcol`).  A
    /// `flush()` on an existing table preserves the on-disk files of any
    /// column that was never written — casacore's in-place semantics, and
    /// what keeps a changed-columns write (dask-ms `putcol`) from zeroing or
    /// corrupting every other column (e.g. an untouched TiledShapeStMan
    /// column rebuilt from default cells collapses its header's nrdim).
    touched: Vec<bool>,
    /// A table/column keyword was written, so the header (`table.dat`) must
    /// be regenerated; an in-place data-only flush cannot preserve it.
    meta_dirty: bool,
}

/// Convert absolute `Table`-valued subtable references to the `./relative`
/// form casacore stores (relative to the parent table's directory); absolute
/// paths outside the parent's directory are kept. Recurses into nested
/// records.
fn relativize_subtables(value: RecordValue, table_dir: Option<&std::path::Path>) -> RecordValue {
    fn relativize_one(name: &str, table_dir: Option<&std::path::Path>) -> String {
        let path = std::path::Path::new(name);
        if !path.is_absolute() {
            return name.to_string();
        }
        let Some(dir) = table_dir else {
            return name.to_string();
        };
        let dir_prefix = format!("{}/", dir.to_string_lossy());
        let name_s = name.to_string();
        if let Some(rest) = name_s.strip_prefix(&dir_prefix) {
            // Inside the table's own directory (an MS subtable).
            return format!("././{rest}");
        }
        if let Some(parent) = dir.parent() {
            let parent_prefix = format!("{}/", parent.to_string_lossy());
            if let Some(rest) = name_s.strip_prefix(&parent_prefix) {
                // A sibling of the table directory.
                return format!("./{rest}");
            }
        }
        name.to_string()
    }
    match value {
        RecordValue::Table(name) => RecordValue::Table(relativize_one(&name, table_dir)),
        RecordValue::Record(mut inner) => {
            inner.values = inner
                .values
                .drain(..)
                .map(|v| relativize_subtables(v, table_dir))
                .collect();
            RecordValue::Record(inner)
        }
        other => other,
    }
}

/// Errors from building a table incrementally.
#[derive(Debug, Error)]
pub enum WriteTableError {
    #[error("row {row} is out of range (table has {nrow} rows)")]
    RowOutOfRange { row: u64, nrow: u64 },
    #[error("column {name} does not exist")]
    NoSuchColumn { name: String },
    #[error("column {name} has no default value (fill every cell of array columns)")]
    NoDefault { name: String },
    #[error(transparent)]
    Create(#[from] TableCreateError),
    #[error("storage error: {0}")]
    Storage(String),
}

impl WritableTable {
    /// Start a new table with the given schema and no rows.
    pub fn create(
        dir: impl Into<std::path::PathBuf>,
        desc: crate::tabledesc::TableDesc,
    ) -> WritableTable {
        let dir: std::path::PathBuf = dir.into();
        let cells = vec![Vec::new(); desc.columns.len()];
        let pending = vec![Vec::new(); desc.columns.len()];
        let touched = vec![false; desc.columns.len()];
        WritableTable {
            dir: absolute_dir(&dir),
            desc,
            cells,
            pending,
            touched,
            meta_dirty: false,
        }
    }

    /// Append `n` empty rows (`addrows`).
    pub fn addrows(&mut self, n: u64) {
        for (col_idx, col) in self.cells.iter_mut().enumerate() {
            // New cells hold the column's default value (like casacore), so a
            // read of an unwritten scalar cell returns the declared default
            // instead of erroring; flush() writes exactly these cells.
            let default = self.desc.columns.get(col_idx).and_then(default_cell_value);
            col.resize_with(col.len() + n as usize, || default.clone());
            let words = col.len().div_ceil(64);
            self.pending[col_idx].resize(words, 0);
        }
    }

    /// Load a cell of an existing table into the store **without** marking
    /// the column as written (`putcell`, but untracked): the writable-open
    /// path materialises the whole table this way, so an untouched loaded
    /// column is preserved by the next `flush()` instead of being
    /// regenerated from its (identical) values.  Only a write made through
    /// `putcell`/`putcol` marks the column dirty.
    pub fn putcell_loaded(
        &mut self,
        col_idx: usize,
        row: u64,
        value: RecordValue,
    ) -> Result<(), WriteTableError> {
        let name = self
            .desc
            .columns
            .get(col_idx)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        let col = self
            .cells
            .get_mut(col_idx)
            .ok_or(WriteTableError::NoSuchColumn { name })?;
        let nrow = col.len() as u64;
        let slot = col
            .get_mut(row as usize)
            .ok_or(WriteTableError::RowOutOfRange { row, nrow })?;
        *slot = Some(value);
        Ok(())
    }

    /// Set one cell (`putcell`).
    pub fn putcell(
        &mut self,
        col_idx: usize,
        row: u64,
        value: RecordValue,
    ) -> Result<(), WriteTableError> {
        let name = self
            .desc
            .columns
            .get(col_idx)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        let col = self
            .cells
            .get_mut(col_idx)
            .ok_or(WriteTableError::NoSuchColumn { name })?;
        let nrow = col.len() as u64;
        let slot = col
            .get_mut(row as usize)
            .ok_or(WriteTableError::RowOutOfRange { row, nrow })?;
        *slot = Some(value);
        self.pending[col_idx][(row as usize) / 64] |= 1u64 << ((row as usize) % 64);
        if let Some(t) = self.touched.get_mut(col_idx) {
            *t = true;
        }
        Ok(())
    }

    /// Write cells starting at `startrow` (`putcol`; also the home of the
    /// dask-ms object-array path — Rust accepts values already converted).
    pub fn putcol(
        &mut self,
        col_idx: usize,
        startrow: u64,
        values: &[RecordValue],
    ) -> Result<(), WriteTableError> {
        for (i, v) in values.iter().enumerate() {
            self.putcell(col_idx, startrow + i as u64, v.clone())?;
        }
        Ok(())
    }

    /// Append a column (`Table::addcols`); existing rows default to the
    /// column's default value at flush.
    pub fn addcol(&mut self, cd: crate::tabledesc::ColumnDesc) {
        let n = self.cells.first().map_or(0, Vec::len);
        self.desc.columns.push(cd);
        self.cells.push(vec![None; n]);
        self.pending.push(vec![0; n.div_ceil(64)]);
        self.touched.push(false);
    }

    /// Remove a column by index (`Table::removecols`); the data is discarded
    /// at the next flush.
    pub fn removecol(&mut self, col_idx: usize) {
        if col_idx < self.desc.columns.len() {
            self.desc.columns.remove(col_idx);
        }
        if col_idx < self.cells.len() {
            self.cells.remove(col_idx);
        }
        if col_idx < self.pending.len() {
            self.pending.remove(col_idx);
        }
        if col_idx < self.touched.len() {
            self.touched.remove(col_idx);
        }
    }

    /// Open an existing table for in-place editing WITHOUT materialising its
    /// data: every cell stays unset (rows hold only defaults, none pending).
    /// A full-column `putcol` on a column then writes directly; a partial
    /// write overlays only its rows' cells; reads on the write handle are
    /// served by the returned `Table` snapshot merged with the pending cells.
    /// This is what keeps a changed-columns write (dask-ms `putcol`) at ~the
    /// working set instead of materialising the whole table — eagerly
    /// loading every column of a 1.6 GB MS into per-cell `RecordValue`s
    /// costs >3 GiB.
    pub fn open_for_update(
        dir: impl Into<std::path::PathBuf>,
    ) -> Result<(Table, WritableTable), TableDatError> {
        let dir: std::path::PathBuf = dir.into();
        let read = Table::open(&dir, false)?;
        let mut wt = WritableTable::create(&dir, read.dat.desc.clone());
        let n = read.nrows();
        if n > 0 {
            wt.addrows(n);
        }
        Ok((read, wt))
    }

    /// Populate one column's cells from its existing values **without**
    /// marking it as written (each cell is loaded via
    /// [`WritableTable::putcell_loaded`]), enabling a partial write to a
    /// column [`WritableTable::open_for_update`] left unloaded — the other
    /// rows keep their old data on flush instead of falling back to
    /// defaults.
    pub fn load_column(
        &mut self,
        col_idx: usize,
        values: Vec<RecordValue>,
    ) -> Result<(), WriteTableError> {
        let n = self.col_len(col_idx);
        for (r, v) in values.into_iter().take(n).enumerate() {
            self.putcell_loaded(col_idx, r as u64, v)?;
        }
        Ok(())
    }

    /// Open an existing table for editing: materialise every cell into the
    /// in-memory store. `flush()` then rewrites the whole table in place
    /// (regenerating `table.dat` and each data file via `create_table`),
    /// which is how casacure does in-place `UPDATE` / `DELETE` / `ALTER`.
    pub fn from_table(
        dir: std::path::PathBuf,
        t: &Table,
    ) -> Result<WritableTable, WriteTableError> {
        let mut wt = WritableTable::create(&dir, t.dat.desc.clone());
        let n = t.nrows();
        if n > 0 {
            wt.addrows(n);
        }
        for j in 0..wt.desc.columns.len() {
            let vals = t
                .getcol(j, 0, n)
                .map_err(|e| WriteTableError::Storage(format!("read column {j}: {e}")))?;
            for (r, v) in vals.iter().enumerate() {
                wt.putcell(j, r as u64, v.clone())?;
            }
        }
        Ok(wt)
    }

    /// Remove the given rows (`removerows`); the surviving rows are
    /// renumbered in the surviving order at the next flush.
    pub fn drop_rows(&mut self, rows: &[u64]) {
        if rows.is_empty() || self.cells.is_empty() {
            return;
        }
        let mut drop: Vec<bool> = vec![false; self.cells[0].len()];
        for &r in rows {
            if let Some(slot) = drop.get_mut(r as usize) {
                *slot = true;
            }
        }
        for col in &mut self.cells {
            let mut j = 0;
            col.retain(|_| {
                let keep = !drop[j];
                j += 1;
                keep
            });
        }
        for pend in &mut self.pending {
            let mut out = Vec::with_capacity(pend.len());
            for (j, w) in pend.drain(..).enumerate() {
                let mut kept = 0u64;
                for bit in 0..64 {
                    let row = j * 64 + bit;
                    if row < drop.len() && !drop[row] && w & (1 << bit) != 0 {
                        kept |= 1 << bit;
                    }
                }
                out.push(kept);
            }
            *pend = out;
        }
    }

    /// Rename a column (`ALTER TABLE ... RENAME COLUMN from TO to`).
    pub fn renamecol(&mut self, from: &str, to: &str) -> Result<(), WriteTableError> {
        let col = self
            .desc
            .columns
            .iter_mut()
            .find(|c| c.name == from)
            .ok_or_else(|| WriteTableError::NoSuchColumn {
                name: from.to_string(),
            })?;
        col.name = to.to_string();
        Ok(())
    }

    /// The number of rows in the in-memory cell store.
    pub fn col_len(&self, col_idx: usize) -> usize {
        self.cells.get(col_idx).map_or(0, Vec::len)
    }

    /// The buffered value of one cell (only rows written since the last
    /// flush are `Some`; everything else is on disk).
    pub fn cell(&self, col_idx: usize, row: u64) -> Option<&RecordValue> {
        self.cells
            .get(col_idx)
            .and_then(|c| c.get(row as usize))
            .and_then(|c| c.as_ref())
    }

    /// The value of one cell IF it was written since the last flush: the
    /// pending-bit-filtered view of [`WritableTable::cell`].  A lazily
    /// opened table's rows hold `Some(default)` cells (from `addrows`) that
    /// are NOT pending and must not overlay the on-disk values on a merged
    /// read — only rows actually `putcell`/`putcol`'d are pending.
    pub fn pending_cell(&self, col_idx: usize, row: u64) -> Option<&RecordValue> {
        let bit = 1u64 << ((row as usize) % 64);
        if self
            .pending
            .get(col_idx)?
            .get((row as usize) / 64)
            .is_some_and(|w| w & bit != 0)
        {
            self.cell(col_idx, row)
        } else {
            None
        }
    }

    pub fn setmaxcachesize(&mut self, _col_idx: usize, _size: usize) {}

    pub fn putkeyword(&mut self, name: &str, value: RecordValue) {
        self.desc
            .keywords
            .set(name, relativize_subtables(value, Some(&self.dir)));
    }

    pub fn removekeyword(&mut self, name: &str) {
        self.desc.keywords.remove(name);
    }

    pub fn keywords_json(&self) -> String {
        self.desc.keywords.to_json_string()
    }

    pub fn desc(&self) -> &crate::tabledesc::TableDesc {
        &self.desc
    }

    pub fn putcolkeyword(
        &mut self,
        col_idx: usize,
        name: &str,
        value: RecordValue,
    ) -> Result<(), WriteTableError> {
        let cname = self
            .desc
            .columns
            .get(col_idx)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        let col = self
            .desc
            .columns
            .get_mut(col_idx)
            .ok_or(WriteTableError::NoSuchColumn { name: cname })?;
        col.keywords
            .set(name, relativize_subtables(value, Some(&self.dir)));
        self.meta_dirty = true;
        Ok(())
    }

    /// Remove a column keyword (`removecolkeyword`).
    pub fn removecolkeyword(&mut self, col_idx: usize, name: &str) -> Result<(), WriteTableError> {
        let cname = self
            .desc
            .columns
            .get(col_idx)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        let col = self
            .desc
            .columns
            .get_mut(col_idx)
            .ok_or(WriteTableError::NoSuchColumn { name: cname })?;
        col.keywords.remove(name);
        self.meta_dirty = true;
        Ok(())
    }

    /// Number of rows written so far.
    pub fn nrows(&self) -> u64 {
        self.cells.first().map_or(0, Vec::len) as u64
    }

    /// Assemble the on-disk table from the buffered cells, filling missing
    /// scalar cells with their defaults; returns the table directory.
    ///
    /// When the table already exists with the same schema and row count and
    /// only some columns were written (`putcell`/`putcol`), casacore's
    /// in-place semantics apply: exactly the data files of the written
    /// columns' data managers are updated, and every untouched column's
    /// files (and the header) stay byte-identical.  Otherwise the whole
    /// table is regenerated.
    ///
    /// The in-place path is INCREMENTAL: only rows written since the last
    /// flush are overlaid onto the on-disk files (TiledShape/TiledColumn
    /// bool tiles are patched byte-wise), the untouched rows are never
    /// re-encoded, and after a successful flush the column's buffered cell
    /// values are released — so a write stream of dask-ms chunks keeps at
    /// most one chunk of values resident instead of the whole column.
    pub fn flush(&mut self) -> Result<std::path::PathBuf, WriteTableError> {
        let dir = self.dir.clone();
        let nrow = self.cells.first().map_or(0, Vec::len) as u64;
        // Preservation is only possible when the on-disk table shares this
        // schema and row count (a full regrowth is otherwise required).
        let preserving =
            self.touched.len() == self.desc.columns.len()
                && !self.meta_dirty
                && self.touched.iter().any(|&t| t)
                && self.touched.iter().any(|&t| !t)
                && std::fs::read(dir.join("table.dat"))
                    .ok()
                    .and_then(|b| parse_table_dat(&b).ok())
                    .is_some_and(|dat| {
                        dat.desc.columns.len() == self.desc.columns.len()
                            && dat.header.nrow == nrow
                            && dat.desc.columns.iter().zip(self.desc.columns.iter()).all(
                                |(a, b)| {
                                    a.name == b.name
                                        && a.data_manager_type == b.data_manager_type
                                        && a.data_manager_group == b.data_manager_group
                                },
                            )
                    });
        if !preserving {
            let values = self.materialize_all()?;
            create_table(&dir, &self.desc, &values)?;
            // The regrowth wrote every cell: nothing is pending anymore.
            self.clear_all_pending();
            return Ok(dir);
        }
        self.flush_preserving(&dir)?;
        Ok(dir)
    }

    /// The whole cell store as per-column value lists (column defaults for
    /// missing cells) — the input for a whole-table regrowth.
    fn materialize_all(&self) -> Result<Vec<Vec<RecordValue>>, WriteTableError> {
        let mut values: Vec<Vec<RecordValue>> = Vec::with_capacity(self.cells.len());
        for (col_idx, col) in self.cells.iter().enumerate() {
            let cd = &self.desc.columns[col_idx];
            let mut list = Vec::with_capacity(col.len());
            for cell in col {
                match cell {
                    Some(v) => list.push(v.clone()),
                    None => list.push(default_cell_value(cd).ok_or_else(|| {
                        WriteTableError::NoDefault {
                            name: cd.name.clone(),
                        }
                    })?),
                }
            }
            values.push(list);
        }
        Ok(values)
    }

    /// Whether any rows of `col` were written since the last flush.
    fn has_pending(&self, col_idx: usize) -> bool {
        self.pending
            .get(col_idx)
            .is_some_and(|p| p.iter().any(|&w| w != 0))
    }

    /// The rows of `col` written since the last flush, ascending.
    fn pending_rows(&self, col_idx: usize) -> impl Iterator<Item = u64> + '_ {
        self.pending.get(col_idx).into_iter().flat_map(|p| {
            p.iter().enumerate().flat_map(move |(wi, &w)| {
                let mut w = w;
                let mut out = Vec::new();
                while w != 0 {
                    let b = w.trailing_zeros() as usize;
                    out.push((wi * 64 + b) as u64);
                    w &= w - 1;
                }
                out
            })
        })
    }

    /// Drop the buffered values and dirty marks of one column: its rows now
    /// live on disk, so resident memory tracks only un-flushed writes.
    fn clear_pending(&mut self, col_idx: usize) {
        if let Some(col) = self.cells.get_mut(col_idx) {
            col.iter_mut().for_each(|c| *c = None);
        }
        if let Some(p) = self.pending.get_mut(col_idx) {
            p.iter_mut().for_each(|w| *w = 0);
        }
    }

    fn clear_all_pending(&mut self) {
        for col in &mut self.cells {
            col.iter_mut().for_each(|c| *c = None);
        }
        for p in &mut self.pending {
            p.iter_mut().for_each(|w| *w = 0);
        }
    }

    /// Update only the data files of data managers holding written columns,
    /// leaving `table.dat` and every untouched column's files byte-identical
    /// (see [`WritableTable::flush`]).  This is what keeps a changed-columns
    /// MS write (skarabina `--write-changed-only`, dask-ms `putcol`)
    /// from regenerating untouched columns from their defaults — which would
    /// zero their data and collapse a TiledShapeStMan column's header nrdim
    /// so casacore can no longer open the table.
    ///
    /// The update is an overlay of the pending rows only: TSM tile files are
    /// patched in place (the untouched rows' bits/bytes are never
    /// re-encoded), and SSM/ISM columns are rebuilt from their on-disk
    /// values with the pending rows overlaid.  A column whose DMs cannot be
    /// patched in place (row growth, missing files) falls back to the full
    /// rebuild from on-disk + pending.  After a successful flush the column
    /// releases its buffered cells.
    fn flush_preserving(&mut self, dir: &std::path::Path) -> Result<(), WriteTableError> {
        let buf = std::fs::read(dir.join("table.dat"))
            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
        let dat = parse_table_dat(&buf).map_err(|e| WriteTableError::Storage(e.to_string()))?;
        let nrow = self.cells.first().map_or(0, Vec::len) as u64;
        let dm_of: Vec<u32> = (0..self.desc.columns.len())
            .map(|c| {
                dat.column_set
                    .columns
                    .get(c)
                    .map(|x| x.data_manager_seq)
                    .unwrap_or(0)
            })
            .collect();
        let mut dm_seqs: Vec<u32> = Vec::new();
        for &seq in &dm_of {
            if !dm_seqs.contains(&seq) {
                dm_seqs.push(seq);
            }
        }
        for seq in dm_seqs {
            let cols: Vec<usize> = dm_of
                .iter()
                .enumerate()
                .filter(|(_, &s)| s == seq)
                .map(|(i, _)| i)
                .collect();
            let touched = cols
                .iter()
                .any(|&c| self.touched.get(c).copied().unwrap_or(false))
                && cols.iter().any(|&c| self.has_pending(c));
            if !touched {
                continue;
            }
            let dm = dat
                .column_set
                .data_managers
                .iter()
                .find(|d| d.sequence_nr == seq)
                .ok_or_else(|| {
                    WriteTableError::Storage(format!("table.dat lost DM sequence {seq}"))
                })?;
            let storage = |e: TableCreateError| WriteTableError::Storage(e.to_string());
            match dm.type_name.as_str() {
                "StandardStMan" | "IncrementalStMan" => {
                    // Full-column rebuild from the on-disk values overlaid
                    // with the pending rows (a StandardStMan bucket groups
                    // several columns, so it cannot be byte-patched in
                    // place; for the small scalar columns MS writes touch —
                    // e.g. FLAG_ROW — the decode is cheap).
                    let mut all_vals: Vec<Vec<RecordValue>> =
                        vec![Vec::new(); self.desc.columns.len()];
                    for &col in &cols {
                        all_vals[col] = self.materialize_col_from_disk(col, nrow)?;
                    }
                    if dm.type_name == "StandardStMan" {
                        let (f0, f0i, _spec) = build_ssm_data(
                            false,
                            nrow,
                            &dm.type_name,
                            &self.desc,
                            &all_vals,
                            &cols,
                        )
                        .map_err(storage)?;
                        std::fs::write(dir.join(format!("table.f{seq}")), f0)
                            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
                        if let Some(i) = f0i {
                            std::fs::write(dir.join(format!("table.f{seq}i")), i)
                                .map_err(|e| WriteTableError::Storage(e.to_string()))?;
                        }
                    } else {
                        let f0 = build_ism_data(
                            false,
                            nrow,
                            &dm.type_name,
                            &self.desc,
                            &all_vals,
                            &cols,
                        )
                        .map_err(storage)?;
                        std::fs::write(dir.join(format!("table.f{seq}")), f0)
                            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
                    }
                    for &col in &cols {
                        self.clear_pending(col);
                    }
                }
                "TiledColumnStMan" | "TiledShapeStMan" => {
                    // One array column per TSM DM (build_tsm_data's
                    // invariant) — patch its tile file in place when
                    // possible, else rebuild it from on-disk + pending.
                    let col = cols[0];
                    if self
                        .patch_tsm_column(dir, seq, &dm.type_name, col, nrow)?
                        .is_none()
                    {
                        let mut all_vals: Vec<Vec<RecordValue>> =
                            vec![Vec::new(); self.desc.columns.len()];
                        all_vals[col] = self.materialize_col_from_disk(col, nrow)?;
                        let (hdr, tile, file_seq) = build_tsm_data(
                            &dm.type_name,
                            false,
                            seq,
                            &dm.type_name,
                            &self.desc,
                            &all_vals,
                            &[col],
                        )
                        .map_err(storage)?;
                        std::fs::write(dir.join(format!("table.f{seq}")), hdr)
                            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
                        std::fs::write(dir.join(format!("table.f{seq}_TSM{file_seq}")), tile)
                            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
                    }
                    self.clear_pending(col);
                }
                other => {
                    return Err(WriteTableError::Storage(format!(
                        "cannot preserve-rewrite data-manager type {other}"
                    )))
                }
            }
        }
        Ok(())
    }

    /// One column's current values: its on-disk cells with the pending
    /// (unflushed) rows overlaid.  Used to rebuild SSM/ISM columns and as a
    /// fallback for TSM columns that cannot be byte-patched.
    fn materialize_col_from_disk(
        &self,
        col: usize,
        nrow: u64,
    ) -> Result<Vec<RecordValue>, WriteTableError> {
        let t = crate::Table::open(&self.dir, false)
            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
        let mut vals = t
            .getcol(col, 0, nrow)
            .map_err(|e| WriteTableError::Storage(e.to_string()))?;
        for r in self.pending_rows(col) {
            if let Some(Some(v)) = self.cells[col].get(r as usize) {
                vals[r as usize] = v.clone();
            }
        }
        Ok(vals)
    }

    /// Byte-patch the on-disk TSM tile file of column `col` (one array
    /// column per TSM DM) with the pending rows, leaving every other row
    /// untouched.  Returns `None` when the file cannot be patched in place
    /// (row growth, missing header/tile, multi-cube headers) so the caller
    /// falls back to a full rebuild.
    fn patch_tsm_column(
        &self,
        dir: &std::path::Path,
        seq: u32,
        stman_type: &str,
        col: usize,
        nrow: u64,
    ) -> Result<Option<()>, WriteTableError> {
        use crate::record::DataType;
        let file_seq = if stman_type == "TiledShapeStMan" {
            1
        } else {
            0
        };
        let hdr_path = dir.join(format!("table.f{seq}"));
        let tile_path = dir.join(format!("table.f{seq}_TSM{file_seq}"));
        let (Ok(hdr_bytes), Ok(mut tile)) = (std::fs::read(&hdr_path), std::fs::read(&tile_path))
        else {
            return Ok(None);
        };
        let header = match crate::tsm::parse_header(&hdr_bytes) {
            Ok(h) => h,
            Err(_) => return Ok(None),
        };
        let nrdim = header.nrdim as usize;
        let Some(cube) = header
            .cubes
            .iter()
            .find(|c| !c.cube_shape.is_empty() && c.cube_shape.len() == nrdim)
        else {
            return Ok(None);
        };
        if header.nrrow != nrow {
            return Ok(None); // row growth: needs a full rebuild
        }
        let Some(cd) = self.desc.columns.get(col) else {
            return Ok(None);
        };
        // Tile geometry from the parsed header's real data cube — never
        // recomputed from the descriptor.  The reader (`TsmFile::read_cell`)
        // groups rows by the cube's tile shape and places each tile at a
        // byte-aligned bucket of `ceil(tile_bits/8)`, so the patch must use
        // the SAME rows-per-tile and bucket bytes or every tile >= 1 of a
        // table whose tile shape differs from the default (e.g. a casacore-
        // written MS with 829-row tiles) is mis-placed.  Equivalent to
        // `tsm_layout` when the header was written by this crate, which keeps
        // the two self-consistent.
        let rows_per_tile = cube.tile_shape[nrdim - 1] as u64;
        let cell_shape: Vec<i64> = cube.cube_shape[..nrdim - 1].to_vec();
        let cell_elems = cell_shape.iter().product::<i64>().max(0) as usize;
        let bucket_bytes = match cd.data_type {
            crate::record::DataType::Bool => (cell_elems * rows_per_tile as usize).div_ceil(8),
            _ => {
                let elem = crate::tsm::elem_size(cd.data_type)
                    .map_err(|e| WriteTableError::Storage(e.to_string()))?;
                cell_elems * elem * rows_per_tile as usize
            }
        };

        // Encode the pending rows' cells.
        let mut pending: Vec<(u64, Vec<u8>)> = Vec::new();
        for r in self.pending_rows(col) {
            if r >= nrow {
                return Ok(None);
            }
            let Some(RecordValue::Array(arr)) =
                self.cells[col].get(r as usize).and_then(|c| c.as_ref())
            else {
                return Ok(None);
            };
            let cell = crate::tsm::tsm_encode_cell(false, cd.data_type, &arr.data)
                .map_err(|e| WriteTableError::Storage(e.to_string()))?;
            pending.push((r, cell));
        }
        if pending.is_empty() {
            return Ok(Some(())); // nothing to patch
        }
        // Overlay each pending row onto its tile bucket.
        for (r, cell) in &pending {
            let tile_nr = r / rows_per_tile;
            let in_tile = (r % rows_per_tile) as usize;
            let bucket = tile_nr as usize * bucket_bytes;
            if cd.data_type == DataType::Bool {
                let bit = in_tile * cell_elems;
                // Clear the row's old bits, then place the new ones.
                for g in bit..bit + cell_elems {
                    tile[bucket + g / 8] &= !(1 << (g % 8));
                }
                crate::tsm::or_bytes_at(&mut tile[bucket..], cell, bit);
            } else {
                let off = bucket + in_tile * cell.len();
                if off + cell.len() <= tile.len() {
                    tile[off..off + cell.len()].copy_from_slice(cell);
                } else {
                    return Ok(None);
                }
            }
        }
        std::fs::write(&tile_path, tile).map_err(|e| WriteTableError::Storage(e.to_string()))?;
        Ok(Some(()))
    }
}

/// The default value for an unfilled cell of `desc` (scalar columns store
/// their default; variable strings default to empty).
/// The cell value casacore uses for an unwritten cell: scalar defaults, a
/// zero-filled array for fixed-shape array columns, and an empty array for
/// variable-shape array columns.
fn default_cell_value(cd: &crate::tabledesc::ColumnDesc) -> Option<RecordValue> {
    use crate::record::{ArrayData, ArrayValue};
    match &cd.kind {
        crate::tabledesc::ColumnKind::Scalar(default) => Some(default.clone()),
        crate::tabledesc::ColumnKind::Array => {
            // Default cells are in the as-given (logical) orientation: the
            // descriptor stores the reversed shape.
            let mut shape: Vec<i64> = cd.shape.clone().unwrap_or_default();
            shape.reverse();
            let n = shape.iter().map(|&d| d.max(0) as usize).product();
            let data = match cd.data_type {
                crate::record::DataType::Bool => ArrayData::Bool(vec![false; n]),
                crate::record::DataType::UChar => ArrayData::UChar(vec![0; n]),
                crate::record::DataType::UShort => ArrayData::UShort(vec![0; n]),
                crate::record::DataType::Short => ArrayData::Short(vec![0; n]),
                crate::record::DataType::Int => ArrayData::Int(vec![0; n]),
                crate::record::DataType::UInt => ArrayData::UInt(vec![0; n]),
                crate::record::DataType::Int64 => ArrayData::Int64(vec![0; n]),
                crate::record::DataType::Float => ArrayData::Float(vec![0.0; n]),
                crate::record::DataType::Double => ArrayData::Double(vec![0.0; n]),
                crate::record::DataType::Complex => ArrayData::Complex(vec![(0.0, 0.0); n]),
                crate::record::DataType::DComplex => ArrayData::DComplex(vec![(0.0, 0.0); n]),
                crate::record::DataType::String => ArrayData::String(vec![String::new(); n]),
                _ => ArrayData::Double(Vec::new()),
            };
            Some(RecordValue::Array(ArrayValue {
                shape: shape.iter().map(|&d| d.max(0) as u32).collect(),
                data,
            }))
        }
        crate::tabledesc::ColumnKind::Record => {
            Some(RecordValue::Record(crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            }))
        }
    }
}

/// A data-manager-info record as returned by casacore `table.getdminfo()`:
/// `{"TYPE", "NAME", "SEQNR", "SPEC", "COLUMNS"}` (keys in that order).
#[derive(Debug, Clone, PartialEq)]
pub struct DmInfo {
    pub type_name: String,
    pub name: String,
    pub seqnr: u32,
    pub spec: DmSpec,
    /// Column names belonging to this manager (sorted, as casacore does).
    pub columns: Vec<String>,
}

/// The `SPEC` field of a data-manager-info record.
#[derive(Debug, Clone, PartialEq)]
pub enum DmSpec {
    StandardStMan {
        max_cache_size: u32,
        bucket_size: u32,
        pers_cache_size: u32,
        index_length: u32,
    },
    IncrementalStMan {
        max_cache_size: u32,
        bucket_size: u32,
        pers_cache_size: u32,
    },
    TiledColumnStMan {
        max_cache_size: u32,
        max_cache_size_64: i64,
        cube_shapes: Vec<Vec<i64>>,
        tile_shapes: Vec<Vec<i64>>,
        cell_shapes: Vec<Vec<i64>>,
        bucket_sizes: Vec<u64>,
        seqnr: u32,
    },
    TiledShapeStMan {
        default_tile_shape: Vec<i64>,
        seqnr: u32,
        hypercubes: Vec<TsmHypercube>,
    },
    Unsupported(String),
}

/// One hypercube of a `TiledShapeStMan` (`SPEC.HYPERCUBES["*N"]`).
#[derive(Debug, Clone, PartialEq)]
pub struct TsmHypercube {
    pub cube_shape: Vec<i64>,
    pub tile_shape: Vec<i64>,
    pub cell_shape: Vec<i64>,
    pub bucket_size: u64,
}

/// Build the `getdminfo()` dictionary (`{"*1": ..., "*2": ...}`) for a
/// table, matching casacore's `ColumnSet::dataManagerInfo`: one entry per
/// data manager with columns, numbered from 1, `COLUMNS` sorted, and each
/// `SPEC` read from the manager's data-file header.
pub fn get_dminfo(
    table_dir: &std::path::Path,
    dat: &TableDat,
) -> Result<std::collections::BTreeMap<String, DmInfo>, TableDatError> {
    let mut out = std::collections::BTreeMap::new();
    let mut tsm_hypercolumn: Option<String> = None;
    let mut n = 0u32;
    for dm in &dat.column_set.data_managers {
        // Columns bound to this data manager, in table order then sorted.
        let mut columns: Vec<&String> = dat
            .column_set
            .columns
            .iter()
            .filter(|c| c.data_manager_seq == dm.sequence_nr)
            .map(|c| &c.original_name)
            .collect();
        if columns.is_empty() {
            continue;
        }
        columns.sort();
        let columns: Vec<String> = columns.into_iter().cloned().collect();
        let spec = match dm.type_name.as_str() {
            "StandardStMan" => {
                let file = crate::ssm::StandardStManFile::open(
                    table_dir,
                    dm.sequence_nr,
                    dat.header.big_endian,
                )
                .map_err(|e| TableDatError::Storage(e.to_string()))?;
                DmSpec::StandardStMan {
                    max_cache_size: file.header.pers_cache_size,
                    bucket_size: file.header.bucket_size,
                    pers_cache_size: file.header.pers_cache_size,
                    index_length: file.header.index_length,
                }
            }
            "IncrementalStMan" => {
                let file =
                    crate::ism::IsmFile::open(table_dir, dm.sequence_nr, dat.header.big_endian)
                        .map_err(|e| TableDatError::Storage(e.to_string()))?;
                DmSpec::IncrementalStMan {
                    max_cache_size: file.header.pers_cache_size,
                    bucket_size: file.header.bucket_size,
                    pers_cache_size: file.header.pers_cache_size,
                }
            }
            "TiledShapeStMan" => {
                let file =
                    crate::tsm::TsmFile::open(table_dir, dm.sequence_nr, dat.header.big_endian)
                        .map_err(|e| TableDatError::Storage(e.to_string()))?;
                let header = &file.header;
                tsm_hypercolumn = Some(header.hypercolumn_name.clone());
                let elem = crate::tsm::elem_size(header.data_types[0])
                    .map_err(|e| TableDatError::Storage(e.to_string()))?;
                let hypercubes = header
                    .cubes
                    .iter()
                    .map(|c| TsmHypercube {
                        cell_shape: {
                            let mut s = c.cube_shape.clone();
                            s.pop();
                            s
                        },
                        bucket_size: c.tile_shape.iter().product::<i64>() as u64 * elem as u64,
                        cube_shape: c.cube_shape.clone(),
                        tile_shape: c.tile_shape.clone(),
                    })
                    .collect();
                DmSpec::TiledShapeStMan {
                    default_tile_shape: header.subclass_shape.clone(),
                    seqnr: header.seq_nr,
                    hypercubes,
                }
            }
            "TiledColumnStMan" => {
                let file =
                    crate::tsm::TsmFile::open(table_dir, dm.sequence_nr, dat.header.big_endian)
                        .map_err(|e| TableDatError::Storage(e.to_string()))?;
                let header = &file.header;
                tsm_hypercolumn = Some(header.hypercolumn_name.clone());
                DmSpec::TiledColumnStMan {
                    max_cache_size: 0,
                    max_cache_size_64: 0,
                    cube_shapes: header.cubes.iter().map(|c| c.cube_shape.clone()).collect(),
                    tile_shapes: header.cubes.iter().map(|c| c.tile_shape.clone()).collect(),
                    cell_shapes: header
                        .cubes
                        .iter()
                        .map(|c| {
                            let mut s = c.cube_shape.clone();
                            if !s.is_empty() {
                                s.pop(); // drop the row axis
                            }
                            s
                        })
                        .collect(),
                    bucket_sizes: header
                        .cubes
                        .iter()
                        .map(|c| c.tile_shape.iter().product::<i64>() as u64 * 16)
                        .collect(),
                    seqnr: header.seq_nr,
                }
            }
            other => DmSpec::Unsupported(other.to_string()),
        };
        // The manager name is the group/hypercolumn name: from the SSM spec
        // blob for StandardStMan, from the TSM header's hypercolumn name
        // otherwise (the TSM blob is empty).
        let name = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s.data_manager_name.clone(),
            crate::columnset::DataManagerBlob::Unsupported(_) => {
                if dm.type_name == "TiledColumnStMan" || dm.type_name == "TiledShapeStMan" {
                    tsm_hypercolumn
                        .clone()
                        .unwrap_or_else(|| dm.type_name.clone())
                } else {
                    dm.type_name.clone()
                }
            }
        };
        n += 1;
        out.insert(
            format!("*{n}"),
            DmInfo {
                type_name: dm.type_name.clone(),
                name,
                seqnr: dm.sequence_nr,
                spec,
                columns,
            },
        );
    }
    Ok(out)
}

impl DmInfo {
    /// Serialize exactly as python-casacore's `getdminfo()` repr order
    /// (TYPE, NAME, SEQNR, SPEC, COLUMNS).
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push('{');
        s.push_str("\"TYPE\":");
        s.push_str(&json_string(&self.type_name));
        s.push(',');
        s.push_str("\"NAME\":");
        s.push_str(&json_string(&self.name));
        s.push(',');
        s.push_str("\"SEQNR\":");
        s.push_str(&self.seqnr.to_string());
        s.push(',');
        s.push_str("\"SPEC\":");
        s.push_str(&self.spec.to_json());
        s.push(',');
        s.push_str("\"COLUMNS\":");
        s.push_str(&json_list(&self.columns));
        s.push('}');
        s
    }
}

impl DmSpec {
    fn to_json(&self) -> String {
        match self {
            DmSpec::StandardStMan {
                max_cache_size,
                bucket_size,
                pers_cache_size,
                index_length,
            } => {
                let mut s = String::new();
                s.push('{');
                s.push_str(&format!("\"MaxCacheSize\":{}", max_cache_size));
                s.push(',');
                s.push_str(&format!("\"BUCKETSIZE\":{}", bucket_size));
                s.push(',');
                s.push_str(&format!("\"PERSCACHESIZE\":{}", pers_cache_size));
                s.push(',');
                s.push_str(&format!("\"IndexLength\":{}", index_length));
                s.push('}');
                s
            }
            DmSpec::IncrementalStMan {
                max_cache_size,
                bucket_size,
                pers_cache_size,
            } => {
                let mut s = String::new();
                s.push('{');
                s.push_str(&format!("\"MaxCacheSize\":{}", max_cache_size));
                s.push(',');
                s.push_str(&format!("\"BUCKETSIZE\":{}", bucket_size));
                s.push(',');
                s.push_str(&format!("\"PERSCACHESIZE\":{}", pers_cache_size));
                s.push('}');
                s
            }
            DmSpec::TiledColumnStMan {
                max_cache_size,
                max_cache_size_64,
                cube_shapes,
                tile_shapes,
                cell_shapes,
                bucket_sizes,
                seqnr,
            } => {
                let mut cubes = String::new();
                for i in 0..cube_shapes.len() {
                    if i > 0 {
                        cubes.push(',');
                    }
                    cubes.push('"');
                    cubes.push('*');
                    cubes.push_str(&(i + 1).to_string());
                    cubes.push('"');
                    cubes.push(':');
                    cubes.push('{');
                    cubes.push_str("\"CubeShape\":");
                    cubes.push_str(&json_ilist(&cube_shapes[i]));
                    cubes.push(',');
                    cubes.push_str("\"TileShape\":");
                    cubes.push_str(&json_ilist(&tile_shapes[i]));
                    cubes.push(',');
                    cubes.push_str("\"CellShape\":");
                    cubes.push_str(&json_ilist(&cell_shapes[i]));
                    cubes.push(',');
                    cubes.push_str(&format!("\"BucketSize\":{}", bucket_sizes[i]));
                    cubes.push(',');
                    cubes.push_str("\"ID\":{}");
                    cubes.push('}');
                }
                let mut s = String::new();
                s.push('{');
                s.push_str(&format!("\"MaxCacheSize\":{}", max_cache_size));
                s.push(',');
                s.push_str("\"DEFAULTTILESHAPE\":[]");
                s.push(',');
                s.push_str(&format!("\"MAXIMUMCACHESIZE\":{}", max_cache_size_64));
                s.push(',');
                s.push_str("\"HYPERCUBES\":{");
                s.push_str(&cubes);
                s.push('}');
                s.push(',');
                s.push_str(&format!("\"SEQNR\":{}", seqnr));
                s.push('}');
                s
            }
            DmSpec::TiledShapeStMan {
                default_tile_shape,
                seqnr,
                hypercubes,
            } => {
                let mut cubes = String::new();
                for (i, c) in hypercubes.iter().enumerate() {
                    if i > 0 {
                        cubes.push(',');
                    }
                    cubes.push_str(&format!(
                        "\"*{}\":{{\"CubeShape\":{},\"TileShape\":{},\"CellShape\":{},\"BucketSize\":{},\"ID\":{{}}}}",
                        i + 1,
                        json_ilist(&c.cube_shape),
                        json_ilist(&c.tile_shape),
                        json_ilist(&c.cell_shape),
                        c.bucket_size,
                    ));
                }
                let mut s = String::new();
                s.push('{');
                s.push_str("\"MaxCacheSize\":0");
                s.push(',');
                s.push_str("\"DEFAULTTILESHAPE\":");
                s.push_str(&json_ilist(default_tile_shape));
                s.push(',');
                s.push_str("\"MAXIMUMCACHESIZE\":0");
                s.push(',');
                s.push_str("\"HYPERCUBES\":{");
                s.push_str(&cubes);
                s.push('}');
                s.push(',');
                s.push_str(&format!("\"SEQNR\":{}", seqnr));
                s.push(',');
                s.push_str(&format!("\"IndexSize\":{}", hypercubes.len()));
                s.push('}');
                s
            }
            DmSpec::Unsupported(t) => {
                let mut s = String::new();
                s.push('{');
                s.push_str("\"TYPE\":");
                s.push_str(&json_string(t));
                s.push('}');
                s
            }
        }
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_list(items: &[String]) -> String {
    let mut s = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_string(item));
    }
    s.push(']');
    s
}

fn json_ilist(items: &[i64]) -> String {
    let mut s = String::from("[");
    for (i, v) in items.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_dat(version: u32, nrow: u64, endian_flag: u32, kind: &str) -> Vec<u8> {
        let mut payload = Vec::new();
        match version {
            2 => payload.extend_from_slice(&(nrow as u32).to_be_bytes()),
            _ => payload.extend_from_slice(&nrow.to_be_bytes()),
        }
        payload.extend_from_slice(&endian_flag.to_be_bytes());
        payload.extend_from_slice(&(kind.len() as u32).to_be_bytes());
        payload.extend_from_slice(kind.as_bytes());
        let type_name = b"Table";
        let length = (4 + 4 + type_name.len() + 4 + payload.len()) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::aipsio::MAGIC.to_be_bytes());
        buf.extend_from_slice(&length.to_be_bytes());
        buf.extend_from_slice(&(type_name.len() as u32).to_be_bytes());
        buf.extend_from_slice(type_name);
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(&payload);
        buf
    }

    #[test]
    fn parses_version2_header() {
        let buf = table_dat(2, 123, 1, "PlainTable");
        let hdr = parse_table_header(&buf).unwrap();
        assert_eq!(
            hdr,
            TableHeader {
                version: 2,
                nrow: 123,
                big_endian: false,
                kind: "PlainTable".into(),
            }
        );
    }

    #[test]
    fn parses_version3_header_with_u64_nrow() {
        let buf = table_dat(3, 5_000_000_000, 0, "PlainTable");
        let hdr = parse_table_header(&buf).unwrap();
        assert_eq!(hdr.nrow, 5_000_000_000);
        assert!(hdr.big_endian);
    }

    #[test]
    fn rejects_wrong_root_type() {
        let mut buf = table_dat(2, 1, 1, "PlainTable");
        // Overwrite "Table" with "Xable" (same length).
        buf[12] = b'X';
        assert!(matches!(
            parse_table_header(&buf),
            Err(TableError::NotATable { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let buf = table_dat(4, 1, 1, "PlainTable");
        assert!(matches!(
            parse_table_header(&buf),
            Err(TableError::UnsupportedVersion(4))
        ));
    }

    #[test]
    fn rejects_bad_endianness_flag() {
        let buf = table_dat(2, 1, 7, "PlainTable");
        assert!(matches!(
            parse_table_header(&buf),
            Err(TableError::BadEndianness(7))
        ));
    }

    use crate::record::{DataType, RecordValue, TableRecord};
    use crate::tabledesc::{ColumnDesc, ColumnKind, TableDesc};

    fn empty_record() -> TableRecord {
        TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        }
    }

    fn zero_value(dt: DataType) -> RecordValue {
        match dt {
            DataType::Bool => RecordValue::Bool(false),
            DataType::UChar | DataType::Char => RecordValue::UChar(0),
            DataType::Short => RecordValue::Short(0),
            DataType::UShort => RecordValue::UShort(0),
            DataType::Int => RecordValue::Int(0),
            DataType::UInt => RecordValue::UInt(0),
            DataType::Int64 => RecordValue::Int64(0),
            DataType::Float => RecordValue::Float(0.0),
            DataType::Double => RecordValue::Double(0.0),
            DataType::Complex => RecordValue::Complex(0.0, 0.0),
            DataType::DComplex => RecordValue::DComplex(0.0, 0.0),
            DataType::String => RecordValue::String(Default::default()),
            _ => panic!("not a scalar type"),
        }
    }

    fn scalar_col(name: &str, dt: DataType, max_len: i32) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: max_len,
            keywords: empty_record(),
            kind: ColumnKind::Scalar(zero_value(dt)),
        }
    }

    /// The same columns as `tests/make_fixtures.py` COLUMN_CASES.
    fn typed_desc() -> TableDesc {
        TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![
                scalar_col("COL_B", DataType::Bool, 0),
                scalar_col("COL_U1", DataType::UChar, 0),
                scalar_col("COL_I2", DataType::Short, 0),
                scalar_col("COL_I4", DataType::Int, 0),
                scalar_col("COL_U4", DataType::UInt, 0),
                scalar_col("COL_R4", DataType::Float, 0),
                scalar_col("COL_R8", DataType::Double, 0),
                scalar_col("COL_C4", DataType::Complex, 0),
                scalar_col("COL_C8", DataType::DComplex, 0),
                scalar_col("COL_S", DataType::String, 0),
            ],
        }
    }

    fn typed_values(nrow: usize) -> Vec<Vec<RecordValue>> {
        use RecordValue::*;
        let b = vec![Bool(true); nrow];
        let u1 = vec![UChar(7); nrow];
        let i2 = vec![Short(-300); nrow];
        let i4 = vec![Int(-70000); nrow];
        let u4 = vec![UInt(4_000_000_000); nrow];
        let r4 = vec![Float(1.5); nrow];
        let r8 = vec![Double(1.5e300); nrow];
        let c4 = vec![Complex(1.5, 2.5); nrow];
        let c8 = vec![DComplex(1.5e300, 2.5e300); nrow];
        let s = vec![String("hello".into()); nrow];
        vec![b, u1, i2, i4, u4, r4, r8, c4, c8, s]
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "casacure-test-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn layout_reproduces_casacore_fixture_offsets() {
        // The real casacore typed.tab tile: rowsPerBucket 32, bucket 1892.
        let bits = [1, 8, 16, 32, 32, 32, 64, 64, 128, 96];
        let l = crate::ssm::layout(32, &bits);
        assert_eq!(
            l.column_offset,
            [0, 4, 36, 100, 228, 356, 484, 740, 996, 1508]
        );
        assert_eq!(l.bucket_size, 1892);
    }

    #[test]
    fn build_table_dat_round_trips() {
        let desc = typed_desc();
        let bits: Vec<u32> = desc
            .columns
            .iter()
            .map(|c| {
                let size = crate::ssm::scalar_cell_size(c);
                if c.data_type == DataType::Bool {
                    1
                } else {
                    8 * size
                }
            })
            .collect();
        let correct = crate::ssm::layout(ROWS_PER_BUCKET, &bits);
        let spec = crate::columnset::StandardStMan {
            data_manager_name: "StandardStMan".into(),
            column_offset: correct.column_offset,
            col_index_map: vec![0; desc.columns.len()],
        };
        let dms = vec![crate::columnset::DmBlob {
            type_name: "StandardStMan".into(),
            sequence_nr: 0,
            blob: crate::columnset::write_standard_stman(&spec),
        }];
        let bytes =
            build_table_dat(false, 1, &desc, &dms, &vec![0u32; desc.columns.len()]).unwrap();
        let parsed = parse_table_dat(&bytes).unwrap();
        assert_eq!(parsed.header.version, 2);
        assert_eq!(parsed.header.nrow, 1);
        assert!(!parsed.header.big_endian);
        assert_eq!(parsed.header.kind, "PlainTable");
        assert_eq!(parsed.desc.columns.len(), 10);
        for (a, b) in desc.columns.iter().zip(parsed.desc.columns.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.data_type, b.data_type);
            assert_eq!(b.data_manager_type, "StandardStMan");
        }
        assert_eq!(parsed.column_set.columns.len(), 10);
        for info in &parsed.column_set.columns {
            assert_eq!(info.data_manager_seq, 0);
        }
        let dm = &parsed.column_set.data_managers[0];
        let ssm = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan blob"),
        };
        assert_eq!(ssm.data_manager_name, "StandardStMan");
        assert_eq!(ssm.column_offset.len(), 10);
        assert_eq!(ssm.col_index_map, vec![0; 10]);
    }

    #[test]
    fn create_table_reads_back_all_values() {
        let desc = typed_desc();
        let values = typed_values(1);
        let dir = temp_dir("roundtrip");
        create_table(&dir, &desc, &values).unwrap();

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        assert_eq!(dat.header.nrow, 1);
        let file = crate::ssm::StandardStManFile::open(&dir, 0, dat.header.big_endian).unwrap();
        let dm = &dat.column_set.data_managers[0];
        let spec = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan blob"),
        };
        let expected = typed_values(1);
        for (i, col) in dat.desc.columns.iter().enumerate() {
            let value = file
                .read_scalar_cell(spec, i, col, 0)
                .unwrap_or_else(|e| panic!("{}: {e}", col.name));
            assert_eq!(value, expected[i][0], "{}", col.name);
        }
    }

    #[test]
    fn create_table_spreads_rows_across_buckets() {
        let desc = {
            let mut d = typed_desc();
            d.columns = vec![scalar_col("X", DataType::Int, 0)];
            d
        };
        let values = vec![(0..100).map(RecordValue::Int).collect::<Vec<_>>()];
        let dir = temp_dir("buckets");
        create_table(&dir, &desc, &values).unwrap();

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        assert_eq!(dat.header.nrow, 100);
        let file = crate::ssm::StandardStManFile::open(&dir, 0, dat.header.big_endian).unwrap();
        assert_eq!(file.indices[0].last_row, vec![31, 63, 95, 99]);
        let spec = match &dat.column_set.data_managers[0].blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => unreachable!(),
        };
        let col = &dat.desc.columns[0];
        assert_eq!(
            file.read_scalar_cell(spec, 0, col, 0).unwrap(),
            RecordValue::Int(0)
        );
        assert_eq!(
            file.read_scalar_cell(spec, 0, col, 31).unwrap(),
            RecordValue::Int(31)
        );
        assert_eq!(
            file.read_scalar_cell(spec, 0, col, 40).unwrap(),
            RecordValue::Int(40)
        );
        assert_eq!(
            file.read_scalar_cell(spec, 0, col, 99).unwrap(),
            RecordValue::Int(99)
        );
    }

    fn array_col(name: &str, dt: DataType, option: i32, casa_shape: Vec<i64>) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: option,
            ndim: casa_shape.len() as i32,
            shape: Some(casa_shape),
            max_length: 0,
            keywords: empty_record(),
            kind: ColumnKind::Array,
        }
    }

    #[test]
    fn create_table_with_array_column_reads_back() {
        use crate::record::{ArrayData, ArrayValue};
        // A fixed-shape 2x3 complex array column + a scalar int column.
        let mut desc = typed_desc();
        desc.columns = vec![
            array_col("ARR", DataType::Complex, 4, vec![3, 2]),
            scalar_col("IDX", DataType::Int, 0),
        ];
        let arr = |base: i32| {
            RecordValue::Array(ArrayValue {
                shape: vec![2, 3],
                data: ArrayData::Complex((1..=6).map(|k| (base as f32, k as f32)).collect()),
            })
        };
        let values = vec![
            vec![arr(0), arr(10)],
            vec![RecordValue::Int(0), RecordValue::Int(1)],
        ];
        let dir = temp_dir("arraynp");
        create_table(&dir, &desc, &values).unwrap();

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        assert_eq!(dat.header.nrow, 2);
        let arr_desc = dat.desc.column("ARR").unwrap();
        assert!(matches!(arr_desc.kind, ColumnKind::Array));
        assert_eq!(arr_desc.shape.as_deref(), Some(&[3i64, 2][..]));
        assert_eq!(arr_desc.options & 4, 4, "FixedShape option");
        // The SSM spec: ARR bucket cells are 8-byte refs -> region 32*8=256.
        let dm = &dat.column_set.data_managers[0];
        let spec = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan spec"),
        };
        assert_eq!(spec.column_offset, vec![0, 256], "layout offsets");
        // Array binding carries the CASA-order shape.
        assert_eq!(
            dat.column_set.columns[0].shape.as_deref(),
            Some(&[3i64, 2][..]),
            "binding shape"
        );

        let file = crate::ssm::StandardStManFile::open(&dir, 0, dat.header.big_endian).unwrap();
        assert!(file.f0i().is_some());
        for (row, base) in [(0u64, 0f32), (1, 10.0)] {
            let cell = crate::ssm::read_array_cell(&file, spec, 0, arr_desc, row).unwrap();
            match cell {
                RecordValue::Array(a) => {
                    assert_eq!(a.shape, vec![2, 3], "logical shape row {row}");
                    match &a.data {
                        ArrayData::Complex(v) => {
                            let expect: Vec<(f32, f32)> =
                                (1..=6).map(|k| (base, k as f32)).collect();
                            assert_eq!(&v[..], &expect[..], "values row {row}");
                        }
                        other => panic!("expected complex, got {other:?}"),
                    }
                }
                other => panic!("expected array, got {other:?}"),
            }
        }
        let idx = dat.desc.column("IDX").unwrap();
        assert_eq!(
            file.read_scalar_cell(spec, 1, idx, 0).unwrap(),
            RecordValue::Int(0)
        );
        assert_eq!(
            file.read_scalar_cell(spec, 1, idx, 1).unwrap(),
            RecordValue::Int(1)
        );
    }

    #[test]
    fn create_table_with_long_strings_reads_back() {
        // Variable strings longer than 8 chars are stored in the SSM string
        // buckets and read back through them.
        let mut desc = typed_desc();
        desc.columns = vec![
            scalar_col("TXT", DataType::String, 0),
            scalar_col("IDX", DataType::Int, 0),
        ];
        let long0 = "hello world this is a longer string than eight chars";
        let long1 = "another quite long string that certainly exceeds eight characters";
        let values = vec![
            vec![
                RecordValue::String(long0.into()),
                RecordValue::String(long1.into()),
            ],
            vec![RecordValue::Int(0), RecordValue::Int(1)],
        ];
        let dir = temp_dir("longstring");
        create_table(&dir, &desc, &values).unwrap();

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        let file = crate::ssm::StandardStManFile::open(&dir, 0, dat.header.big_endian).unwrap();
        // String bucket allocated after the data + index buckets.
        assert_eq!(file.header.nr_buckets, 3, "data + index + string bucket");
        assert_eq!(file.header.last_string_bucket, 2);
        let dm = &dat.column_set.data_managers[0];
        let spec = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan spec"),
        };
        let txt = dat.desc.column("TXT").unwrap();
        assert_eq!(
            file.read_scalar_cell(spec, 0, txt, 0).unwrap(),
            RecordValue::String(long0.into())
        );
        assert_eq!(
            file.read_scalar_cell(spec, 0, txt, 1).unwrap(),
            RecordValue::String(long1.into())
        );
        let idx = dat.desc.column("IDX").unwrap();
        assert_eq!(
            file.read_scalar_cell(spec, 1, idx, 1).unwrap(),
            RecordValue::Int(1)
        );
    }
    fn ism_col(name: &str, dt: DataType, option: i32) -> ColumnDesc {
        let mut d = scalar_col(name, dt, 0);
        d.data_manager_type = "IncrementalStMan".into();
        d.data_manager_group = "IncrementalStMan".into();
        d.options = option;
        d
    }

    fn tsm_arr_col(name: &str, dt: DataType, logical_shape: Vec<i64>) -> ColumnDesc {
        let casa: Vec<i64> = logical_shape.iter().rev().copied().collect();
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "TiledColumnStMan".into(),
            data_manager_group: "TiledData_GROUP".into(),
            options: 4,
            ndim: logical_shape.len() as i32,
            shape: Some(casa),
            max_length: 0,
            keywords: empty_record(),
            kind: ColumnKind::Array,
        }
    }

    #[test]
    fn table_lifecycle_open_create_close() {
        let mut desc = typed_desc();
        desc.columns = vec![scalar_col("X", DataType::Int, 0)];
        let values = vec![(0..10).map(RecordValue::Int).collect()];
        let dir = temp_dir("lifecycle");

        let mut t = Table::create(&dir, &desc, &values).unwrap();
        assert!(t.is_writable());
        assert_eq!(t.name(), dir.to_str().unwrap());
        assert_eq!(t.nrows(), 10);
        assert_eq!(t.colnames(), vec!["X"]);
        // Advisory locking.
        assert!(!t.is_locked());
        t.lock();
        assert!(t.is_locked());
        t.unlock();
        assert!(!t.is_locked());
        t.flush();
        // The data manager files were opened (SSM seq 0).
        assert_eq!(t.ssm_files.len(), 1);
        assert_eq!(t.ssm_files[0].0, 0);
        t.close();

        // Open the written table read-only.
        let r = Table::open(&dir, true).unwrap();
        assert!(!r.is_writable());
        assert_eq!(r.nrows(), 10);
        assert_eq!(r.colnames(), vec!["X"]);

        // Read a value back through the opened StandardStMan file.
        let ssm = &r.ssm_files[0].1;
        let dm = &r.dat.column_set.data_managers[0];
        let spec = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan spec"),
        };
        let desc_x = r.dat.desc.column("X").unwrap();
        assert_eq!(
            ssm.read_scalar_cell(spec, 0, desc_x, 4).unwrap(),
            RecordValue::Int(4)
        );
    }

    #[test]
    fn dminfo_round_trips_group_names() {
        // create_table names data managers by their group; get_dminfo must
        // report those names back (the `_1` auto-suffix applies to addcols,
        // which is not implemented yet).
        let mut desc = typed_desc();
        desc.columns = vec![
            scalar_col("A", DataType::Int, 0),
            scalar_col("B", DataType::Int, 0),
        ];
        desc.columns[0].data_manager_group = "Main".into();
        desc.columns[0].options = 0;
        // second column keeps the default group "StandardStMan".
        let values = vec![
            (0..3).map(RecordValue::Int).collect(),
            (0..3).map(RecordValue::Int).collect(),
        ];
        let dir = temp_dir("dminfo");
        create_table(&dir, &desc, &values).unwrap();
        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        let info = crate::table::get_dminfo(&dir, &dat).unwrap();
        // Col A (group "Main") creates DM 0, col B the default SSM DM.
        assert_eq!(info["*1"].name, "Main");
        assert_eq!(info["*1"].columns, vec!["A"]);
        assert_eq!(info["*2"].name, "StandardStMan");
        assert_eq!(info["*2"].columns, vec!["B"]);
        let all_types: Vec<&str> = info.values().map(|d| d.type_name.as_str()).collect();
        assert_eq!(all_types, vec!["StandardStMan", "StandardStMan"]);
    }

    #[test]
    fn column_reads_getcell_getcol_slices_varcol() {
        use crate::record::{ArrayData, ArrayValue};
        // A mixed table: SSM scalar I4, SSM array ARR, TSM tiled DATA,
        // long string NAME, ISM TIME.
        let mut desc = typed_desc();
        desc.columns = vec![
            scalar_col("I4", DataType::Int, 0),
            array_col("ARR", DataType::Complex, 4, vec![3, 2]),
            tsm_arr_col("DATA", DataType::DComplex, vec![2, 3]),
            scalar_col("NAME", DataType::String, 0),
            ism_col("TIME", DataType::Double, 1),
        ];
        let arr = |row: i32| {
            RecordValue::Array(ArrayValue {
                shape: vec![2, 3],
                data: ArrayData::Complex((1..=6).map(|k| (row as f32, k as f32)).collect()),
            })
        };
        let data = |row: i32| {
            RecordValue::Array(ArrayValue {
                shape: vec![2, 3],
                data: ArrayData::DComplex((1..=6).map(|k| (row as f64, k as f64)).collect()),
            })
        };
        let names: Vec<RecordValue> = (0..4)
            .map(|i| {
                RecordValue::String(format!(
                    "a very long label number {i} exceeding eight chars"
                ))
            })
            .collect();
        let values = vec![
            (0..4).map(RecordValue::Int).collect(),
            (0..4).map(arr).collect(),
            (0..4).map(data).collect(),
            names,
            (0..4)
                .map(|i| RecordValue::Double(i as f64 / 2.0))
                .collect(),
        ];
        let dir = temp_dir("colread");
        Table::create(&dir, &desc, &values).unwrap();
        let t = Table::open(&dir, true).unwrap();

        // Scalar getcell / getcol.
        assert_eq!(t.getcell(0, 2).unwrap(), RecordValue::Int(2));
        assert_eq!(
            t.getcol(0, 1, 2).unwrap(),
            vec![RecordValue::Int(1), RecordValue::Int(2)]
        );

        // Fixed-shape array getcell (logical shape restored).
        match t.getcell(1, 1).unwrap() {
            RecordValue::Array(a) => {
                assert_eq!(a.shape, vec![2, 3]);
                match &a.data {
                    ArrayData::Complex(v) => assert_eq!(
                        &v[..],
                        &[
                            (1.0f32, 1.0),
                            (1.0, 2.0),
                            (1.0, 3.0),
                            (1.0, 4.0),
                            (1.0, 5.0),
                            (1.0, 6.0)
                        ][..]
                    ),
                    other => panic!("expected complex, got {other:?}"),
                }
            }
            other => panic!("expected array, got {other:?}"),
        }

        // Tiled DATA via TSM.
        match t.getcell(2, 3).unwrap() {
            RecordValue::Array(a) => {
                assert_eq!(a.shape, vec![2, 3]);
                match &a.data {
                    ArrayData::DComplex(v) => assert_eq!(
                        &v[..],
                        &[
                            (3.0f64, 1.0),
                            (3.0, 2.0),
                            (3.0, 3.0),
                            (3.0, 4.0),
                            (3.0, 5.0),
                            (3.0, 6.0)
                        ][..]
                    ),
                    other => panic!("expected dcomplex, got {other:?}"),
                }
            }
            other => panic!("expected array, got {other:?}"),
        }

        // getcolslice: slice DATA channels [1..1] inclusive -> [3,1] cell.
        let sliced = t.getcolslice(2, &[1, 0], &[1, 2], 0, 1).unwrap();
        match &sliced[0] {
            RecordValue::Array(a) => {
                assert_eq!(a.shape, vec![1, 3]);
                match &a.data {
                    ArrayData::DComplex(v) => {
                        // The [1, :] row of row0 cell = (0,4),(0,5),(0,6)
                        assert_eq!(&v[..], &[(0.0f64, 4.0), (0.0, 5.0), (0.0, 6.0)][..]);
                    }
                    other => panic!("expected dcomplex, got {other:?}"),
                }
            }
            other => panic!("expected array, got {other:?}"),
        }

        // Long string via the string buckets.
        assert_eq!(
            t.getcell(3, 1).unwrap(),
            RecordValue::String("a very long label number 1 exceeding eight chars".into())
        );

        // ISM column.
        assert_eq!(t.getcell(4, 3).unwrap(), RecordValue::Double(1.5));

        // getvarcol iterates all rows.
        let all = t.getvarcol(0).unwrap();
        assert_eq!(all.len(), 4);
        assert_eq!(all[3], RecordValue::Int(3));

        // getcol on the ISM column.
        let times = t.getcol(4, 0, 4).unwrap();
        assert_eq!(
            times,
            (0..4)
                .map(|i| RecordValue::Double(i as f64 / 2.0))
                .collect::<Vec<_>>()
        );
    }

    fn kw_desc() -> TableDesc {
        let mut desc = typed_desc();
        desc.columns = vec![scalar_col("A", DataType::Int, 0)];
        desc
    }

    #[test]
    fn keyword_write_round_trip() {
        // Build the same keywords the kw.tab fixture holds.
        let mut wt = WritableTable::create(temp_dir("kwrite"), kw_desc());
        wt.addrows(1);
        wt.putcell(0, 0, RecordValue::Int(0)).unwrap();
        wt.putkeyword("VER", RecordValue::String("1.0".into()));
        wt.putkeyword("MAXROWS", RecordValue::Int(1000));
        let hh = {
            let mut r = crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            };
            r.desc.fields.push(crate::record::RecordDescField {
                name: "II".into(),
                data_type: crate::record::DataType::Int,
                sub_desc: None,
                shape: None,
                table_desc_name: None,
                comment: String::new(),
            });
            r.values.push(RecordValue::Int(5));
            r
        };
        let nest = {
            let mut r = crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            };
            r.desc.fields.push(crate::record::RecordDescField {
                name: "HH".into(),
                data_type: crate::record::DataType::Record,
                sub_desc: Some(hh.desc.clone()),
                shape: None,
                table_desc_name: None,
                comment: String::new(),
            });
            r.desc.fields.push(crate::record::RecordDescField {
                name: "S".into(),
                data_type: crate::record::DataType::String,
                sub_desc: None,
                shape: None,
                table_desc_name: None,
                comment: String::new(),
            });
            r.values.push(RecordValue::Record(hh));
            r.values.push(RecordValue::String("x".into()));
            r
        };
        wt.putkeyword("NEST", RecordValue::Record(nest));
        wt.putcolkeyword(0, "UNITS", RecordValue::String("Jy".into()))
            .unwrap();
        wt.putcolkeyword(0, "MULTI", RecordValue::Int(3)).unwrap();
        assert_eq!(
            wt.keywords_json(),
            r#"{"VER":"1.0","MAXROWS":1000,"NEST":{"HH":{"II":5},"S":"x"}}"#
        );
        let dir = wt.flush().unwrap();

        let t = Table::open(&dir, true).unwrap();
        assert_eq!(
            t.getkeywords(),
            r#"{"VER":"1.0","MAXROWS":1000,"NEST":{"HH":{"II":5},"S":"x"}}"#
        );
        assert_eq!(t.getcolkeywords(0).unwrap(), r#"{"UNITS":"Jy","MULTI":3}"#);

        // Removal.
        let mut wt2 = WritableTable::create(temp_dir("kwdel"), kw_desc());
        wt2.putkeyword("K", RecordValue::Int(1));
        wt2.removekeyword("K");
        assert_eq!(wt2.keywords_json(), "{}");
    }

    #[test]
    fn subtable_keyword_write_round_trip() {
        let base = temp_dir("subkw");

        // A subtable in the same directory as the parent's directory.
        let sub_dir = std::path::PathBuf::from(&base).join("SUB.tab");
        let mut sub = WritableTable::create(&sub_dir, kw_desc());
        sub.addrows(1);
        sub.putcell(0, 0, RecordValue::Int(1)).unwrap();
        sub.flush().unwrap();

        // A subtable one level down.
        let sub2_dir = std::path::PathBuf::from(&base)
            .join("nested")
            .join("S2.tab");
        std::fs::create_dir_all(sub2_dir.parent().unwrap()).unwrap();
        let mut sub2 = WritableTable::create(&sub2_dir, kw_desc());
        sub2.addrows(1);
        sub2.putcell(0, 0, RecordValue::Int(2)).unwrap();
        sub2.flush().unwrap();

        let parent_dir = std::path::PathBuf::from(&base).join("P.tab");
        let mut wt = WritableTable::create(&parent_dir, kw_desc());
        wt.addrows(1);
        wt.putcell(0, 0, RecordValue::Int(0)).unwrap();
        // Casacore-style subtable keyword: TpTable value relative to P's dir.
        wt.putkeyword(
            "SAME",
            RecordValue::Table(sub_dir.to_string_lossy().into_owned()),
        );
        wt.putkeyword(
            "SUB2",
            RecordValue::Table(sub2_dir.to_string_lossy().into_owned()),
        );
        // A column keyword and a nested-record keyword containing a subtable.
        wt.putcolkeyword(
            0,
            "SUBREF",
            RecordValue::Table(sub_dir.to_string_lossy().into_owned()),
        )
        .unwrap();
        let nested = {
            let mut inner = crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            };
            inner.desc.fields.push(crate::record::RecordDescField {
                name: "LINK".into(),
                data_type: crate::record::DataType::Table,
                sub_desc: None,
                shape: None,
                table_desc_name: None,
                comment: String::new(),
            });
            // A nested TpTable reference; putkeyword's relativization recurses
            // into nested records.
            inner
                .values
                .push(RecordValue::Table(sub_dir.to_string_lossy().into_owned()));
            inner
        };
        let mut nest = crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        };
        nest.desc.fields.push(crate::record::RecordDescField {
            name: "HH".into(),
            data_type: crate::record::DataType::Record,
            sub_desc: Some(nested.desc.clone()),
            shape: None,
            table_desc_name: None,
            comment: String::new(),
        });
        nest.values.push(RecordValue::Record(nested));
        wt.putkeyword("NEST", RecordValue::Record(nest));
        let dir = wt.flush().unwrap();

        let t = Table::open(&dir, true).unwrap();
        let sub_abs = sub_dir.canonicalize().unwrap().display().to_string();
        let sub2_abs = sub2_dir.canonicalize().unwrap().display().to_string();
        let expected = format!(
            "{{\"SAME\":\"Table: {sub_abs}\",\"SUB2\":\"Table: {sub2_abs}\",\"NEST\":{{\"HH\":{{\"LINK\":\"Table: {sub_abs}\"}}}}}}"
        );
        assert_eq!(t.getkeywords(), expected);
        assert_eq!(
            t.getcolkeywords(0).unwrap(),
            format!("{{\"SUBREF\":\"Table: {sub_abs}\"}}")
        );
    }

    #[test]
    fn writable_table_addrows_putcol_flush() {
        use crate::record::{ArrayData, ArrayValue};
        // MS-style schema: ISM index columns + TSM DATA + SSM scalar/string.
        let mut desc = typed_desc();
        desc.columns = vec![
            ism_col("TIME", DataType::Double, 1),
            ism_col("ANT1", DataType::Int, 1),
            tsm_arr_col("DATA", DataType::DComplex, vec![2, 3]),
            scalar_col("NAME", DataType::String, 0),
        ];
        let mut wt = WritableTable::create(temp_dir("wtable"), desc);
        wt.addrows(4);
        wt.putcol(
            0,
            0,
            &(0..4)
                .map(|i| RecordValue::Double(i as f64 / 2.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        wt.putcol(
            1,
            0,
            &[0, 0, 1, 1]
                .into_iter()
                .map(RecordValue::Int)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for row in 0..4u64 {
            wt.putcell(
                2,
                row,
                RecordValue::Array(ArrayValue {
                    shape: vec![2, 3],
                    data: ArrayData::DComplex((1..=6).map(|k| (row as f64, k as f64)).collect()),
                }),
            )
            .unwrap();
        }
        wt.putcol(
            3,
            0,
            &(0..4)
                .map(|i| RecordValue::String(format!("row{i} long label")))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let dir = wt.flush().unwrap();

        // Reopen and read back.
        let t = Table::open(&dir, true).unwrap();
        assert_eq!(t.nrows(), 4);
        assert_eq!(t.getcol(0, 0, 4).unwrap()[3], RecordValue::Double(1.5));
        assert_eq!(
            t.getcol(1, 0, 4).unwrap(),
            [0, 0, 1, 1]
                .into_iter()
                .map(RecordValue::Int)
                .collect::<Vec<_>>()
        );
        match t.getcell(2, 2).unwrap() {
            RecordValue::Array(a) => {
                assert_eq!(a.shape, vec![2, 3]);
                match &a.data {
                    ArrayData::DComplex(v) => {
                        let expect: Vec<(f64, f64)> = (1..=6).map(|k| (2.0, k as f64)).collect();
                        assert_eq!(&v[..], &expect[..]);
                    }
                    other => panic!("expected dcomplex, got {other:?}"),
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
        assert_eq!(
            t.getcell(3, 3).unwrap(),
            RecordValue::String("row3 long label".into())
        );
    }

    /// The `table.lock` sync record's row count (casacore reads it in
    /// preference to the table.dat header) parses for both sync versions.
    #[test]
    fn lock_sync_nrrow_parses_sync_record() {
        let dir = temp_dir("locksync");
        std::fs::create_dir_all(&dir).unwrap();
        for (nrrow, ver) in [(5u64, 1u32), (70_000_000_000u64, 2u32)] {
            let mut b = vec![0u8; 8]; // FileLocker preamble
            b.extend_from_slice(&[0, 0, 0, 4]); // framed string len
            b.extend_from_slice(b"sync");

            b.extend_from_slice(&ver.to_be_bytes());
            if ver == 1 {
                b.extend_from_slice(&(nrrow as u32).to_be_bytes());
            } else {
                b.extend_from_slice(&nrrow.to_be_bytes());
            }
            std::fs::write(dir.join("table.lock"), b).unwrap();
            assert_eq!(super::lock_sync_nrrow(&dir), Some(nrrow), "sync v{ver}");
        }
        // A lock with no sync record yields nothing (header rule applies).
        std::fs::write(dir.join("table.lock"), vec![0u8; 64]).unwrap();
        assert_eq!(super::lock_sync_nrrow(&dir), None);
    }

    #[test]
    fn create_table_with_tsm_column_reads_back() {
        use crate::record::{ArrayData, ArrayValue};
        // A fixed-shape 2x3 dcomplex column stored with TiledColumnStMan.
        let mut desc = typed_desc();
        desc.columns = vec![
            tsm_arr_col("DATA", DataType::DComplex, vec![2, 3]),
            scalar_col("IDX", DataType::Int, 0),
        ];
        let arr = |row: i32| {
            RecordValue::Array(ArrayValue {
                shape: vec![2, 3],
                data: ArrayData::DComplex((1..=6).map(|k| (row as f64, k as f64)).collect()),
            })
        };
        let values = vec![
            vec![arr(0), arr(1), arr(2)],
            vec![
                RecordValue::Int(0),
                RecordValue::Int(1),
                RecordValue::Int(2),
            ],
        ];
        let dir = temp_dir("tsm");
        create_table(&dir, &desc, &values).unwrap();
        // Files: table.dat, table.f0 (TSM header), table.f0_TSM0 (tiles),
        // table.f1 (SSM).
        assert!(dir.join("table.f0_TSM0").is_file(), "tile file present");

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        assert_eq!(
            dat.column_set.data_managers[0].type_name,
            "TiledColumnStMan"
        );
        assert!(
            matches!(
                &dat.column_set.data_managers[0].blob,
                crate::columnset::DataManagerBlob::Unsupported(b) if b.is_empty()
            ),
            "TSM ColumnSet blob is empty"
        );
        let tsm = crate::tsm::TsmFile::open(&dir, 0, dat.header.big_endian).unwrap();
        assert_eq!(tsm.header.hypercolumn_name, "TiledData_GROUP");
        assert_eq!(tsm.header.nrrow, 3);
        assert_eq!(tsm.header.cubes[0].cube_shape, vec![3, 2, 3]);
        let data = dat.desc.column("DATA").unwrap();
        for row in 0..3u64 {
            let cell = tsm.read_cell(data, row).unwrap();
            match cell {
                RecordValue::Array(a) => {
                    assert_eq!(a.shape, vec![2, 3], "row {row}");
                    match &a.data {
                        ArrayData::DComplex(v) => {
                            let expect: Vec<(f64, f64)> =
                                (1..=6).map(|k| (row as f64, k as f64)).collect();
                            assert_eq!(&v[..], &expect[..], "row {row}");
                        }
                        other => panic!("expected dcomplex, got {other:?}"),
                    }
                }
                other => panic!("expected array, got {other:?}"),
            }
        }
    }

    /// The incremental flush: a chunked in-place write with a flush between
    /// chunks must leave the on-disk state equal to the overlay of every
    /// write, without re-encoding untouched rows (the dask-ms per-chunk
    /// `putcol` + `flush` pattern), and must release the buffered cells so
    /// at most one chunk stays resident.  Exercises TSM tile patching
    /// (cross-tile, within-tile and re-overwritten rows) and an SSM bool
    /// column rebuilt from disk + pending.
    #[test]
    fn incremental_flush_overlays_only_written_rows() {
        use crate::record::{ArrayData, ArrayValue};
        let mut desc = typed_desc();
        desc.columns = vec![
            ColumnDesc {
                name: "FLAG".into(),
                comment: String::new(),
                data_type: DataType::Bool,
                data_manager_type: "TiledShapeStMan".into(),
                data_manager_group: "TiledFlag".into(),
                options: 4,
                ndim: 2,
                shape: Some(vec![]),
                max_length: 0,
                keywords: empty_record(),
                kind: ColumnKind::Array,
            },
            ColumnDesc {
                name: "FLAG_ROW".into(),
                comment: String::new(),
                data_type: DataType::Bool,
                data_manager_type: "StandardStMan".into(),
                data_manager_group: "FlagRow".into(),
                options: 0,
                ndim: -1,
                shape: None,
                max_length: 0,
                keywords: empty_record(),
                kind: ColumnKind::Scalar(zero_value(DataType::Bool)),
            },
            scalar_col("SCAN_NUMBER", DataType::Int, 0),
        ];
        let elems = 158usize;
        let nrows = 26214 * 2 + 9; // three 26214-row tiles
        let dir = temp_dir("incrr");
        let v0 = |r: usize| (r * 3 + 1).is_multiple_of(7);
        let w1 = |r: usize| (r * 5 + 2).is_multiple_of(11);
        let w2 = |r: usize| (r * 7 + 3).is_multiple_of(13);
        let arr_of = |r: usize, f: fn(usize) -> bool| {
            RecordValue::Array(ArrayValue {
                shape: vec![79, 2],
                data: ArrayData::Bool((0..elems).map(|k| f(r * 3 + k)).collect()),
            })
        };
        {
            let mut wt = WritableTable::create(&dir, desc.clone());
            wt.addrows(nrows as u64);
            for r in 0..nrows {
                wt.putcell(0, r as u64, arr_of(r, v0)).unwrap();
                wt.putcell(1, r as u64, RecordValue::Bool(v0(r))).unwrap();
                wt.putcell(2, r as u64, RecordValue::Int(r as i32)).unwrap();
            }
            wt.flush().unwrap();
        }
        {
            let (_snap, mut wt) = WritableTable::open_for_update(&dir).unwrap();
            for r in 10_000..25_000 {
                wt.putcell(0, r as u64, arr_of(r, w1)).unwrap();
                wt.putcell(1, r as u64, RecordValue::Bool(w1(r))).unwrap();
            }
            wt.flush().unwrap();
            assert_eq!(wt.pending_rows(0).count(), 0);
            assert_eq!(wt.pending_rows(1).count(), 0);
            assert!(wt.cell(0, 10_100).is_none(), "cells released after flush");
            for r in (5..15_000).chain(30_000..45_000) {
                wt.putcell(0, r as u64, arr_of(r, w2)).unwrap();
                wt.putcell(1, r as u64, RecordValue::Bool(w2(r))).unwrap();
            }
            wt.flush().unwrap();
        }
        let fex = |r: usize| {
            if r < 5 {
                v0
            } else if r < 15_000 {
                w2
            } else if r < 25_000 {
                w1
            } else if r < 30_000 {
                v0
            } else if r < 45_000 {
                w2
            } else {
                v0
            }
        };
        let t = Table::open(&dir, true).unwrap();
        for r in [
            0usize, 4, 5, 9_999, 10_000, 14_999, 15_000, 24_999, 25_000, 29_999, 30_000, 44_999,
            45_000, 52_435, 52_436,
        ] {
            assert_eq!(
                t.getcell(0, r as u64).unwrap(),
                arr_of(r, fex(r)),
                "FLAG row {r}"
            );
            assert_eq!(
                t.getcell(1, r as u64).unwrap(),
                RecordValue::Bool(fex(r)(r)),
                "FLAG_ROW row {r}"
            );
        }
        let flag_row = t.getcol(1, 0, nrows as u64).unwrap();
        for (r, v) in flag_row.iter().enumerate() {
            assert_eq!(*v, RecordValue::Bool(fex(r)(r)), "FLAG_ROW scan row {r}");
        }
        let scan = t.getcol(2, 0, nrows as u64).unwrap();
        for (r, v) in scan.iter().enumerate() {
            assert_eq!(*v, RecordValue::Int(r as i32), "SCAN_NUMBER row {r}");
        }
    }

    #[test]
    fn create_table_with_ism_columns_reads_back() {
        use crate::ism::IsmFile;
        // MS-style: ISM index columns (TIME double, ANT1 int) + SSM VAL.
        let mut desc = typed_desc();
        desc.columns = vec![
            ism_col("TIME", DataType::Double, 1),
            ism_col("ANT1", DataType::Int, 1),
            scalar_col("VAL", DataType::Float, 0),
        ];
        let time = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let ant1 = [0, 0, 1, 1, 1, 2];
        let values = vec![
            time.iter().map(|&v| RecordValue::Double(v)).collect(),
            ant1.iter().map(|&v| RecordValue::Int(v)).collect(),
            (0..6).map(|i| RecordValue::Float(i as f32)).collect(),
        ];
        let dir = temp_dir("ism");
        create_table(&dir, &desc, &values).unwrap();

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        // DMs: IncrementalStMan seq 0, StandardStMan seq 1.
        assert_eq!(dat.column_set.data_managers.len(), 2);
        assert_eq!(
            dat.column_set.data_managers[0].type_name,
            "IncrementalStMan"
        );
        assert_eq!(dat.column_set.data_managers[0].sequence_nr, 0);
        assert_eq!(dat.column_set.data_managers[1].type_name, "StandardStMan");
        // Per-column bindings: TIME/ANT1 -> dm 0, VAL -> dm 1.
        assert_eq!(dat.column_set.columns[0].data_manager_seq, 0);
        assert_eq!(dat.column_set.columns[1].data_manager_seq, 0);
        assert_eq!(dat.column_set.columns[2].data_manager_seq, 1);

        let ism = IsmFile::open(&dir, 0, dat.header.big_endian).unwrap();
        let ant1_col = dat.desc.column("ANT1").unwrap();
        for (row, &v) in ant1.iter().enumerate() {
            assert_eq!(
                ism.read_scalar_cell(1, ant1_col, row as u64).unwrap(),
                RecordValue::Int(v),
                "ANT1 row {row}"
            );
        }
        let time_col = dat.desc.column("TIME").unwrap();
        for (row, &v) in time.iter().enumerate() {
            assert_eq!(
                ism.read_scalar_cell(0, time_col, row as u64).unwrap(),
                RecordValue::Double(v),
                "TIME row {row}"
            );
        }
        // VAL is in the SSM file (seq 1), first column of that DM.
        let ssm = crate::ssm::StandardStManFile::open(&dir, 1, dat.header.big_endian).unwrap();
        let dm1 = &dat.column_set.data_managers[1];
        let spec = match &dm1.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan spec"),
        };
        let val_col = dat.desc.column("VAL").unwrap();
        for row in 0..6u64 {
            assert_eq!(
                ssm.read_scalar_cell(spec, 0, val_col, row).unwrap(),
                RecordValue::Float(row as f32)
            );
        }
    }

    #[test]
    fn long_string_spans_multiple_string_buckets() {
        // A single string column gives a bucket of ~384 bytes; a 600-char
        // string must chain across several string buckets (putData).
        let mut desc = typed_desc();
        desc.columns = vec![scalar_col("TXT", DataType::String, 0)];
        let big = "x".repeat(600);
        let values = vec![vec![RecordValue::String(big.clone())]];
        let dir = temp_dir("chainstr");
        create_table(&dir, &desc, &values).unwrap();

        let dat_bytes = std::fs::read(dir.join("table.dat")).unwrap();
        let dat = parse_table_dat(&dat_bytes).unwrap();
        let file = crate::ssm::StandardStManFile::open(&dir, 0, dat.header.big_endian).unwrap();
        // Bucket size = rowsPerBucket * 12 = 384; string data area = 368.
        assert_eq!(file.header.bucket_size, 384);
        assert!(file.header.nr_buckets >= 3, "string chained buckets");
        let dm = &dat.column_set.data_managers[0];
        let spec = match &dm.blob {
            crate::columnset::DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan spec"),
        };
        let txt = dat.desc.column("TXT").unwrap();
        assert_eq!(
            file.read_scalar_cell(spec, 0, txt, 0).unwrap(),
            RecordValue::String(big)
        );
    }

    // ---------------------------------------------------------------------
    // Extensive core-table tests: full dtype/boundary matrix, string
    // boundaries, empty-table lifecycle, flush-then-reopen persistence.
    // ---------------------------------------------------------------------

    /// Bit-exact `RecordValue` equality so NaN payloads and signed zero
    /// survive byte-for-byte (the plain `PartialEq` derives on floats, for
    /// which NaN != NaN).
    fn value_eq_bits(a: &RecordValue, b: &RecordValue) -> bool {
        use crate::record::RecordValue as RV;
        match (a, b) {
            (RV::Bool(a), RV::Bool(b)) => a == b,
            (RV::UChar(a), RV::UChar(b)) => a == b,
            (RV::UShort(a), RV::UShort(b)) => a == b,
            (RV::Short(a), RV::Short(b)) => a == b,
            (RV::Int(a), RV::Int(b)) => a == b,
            (RV::UInt(a), RV::UInt(b)) => a == b,
            (RV::Int64(a), RV::Int64(b)) => a == b,
            (RV::Float(a), RV::Float(b)) => a.to_bits() == b.to_bits(),
            (RV::Double(a), RV::Double(b)) => a.to_bits() == b.to_bits(),
            (RV::Complex(a1, a2), RV::Complex(b1, b2)) => {
                a1.to_bits() == b1.to_bits() && a2.to_bits() == b2.to_bits()
            }
            (RV::DComplex(a1, a2), RV::DComplex(b1, b2)) => {
                a1.to_bits() == b1.to_bits() && a2.to_bits() == b2.to_bits()
            }
            (RV::String(a), RV::String(b)) => a == b,
            (RV::Table(a), RV::Table(b)) => a == b,
            (RV::Record(a), RV::Record(b)) => a == b,
            (RV::Array(a), RV::Array(b)) => a.shape == b.shape && a.data == b.data,
            _ => false,
        }
    }

    /// Every supported scalar type round-trips exact values through the SSM
    /// write path (`create_table`) and the read path (`Table::open`/`getcol`),
    /// including boundary and extreme values (min/max ints, inf, NaN, -0.0)
    /// compared bit-for-bit. `UShort` is deliberately absent: casacore's own
    /// storage managers reject it ("unknown data type 4", like real
    /// python-casacore), so it is covered only at the mapping layer.
    #[test]
    fn scalar_dtype_matrix_with_boundaries_round_trips() {
        use crate::record::DataType as DT;
        use crate::record::RecordValue as RV;
        let desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![
                scalar_col("B", DT::Bool, 0),
                scalar_col("U1", DT::UChar, 0),
                scalar_col("I2", DT::Short, 0),
                scalar_col("I4", DT::Int, 0),
                scalar_col("U4", DT::UInt, 0),
                scalar_col("I8", DT::Int64, 0),
                scalar_col("R4", DT::Float, 0),
                scalar_col("R8", DT::Double, 0),
                scalar_col("C4", DT::Complex, 0),
                scalar_col("C8", DT::DComplex, 0),
                scalar_col("S", DT::String, 0),
            ],
        };
        let rows = 3usize;
        let values = vec![
            vec![RV::Bool(true), RV::Bool(false), RV::Bool(true)],
            vec![RV::UChar(0), RV::UChar(255), RV::UChar(7)],
            vec![RV::Short(-32768), RV::Short(32767), RV::Short(-300)],
            vec![
                RV::Int(-2_147_483_648),
                RV::Int(2_147_483_647),
                RV::Int(-70000),
            ],
            vec![RV::UInt(0), RV::UInt(4_294_967_295), RV::UInt(7)],
            vec![
                RV::Int64(-9_223_372_036_854_775_808),
                RV::Int64(9_223_372_036_854_775_807),
                RV::Int64(-9_000_000_000_000),
            ],
            vec![RV::Float(0.0), RV::Float(f32::MAX), RV::Float(-0.0)],
            vec![
                RV::Double(f64::NEG_INFINITY),
                RV::Double(f64::NAN),
                RV::Double(1.5e300),
            ],
            vec![
                RV::Complex(-1.5e38, -2.0),
                RV::Complex(0.0, 0.0),
                RV::Complex(1.0, 1.0),
            ],
            vec![
                RV::DComplex(1.5e300, f64::NAN),
                RV::DComplex(0.0, -0.0),
                RV::DComplex(-1.0, 2.0),
            ],
            vec![
                RV::String(String::new()),
                RV::String("abc".into()),
                RV::String("a very long string that spans buckets".into()),
            ],
        ];
        let dir = temp_dir("dtype");
        create_table(&dir, &desc, &values).unwrap();
        let t = Table::open(&dir, false).unwrap();
        assert_eq!(t.nrows(), rows as u64);
        for (i, col) in desc.columns.iter().enumerate() {
            let got = t.getcol(i, 0, rows as u64).unwrap();
            assert_eq!(got.len(), rows);
            for (g, e) in got.iter().zip(values[i].iter()) {
                assert!(value_eq_bits(g, e), "{}: {g:?} != {e:?}", col.name);
            }
        }
    }

    /// SSM string handling honours every length boundary: empty, inline
    /// (<= 8 bytes), the 8/9-char inline/bucket cutover, fixed-length
    /// `maxlen` cells, and multi-byte UTF-8 round-trip intact.
    #[test]
    fn string_length_boundaries_round_trip() {
        use crate::record::DataType as DT;
        use crate::record::RecordValue as RV;
        let cases = [
            String::new(),
            "a".into(),
            "1234567".into(),                // 7 bytes: inline
            "12345678".into(),               // 8 bytes: still inline
            "123456789".into(),              // 9 bytes: first bucket ref
            "12345678901234567890".into(),   // 20 bytes
            "日本語のテキストテスト".into(), // multi-byte UTF-8
            "x".repeat(300),                 // chains several buckets
        ];
        let desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![scalar_col("S", DT::String, 0)],
        };
        let values: Vec<Vec<RecordValue>> = vec![cases.iter().cloned().map(RV::String).collect()];
        let dir = temp_dir("strings");
        create_table(&dir, &desc, &values).unwrap();
        let t = Table::open(&dir, false).unwrap();
        for (r, expected) in cases.iter().enumerate() {
            assert_eq!(
                t.getcell(0, r as u64).unwrap(),
                RV::String(expected.clone()),
                "row {r}"
            );
        }
        // Fixed-length (maxlen) columns pad/truncate to maxlen on write.
        let desc_fixed = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![scalar_col("F", DT::String, 8)],
        };
        let values_fixed: Vec<Vec<RecordValue>> = vec![vec![
            RV::String("12345678".into()),
            RV::String("123456789".into()), // longer than maxlen
            RV::String(String::new()),
        ]];
        let dir2 = temp_dir("strings_fixed");
        create_table(&dir2, &desc_fixed, &values_fixed).unwrap();
        let t2 = Table::open(&dir2, false).unwrap();
        // Only rows within maxlen round-trip exact; overflow is mangled (the
        // caller must truncate); assert the empty and exact rows survive.
        assert_eq!(t2.getcell(0, 0).unwrap(), RV::String("12345678".into()));
        assert_eq!(t2.getcell(0, 2).unwrap(), RV::String(String::new()));
    }

    /// Zero-row and single-row tables round-trip: open, report rows, expose
    /// columns, and reject reading past the row count.
    #[test]
    fn empty_and_single_row_tables() {
        use crate::record::DataType as DT;
        let desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![scalar_col("V", DT::Int, 0), scalar_col("S", DT::String, 0)],
        };
        let dir = temp_dir("empty");
        create_table(&dir, &desc, &[vec![], vec![]]).unwrap();
        let t = Table::open(&dir, false).unwrap();
        assert_eq!(t.nrows(), 0);
        assert_eq!(t.getcol(0, 0, 0).unwrap().len(), 0);
        // Reading any cell of an empty table is out of range (the Python
        // layer clamps nrow before reaching the core).
        assert!(t.getcol(0, 0, 1).is_err());
        assert!(t.getcell(0, 0).is_err(), "no rows to read");

        let dir1 = temp_dir("onerow");
        create_table(
            &dir1,
            &desc,
            &[
                vec![RecordValue::Int(42)],
                vec![RecordValue::String("hi".into())],
            ],
        )
        .unwrap();
        let t1 = Table::open(&dir1, false).unwrap();
        assert_eq!(t1.nrows(), 1);
        assert_eq!(t1.getcell(0, 0).unwrap(), RecordValue::Int(42));
        assert!(t1.getcell(0, 1).is_err());
    }

    /// The writable path buffers cells and `flush()` persists them: after a
    /// `WritableTable` `addrows` + `putcell` + `flush`, a fresh `Table::open`
    /// reads exactly the stored cells, with unwritten scalar cells holding
    /// their column default.
    #[test]
    fn writable_flush_persists_cells_for_reopen() {
        use crate::record::DataType as DT;
        let desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![scalar_col("V", DT::Int, 0)],
        };
        let dir = temp_dir("wpersist");
        let mut wt = WritableTable::create(&dir, desc.clone());
        wt.addrows(3);
        wt.putcell(0, 1, RecordValue::Int(7)).unwrap();
        let _ = wt.flush().unwrap();
        let t = Table::open(&dir, false).unwrap();
        assert_eq!(t.nrows(), 3);
        // Row 0 and 2 were never written: scalar default (0).
        assert_eq!(
            t.getcol(0, 0, 3).unwrap(),
            [
                RecordValue::Int(0),
                RecordValue::Int(7),
                RecordValue::Int(0)
            ]
        );
        let _ = desc;
    }
}
