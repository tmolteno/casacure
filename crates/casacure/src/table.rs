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
            "TiledColumnStMan" => {
                let (header, tile) =
                    build_tsm_data(big_endian, dm as u32, dm_name, desc, values, &dm_cols)?;
                dms.push(DmBlob {
                    type_name: type_name.clone(),
                    sequence_nr: dm as u32,
                    blob: Vec::new(), // TSM writes its spec to the header file
                });
                data_files.push((dm as u32, header));
                // Tile data lives in `table.f{dm}_TSM0`.
                tile_files.push((dm as u32, 0, tile));
            }
            other => {
                return Err(TableCreateError::Io(std::io::Error::other(format!(
                    "unsupported data-manager type {other}"
                ))))
            }
        }
    }

    let table_dat = build_table_dat(big_endian, nrow, desc, &dms, &col_dm_seq)?;

    std::fs::create_dir_all(table_dir)?;
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
                // One byte per row including Bool scalars: casacore's SSM
                // stores scalar Bool cells as a byte (verified against a
                // 100-row casacore table), unlike the bit-packed array
                // index. The Bucket layout must reserve the same space the
                // writer fills.
                let bits = 8 * size;
                (size, bits)
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
                return Err(TableCreateError::NotScalar(cd.name.clone()))
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
                crate::tabledesc::ColumnKind::Record => unreachable!(),
            }
        }
        encoded.push(bytes);
    }

    let cols: Vec<WriteColumn<'_>> = encoded
        .iter()
        .zip(cell_bytes.iter())
        .map(|(bytes, size)| WriteColumn {
            cell_size: *size,
            cell_bits: 0,
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
    big_endian: bool,
    seq_nr: u32,
    dm_name: &str,
    desc: &crate::tabledesc::TableDesc,
    values: &[Vec<crate::record::RecordValue>],
    dm_cols: &[usize],
) -> Result<(Vec<u8>, Vec<u8>), TableCreateError> {
    use crate::record::RecordValue;
    if dm_cols.len() != 1 {
        return Err(TableCreateError::Io(std::io::Error::other(format!(
            "TiledColumnStMan with {} columns is not supported (one array column per group)",
            dm_cols.len()
        ))));
    }
    let col = dm_cols[0];
    let cd = &desc.columns[col];
    let cradle = cd.shape.clone().ok_or_else(|| {
        TableCreateError::NotScalar(format!(
            "{}.{}: TiledColumnStMan needs a fixed-shape array column",
            desc.name, cd.name
        ))
    })?;
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
    crate::tsm::write_tsm_file(big_endian, seq_nr, dm_name, cd.data_type, &cradle, &cells).map_err(
        |e| {
            TableCreateError::Io(std::io::Error::other(format!(
                "encode {}.{}: {e}",
                desc.name, cd.name
            )))
        },
    )
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

impl Table {
    /// Open a table directory (`<dir>/table.dat` + data files).
    pub fn open(
        dir: impl Into<std::path::PathBuf>,
        readonly: bool,
    ) -> Result<Table, TableDatError> {
        let path = dir.into();
        let buf = std::fs::read(path.join("table.dat"))?;
        let dat = parse_table_dat(&buf)?;
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
                "TiledColumnStMan" => tsm_files.push((
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
            &cd.keywords.to_json_string_ctx(self.path.parent()),
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
        s.push_str(
            &self
                .dat
                .desc
                .keywords
                .to_json_string_ctx(self.path.parent()),
        );
        s.push_str(",\"_private_keywords_\":");
        s.push_str(
            &self
                .dat
                .desc
                .private_keywords
                .to_json_string_ctx(self.path.parent()),
        );
        s.push('}');
        s
    }

    /// The table keyword record as a JSON object (python-casacore
    /// `table.getkeywords()`).
    pub fn getkeywords(&self) -> String {
        self.dat
            .desc
            .keywords
            .to_json_string_ctx(self.path.parent())
    }

    /// A column's keyword record as a JSON object (python-casacore
    /// `table.getcolkeywords(col)`).
    pub fn getcolkeywords(&self, col_idx: usize) -> Option<String> {
        self.dat
            .desc
            .columns
            .get(col_idx)
            .map(|c| c.keywords.to_json_string_ctx(self.path.parent()))
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
        match desc.data_manager_type.as_str() {
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
            "TiledColumnStMan" => {
                let file = self
                    .tsm_files
                    .iter()
                    .find(|(s, _)| *s == seq)
                    .map(|(_, f)| f)
                    .ok_or_else(|| {
                        TableReadError::UnsupportedColumn(
                            desc.name.clone(),
                            "TiledColumnStMan".into(),
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
pub struct WritableTable {
    dir: std::path::PathBuf,
    desc: crate::tabledesc::TableDesc,
    /// `cells[col][row]`.
    cells: Vec<Vec<Option<RecordValue>>>,
}

/// Convert absolute `Table`-valued subtable references to the `./relative`
/// form casacore stores (relative to the parent table's directory); absolute
/// paths outside the parent's directory are kept. Recurses into nested
/// records.
fn relativize_subtables(value: RecordValue, base: Option<&std::path::Path>) -> RecordValue {
    match value {
        RecordValue::Table(name) => {
            let path = std::path::Path::new(&name);
            let relativized = match (path.is_absolute(), base) {
                (true, Some(b)) => match path.strip_prefix(b) {
                    Ok(rest) => format!("./{}", rest.display()),
                    Err(_) => name,
                },
                _ => name,
            };
            RecordValue::Table(relativized)
        }
        RecordValue::Record(mut inner) => {
            inner.values = inner
                .values
                .drain(..)
                .map(|v| relativize_subtables(v, base))
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
        let cells = vec![Vec::new(); desc.columns.len()];
        WritableTable {
            dir: dir.into(),
            desc,
            cells,
        }
    }

    /// Append `n` empty rows (`addrows`).
    pub fn addrows(&mut self, n: u64) {
        for col in &mut self.cells {
            col.resize(col.len() + n as usize, None);
        }
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
    }

    /// The number of rows in the in-memory cell store.
    pub fn col_len(&self, col_idx: usize) -> usize {
        self.cells.get(col_idx).map_or(0, Vec::len)
    }

    /// The in-memory value at (col, row), if set.
    pub fn cell(&self, col_idx: usize, row: u64) -> Option<&RecordValue> {
        self.cells.get(col_idx)?.get(row as usize)?.as_ref()
    }

    /// No-op, as casacore's `setmaxcachesize` should be for the replacement
    /// (no caches exist).
    pub fn setmaxcachesize(&mut self, _col_idx: usize, _size: usize) {}

    /// Set a table keyword (`putkeyword`); nested records are supported.
    /// `Table` values (subtable references) are stored relative to the
    /// parent table's directory with a `./` prefix, like casacore.
    pub fn putkeyword(&mut self, name: &str, value: RecordValue) {
        let value = relativize_subtables(value, self.dir.parent());
        self.desc.keywords.set(name, value);
    }

    /// Remove a table keyword (`removekeyword`).
    pub fn removekeyword(&mut self, name: &str) {
        self.desc.keywords.remove(name);
    }

    /// The current table keywords as a JSON object.
    pub fn keywords_json(&self) -> String {
        self.desc.keywords.to_json_string()
    }

    /// The schema being built (columns, keywords, storage managers).
    pub fn desc(&self) -> &crate::tabledesc::TableDesc {
        &self.desc
    }

    /// Set a column keyword (`putcolkeyword`).
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
            .set(name, relativize_subtables(value, self.dir.parent()));
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
        Ok(())
    }

    /// Number of rows written so far.
    pub fn nrows(&self) -> u64 {
        self.cells.first().map_or(0, Vec::len) as u64
    }

    /// Assemble the on-disk table from the buffered cells, filling missing
    /// scalar cells with their defaults; returns the table directory.
    pub fn flush(&mut self) -> Result<std::path::PathBuf, WriteTableError> {
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
        let dir = self.dir.clone();
        create_table(&dir, &self.desc, &values)?;
        Ok(dir)
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
        crate::tabledesc::ColumnKind::Record => None,
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
    Unsupported(String),
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
                if dm.type_name == "TiledColumnStMan" {
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
}
