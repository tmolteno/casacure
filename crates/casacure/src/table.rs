//! Parsing of the `table.dat` header of a CASA table.
//!
//! Layout (`casacore/tables/Tables/BaseTable.cc::writeStart`,
//! `PlainTable.cc::putFile`): a root AipsIO object of type `"Table"`, whose
//! payload is the row count, an endianness flag describing the *data* files
//! (`table.f*` — the `table.dat` stream itself is always big-endian
//! canonical AipsIO), and a table-kind string (`"PlainTable"`).

use crate::aipsio::{AipsIoError, Reader};
use crate::columnset::{parse_column_set, ColumnSet, ColumnSetError};
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
pub fn build_table_dat(
    big_endian: bool,
    nrow: u64,
    desc: &crate::tabledesc::TableDesc,
    spec: &crate::columnset::StandardStMan,
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
    crate::columnset::write_column_set(&mut w, nrow, 1, &desc.columns, spec);
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

pub fn create_table(
    table_dir: &std::path::Path,
    desc: &crate::tabledesc::TableDesc,
    values: &[Vec<crate::record::RecordValue>],
) -> Result<Vec<std::path::PathBuf>, TableCreateError> {
    use crate::ssm::{layout, write_standard_stman_file, WriteColumn};
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
    // (little endian; matches casacore on any little-endian host)
    let big_endian = false;

    let mut cell_bits = Vec::with_capacity(desc.columns.len());
    let mut cell_bytes = Vec::with_capacity(desc.columns.len());
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(desc.columns.len());
    for (col, list) in values.iter().enumerate() {
        let cd = &desc.columns[col];
        if cd.data_manager_type != "StandardStMan" {
            return Err(TableCreateError::NotStandardStMan(cd.name.clone()));
        }
        if !matches!(cd.kind, crate::tabledesc::ColumnKind::Scalar(_)) {
            return Err(TableCreateError::NotScalar(cd.name.clone()));
        }
        let size = crate::ssm::scalar_cell_size(cd);
        let mut bytes = Vec::with_capacity(list.len() * size as usize);
        for value in list {
            bytes.extend_from_slice(
                &crate::ssm::encode_scalar_cell(big_endian, cd, value).map_err(|e| match e {
                    crate::ssm::SsmError::StringBucketUnsupported { .. } => {
                        TableCreateError::LongString(cd.name.clone())
                    }
                    other => TableCreateError::Io(std::io::Error::other(format!(
                        "encode {}.{}: {other}",
                        desc.name, cd.name
                    ))),
                })?,
            );
        }
        cell_bits.push(if cd.data_type == crate::record::DataType::Bool {
            1
        } else {
            8 * size
        });
        cell_bytes.push(size);
        encoded.push(bytes);
    }

    let l = layout(ROWS_PER_BUCKET, &cell_bits);
    let cols: Vec<WriteColumn<'_>> = encoded
        .iter()
        .zip(cell_bytes.iter())
        .map(|(bytes, size)| WriteColumn {
            cell_size: *size,
            cell_bits: 0, // unused by the file writer
            bytes,
        })
        .collect();
    let data_file = write_standard_stman_file(big_endian, nrow, &cols, &l);

    let spec = crate::columnset::StandardStMan {
        data_manager_name: "StandardStMan".into(),
        column_offset: l.column_offset.clone(),
        col_index_map: vec![0; desc.columns.len()],
    };
    let table_dat = build_table_dat(big_endian, nrow, desc, &spec)?;

    std::fs::create_dir_all(table_dir)?;
    let dat_path = table_dir.join("table.dat");
    std::fs::write(&dat_path, table_dat)?;
    let f0_path = table_dir.join("table.f0");
    std::fs::write(&f0_path, data_file)?;
    Ok(vec![dat_path, f0_path])
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
        let bytes = build_table_dat(false, 1, &desc, &spec).unwrap();
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

    #[test]
    fn create_table_rejects_long_strings() {
        let desc = typed_desc();
        let mut values = typed_values(1);
        values[9] = vec![RecordValue::String("way longer than eight chars".into())];
        let dir = temp_dir("longstring");
        assert!(matches!(
            create_table(&dir, &desc, &values),
            Err(TableCreateError::LongString(_))
        ));
    }
}
