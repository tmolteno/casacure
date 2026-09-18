//! Parsing of the `table.dat` header of a CASA table.
//!
//! Layout (`casacore/tables/Tables/BaseTable.cc::writeStart`,
//! `PlainTable.cc::putFile`): a root AipsIO object of type `"Table"`, whose
//! payload is the row count, an endianness flag describing the *data* files
//! (`table.f*` — the `table.dat` stream itself is always big-endian
//! canonical AipsIO), and a table-kind string (`"PlainTable"`).

use crate::aipsio::{AipsIoError, Reader};
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

/// Parse the header from the start of a `table.dat` buffer.
pub fn parse_table_header(buf: &[u8]) -> Result<TableHeader, TableError> {
    let mut r = Reader::new(buf);
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
}
