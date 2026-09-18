//! Parsing of the data-manager info inside `table.dat`
//! (`casacore/tables/Tables/ColumnSet.cc::putFile`/`getFile`, plus the
//! per-data-manager spec blobs written by `DataManager::flush`).
//!
//! Layout: immediately after the `TableDesc` object (still inside the root
//! `"Table"` object; **not** itself framed) is the ColumnSet payload:
//!
//! - `Int` version: negative when written by modern casacore, else the row
//!   count of an ancient version-1 table. The absolute value is the version:
//!   - `<= 2`: followed by a `u32` row count and no storage option;
//!   - `>= 3`: followed by a `u64` row count, then `Int` storage option and
//!     `Int` block size.
//! - `u32` nrman: total data managers ever created (sequence counter).
//! - `u32` nr: data managers with columns, then `nr` ×
//!   (`String` data-manager type, `u32` sequence number).
//! - One `PlainColumn::putFile` record per column, in table column order:
//!   `u32` class version (2), `String` original name, then the class-specific
//!   binding: for scalar/record columns `u32` 1 + `u32` data-manager
//!   sequence number; for array columns additionally a `Bool` shape-column
//!   flag and, when set, the `String` shape-column name.
//! - `nr` opaque blobs (`u32` length + bytes), one per data manager, each a
//!   self-contained AipsIO stream holding that manager's spec — for
//!   StandardStMan a framed `"SSM"` object (`SSMBase::flush`) with the
//!   manager name and the `"Block"`-framed column offset / index map tables.

use crate::aipsio::{AipsIoError, Reader};
use crate::tabledesc::{ColumnDesc, ColumnKind};
use thiserror::Error;

/// Errors from parsing the data-manager info.
#[derive(Debug, Error)]
pub enum ColumnSetError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error("unexpected object type {found:?}, expected {expected:?}")]
    UnexpectedType { expected: String, found: String },
    #[error("unsupported PlainColumn class version {0}")]
    UnsupportedColumnVersion(u32),
    #[error("data-manager blob has type {found:?}, expected \"SSM\" for {type_name:?}")]
    UnexpectedBlobType { type_name: String, found: String },
}

/// Binding of one table column to its data manager (`PlainColumn`).
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnInfo {
    /// Name the column was created with.
    pub original_name: String,
    /// Sequence number of the data manager storing the column data.
    pub data_manager_seq: u32,
    /// Name of the companion shape column for variable-shape array columns.
    pub shape_column: Option<String>,
}

/// A data manager listed in the ColumnSet, with its spec blob.
#[derive(Debug, Clone, PartialEq)]
pub struct DataManager {
    pub type_name: String,
    pub sequence_nr: u32,
    pub blob: DataManagerBlob,
}

/// The decoded per-data-manager spec blob.
#[derive(Debug, Clone, PartialEq)]
pub enum DataManagerBlob {
    /// StandardStMan spec (`SSM` object): manager name + column offset table.
    StandardStMan(StandardStMan),
    /// Other data managers (IncrementalStMan, TiledColumnStMan, ...) whose
    /// on-disk spec is not decoded yet; the raw blob is preserved.
    Unsupported(Vec<u8>),
}

/// StandardStMan spec blob (`SSMBase::flush`/`open64`).
#[derive(Debug, Clone, PartialEq)]
pub struct StandardStMan {
    /// Storage manager name as written at creation.
    pub data_manager_name: String,
    /// Byte offsets into the `.f` data file where each column's data region
    /// starts (one per column this manager owns).
    pub column_offset: Vec<u32>,
    /// Maps each owned column to a stable column index.
    pub col_index_map: Vec<u32>,
}

/// The parsed data-manager info of a CASA table.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSet {
    /// ColumnSet format version (absolute value on disk).
    pub version: u32,
    /// Row count written inside the ColumnSet (redundant with `TableHeader`).
    pub nrow: u64,
    /// (storage option, block size) — only present for version >= 3.
    pub storage_option: Option<(i32, i32)>,
    /// Total data managers ever created (sequence counter).
    pub seq_count: u32,
    /// Data managers with columns, in writing order.
    pub data_managers: Vec<DataManager>,
    /// Per-column data-manager bindings, in table column order.
    pub columns: Vec<ColumnInfo>,
}

impl ColumnSet {
    /// The data manager storing `column`, by its sequence number.
    pub fn data_manager_for<'a>(&'a self, column: &ColumnInfo) -> Option<&'a DataManager> {
        self.data_managers
            .iter()
            .find(|dm| dm.sequence_nr == column.data_manager_seq)
    }
}

/// Parse the ColumnSet payload following the `TableDesc` object.
///
/// `columns` supplies each column's kind (scalar / record / array), which
/// determines the shape of the per-column binding record — mirroring
/// `ColumnSet::getFile` iterating the columns in TableDesc order.
pub fn parse_column_set(
    r: &mut Reader<'_>,
    columns: &[ColumnDesc],
) -> Result<ColumnSet, ColumnSetError> {
    let version_raw = r.read_i32()?;
    let (version, nrow) = if version_raw < 0 {
        let v = version_raw.unsigned_abs();
        if v <= 2 {
            (v, u64::from(r.read_u32()?))
        } else {
            (v, r.read_u64()?)
        }
    } else {
        // Ancient version 1: the first Int is the row count itself.
        (1, u64::from(version_raw as u32))
    };
    let storage_option = if version >= 3 {
        Some((r.read_i32()?, r.read_i32()?))
    } else {
        None
    };
    let seq_count = r.read_u32()?;
    let nr = r.read_u32()?;

    let mut data_managers = Vec::with_capacity(nr as usize);
    for _ in 0..nr {
        let type_name = r.read_string()?;
        let sequence_nr = r.read_u32()?;
        data_managers.push(DataManager {
            type_name,
            sequence_nr,
            blob: DataManagerBlob::Unsupported(Vec::new()),
        });
    }

    let mut column_info = Vec::with_capacity(columns.len());
    for desc in columns {
        // PlainColumn::putFile: class version + original name + derived data.
        let class_version = r.read_u32()?;
        if class_version != 2 {
            return Err(ColumnSetError::UnsupportedColumnVersion(class_version));
        }
        let original_name = r.read_string()?;
        // Derived binding (ScaColData.tcc / ArrColData.cc): a subclass
        // version and the data-manager sequence number; array columns add a
        // shape-column flag and name.
        let _subclass_version = r.read_u32()?;
        let data_manager_seq = r.read_u32()?;
        let shape_column = match desc.kind {
            ColumnKind::Array => {
                let has_shape_col = r.read_bool()?;
                if has_shape_col {
                    Some(r.read_string()?)
                } else {
                    None
                }
            }
            ColumnKind::Scalar(_) | ColumnKind::Record => None,
        };
        column_info.push(ColumnInfo {
            original_name,
            data_manager_seq,
            shape_column,
        });
    }

    // Read and decode the per-data-manager spec blobs, in manager order.
    for dm in &mut data_managers {
        let blob = r.read_opaque()?.to_vec();
        dm.blob = if dm.type_name == "StandardStMan" {
            DataManagerBlob::StandardStMan(parse_standard_stman(&blob, &dm.type_name)?)
        } else {
            DataManagerBlob::Unsupported(blob)
        };
    }

    Ok(ColumnSet {
        version,
        nrow,
        storage_option,
        seq_count,
        data_managers,
        columns: column_info,
    })
}

/// Decode a StandardStMan `"SSM"` spec blob (`SSMBase::open64`).
pub fn parse_standard_stman(blob: &[u8], type_name: &str) -> Result<StandardStMan, ColumnSetError> {
    let mut r = Reader::new(blob);
    let obj = r.read_object_start(true)?;
    if obj.type_name != "SSM" {
        return Err(ColumnSetError::UnexpectedBlobType {
            type_name: type_name.to_string(),
            found: obj.type_name,
        });
    }
    let data_manager_name = r.read_string()?;
    let column_offset = read_block_u32(&mut r)?;
    let col_index_map = read_block_u32(&mut r)?;
    Ok(StandardStMan {
        data_manager_name,
        column_offset,
        col_index_map,
    })
}

/// Read a framed `"Block"` v1 of `uInt`s (`putBlock`/`getBlock` in
/// `casacore/casa/Containers/BlockIO.tcc`): `u32` element count then data.
fn read_block_u32(r: &mut Reader<'_>) -> Result<Vec<u32>, ColumnSetError> {
    let obj = r.read_object_start(false)?;
    if obj.type_name != "Block" {
        return Err(ColumnSetError::UnexpectedType {
            expected: "Block".into(),
            found: obj.type_name,
        });
    }
    let n = r.read_u32()? as usize;
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(r.read_u32()?);
    }
    Ok(values)
}

/// Write a framed `"Block"` v1 of `uInt`s (`putBlock` in
/// `casacore/casa/Containers/BlockIO.tcc`): `u32` element count then data.
fn write_block_u32(w: &mut crate::aipsio::Writer, values: &[u32]) {
    w.put_object_start("Block", 1);
    w.put_u32(values.len() as u32);
    for v in values {
        w.put_u32(*v);
    }
    w.put_object_end();
}

/// Serialize a StandardStMan spec blob (`SSMBase::flush`): a root `"SSM"`
/// v2 AipsIO stream with the manager name and the column-offset /
/// column-index-map `Block`s. `table.dat` is always canonical big endian,
/// so the blob is too — byte-identical to what casacore writes.
pub fn write_standard_stman(spec: &StandardStMan) -> Vec<u8> {
    let mut w = crate::aipsio::Writer::new();
    w.put_root_object_start("SSM", 2);
    w.put_string(&spec.data_manager_name);
    write_block_u32(&mut w, &spec.column_offset);
    write_block_u32(&mut w, &spec.col_index_map);
    w.put_object_end();
    w.into_bytes()
}

/// Serialize the ColumnSet payload that follows the `TableDesc` in
/// `table.dat` (`ColumnSet::putFile` with `writeTable` set), for the
/// supported layout: a single StandardStMan data manager (sequence 0)
/// owning every column, written in the v2 (u32 row count) form.
pub fn write_column_set(
    w: &mut crate::aipsio::Writer,
    nrow: u64,
    seq_count: u32,
    columns: &[ColumnDesc],
    spec: &StandardStMan,
) {
    w.put_i32(-2); // v2: u32 row count follows
    w.put_u32(u32::try_from(nrow).expect("row count exceeds the u32 range of a v2 ColumnSet"));
    w.put_u32(seq_count);
    w.put_u32(1); // one data manager with columns
    w.put_string("StandardStMan");
    w.put_u32(0); // sequence number
    for desc in columns {
        // PlainColumn::putFile
        w.put_u32(2); // class version
        w.put_string(&desc.name);
        match desc.kind {
            ColumnKind::Scalar(_) | ColumnKind::Record => {
                w.put_u32(1); // ScalarColumnData class version
                w.put_u32(0); // data-manager sequence number
            }
            ColumnKind::Array => {
                unimplemented!("array columns are not writable yet")
            }
        }
    }
    w.put_opaque(&write_standard_stman(spec));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aipsio::MAGIC;
    use crate::record::RecordValue;

    fn framed(type_name: &str, version: u32, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        let length = (4 + 4 + type_name.len() + 4 + payload.len()) as u32;
        buf.extend_from_slice(&length.to_be_bytes());
        buf.extend_from_slice(&(type_name.len() as u32).to_be_bytes());
        buf.extend_from_slice(type_name.as_bytes());
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    fn root_framed(type_name: &str, version: u32, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.extend_from_slice(&framed(type_name, version, payload));
        buf
    }

    fn string_bytes(s: &str) -> Vec<u8> {
        let mut buf = (s.len() as u32).to_be_bytes().to_vec();
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    fn block_bytes(values: &[u32]) -> Vec<u8> {
        let mut payload = (values.len() as u32).to_be_bytes().to_vec();
        for v in values {
            payload.extend_from_slice(&v.to_be_bytes());
        }
        framed("Block", 1, &payload)
    }

    fn ssm_blob(name: &str, offsets: &[u32], index_map: &[u32]) -> Vec<u8> {
        let mut payload = string_bytes(name);
        payload.extend_from_slice(&block_bytes(offsets));
        payload.extend_from_slice(&block_bytes(index_map));
        root_framed("SSM", 2, &payload)
    }

    fn scalar_column(name: &str, seq: u32) -> Vec<u8> {
        let mut payload = 2u32.to_be_bytes().to_vec(); // PlainColumn class version
        payload.extend_from_slice(&string_bytes(name));
        payload.extend_from_slice(&1u32.to_be_bytes()); // subclass version
        payload.extend_from_slice(&seq.to_be_bytes());
        payload
    }

    fn array_column(name: &str, seq: u32, shape_col: Option<&str>) -> Vec<u8> {
        let mut payload = 2u32.to_be_bytes().to_vec();
        payload.extend_from_slice(&string_bytes(name));
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&seq.to_be_bytes());
        match shape_col {
            Some(s) => {
                payload.push(1);
                payload.extend_from_slice(&string_bytes(s));
            }
            None => payload.push(0),
        }
        payload
    }

    fn opaque(bytes: &[u8]) -> Vec<u8> {
        let mut buf = (bytes.len() as u32).to_be_bytes().to_vec();
        buf.extend_from_slice(bytes);
        buf
    }

    /// A `ColumnDesc` shim for the per-column kind dispatch.
    fn desc(name: &str, kind: ColumnKind) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: crate::record::DataType::Int,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: 0,
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind,
        }
    }

    #[test]
    fn parses_version2_column_set() {
        // version -2, u32 nrow, seq_count, nr, one DM, column info, one blob.
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-2i32).to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes()); // nrow
        payload.extend_from_slice(&1u32.to_be_bytes()); // seq_count
        payload.extend_from_slice(&1u32.to_be_bytes()); // nr dms
        payload.extend_from_slice(&string_bytes("StandardStMan"));
        payload.extend_from_slice(&0u32.to_be_bytes()); // seqnr 0
        payload.extend_from_slice(&scalar_column("COL", 0));
        let blob = ssm_blob("StandardStMan", &[0, 4, 36], &[0, 1, 2]);
        payload.extend_from_slice(&opaque(&blob));

        let mut r = Reader::new(&payload);
        let cs = parse_column_set(
            &mut r,
            &[desc("COL", ColumnKind::Scalar(RecordValue::Int(0)))],
        )
        .unwrap();
        assert_eq!(cs.version, 2);
        assert_eq!(cs.nrow, 1);
        assert_eq!(cs.storage_option, None);
        assert_eq!(cs.seq_count, 1);
        assert_eq!(cs.data_managers.len(), 1);
        assert_eq!(cs.data_managers[0].type_name, "StandardStMan");
        assert_eq!(cs.data_managers[0].sequence_nr, 0);
        assert_eq!(
            cs.columns,
            vec![ColumnInfo {
                original_name: "COL".into(),
                data_manager_seq: 0,
                shape_column: None,
            }]
        );
        let dm = cs.data_manager_for(&cs.columns[0]).unwrap();
        let ssm = match &dm.blob {
            DataManagerBlob::StandardStMan(s) => s,
            _ => panic!("expected StandardStMan blob"),
        };
        assert_eq!(ssm.data_manager_name, "StandardStMan");
        assert_eq!(ssm.column_offset, vec![0, 4, 36]);
        assert_eq!(ssm.col_index_map, vec![0, 1, 2]);
        // All bytes consumed.
        assert_eq!(r.position(), payload.len());
    }

    #[test]
    fn parses_version3_column_set_with_storage_option() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-3i32).to_be_bytes());
        payload.extend_from_slice(&5u64.to_be_bytes()); // nrow
        payload.extend_from_slice(&2i32.to_be_bytes()); // opt = SepFile
        payload.extend_from_slice(&(-1i32).to_be_bytes()); // block size
        payload.extend_from_slice(&1u32.to_be_bytes()); // seq_count
        payload.extend_from_slice(&1u32.to_be_bytes()); // nr
        payload.extend_from_slice(&string_bytes("StandardStMan"));
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&scalar_column("A", 0));
        payload.extend_from_slice(&opaque(&ssm_blob("S", &[0, 2], &[0, 0])));

        let mut r = Reader::new(&payload);
        let cs = parse_column_set(
            &mut r,
            &[desc("A", ColumnKind::Scalar(RecordValue::Int(0)))],
        )
        .unwrap();
        assert_eq!(cs.version, 3);
        assert_eq!(cs.nrow, 5);
        assert_eq!(cs.storage_option, Some((2, -1)));
    }

    #[test]
    fn parses_array_column_with_shape_column() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-2i32).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes()); // nrow
        payload.extend_from_slice(&1u32.to_be_bytes()); // seq_count
        payload.extend_from_slice(&1u32.to_be_bytes()); // nr
        payload.extend_from_slice(&string_bytes("StandardStMan"));
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&array_column("ARR", 0, Some("ARR_IDX")));
        payload.extend_from_slice(&opaque(&ssm_blob("S", &[0], &[0])));

        let mut r = Reader::new(&payload);
        let cs = parse_column_set(&mut r, &[desc("ARR", ColumnKind::Array)]).unwrap();
        assert_eq!(
            cs.columns,
            vec![ColumnInfo {
                original_name: "ARR".into(),
                data_manager_seq: 0,
                shape_column: Some("ARR_IDX".into()),
            }]
        );
    }

    #[test]
    fn parses_array_column_without_shape_column() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-2i32).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&string_bytes("StandardStMan"));
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&array_column("ARR", 0, None));
        payload.extend_from_slice(&opaque(&ssm_blob("S", &[0], &[0])));

        let mut r = Reader::new(&payload);
        let cs = parse_column_set(&mut r, &[desc("ARR", ColumnKind::Array)]).unwrap();
        assert_eq!(cs.columns[0].shape_column, None);
    }

    #[test]
    fn leaves_unknown_data_managers_raw() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-2i32).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&string_bytes("IncrementalStMan"));
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&scalar_column("A", 0));
        let blob = b"\xde\xad\xbe\xef";
        payload.extend_from_slice(&opaque(blob));

        let mut r = Reader::new(&payload);
        let cs = parse_column_set(
            &mut r,
            &[desc("A", ColumnKind::Scalar(RecordValue::Int(0)))],
        )
        .unwrap();
        assert_eq!(cs.data_managers[0].type_name, "IncrementalStMan");
        assert_eq!(
            cs.data_managers[0].blob,
            DataManagerBlob::Unsupported(blob.to_vec())
        );
    }

    #[test]
    fn rejects_wrong_plaincolumn_version() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-2i32).to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&string_bytes("StandardStMan"));
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&3u32.to_be_bytes()); // unsupported version
        payload.extend_from_slice(&string_bytes("A"));

        let mut r = Reader::new(&payload);
        assert!(matches!(
            parse_column_set(
                &mut r,
                &[desc("A", ColumnKind::Scalar(RecordValue::Int(0)))]
            ),
            Err(ColumnSetError::UnsupportedColumnVersion(3))
        ));
    }

    #[test]
    fn rejects_non_ssm_standardstman_blob() {
        let blob = root_framed("XXY", 2, b"");
        let err = parse_standard_stman(&blob, "StandardStMan").unwrap_err();
        assert!(matches!(
            err,
            ColumnSetError::UnexpectedBlobType {
                type_name,
                found
            } if type_name == "StandardStMan" && found == "XXY"
        ));
    }
}
