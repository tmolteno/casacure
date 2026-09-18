//! Parsing of the `TableDesc` object inside `table.dat`
//! (`casacore/tables/Tables/TableDesc.cc`, `ColumnDesc.cc`,
//! `BaseColDesc.cc`, `ScaColDesc.tcc`, `ArrColDesc.cc`).
//!
//! Layout: framed `"TableDesc"` v2 object containing name/version/comment
//! strings, the public and private keyword TableRecords, then the
//! ColumnDescSet: a `u32` column count followed by that many column
//! descriptors. Each column descriptor is `u32 classversion`, a class name
//! string (e.g. `"ScalarColumnDesc<Int   "`), the BaseColumnDesc fields,
//! a keyword TableRecord, and class-specific data (scalar default value or
//! array flag).

use crate::aipsio::Reader;
use crate::record::{DataType, RecordError, RecordValue, TableRecord};
use crate::types::ValueType;
use thiserror::Error;

/// Errors from parsing a TableDesc.
#[derive(Debug, Error)]
pub enum TableDescError {
    #[error(transparent)]
    AipsIo(#[from] crate::aipsio::AipsIoError),
    #[error(transparent)]
    Record(#[from] RecordError),
    #[error("unexpected object type {found:?}, expected {expected:?}")]
    UnexpectedType { expected: String, found: String },
    #[error("unknown column description class {0:?}")]
    UnknownColumnClass(String),
}

/// What kind of column a descriptor describes.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnKind {
    /// Scalar column, with its default value.
    Scalar(RecordValue),
    /// Array column (fixed or variable shape).
    Array,
    /// Scalar record column (no default stored).
    Record,
}

/// A parsed column descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDesc {
    pub name: String,
    pub comment: String,
    /// Element data type (scalar DataType, e.g. `DataType::Int`).
    pub data_type: DataType,
    /// Data manager type name, e.g. `"StandardStMan"`.
    pub data_manager_type: String,
    /// Data manager group name.
    pub data_manager_group: String,
    /// `ColumnDesc::Options` bitmask.
    pub options: i32,
    /// Number of dimensions (-1 = scalar or variable).
    pub ndim: i32,
    /// Fixed shape for array columns.
    pub shape: Option<Vec<i64>>,
    pub max_length: i32,
    pub keywords: TableRecord,
    pub kind: ColumnKind,
}

impl ColumnDesc {
    /// The corresponding CASA value type, if the column holds a plain
    /// value type (not a record).
    pub fn value_type(&self) -> Option<ValueType> {
        Some(match self.data_type {
            DataType::Bool => ValueType::Bool,
            DataType::UChar => ValueType::Byte,
            DataType::Short => ValueType::Short,
            DataType::UShort => ValueType::UShort,
            DataType::Int => ValueType::Int,
            DataType::UInt => ValueType::UInt,
            DataType::Float => ValueType::Float,
            DataType::Double => ValueType::Double,
            DataType::Complex => ValueType::Complex,
            DataType::DComplex => ValueType::DComplex,
            DataType::String => ValueType::String,
            _ => return None,
        })
    }
}

/// A parsed table description.
#[derive(Debug, Clone, PartialEq)]
pub struct TableDesc {
    pub name: String,
    pub version: String,
    pub comment: String,
    pub keywords: TableRecord,
    pub private_keywords: TableRecord,
    pub columns: Vec<ColumnDesc>,
}

impl TableDesc {
    pub fn column(&self, name: &str) -> Option<&ColumnDesc> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// Parse a framed `"TableDesc"` object from the stream.
pub fn parse_table_desc(r: &mut Reader<'_>) -> Result<TableDesc, TableDescError> {
    let obj = r.read_object_start(false)?;
    if obj.type_name != "TableDesc" {
        return Err(TableDescError::UnexpectedType {
            expected: "TableDesc".into(),
            found: obj.type_name,
        });
    }
    let name = r.read_string()?;
    let version = r.read_string()?;
    let comment = r.read_string()?;
    let keywords = TableRecord::read(r)?;
    // TableDesc version 1 has no private keyword set.
    let private_keywords = if obj.version != 1 {
        TableRecord::read(r)?
    } else {
        TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        }
    };
    let ncols = r.read_u32()?;
    let mut columns = Vec::with_capacity(ncols as usize);
    for _ in 0..ncols {
        columns.push(parse_column_desc(r)?);
    }
    Ok(TableDesc {
        name,
        version,
        comment,
        keywords,
        private_keywords,
        columns,
    })
}

fn parse_column_desc(r: &mut Reader<'_>) -> Result<ColumnDesc, TableDescError> {
    let _class_version = r.read_u32()?;
    let class_name = r.read_string()?;
    let is_scalar = if class_name.starts_with("ScalarColumnDesc<")
        || class_name.starts_with("ScalarRecordColumnDesc")
    {
        true
    } else if class_name.starts_with("ArrayColumnDesc<") {
        false
    } else {
        return Err(TableDescError::UnknownColumnClass(class_name));
    };

    // BaseColumnDesc fields.
    let _base_version = r.read_u32()?;
    let name = r.read_string()?;
    let comment = r.read_string()?;
    let data_manager_type = r.read_string()?;
    let data_manager_group = r.read_string()?;
    let data_type = DataType::from_i32(r.read_i32()?)?;
    let options = r.read_i32()?;
    let ndim = r.read_i32()?;
    let shape = if is_scalar {
        None
    } else {
        Some(r.read_iposition()?)
    };
    let max_length = r.read_i32()?;
    let keywords = TableRecord::read(r)?;

    // Class-specific data (`putDesc`).
    let kind = if class_name.starts_with("ScalarRecordColumnDesc") {
        let _version = r.read_u32()?;
        ColumnKind::Record
    } else if is_scalar {
        let _version = r.read_u32()?;
        ColumnKind::Scalar(crate::record::read_scalar_value(r, data_type)?)
    } else {
        let _version = r.read_u32()?;
        let _has_default = r.read_bool()?;
        ColumnKind::Array
    };

    Ok(ColumnDesc {
        name,
        comment,
        data_type,
        data_manager_type,
        data_manager_group,
        options,
        ndim,
        shape,
        max_length,
        keywords,
        kind,
    })
}
