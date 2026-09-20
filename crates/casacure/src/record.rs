//! TableRecord / RecordDesc parsing (casacore `casacore/casa/Containers/
//! RecordDesc.cc`, `RecordRep.cc` and `casacore/tables/Tables/
//! TableRecordRep.cc`).
//!
//! A `TableRecord` on disk is a framed `"TableRecord"` v1 object: a framed
//! `"RecordDesc"` v2, an `Int` record type (0 = Fixed, 1 = Variable), then
//! the field values in RecordDesc order. Scalar values are written inline;
//! arrays are framed `"Array<...>"` v3 objects; a sub-record with an empty
//! description is a fully framed nested TableRecord, while one with a
//! non-empty description is written as bare field data; a `TpTable` field is
//! a plain string (the table name).

use crate::aipsio::{AipsIoError, Reader};
use thiserror::Error;

/// casacore `DataType` enum (`casacore/casa/Utilities/DataType.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bool,
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
    Complex,
    DComplex,
    String,
    Table,
    Record,
    ArrayBool,
    ArrayChar,
    ArrayUChar,
    ArrayShort,
    ArrayUShort,
    ArrayInt,
    ArrayUInt,
    ArrayFloat,
    ArrayDouble,
    ArrayComplex,
    ArrayDComplex,
    ArrayString,
    Other,
    Quantity,
    ArrayQuantity,
    Int64,
    ArrayInt64,
}

/// Errors from parsing records and descriptors.
#[derive(Debug, Error)]
pub enum RecordError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error("unknown DataType value {0}")]
    UnknownDataType(i32),
    #[error("unsupported legacy keyword set object {0:?}")]
    LegacyKeywordSet(String),
}

impl DataType {
    pub fn from_i32(v: i32) -> Result<Self, RecordError> {
        use DataType::*;
        Ok(match v {
            0 => Bool,
            1 => Char,
            2 => UChar,
            3 => Short,
            4 => UShort,
            5 => Int,
            6 => UInt,
            7 => Float,
            8 => Double,
            9 => Complex,
            10 => DComplex,
            11 => String,
            12 => Table,
            13 => ArrayBool,
            14 => ArrayChar,
            15 => ArrayUChar,
            16 => ArrayShort,
            17 => ArrayUShort,
            18 => ArrayInt,
            19 => ArrayUInt,
            20 => ArrayFloat,
            21 => ArrayDouble,
            22 => ArrayComplex,
            23 => ArrayDComplex,
            24 => ArrayString,
            25 => Record,
            26 => Other,
            27 => Quantity,
            28 => ArrayQuantity,
            29 => Int64,
            30 => ArrayInt64,
            v => return Err(RecordError::UnknownDataType(v)),
        })
    }

    /// The scalar element type of an array type, or itself for scalars.
    pub fn element_type(self) -> DataType {
        use DataType::*;
        match self {
            ArrayBool => Bool,
            ArrayChar => Char,
            ArrayUChar => UChar,
            ArrayShort => Short,
            ArrayUShort => UShort,
            ArrayInt => Int,
            ArrayUInt => UInt,
            ArrayFloat => Float,
            ArrayDouble => Double,
            ArrayComplex => Complex,
            ArrayDComplex => DComplex,
            ArrayString => String,
            ArrayInt64 => Int64,
            other => other,
        }
    }

    pub fn is_array(self) -> bool {
        use DataType::*;
        matches!(
            self,
            ArrayBool
                | ArrayChar
                | ArrayUChar
                | ArrayShort
                | ArrayUShort
                | ArrayInt
                | ArrayUInt
                | ArrayFloat
                | ArrayDouble
                | ArrayComplex
                | ArrayDComplex
                | ArrayString
                | ArrayQuantity
                | ArrayInt64
        )
    }
}

/// One field of a record description.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordDescField {
    pub name: String,
    pub data_type: DataType,
    /// Sub-record description (TpRecord fields).
    pub sub_desc: Option<RecordDesc>,
    /// Fixed shape (array fields with a set shape; empty when unfixed).
    pub shape: Option<Vec<i64>>,
    /// Referenced table description name (TpTable fields).
    pub table_desc_name: Option<String>,
    pub comment: String,
}

/// A record description (`RecordDesc` v2 on disk).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecordDesc {
    pub fields: Vec<RecordDescField>,
}

impl RecordDesc {
    pub fn field(&self, name: &str) -> Option<&RecordDescField> {
        self.fields.iter().find(|f| f.name == name)
    }

    fn read(r: &mut Reader<'_>) -> Result<RecordDesc, RecordError> {
        let (version, _) = r.read_object(false, "RecordDesc")?;
        let nfields = r.read_i32()?;
        let mut fields = Vec::with_capacity(nfields.max(0) as usize);
        for _ in 0..nfields.max(0) {
            let name = r.read_string()?;
            let data_type = DataType::from_i32(r.read_i32()?)?;
            let mut sub_desc = None;
            let mut shape = None;
            let mut table_desc_name = None;
            match data_type {
                DataType::Record => sub_desc = Some(RecordDesc::read(r)?),
                DataType::Table => table_desc_name = Some(r.read_string()?),
                dt if dt.is_array() => shape = Some(r.read_iposition()?),
                _ => {}
            }
            // Comments were added in RecordDesc version 2.
            let comment = if version > 1 {
                r.read_string()?
            } else {
                String::new()
            };
            fields.push(RecordDescField {
                name,
                data_type,
                sub_desc,
                shape,
                table_desc_name,
                comment,
            });
        }
        Ok(RecordDesc { fields })
    }
}

/// A decoded array value (framed `"Array<...>"` object).
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayValue {
    pub shape: Vec<u32>,
    pub data: ArrayData,
}

impl ArrayValue {
    /// The flat element values in row-major order.
    pub fn elements(&self) -> Vec<RecordValue> {
        let d = &self.data;
        match d {
            ArrayData::Bool(v) => v.iter().map(|b| RecordValue::Bool(*b)).collect(),
            ArrayData::UChar(v) => v.iter().map(|x| RecordValue::UChar(*x)).collect(),
            ArrayData::Short(v) => v.iter().map(|x| RecordValue::Short(*x)).collect(),
            ArrayData::UShort(v) => v.iter().map(|x| RecordValue::UShort(*x)).collect(),
            ArrayData::Int(v) => v.iter().map(|x| RecordValue::Int(*x)).collect(),
            ArrayData::UInt(v) => v.iter().map(|x| RecordValue::UInt(*x)).collect(),
            ArrayData::Int64(v) => v.iter().map(|x| RecordValue::Int64(*x)).collect(),
            ArrayData::Float(v) => v.iter().map(|x| RecordValue::Float(*x)).collect(),
            ArrayData::Double(v) => v.iter().map(|x| RecordValue::Double(*x)).collect(),
            ArrayData::Complex(v) => v
                .iter()
                .map(|(re, im)| RecordValue::Complex(*re, *im))
                .collect(),
            ArrayData::DComplex(v) => v
                .iter()
                .map(|(re, im)| RecordValue::DComplex(*re, *im))
                .collect(),
            ArrayData::String(v) => v.iter().map(|x| RecordValue::String(x.clone())).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    Bool(Vec<bool>),
    UChar(Vec<u8>),
    Short(Vec<i16>),
    UShort(Vec<u16>),
    Int(Vec<i32>),
    UInt(Vec<u32>),
    Int64(Vec<i64>),
    Float(Vec<f32>),
    Double(Vec<f64>),
    /// (real, imag) pairs.
    Complex(Vec<(f32, f32)>),
    DComplex(Vec<(f64, f64)>),
    String(Vec<String>),
}

/// A decoded record field value.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordValue {
    Bool(bool),
    UChar(u8),
    Short(i16),
    UShort(u16),
    Int(i32),
    UInt(u32),
    Int64(i64),
    Float(f32),
    Double(f64),
    Complex(f32, f32),
    DComplex(f64, f64),
    String(String),
    Record(TableRecord),
    /// Name of the referenced table.
    Table(String),
    Array(ArrayValue),
}

/// A decoded TableRecord: description, record type, and values aligned with
/// `desc.fields`.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRecord {
    pub desc: RecordDesc,
    /// 0 = Fixed, 1 = Variable (`RecordInterface::RecordType`).
    pub record_type: i32,
    pub values: Vec<RecordValue>,
}

impl TableRecord {
    /// Read a framed TableRecord object (`TableRecordRep::getRecord`).
    pub fn read(r: &mut Reader<'_>) -> Result<TableRecord, RecordError> {
        let obj = r.read_object_start(false)?;
        if obj.type_name != "TableRecord" {
            // Legacy ScalarKeywordSet/ArrayKeywordSet/TableKeywordSet objects
            // are still accepted by casacore; we don't support them yet.
            return Err(RecordError::LegacyKeywordSet(obj.type_name));
        }
        let desc = RecordDesc::read(r)?;
        let record_type = r.read_i32()?;
        let values = read_record_data(r, &desc)?;
        Ok(TableRecord {
            desc,
            record_type,
            values,
        })
    }

    /// Value of a named field.
    pub fn get(&self, name: &str) -> Option<&RecordValue> {
        let idx = self.desc.fields.iter().position(|f| f.name == name)?;
        self.values.get(idx)
    }
}

/// Read the data fields of a record (no framing), in RecordDesc order
/// (`TableRecordRep::getData`).
fn read_record_data(
    r: &mut Reader<'_>,
    desc: &RecordDesc,
) -> Result<Vec<RecordValue>, RecordError> {
    let mut values = Vec::with_capacity(desc.fields.len());
    for field in &desc.fields {
        values.push(read_field_value(r, field)?);
    }
    Ok(values)
}

fn read_field_value(
    r: &mut Reader<'_>,
    field: &RecordDescField,
) -> Result<RecordValue, RecordError> {
    use DataType::*;
    match field.data_type {
        Table => Ok(RecordValue::Table(r.read_string()?)),
        Record => {
            let sub = field.sub_desc.as_ref().expect("record field has subdesc");
            if sub.fields.is_empty() {
                Ok(RecordValue::Record(TableRecord::read(r)?))
            } else {
                // Non-empty subdesc: bare field data, no framing; the record
                // type is not stored.
                Ok(RecordValue::Record(TableRecord {
                    desc: sub.clone(),
                    record_type: 0,
                    values: read_record_data(r, sub)?,
                }))
            }
        }
        dt if dt.is_array() => Ok(RecordValue::Array(read_array(r, dt.element_type())?)),
        dt => read_scalar_value(r, dt),
    }
}

/// Read a scalar value of the given (non-array) type. Also used for column
/// default values in `ScalarColumnDesc`.
pub(crate) fn read_scalar_value(
    r: &mut Reader<'_>,
    dt: DataType,
) -> Result<RecordValue, RecordError> {
    use DataType::*;
    Ok(match dt {
        Bool => RecordValue::Bool(r.read_bool()?),
        Char | UChar => RecordValue::UChar(r.read_u8()?),
        Short => RecordValue::Short(r.read_i16()?),
        UShort => RecordValue::UShort(r.read_u16()?),
        Int => RecordValue::Int(r.read_i32()?),
        UInt => RecordValue::UInt(r.read_u32()?),
        Int64 => RecordValue::Int64(r.read_i64()?),
        Float => RecordValue::Float(r.read_f32()?),
        Double => RecordValue::Double(r.read_f64()?),
        Complex => RecordValue::Complex(r.read_f32()?, r.read_f32()?),
        DComplex => RecordValue::DComplex(r.read_f64()?, r.read_f64()?),
        String => RecordValue::String(r.read_string()?),
        dt => return Err(RecordError::UnknownDataType(data_type_code(dt))),
    })
}

fn data_type_code(dt: DataType) -> i32 {
    use DataType::*;
    match dt {
        Bool => 0,
        Char => 1,
        UChar => 2,
        Short => 3,
        UShort => 4,
        Int => 5,
        UInt => 6,
        Float => 7,
        Double => 8,
        Complex => 9,
        DComplex => 10,
        String => 11,
        Table => 12,
        ArrayBool => 13,
        ArrayChar => 14,
        ArrayUChar => 15,
        ArrayShort => 16,
        ArrayUShort => 17,
        ArrayInt => 18,
        ArrayUInt => 19,
        ArrayFloat => 20,
        ArrayDouble => 21,
        ArrayComplex => 22,
        ArrayDComplex => 23,
        ArrayString => 24,
        Record => 25,
        Other => 26,
        Quantity => 27,
        ArrayQuantity => 28,
        Int64 => 29,
        ArrayInt64 => 30,
    }
}

/// Read a framed `"Array<...>"` v3 object (`casacore/casa/IO/ArrayIO.tcc`).
fn read_array(r: &mut Reader<'_>, elem: DataType) -> Result<ArrayValue, RecordError> {
    let (version, obj) = {
        let obj = r.read_object_start(false)?;
        (obj.version, obj)
    };
    if !obj.type_name.starts_with("Array") {
        return Err(RecordError::AipsIo(AipsIoError::UnexpectedType {
            offset: obj.payload_offset,
            expected: "Array<...>".to_string(),
            found: obj.type_name,
        }));
    }
    let ndim = r.read_u32()? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(r.read_u32()?);
    }
    if version < 3 {
        // Old versions stored a per-dimension origin (always 1); discard.
        for _ in 0..ndim {
            r.read_u32()?;
        }
    }
    let nelem = r.read_u32()? as usize;
    let data = match elem {
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
        DataType::Complex => ArrayData::Complex(
            (0..nelem)
                .map(|_| Ok((r.read_f32()?, r.read_f32()?)))
                .collect::<Result<_, AipsIoError>>()?,
        ),
        DataType::DComplex => ArrayData::DComplex(
            (0..nelem)
                .map(|_| Ok((r.read_f64()?, r.read_f64()?)))
                .collect::<Result<_, AipsIoError>>()?,
        ),
        DataType::String => ArrayData::String(
            (0..nelem)
                .map(|_| r.read_string())
                .collect::<Result<_, _>>()?,
        ),
        dt => return Err(RecordError::UnknownDataType(data_type_code(dt))),
    };
    Ok(ArrayValue { shape, data })
}

/// Write a scalar value of the given (non-array) type, as produced by
/// `read_scalar_value`. Used for column default values in
/// `ScalarColumnDesc` and for record fields.
pub fn write_scalar_value(w: &mut crate::aipsio::Writer, dt: DataType, value: &RecordValue) {
    match dt {
        DataType::Bool => match value {
            RecordValue::Bool(b) => w.put_bool(*b),
            RecordValue::UChar(u) => w.put_bool(*u != 0),
            RecordValue::Int(i) => w.put_bool(*i != 0),
            _ => w.put_bool(false),
        },
        DataType::Char | DataType::UChar => match value {
            RecordValue::UChar(u) => w.put_u8(*u),
            RecordValue::Bool(b) => w.put_u8(u8::from(*b)),
            RecordValue::Int(i) => w.put_u8(*i as u8),
            RecordValue::Short(i) => w.put_u8(*i as u8),
            _ => w.put_u8(0),
        },
        DataType::Short => match value {
            RecordValue::Short(i) => w.put_i16(*i),
            RecordValue::Int(i) => w.put_i16(*i as i16),
            RecordValue::UShort(u) => w.put_i16(*u as i16),
            _ => w.put_i16(0),
        },
        DataType::UShort => match value {
            RecordValue::UShort(u) => w.put_u16(*u),
            RecordValue::Short(i) => w.put_u16(*i as u16),
            RecordValue::UInt(u) => w.put_u16(*u as u16),
            RecordValue::Int(i) => w.put_u16(*i as u16),
            _ => w.put_u16(0),
        },
        DataType::Int => match value {
            RecordValue::Int(i) => w.put_i32(*i),
            RecordValue::Short(i) => w.put_i32(i32::from(*i)),
            RecordValue::UShort(u) => w.put_i32(i32::from(*u)),
            RecordValue::UChar(u) => w.put_i32(i32::from(*u)),
            RecordValue::Bool(b) => w.put_i32(i32::from(*b)),
            _ => w.put_i32(0),
        },
        DataType::UInt => match value {
            RecordValue::UInt(u) => w.put_u32(*u),
            RecordValue::Int(i) => w.put_u32(*i as u32),
            _ => w.put_u32(0),
        },
        DataType::Int64 => match value {
            RecordValue::Int(i) => w.put_i64(i64::from(*i)),
            RecordValue::Int64(i) => w.put_i64(*i),
            _ => w.put_i64(0),
        },
        DataType::Float => match value {
            RecordValue::Float(f) => w.put_f32(*f),
            RecordValue::Double(d) => w.put_f32(*d as f32),
            _ => w.put_f32(0.0),
        },
        DataType::Double => match value {
            RecordValue::Double(d) => w.put_f64(*d),
            RecordValue::Float(f) => w.put_f64(*f as f64),
            _ => w.put_f64(0.0),
        },
        DataType::Complex => match value {
            RecordValue::Complex(re, im) => {
                w.put_f32(*re);
                w.put_f32(*im);
            }
            RecordValue::DComplex(re, im) => {
                w.put_f32(*re as f32);
                w.put_f32(*im as f32);
            }
            _ => {
                w.put_f32(0.0);
                w.put_f32(0.0);
            }
        },
        DataType::DComplex => match value {
            RecordValue::DComplex(re, im) => {
                w.put_f64(*re);
                w.put_f64(*im);
            }
            RecordValue::Complex(re, im) => {
                w.put_f64(*re as f64);
                w.put_f64(*im as f64);
            }
            _ => {
                w.put_f64(0.0);
                w.put_f64(0.0);
            }
        },
        DataType::String | DataType::Table => match value {
            RecordValue::String(s) | RecordValue::Table(s) => w.put_string(s),
            _ => w.put_string(""),
        },
        _ => {}
    }
}

/// Serialize a `TableRecord` (`TableRecordRep::putRecord`).
///
/// Currently supports the field types this project writes: scalars, strings,
/// and nested records. Array fields are rejected.
pub(crate) fn write_table_record(
    w: &mut crate::aipsio::Writer,
    record: &TableRecord,
) -> Result<(), RecordError> {
    w.put_object_start("TableRecord", 1);
    w.put_object_start("RecordDesc", 2);
    w.put_i32(record.desc.fields.len() as i32);
    for field in &record.desc.fields {
        w.put_string(&field.name);
        w.put_i32(data_type_code(field.data_type));
        match field.data_type {
            DataType::Record => {
                let sub = field
                    .sub_desc
                    .as_ref()
                    .ok_or(RecordError::LegacyKeywordSet("missing sub-desc".into()))?;
                write_record_desc(w, sub)?;
            }
            DataType::Table => w.put_string(field.table_desc_name.as_deref().unwrap_or("")),
            dt if dt.is_array() => write_iposition(w, field.shape.as_deref().unwrap_or(&[])),
            _ => {}
        }
        w.put_string(&field.comment);
    }
    w.put_object_end(); // RecordDesc
    w.put_i32(record.record_type);
    for field in &record.desc.fields {
        match field.data_type {
            dt if dt.is_array() => {
                let Some(RecordValue::Array(a)) = record.get(&field.name) else {
                    return Err(RecordError::LegacyKeywordSet(format!(
                        "missing array value for {}",
                        field.name
                    )));
                };
                write_array_value(w, &a.shape, &a.data);
            }
            DataType::Record => {
                let Some(RecordValue::Record(sub)) = record.get(&field.name) else {
                    return Err(RecordError::LegacyKeywordSet("missing record value".into()));
                };
                // Framed iff the *field's* sub-descriptor is empty (casacore
                // frames nested keyword records whose header sub-desc is
                // empty, regardless of the value's own desc).
                let field_empty = field.sub_desc.as_ref().is_none_or(|s| s.fields.is_empty());
                if field_empty {
                    write_table_record(w, sub)?;
                } else {
                    write_record_data_values(w, sub)?;
                }
            }
            _ => {
                let value = record
                    .get(&field.name)
                    .ok_or_else(|| RecordError::LegacyKeywordSet("missing field value".into()))?;
                write_scalar_value(w, field.data_type, value);
            }
        }
    }
    w.put_object_end(); // TableRecord
    Ok(())
}

/// Serialize a framed `"Array<...>"` v3 object (casacore `ArrayIO`),
/// the inverse of `read_array`.
pub fn write_array_value(w: &mut crate::aipsio::Writer, shape: &[u32], data: &ArrayData) {
    let nelem = match data {
        ArrayData::Bool(v) => v.len(),
        ArrayData::UChar(v) => v.len(),
        ArrayData::Short(v) => v.len(),
        ArrayData::UShort(v) => v.len(),
        ArrayData::Int(v) => v.len(),
        ArrayData::UInt(v) => v.len(),
        ArrayData::Int64(v) => v.len(),
        ArrayData::Float(v) => v.len(),
        ArrayData::Double(v) => v.len(),
        ArrayData::Complex(v) => v.len(),
        ArrayData::DComplex(v) => v.len(),
        ArrayData::String(v) => v.len(),
    };
    w.put_object_start("Array<void>", 3);
    w.put_u32(shape.len() as u32);
    for d in shape {
        w.put_u32(*d);
    }
    w.put_u32(nelem as u32);
    match data {
        ArrayData::Bool(v) => {
            // Bit-packed, LSB first.
            let mut packed = vec![0u8; v.len().div_ceil(8)];
            for (i, b) in v.iter().enumerate() {
                if *b {
                    packed[i / 8] |= 1 << (i % 8);
                }
            }
            for byte in packed {
                w.put_u8(byte);
            }
        }
        ArrayData::UChar(v) => {
            for x in v {
                w.put_u8(*x);
            }
        }
        ArrayData::Short(v) => {
            for x in v {
                w.put_i16(*x);
            }
        }
        ArrayData::UShort(v) => {
            for x in v {
                w.put_u16(*x);
            }
        }
        ArrayData::Int(v) => {
            for x in v {
                w.put_i32(*x);
            }
        }
        ArrayData::UInt(v) => {
            for x in v {
                w.put_u32(*x);
            }
        }
        ArrayData::Int64(v) => {
            for x in v {
                w.put_i64(*x);
            }
        }
        ArrayData::Float(v) => {
            for x in v {
                w.put_f32(*x);
            }
        }
        ArrayData::Double(v) => {
            for x in v {
                w.put_f64(*x);
            }
        }
        ArrayData::Complex(v) => {
            for (re, im) in v {
                w.put_f32(*re);
                w.put_f32(*im);
            }
        }
        ArrayData::DComplex(v) => {
            for (re, im) in v {
                w.put_f64(*re);
                w.put_f64(*im);
            }
        }
        ArrayData::String(v) => {
            for s in v {
                w.put_string(s);
            }
        }
    }
    w.put_object_end();
}

/// Write just the record-description object (for nested sub-records).
fn write_record_desc(w: &mut crate::aipsio::Writer, desc: &RecordDesc) -> Result<(), RecordError> {
    w.put_object_start("RecordDesc", 2);
    w.put_i32(desc.fields.len() as i32);
    for field in &desc.fields {
        w.put_string(&field.name);
        w.put_i32(data_type_code(field.data_type));
        match field.data_type {
            DataType::Record => {
                write_record_desc(w, field.sub_desc.as_ref().unwrap())?;
            }
            DataType::Table => w.put_string(field.table_desc_name.as_deref().unwrap_or("")),
            dt if dt.is_array() => write_iposition(w, field.shape.as_deref().unwrap_or(&[])),
            _ => {}
        }
        w.put_string(&field.comment);
    }
    w.put_object_end();
    Ok(())
}

/// Write bare field values of a nested record with a non-empty description,
/// recursing into nested records (casacore `RecordRep::putData`).
fn write_record_data_values(
    w: &mut crate::aipsio::Writer,
    record: &TableRecord,
) -> Result<(), RecordError> {
    for field in &record.desc.fields {
        let value = record
            .get(&field.name)
            .ok_or_else(|| RecordError::LegacyKeywordSet("missing nested value".into()))?;
        match value {
            RecordValue::Record(sub) => {
                let field_empty = field.sub_desc.as_ref().is_none_or(|s| s.fields.is_empty());
                if field_empty {
                    write_table_record(w, sub)?;
                } else {
                    write_record_data_values(w, sub)?;
                }
            }
            RecordValue::Array(a) if field.data_type.is_array() => {
                write_array_value(w, &a.shape, &a.data);
            }
            _ => write_scalar_value(w, field.data_type, value),
        }
    }
    Ok(())
}

/// The CASA `DataType` that best represents a constructed `RecordValue`
/// (used when adding keyword fields).
pub fn infer_data_type(value: &RecordValue) -> DataType {
    use DataType::*;
    match value {
        RecordValue::Bool(_) => Bool,
        RecordValue::UChar(_) => UChar,
        RecordValue::Short(_) => Short,
        RecordValue::UShort(_) => UShort,
        RecordValue::Int(_) => Int,
        RecordValue::UInt(_) => UInt,
        RecordValue::Int64(_) => Int64,
        RecordValue::Float(_) => Float,
        RecordValue::Double(_) => Double,
        RecordValue::Complex(_, _) => Complex,
        RecordValue::DComplex(_, _) => DComplex,
        RecordValue::String(_) => String,
        RecordValue::Table(_) => Table,
        RecordValue::Record(_) => Record,
        RecordValue::Array(a) => match &a.data {
            ArrayData::Bool(_) => ArrayBool,
            ArrayData::UChar(_) => ArrayUChar,
            ArrayData::Short(_) => ArrayShort,
            ArrayData::UShort(_) => ArrayUShort,
            ArrayData::Int(_) => ArrayInt,
            ArrayData::UInt(_) => ArrayUInt,
            ArrayData::Int64(_) => ArrayInt64,
            ArrayData::Float(_) => ArrayFloat,
            ArrayData::Double(_) => ArrayDouble,
            ArrayData::Complex(_) => ArrayComplex,
            ArrayData::DComplex(_) => ArrayDComplex,
            ArrayData::String(_) => ArrayString,
        },
    }
}

impl TableRecord {
    /// Set (or append) a keyword field, inferring its data type from the
    /// value (`putkeyword`/`putcolkeyword`).
    pub fn set(&mut self, name: &str, value: RecordValue) {
        let dt = infer_data_type(&value);
        if let Some(i) = self.desc.fields.iter().position(|f| f.name == name) {
            self.desc.fields[i].data_type = dt;
            if let RecordValue::Record(sub) = &value {
                self.desc.fields[i].sub_desc = Some(sub.desc.clone());
            }
            self.values[i] = value;
        } else {
            let sub_desc = match &value {
                RecordValue::Record(sub) => Some(sub.desc.clone()),
                _ => None,
            };
            self.desc.fields.push(RecordDescField {
                name: name.to_string(),
                data_type: dt,
                sub_desc,
                shape: None,
                table_desc_name: None,
                comment: String::new(),
            });
            self.values.push(value);
        }
    }

    /// Remove a field by name, if present (`removekeyword`).
    pub fn remove(&mut self, name: &str) {
        if let Some(i) = self.desc.fields.iter().position(|f| f.name == name) {
            self.desc.fields.remove(i);
            self.values.remove(i);
        }
    }
}

/// Serialize an `IPosition` (framed `"IPosition"` v1).
fn write_iposition(w: &mut crate::aipsio::Writer, dims: &[i64]) {
    w.put_object_start("IPosition", 1);
    w.put_u32(dims.len() as u32);
    for d in dims {
        w.put_i32(*d as i32);
    }
    w.put_object_end();
}

/// JSON serialization for the metadata API (keywords, column descriptors).
/// Numbers use Rust's `{}` formatting; floats are emitted as e.g. `1.5`.
impl RecordValue {
    /// The value as a JSON fragment (matching the dict values python-casacore
    /// returns: bools, integers, floats, strings, nested records, arrays).
    pub fn to_json_string(&self) -> String {
        self.to_json_string_ctx(None)
    }

    /// Like `to_json_string`, but resolves subtable (`TpTable`) fields to the
    /// `"Table: <path>"` string python-casacore returns, joining a relative
    /// stored path against `base` (the directory containing the parent
    /// table).
    pub fn to_json_string_ctx(&self, base: Option<&std::path::Path>) -> String {
        match self {
            RecordValue::Bool(b) => b.to_string(),
            RecordValue::UChar(u) => u.to_string(),
            RecordValue::Short(i) => i.to_string(),
            RecordValue::UShort(u) => u.to_string(),
            RecordValue::Int(i) => i.to_string(),
            RecordValue::UInt(u) => u.to_string(),
            RecordValue::Int64(i) => i.to_string(),
            RecordValue::Float(f) => format_float(*f),
            RecordValue::Double(d) => format_float(*d),
            RecordValue::Complex(re, im) => {
                format!("[{}, {}]", format_float(*re), format_float(*im))
            }
            RecordValue::DComplex(re, im) => {
                format!("[{}, {}]", format_float(*re), format_float(*im))
            }
            RecordValue::String(s) => json_string(s),
            RecordValue::Table(name) => match base {
                Some(b) => json_string(&format!("Table: {}", resolve_subtable(name, b))),
                None => json_string(name),
            },
            RecordValue::Record(r) => r.to_json_string_ctx(base),
            RecordValue::Array(a) => {
                let mut s = String::from("{\"shape\":[");
                for (i, d) in a.shape.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&d.to_string());
                }
                s.push_str("],\"array\":[");
                let mut first = true;
                for v in array_data_flat(&a.data).iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    s.push_str(v);
                }
                s.push_str("]}");
                s
            }
        }
    }
}

/// Resolve a stored subtable name against the directory containing the
/// parent table, lexically (no filesystem access, matching casacore's
/// dynamic resolution).
fn resolve_subtable(name: &str, base: &std::path::Path) -> String {
    let path = std::path::Path::new(name);
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    lexical_normalize(&joined).display().to_string()
}

/// Strip `.` components and resolve `..` without touching the filesystem.
pub fn lexical_normalize(path: &std::path::Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The flat element JSON fragments of an `ArrayData`.
pub fn array_data_flat(data: &ArrayData) -> Vec<String> {
    match data {
        ArrayData::Bool(v) => v.iter().map(|b| b.to_string()).collect(),
        ArrayData::UChar(v) => v.iter().map(|x| x.to_string()).collect(),
        ArrayData::Short(v) => v.iter().map(|x| x.to_string()).collect(),
        ArrayData::UShort(v) => v.iter().map(|x| x.to_string()).collect(),
        ArrayData::Int(v) => v.iter().map(|x| x.to_string()).collect(),
        ArrayData::UInt(v) => v.iter().map(|x| x.to_string()).collect(),
        ArrayData::Int64(v) => v.iter().map(|x| x.to_string()).collect(),
        ArrayData::Float(v) => v.iter().map(|x| format_float(*x)).collect(),
        ArrayData::Double(v) => v.iter().map(|x| format_float(*x)).collect(),
        ArrayData::Complex(v) => v
            .iter()
            .map(|(re, im)| format!("[{}, {}]", format_float(*re), format_float(*im)))
            .collect(),
        ArrayData::DComplex(v) => v
            .iter()
            .map(|(re, im)| format!("[{}, {}]", format_float(*re), format_float(*im)))
            .collect(),
        ArrayData::String(v) => v.iter().map(|x| json_string(x)).collect(),
    }
}

fn format_float<F: Into<f64>>(f: F) -> String {
    format!("{}", f.into())
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

impl TableRecord {
    /// The record as a JSON object `{"key": value, ...}` in field order.
    pub fn to_json_string(&self) -> String {
        self.to_json_string_ctx(None)
    }

    /// Like `to_json_string`. With a `base` (the directory containing the
    /// parent table), subtable `TpTable` fields are rendered as the
    /// `"Table: <resolved-path>"` string python-casacore returns.
    pub fn to_json_string_ctx(&self, base: Option<&std::path::Path>) -> String {
        let mut s = String::from("{");
        for (i, (field, value)) in self.desc.fields.iter().zip(self.values.iter()).enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_string(&field.name));
            s.push(':');
            s.push_str(&value.to_json_string_ctx(base));
        }
        s.push('}');
        s
    }
}
/// Parse a JSON object into a `TableRecord` (`{"name": value, ...}`).
/// This is the inverse of `TableRecord::to_json_string`: it reconstructs the
/// typed keyword/descriptor structures from the dict representation
/// python-casacore produces (used for the vendored MS schemas and for
/// accepting dict inputs in the bindings layer).
pub fn parse_json_record(input: &str) -> Result<TableRecord, RecordError> {
    let mut p = JsonParser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let (desc, values) = p.parse_object()?;
    p.skip_ws();
    if p.pos < p.bytes.len() {
        return Err(RecordError::LegacyKeywordSet(
            "trailing JSON after object".into(),
        ));
    }
    Ok(TableRecord {
        desc,
        record_type: 0,
        values,
    })
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && (self.bytes[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), RecordError> {
        let got = self.peek().ok_or_else(|| {
            RecordError::LegacyKeywordSet(format!("JSON: expected `{}`, found end", c as char))
        })?;
        if got != c {
            return Err(RecordError::LegacyKeywordSet(format!(
                "JSON: expected `{}`, found `{}`",
                c as char, got as char
            )));
        }
        self.pos += 1;
        Ok(())
    }

    fn parse_object(&mut self) -> Result<(RecordDesc, Vec<RecordValue>), RecordError> {
        self.expect(b'{')?;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok((RecordDesc { fields: Vec::new() }, Vec::new()));
        }
        let mut desc = RecordDesc { fields: Vec::new() };
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            let name = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let (dt, sub_desc, value) = self.parse_value()?;
            desc.fields.push(RecordDescField {
                name,
                data_type: dt,
                sub_desc,
                shape: None,
                table_desc_name: None,
                comment: String::new(),
            });
            values.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                other => {
                    return Err(RecordError::LegacyKeywordSet(format!(
                        "JSON: expected `,` or `}}` in object, found {other:?}"
                    )));
                }
            }
        }
        Ok((desc, values))
    }

    fn parse_value(&mut self) -> Result<(DataType, Option<RecordDesc>, RecordValue), RecordError> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                let s = self.parse_string()?;
                Ok((DataType::String, None, RecordValue::String(s)))
            }
            Some(b'{') => {
                let (sub_desc, values) = self.parse_object()?;
                let rec = TableRecord {
                    desc: sub_desc.clone(),
                    record_type: 0,
                    values,
                };
                Ok((DataType::Record, Some(sub_desc), RecordValue::Record(rec)))
            }
            Some(b'[') => self.parse_array(),
            Some(b't') => {
                self.expect_str("true")?;
                Ok((DataType::Bool, None, RecordValue::Bool(true)))
            }
            Some(b'f') => {
                self.expect_str("false")?;
                Ok((DataType::Bool, None, RecordValue::Bool(false)))
            }
            Some(b'n') => {
                self.expect_str("null")?;
                Ok((DataType::String, None, RecordValue::String(String::new())))
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            other => Err(RecordError::LegacyKeywordSet(format!(
                "JSON: unexpected value start {other:?}"
            ))),
        }
    }

    fn parse_array(&mut self) -> Result<(DataType, Option<RecordDesc>, RecordValue), RecordError> {
        self.expect(b'[')?;
        self.skip_ws();
        let mut items = Vec::new();
        let mut kind: Option<u8> = None;
        loop {
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                break;
            }
            let (_dt, _sub, v) = self.parse_value()?;
            let item_kind = match &v {
                RecordValue::Bool(_) => 4u8,
                RecordValue::Int(_) | RecordValue::Int64(_) => 1u8,
                RecordValue::Double(_) | RecordValue::Float(_) => 2u8,
                RecordValue::String(_) => 3u8,
                _ => 0u8,
            };
            if let Some(k) = kind {
                if k != item_kind {
                    return Err(RecordError::LegacyKeywordSet(
                        "JSON: mixed-type array in keyword".into(),
                    ));
                }
            } else {
                kind = Some(item_kind);
            }
            items.push(v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                other => {
                    return Err(RecordError::LegacyKeywordSet(format!(
                        "JSON: expected `,` or `]` in array, found {other:?}"
                    )));
                }
            }
        }
        let (dt, data) = match kind {
            Some(1) => {
                let vals: Vec<i64> = items
                    .iter()
                    .map(|v| match v {
                        RecordValue::Int(i) => i64::from(*i),
                        RecordValue::Int64(i) => *i,
                        _ => unreachable!(),
                    })
                    .collect();
                if vals.iter().all(|&i| i32::try_from(i).is_ok()) {
                    (
                        DataType::ArrayInt,
                        ArrayData::Int(vals.iter().map(|&i| i as i32).collect()),
                    )
                } else {
                    (DataType::ArrayInt64, ArrayData::Int64(vals))
                }
            }
            Some(2) => (
                DataType::ArrayDouble,
                ArrayData::Double(
                    items
                        .iter()
                        .map(|v| match v {
                            RecordValue::Double(d) => *d,
                            RecordValue::Float(f) => f64::from(*f),
                            _ => unreachable!(),
                        })
                        .collect(),
                ),
            ),
            Some(3) => (
                DataType::ArrayString,
                ArrayData::String(
                    items
                        .iter()
                        .map(|v| match v {
                            RecordValue::String(s) => s.clone(),
                            _ => unreachable!(),
                        })
                        .collect(),
                ),
            ),
            Some(4) => (
                DataType::ArrayBool,
                ArrayData::Bool(
                    items
                        .iter()
                        .map(|v| match v {
                            RecordValue::Bool(b) => *b,
                            _ => unreachable!(),
                        })
                        .collect(),
                ),
            ),
            _ => (DataType::ArrayString, ArrayData::String(Vec::new())),
        };
        Ok((
            dt,
            None,
            RecordValue::Array(ArrayValue {
                shape: vec![items.len() as u32],
                data,
            }),
        ))
    }

    fn parse_number(&mut self) -> Result<(DataType, Option<RecordDesc>, RecordValue), RecordError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.pos < self.bytes.len()
            && ((self.bytes[self.pos] as char).is_ascii_digit()
                || matches!(self.bytes[self.pos], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| RecordError::LegacyKeywordSet("JSON: bad number".into()))?;
        if text.contains('.') || text.contains('e') || text.contains('E') {
            let f: f64 = text
                .parse()
                .map_err(|_| RecordError::LegacyKeywordSet("JSON: bad float".into()))?;
            Ok((DataType::Double, None, RecordValue::Double(f)))
        } else {
            let i: i64 = text
                .parse()
                .map_err(|_| RecordError::LegacyKeywordSet("JSON: bad int".into()))?;
            if let Ok(i32v) = i32::try_from(i) {
                Ok((DataType::Int, None, RecordValue::Int(i32v)))
            } else {
                Ok((DataType::Int64, None, RecordValue::Int64(i)))
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, RecordError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| RecordError::LegacyKeywordSet("JSON: unterminated string".into()))?;
            match c {
                b'"' => {
                    self.pos += 1;
                    return Ok(s);
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self
                        .peek()
                        .ok_or_else(|| RecordError::LegacyKeywordSet("JSON: bad escape".into()))?;
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            if self.pos + 4 > self.bytes.len() {
                                return Err(RecordError::LegacyKeywordSet(
                                    "JSON: bad \\u escape".into(),
                                ));
                            }
                            let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                .map_err(|_| {
                                    RecordError::LegacyKeywordSet("JSON: bad \\u".into())
                                })?;
                            let code = u32::from_str_radix(hex, 16).map_err(|_| {
                                RecordError::LegacyKeywordSet("JSON: bad \\u hex".into())
                            })?;
                            if let Some(ch) = char::from_u32(code) {
                                s.push(ch);
                            }
                            self.pos += 4;
                        }
                        _ => {
                            return Err(RecordError::LegacyKeywordSet(
                                "JSON: unknown escape".into(),
                            ));
                        }
                    }
                }
                b'\n' | b'\r' => {
                    return Err(RecordError::LegacyKeywordSet(
                        "JSON: newline in string".into(),
                    ));
                }
                _ => {
                    let ch = c as char;
                    if ch.is_control() {
                        return Err(RecordError::LegacyKeywordSet(
                            "JSON: unescaped control char in string".into(),
                        ));
                    }
                    s.push(ch);
                    self.pos += 1;
                }
            }
        }
    }

    fn expect_str(&mut self, s: &str) -> Result<(), RecordError> {
        if self.bytes[self.pos..].starts_with(s.as_bytes()) {
            self.pos += s.len();
            Ok(())
        } else {
            Err(RecordError::LegacyKeywordSet(format!(
                "JSON: expected `{s}`"
            )))
        }
    }
}

/// Map a CASA `DataType` to the numpy dtype string casacore would coerce an
/// incoming numeric array to (None for string/record columns).
pub fn data_type_to_np(dt: Option<&DataType>) -> Option<&'static str> {
    use DataType::*;
    match dt? {
        Bool => Some("bool"),
        UChar => Some("uint8"),
        Short => Some("int16"),
        UShort => Some("uint16"),
        Int => Some("int32"),
        UInt => Some("uint32"),
        Int64 => Some("int64"),
        Float => Some("float32"),
        Double => Some("float64"),
        Complex => Some("complex64"),
        DComplex => Some("complex128"),
        _ => None,
    }
}

/// The numpy dtype string matching a `(kind, itemsize)` pair from numpy's
/// `dtype.kind` / `dtype.itemsize`.
pub fn np_kind_itemsize(kind: &str, itemsize: i64) -> Option<&'static str> {
    match (kind, itemsize) {
        ("b", _) => Some("bool"),
        ("u", 1) => Some("uint8"),
        ("u", 2) => Some("uint16"),
        ("u", 4) => Some("uint32"),
        ("u", 8) => Some("uint64"),
        ("i", 1) => Some("int8"),
        ("i", 2) => Some("int16"),
        ("i", 4) => Some("int32"),
        ("i", 8) => Some("int64"),
        ("f", 4) => Some("float32"),
        ("f", 8) => Some("float64"),
        ("c", 8) => Some("complex64"),
        ("c", 16) => Some("complex128"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aipsio::Writer;

    fn put_object(buf: &mut Vec<u8>, type_name: &str, version: u32, payload: &[u8]) {
        let length = (4 + 4 + type_name.len() + 4 + payload.len()) as u32;
        buf.extend_from_slice(&length.to_be_bytes());
        buf.extend_from_slice(&(type_name.len() as u32).to_be_bytes());
        buf.extend_from_slice(type_name.as_bytes());
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(payload);
    }

    fn put_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn put_iposition(buf: &mut Vec<u8>, dims: &[i32]) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(dims.len() as u32).to_be_bytes());
        for d in dims {
            payload.extend_from_slice(&d.to_be_bytes());
        }
        put_object(buf, "IPosition", 1, &payload);
    }

    /// RecordDesc v2 with one scalar Int field and one ArrayString field.
    fn sample_desc() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2i32.to_be_bytes()); // nfields
        put_string(&mut payload, "count");
        payload.extend_from_slice(&5i32.to_be_bytes()); // TpInt
        put_string(&mut payload, "a comment");
        put_string(&mut payload, "names");
        payload.extend_from_slice(&24i32.to_be_bytes()); // TpArrayString
        put_iposition(&mut payload, &[-1]);
        put_string(&mut payload, "");
        let mut buf = Vec::new();
        put_object(&mut buf, "RecordDesc", 2, &payload);
        buf
    }

    #[test]
    fn parses_record_desc() {
        let buf = sample_desc();
        let mut r = Reader::new(&buf);
        let desc = RecordDesc::read(&mut r).unwrap();
        assert_eq!(desc.fields.len(), 2);
        assert_eq!(desc.fields[0].name, "count");
        assert_eq!(desc.fields[0].data_type, DataType::Int);
        assert_eq!(desc.fields[0].comment, "a comment");
        assert_eq!(desc.fields[1].data_type, DataType::ArrayString);
        assert_eq!(desc.fields[1].shape, Some(vec![-1]));
    }

    #[test]
    fn parses_table_record_with_scalar_and_array() {
        // TableRecord v1: desc + recordType + data (Int 42, string array).
        let mut payload = sample_desc();
        payload.extend_from_slice(&1i32.to_be_bytes()); // recordType Variable
        payload.extend_from_slice(&42i32.to_be_bytes());
        // Array<String> v3: ndim 1, shape [2], nelem 2, "foo", "bar"
        let mut arr = Vec::new();
        arr.extend_from_slice(&1u32.to_be_bytes());
        arr.extend_from_slice(&2u32.to_be_bytes());
        arr.extend_from_slice(&2u32.to_be_bytes());
        put_string(&mut arr, "foo");
        put_string(&mut arr, "bar");
        put_object(&mut payload, "Array<String>", 3, &arr);
        let mut buf = Vec::new();
        put_object(&mut buf, "TableRecord", 1, &payload);

        let mut r = Reader::new(&buf);
        let rec = TableRecord::read(&mut r).unwrap();
        assert_eq!(rec.record_type, 1);
        assert_eq!(rec.get("count"), Some(&RecordValue::Int(42)));
        assert_eq!(
            rec.get("names"),
            Some(&RecordValue::Array(ArrayValue {
                shape: vec![2],
                data: ArrayData::String(vec!["foo".into(), "bar".into()]),
            }))
        );
    }

    #[test]
    fn parses_bit_packed_bool_array() {
        let mut arr = Vec::new();
        arr.extend_from_slice(&1u32.to_be_bytes()); // ndim
        arr.extend_from_slice(&10u32.to_be_bytes()); // shape
        arr.extend_from_slice(&10u32.to_be_bytes()); // nelem
                                                     // bits: elements 0,1,9 true -> byte0 = 0b00000011, byte1 = 0b00000010
        arr.extend_from_slice(&[0b0000_0011, 0b0000_0010]);
        let mut buf = Vec::new();
        put_object(&mut buf, "Array<void>", 3, &arr);
        let mut r = Reader::new(&buf);
        let v = read_array(&mut r, DataType::Bool).unwrap();
        let ArrayData::Bool(vals) = v.data else {
            panic!("expected bool array")
        };
        let expected: Vec<bool> = [
            true, true, false, false, false, false, false, false, false, true,
        ]
        .into();
        assert_eq!(vals, expected);
    }

    #[test]
    fn json_parser_round_trip() {
        // Keyword/descriptor dicts: scalars, strings, floats, arrays, and
        // nested records round-trip through the JSON parser.
        let json = r#"{"VER":"1.0","MAXROWS":1000,"MS_VERSION":2.0,"QS":["s","u"],"EMPTY":{},"NEST":{"HH":{"II":5},"S":"x"},"FLAG":true,"AR":[1,2,3]}"#;
        let rec = parse_json_record(json).unwrap();
        assert_eq!(
            rec.to_json_string(),
            r#"{"VER":"1.0","MAXROWS":1000,"MS_VERSION":2,"QS":{"shape":[2],"array":["s","u"]},"EMPTY":{},"NEST":{"HH":{"II":5},"S":"x"},"FLAG":true,"AR":{"shape":[3],"array":[1,2,3]}}"#
        );
        assert_eq!(rec.get("MAXROWS"), Some(&RecordValue::Int(1000)));
        assert_eq!(rec.get("MS_VERSION"), Some(&RecordValue::Double(2.0)));
        assert_eq!(rec.get("FLAG"), Some(&RecordValue::Bool(true)));
        match rec.get("QS") {
            Some(RecordValue::Array(a)) => {
                assert_eq!(a.data, ArrayData::String(vec!["s".into(), "u".into()]));
            }
            other => panic!("QS: {other:?}"),
        }
        match rec.get("NEST") {
            Some(RecordValue::Record(r)) => {
                assert_eq!(r.to_json_string(), r#"{"HH":{"II":5},"S":"x"}"#);
            }
            other => panic!("NEST: {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_data_type() {
        assert!(matches!(
            DataType::from_i32(99),
            Err(RecordError::UnknownDataType(99))
        ));
    }

    /// A representative scalar value for every scalar `DataType`.
    fn scalar_value(dt: DataType) -> RecordValue {
        use RecordValue::*;
        match dt {
            DataType::Bool => Bool(true),
            DataType::UChar => UChar(200),
            DataType::Short => Short(-1_234),
            DataType::UShort => UShort(60_000),
            DataType::Int => Int(-1_234_567),
            DataType::UInt => UInt(4_000_000_000),
            DataType::Int64 => Int64(-9_000_000_000_000),
            DataType::Float => Float(1.25),
            DataType::Double => Double(1.25e300),
            DataType::Complex => Complex(1.5, -2.5),
            DataType::DComplex => DComplex(1.5e300, -2.5e300),
            DataType::String => String("hello".into()),
            other => panic!("{other:?} is not a scalar type"),
        }
    }

    #[test]
    fn scalar_write_read_round_trips_both_endians() {
        for &le in &[false, true] {
            for dt in [
                DataType::Bool,
                DataType::UChar,
                DataType::Short,
                DataType::UShort,
                DataType::Int,
                DataType::UInt,
                DataType::Int64,
                DataType::Float,
                DataType::Double,
                DataType::Complex,
                DataType::DComplex,
                DataType::String,
            ] {
                let value = scalar_value(dt);
                let mut w = if le { Writer::new_le() } else { Writer::new() };
                write_scalar_value(&mut w, dt, &value);
                let bytes = w.into_bytes();
                let mut r = if le {
                    Reader::new_le(&bytes)
                } else {
                    Reader::new(&bytes)
                };
                let back = read_scalar_value(&mut r, dt).unwrap();
                assert_eq!(back, value, "le={le} dt={dt:?}");
            }
        }
    }

    /// The codec writes an IPosition even for unfixed array fields, so `set`'s
    /// shape None comes back as Some([]); normalize to the canonical on-disk
    /// form so the round-trip assertion below is an identity check.
    fn normalize_array_shapes(rec: &mut TableRecord) {
        for (f, v) in rec.desc.fields.iter_mut().zip(rec.values.iter_mut()) {
            if f.data_type.is_array() && f.shape.is_none() {
                f.shape = Some(Vec::new());
            }
            if let RecordValue::Record(sub) = v {
                normalize_array_shapes(sub);
            }
        }
    }

    /// Scalar, array, and nested-record fields, built via `set`. Array
    /// fields are normalized to the canonical on-disk form (`Some([])` for
    /// unfixed) *before* `.set` clones a record's desc into the parent's
    /// sub_desc, so the round trip is an identity.
    fn sample_record() -> TableRecord {
        let mut r = TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        };
        r.set("count", RecordValue::Int(42));
        r.set("ratio", RecordValue::Double(1.5e300));
        r.set("tag", RecordValue::String("x".into()));
        let mut sub = TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        };
        sub.set("inner", RecordValue::Bool(true));
        sub.set(
            "units",
            RecordValue::Array(ArrayValue {
                shape: vec![2],
                data: ArrayData::String(vec!["m".into(), "s".into()]),
            }),
        );
        normalize_array_shapes(&mut sub);
        r.set("nested", RecordValue::Record(sub));
        r.set(
            "arr",
            RecordValue::Array(ArrayValue {
                shape: vec![3],
                data: ArrayData::Int(vec![1, 2, 3]),
            }),
        );
        normalize_array_shapes(&mut r);
        r
    }

    #[test]
    fn table_record_write_read_round_trip() {
        let mut rec = sample_record();
        normalize_array_shapes(&mut rec);
        let mut w = Writer::new();
        write_table_record(&mut w, &rec).unwrap();
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        let back = TableRecord::read(&mut r).unwrap();
        assert_eq!(back, rec);
    }
}
