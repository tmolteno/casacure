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
    /// The tile shape this column's tiled storage uses (CASA order: the cell
    /// dimensions plus the rows per tile), when it is known rather than
    /// derived: set from a dminfo record's `SPEC.DEFAULTTILESHAPE` at
    /// creation, and from the stored TSM header at open, so regenerating the
    /// column (`create_table` / row growth) reproduces the exact tiling
    /// instead of silently re-tiling to the derived shape. Not persisted in
    /// `table.dat` (casacore keeps the tile shape in the TSM header, as does
    /// this crate).
    pub tile_shape: Option<Vec<i64>>,
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
    /// Build a `TableDesc` from the python-casacore table-desc dict (the
    /// format `required_ms_desc`/`getdesc` produce): column keys plus the
    /// `_define_hypercolumn_`/`_keywords_`/`_private_keywords_` entries.
    pub fn from_desc_json(json: &str) -> Result<TableDesc, TableDescError> {
        let record = crate::record::parse_json_record(json)?;
        let mut desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            private_keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            columns: Vec::new(),
        };
        for (field, value) in record.desc.fields.iter().zip(record.values.iter()) {
            match field.name.as_str() {
                "_define_hypercolumn_" => {}
                "_keywords_" => {
                    if let crate::record::RecordValue::Record(r) = value {
                        desc.keywords = r.clone();
                    }
                }
                "_private_keywords_" => {
                    if let crate::record::RecordValue::Record(r) = value {
                        desc.private_keywords = r.clone();
                    }
                }
                name => {
                    let col = column_from_desc_dict(name, value)?;
                    desc.columns.push(col);
                }
            }
        }
        Ok(desc)
    }
}

/// Convert one column's desc dict (a record value) to a `ColumnDesc`.
pub(crate) fn column_from_desc_dict(
    name: &str,
    value: &crate::record::RecordValue,
) -> Result<ColumnDesc, TableDescError> {
    use crate::record::RecordValue;
    use crate::types::ValueType;
    let crate::record::RecordValue::Record(rec) = value else {
        return Err(TableDescError::UnknownColumnClass(format!(
            "column {name}: expected a desc dict, got {value:?}"
        )));
    };
    let get = |key: &str| rec.get(key);
    let data_type = match get("valueType") {
        Some(RecordValue::String(s)) => {
            let vt = ValueType::from_casa_name(s).map_err(|_| {
                TableDescError::UnknownColumnClass(format!("column {name}: bad valueType {s}"))
            })?;
            match vt {
                ValueType::Bool => DataType::Bool,
                ValueType::Byte => DataType::UChar,
                ValueType::Short => DataType::Short,
                ValueType::UShort => DataType::UShort,
                ValueType::Int => DataType::Int,
                ValueType::UInt => DataType::UInt,
                ValueType::Int64 => DataType::Int64,
                ValueType::Float => DataType::Float,
                ValueType::Double => DataType::Double,
                ValueType::Complex => DataType::Complex,
                ValueType::DComplex => DataType::DComplex,
                ValueType::String => DataType::String,
                ValueType::Record => DataType::Record,
            }
        }
        other => {
            return Err(TableDescError::UnknownColumnClass(format!(
                "column {name}: missing string valueType ({other:?})"
            )));
        }
    };

    let comment = match get("comment") {
        Some(RecordValue::String(s)) => s.clone(),
        _ => String::new(),
    };
    // An empty data-manager type/group means StandardStMan (python-casacore's
    // default when `makescacoldesc(..., datamanagertype='')` is used).
    let data_manager_type = match get("dataManagerType") {
        Some(RecordValue::String(s)) if !s.is_empty() => s.clone(),
        _ => "StandardStMan".to_string(),
    };
    // The dict `shape` is the logical (row-major) shape; the descriptor
    // stores it in CASA order (reversed), like `getcoldesc` reports back.
    // JSON represents array values as `{"shape":[..],"array":[..]}` records,
    // so accept both that dict form and a bare array.
    let shape_elems: Option<Vec<i64>> = match get("shape") {
        Some(RecordValue::Array(a)) => Some(
            a.elements()
                .iter()
                .map(|e| match e {
                    RecordValue::Int(i) => i64::from(*i),
                    RecordValue::Int64(i) => *i,
                    _ => 0,
                })
                .collect(),
        ),
        Some(RecordValue::Record(r)) => r.get("array").and_then(|v| match v {
            RecordValue::Array(a) => Some(
                a.elements()
                    .iter()
                    .map(|e| match e {
                        RecordValue::Int(i) => i64::from(*i),
                        RecordValue::Int64(i) => *i,
                        _ => 0,
                    })
                    .collect(),
            ),
            _ => None,
        }),
        _ => None,
    };
    // TiledCellStMan & friends (variable-shape tiled managers) are not
    // implemented by casacure's storage engine; create them with StandardStMan
    // instead (same values round-trip; only the on-disk layout differs).
    // TiledColumnStMan is always kept; TiledShapeStMan needs a fixed cell
    // shape to tile, so a shape-less declaration (skarabina's flag versions)
    // also stores via StandardStMan.
    let fixed_shape_declared = shape_elems.is_some();
    let data_manager_type = match data_manager_type.as_str() {
        "TiledShapeStMan" if !fixed_shape_declared => "StandardStMan".to_string(),
        "TiledCellStMan" | "TSMBoundedStMan" | "TSMExpStMan" => "StandardStMan".to_string(),
        other => other.to_string(),
    };
    let data_manager_group = match get("dataManagerGroup") {
        Some(RecordValue::String(s)) if !s.is_empty() => s.clone(),
        _ => "StandardStMan".to_string(),
    };
    let options = match get("option") {
        Some(RecordValue::Int(i)) => *i,
        Some(RecordValue::Int64(i)) => *i as i32,
        _ => 0,
    };
    let max_length = match get("maxlen") {
        Some(RecordValue::Int(i)) => *i,
        Some(RecordValue::Int64(i)) => *i as i32,
        _ => 0,
    };
    // `ndim` present at all (any value, incl. -1) means an ARRAY column in
    // casacore: -1 = unconstrained variable-shape, >= 0 = declared dims
    // (variable unless `shape` is fixed). Absent means a scalar column.
    let ndim_present = get("ndim").is_some();
    let ndim = match get("ndim") {
        Some(RecordValue::Int(i)) => *i,
        Some(RecordValue::Int64(i)) => *i as i32,
        _ => -1,
    };
    let shape = shape_elems.map(|dims| dims.into_iter().rev().collect());

    let keywords = match get("keywords") {
        Some(RecordValue::Record(r)) => r.clone(),
        _ => crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
    };

    // Scalar vs array: a column with an explicit `ndim >= 0` is an array
    // column (fixed shape when `shape` is present, variable otherwise);
    // `record` columns are scalar records with no stored default.
    let kind = if data_type == DataType::Record {
        ColumnKind::Record
    } else if ndim_present {
        ColumnKind::Array
    } else {
        ColumnKind::Scalar(default_scalar(data_type))
    };

    Ok(ColumnDesc {
        name: name.to_string(),
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
        tile_shape: None,
    })
}

/// The zero default scalar for a column type.
fn default_scalar(dt: DataType) -> RecordValue {
    use crate::record::RecordValue as RV;
    match dt {
        DataType::Bool => RV::Bool(false),
        DataType::UChar | DataType::Char => RV::UChar(0),
        DataType::Short => RV::Short(0),
        DataType::UShort => RV::UShort(0),
        DataType::Int => RV::Int(0),
        DataType::UInt => RV::UInt(0),
        DataType::Int64 => RV::Int64(0),
        DataType::Float => RV::Float(0.0),
        DataType::Double => RV::Double(0.0),
        DataType::Complex => RV::Complex(0.0, 0.0),
        DataType::DComplex => RV::DComplex(0.0, 0.0),
        DataType::String => RV::String(String::new()),
        _ => RV::Int(0),
    }
}

impl TableDesc {
    pub fn column(&self, name: &str) -> Option<&ColumnDesc> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// The data-manager types table creation can write (`create_table`'s
/// supported managers); anything else fails fast rather than at the flush
/// that regenerates the table.
pub const SUPPORTED_DM_TYPES: [&str; 4] = [
    "StandardStMan",
    "IncrementalStMan",
    "TiledColumnStMan",
    "TiledShapeStMan",
];

/// Why a dminfo record could not be applied to a table description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmInfoError {
    /// A `dminfo` mapping value (or its `COLUMNS` field) is not the
    /// expected record/list shape.
    BadRecord,
    /// The record's `TYPE` is a manager table creation cannot write.
    UnsupportedType { type_name: String },
    /// The record's `COLUMNS` names a column the description does not have.
    UnknownColumn { column: String },
    /// The record's `SPEC` carries a field creation cannot honour (as
    /// opposed to the recognised cache-sizing fields, which are accepted
    /// and ignored — they do not change the on-disk layout).
    BadSpecField { manager: String, field: String },
    /// `SPEC.DEFAULTTILESHAPE` is malformed or does not fit the column.
    BadTileShape { detail: String },
}

impl std::fmt::Display for DmInfoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmInfoError::BadRecord => {
                write!(f, "dminfo values must be dicts with a list of COLUMNS")
            }
            DmInfoError::UnsupportedType { type_name } => write!(
                f,
                "unsupported data-manager type {type_name:?} in dminfo; casacure supports {}",
                SUPPORTED_DM_TYPES.join(", ")
            ),
            DmInfoError::UnknownColumn { column } => write!(
                f,
                "dminfo names column {column:?}, which is not among the columns being created"
            ),
            DmInfoError::BadSpecField { manager, field } => write!(
                f,
                "unsupported SPEC field {field:?} for data-manager type {manager:?} in dminfo"
            ),
            DmInfoError::BadTileShape { detail } => {
                write!(f, "bad SPEC.DEFAULTTILESHAPE in dminfo: {detail}")
            }
        }
    }
}

impl std::error::Error for DmInfoError {}

/// Apply a data-manager-info record (the `dminfo` argument of table
/// creation, `addcols` and the deep `tablecopy`) to the columns being
/// created: each record's `TYPE`/`NAME` — and its
/// `SPEC.DEFAULTTILESHAPE` for a tiled manager — land on the columns the
/// record names in `COLUMNS`, or on every column when it names none.
///
/// Both caller shapes are accepted, matching casacore's
/// `Table::addColumns(desc, dminfo)`:
/// * a flat record `{TYPE, NAME, SPEC}` (python-casacore's documented
///   form) applies to every column;
/// * a `getdminfo()`-shaped mapping `{field: {TYPE, NAME, COLUMNS, ...}}`
///   applies each record to the columns it names.
///
/// A mapping record with neither `TYPE` nor `COLUMNS` — python-casacore's
/// `makedminfo(tabdesc, dmgroup_spec)` *input* shape, keyed by group name
/// (`{"DataGroup": {"DEFAULTTILESHAPE": ...}}`) — refines the columns that
/// already declare that `dataManagerGroup` in their descriptors (every
/// column, when no descriptor does): it sets a manager's tile shape or
/// group name without re-routing columns between managers.
///
/// A record given without a `TYPE` keeps its columns' declared manager and
/// only renames the group. `SEQNR` is ignored (managers are numbered in
/// column order), and the cache-sizing `SPEC` fields are accepted and
/// ignored: they tune reads, not the on-disk layout. Any other SPEC field —
/// `HYPERCUBES` above all — fails loudly: silently creating a different
/// layout than the caller asked for is the bug this guards against.
pub fn apply_dminfo(
    dminfo: &crate::record::TableRecord,
    cols: &mut [ColumnDesc],
) -> Result<(), DmInfoError> {
    use crate::record::{ArrayData, RecordValue};
    if dminfo.values.is_empty() || cols.is_empty() {
        return Ok(());
    }
    let field_str = |rec: &crate::record::TableRecord, key: &str| -> Option<String> {
        match rec.get(key) {
            Some(RecordValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    };
    // A flat record has TYPE at the top level and applies to the whole set.
    if dminfo.get("TYPE").is_some() {
        return apply_dm_record(
            dminfo,
            field_str(dminfo, "TYPE").as_deref(),
            field_str(dminfo, "NAME").as_deref(),
            None,
            false,
            cols,
        );
    }
    // getdminfo()-shaped mapping: one record per data manager. Column
    // targeting by declared group (a `dmgroup_spec` entry) reads the
    // groups the descriptors came in with, not those an earlier record
    // set.
    let declared_groups: Vec<String> = cols.iter().map(|c| c.data_manager_group.clone()).collect();
    for (field, value) in dminfo.desc.fields.iter().zip(dminfo.values.iter()) {
        let RecordValue::Record(rec) = value else {
            return Err(DmInfoError::BadRecord);
        };
        let columns: Option<Vec<usize>> = match rec.get("COLUMNS") {
            None => {
                if rec.get("TYPE").is_some() {
                    // A typed record without targets governs every column.
                    None
                } else {
                    // A dmgroup_spec entry: the columns declaring this
                    // group (every column when none does).
                    let key = field.name.as_str();
                    let matched: Vec<usize> = declared_groups
                        .iter()
                        .enumerate()
                        .filter(|(_, g)| g.as_str() == key)
                        .map(|(i, _)| i)
                        .collect();
                    if matched.is_empty() {
                        None
                    } else {
                        Some(matched)
                    }
                }
            }
            Some(RecordValue::Array(av)) => {
                let names = match &av.data {
                    ArrayData::String(names) => names,
                    _ => return Err(DmInfoError::BadRecord),
                };
                let mut out = Vec::with_capacity(names.len());
                for col_name in names {
                    let idx = cols
                        .iter()
                        .position(|c| &c.name == col_name)
                        .ok_or_else(|| DmInfoError::UnknownColumn {
                            column: col_name.clone(),
                        })?;
                    out.push(idx);
                }
                Some(out)
            }
            Some(_) => return Err(DmInfoError::BadRecord),
        };
        apply_dm_record(
            rec,
            field_str(rec, "TYPE").as_deref(),
            field_str(rec, "NAME")
                .as_deref()
                .or(Some(field.name.as_str())),
            columns.as_deref(),
            // A dmgroup_spec entry carries its SPEC fields at the top
            // level — the record itself is the SPEC.
            rec.get("TYPE").is_none() && rec.get("COLUMNS").is_none(),
            cols,
        )?;
    }
    Ok(())
}

/// Apply one dminfo record to its target columns: `columns` selects them by
/// index (`None` = every column), `dm_type`/`name` set the manager type
/// and group, and the record's `SPEC` is validated (its
/// `DEFAULTTILESHAPE` is kept on the columns for the storage writers). A
/// `dmgroup_spec` entry (`record_is_spec`) has no `TYPE`/`COLUMNS`/`SPEC`
/// wrapper — the record itself carries the SPEC fields.
fn apply_dm_record(
    rec: &crate::record::TableRecord,
    dm_type: Option<&str>,
    name: Option<&str>,
    columns: Option<&[usize]>,
    record_is_spec: bool,
    cols: &mut [ColumnDesc],
) -> Result<(), DmInfoError> {
    use crate::record::{ArrayData, RecordValue};
    if let Some(t) = dm_type {
        if !SUPPORTED_DM_TYPES.contains(&t) {
            return Err(DmInfoError::UnsupportedType {
                type_name: t.to_string(),
            });
        }
    }
    // Validate SPEC before touching any column, so a rejected record
    // rejects as a whole.
    let mut tile_shape = None;
    let spec = match rec.get("SPEC") {
        Some(RecordValue::Record(spec)) => Some(spec),
        Some(_) => return Err(DmInfoError::BadRecord),
        None if record_is_spec => Some(rec),
        None => None,
    };
    if let Some(spec) = spec {
        if !spec.values.is_empty() {
            let manager = dm_type.unwrap_or("StandardStMan").to_string();
            for field in &spec.desc.fields {
                match field.name.as_str() {
                    // Cache sizes and the derived cube geometry do not
                    // change the stored bytes; ignore them.
                    "MaxCacheSize" | "MAXIMUMCACHESIZE" | "BUCKETSIZE" | "PERSCACHESIZE"
                    | "IndexLength" | "IndexSize" | "DEFAULTCUBESHAPE" => {}
                    "DEFAULTTILESHAPE" => {
                        let dims: Vec<i64> = match spec.get("DEFAULTTILESHAPE") {
                            Some(RecordValue::Array(av)) => match &av.data {
                                ArrayData::Int(v) => v.iter().map(|&d| d as i64).collect(),
                                ArrayData::Int64(v) => v.clone(),
                                _ => {
                                    return Err(DmInfoError::BadTileShape {
                                        detail: "expected a list of integers".into(),
                                    })
                                }
                            },
                            _ => {
                                return Err(DmInfoError::BadTileShape {
                                    detail: "expected a list of integers".into(),
                                })
                            }
                        };
                        if dims.len() < 2 || dims.iter().any(|&d| d <= 0) {
                            return Err(DmInfoError::BadTileShape {
                                detail: format!("expected positive dimensions, got {dims:?}"),
                            });
                        }
                        // A tile shape on a non-tiled manager cannot be
                        // created.
                        if matches!(dm_type, Some("StandardStMan") | Some("IncrementalStMan")) {
                            return Err(DmInfoError::BadSpecField {
                                manager,
                                field: "DEFAULTTILESHAPE".into(),
                            });
                        }
                        tile_shape = Some(dims);
                    }
                    other => {
                        return Err(DmInfoError::BadSpecField {
                            manager,
                            field: other.to_string(),
                        })
                    }
                }
            }
        }
    }
    // Resolve the targets (all columns, or the selected ones).
    let targets: Vec<usize> = match columns {
        None => (0..cols.len()).collect(),
        Some(idxs) => idxs.to_vec(),
    };
    for idx in targets {
        let cd = &mut cols[idx];
        if let Some(t) = dm_type {
            cd.data_manager_type = t.to_string();
        }
        // The group defaults to the (possibly new) manager type, as
        // casacore names a manager given without an explicit NAME.
        cd.data_manager_group = name
            .map(str::to_string)
            .unwrap_or_else(|| cd.data_manager_type.clone());
        cd.tile_shape = tile_shape.clone();
    }
    Ok(())
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
        tile_shape: None,
    })
}

/// The 8-char data-type id used inside column-description class names
/// (`ScaColDesc.h` `dataTypeId`), e.g. `"Int     "` for
/// `"ScalarColumnDesc<Int     "`.
pub fn data_type_id(dt: DataType) -> String {
    use DataType::*;
    let id = match dt {
        Bool => "Bool",
        Char => "uChar",
        UChar => "uChar",
        Short => "Short",
        UShort => "uShort",
        Int => "Int",
        UInt => "uInt",
        Int64 => "Int64",
        Float => "float",
        Double => "double",
        Complex => "Complex",
        DComplex => "DComplex",
        String => "String",
        _ => "Other",
    };
    format!("{id:8}")
}

/// The class name written for a column description (`ColumnDesc::putFile`
/// via `ScaColumnDesc`/`ArrayColumnDesc` `className`).
fn column_class_name(desc: &ColumnDesc) -> String {
    if matches!(desc.kind, ColumnKind::Array) {
        format!("ArrayColumnDesc<{}", data_type_id(desc.data_type))
    } else if desc.data_type == DataType::Record {
        "ScalarRecordColumnDesc".to_string()
    } else {
        format!("ScalarColumnDesc<{}", data_type_id(desc.data_type))
    }
}

/// Serialize a `ColumnDesc` (`ColumnDesc::putFile` +
/// `BaseColumnDesc::putFile` + the subclass `putDesc`).
pub(crate) fn write_column_desc(w: &mut crate::aipsio::Writer, desc: &ColumnDesc) {
    use crate::record::write_scalar_value;
    w.put_u32(1); // ColumnDesc class version
    w.put_string(&column_class_name(desc));
    // BaseColumnDesc::putFile
    w.put_u32(1); // base class version
    w.put_string(&desc.name);
    w.put_string(&desc.comment);
    w.put_string(&desc.data_manager_type);
    w.put_string(&desc.data_manager_group);
    w.put_i32(data_type_code(desc.data_type));
    w.put_i32(desc.options);
    w.put_i32(desc.ndim);
    if !matches!(desc.kind, ColumnKind::Scalar(_) | ColumnKind::Record) {
        w.put_object_start("IPosition", 1);
        w.put_u32(desc.shape.as_ref().map_or(0, |s| s.len()) as u32);
        for d in desc.shape.as_deref().unwrap_or(&[]) {
            w.put_i32(*d as i32);
        }
        w.put_object_end();
    }
    w.put_i32(desc.max_length);
    crate::record::write_table_record(w, &desc.keywords).expect("write column keyword record");
    // Subclass putDesc.
    match &desc.kind {
        ColumnKind::Scalar(default) => {
            w.put_u32(1); // ScalarColumnDesc class version
            write_scalar_value(w, desc.data_type, default);
        }
        ColumnKind::Array => {
            w.put_u32(1); // ArrayColumnDesc class version
            w.put_bool(false); // no default array
        }
        ColumnKind::Record => {
            w.put_u32(1);
        }
    }
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
        Record => 25,
        Int64 => 29,
        ArrayInt64 => 30,
        _ => 13,
    }
}

/// Serialize a `TableDesc` as the nested `"TableDesc"` v2 object that
/// follows the `"Table"` header in `table.dat`.
pub fn write_table_desc(w: &mut crate::aipsio::Writer, desc: &TableDesc) {
    w.put_object_start("TableDesc", 2);
    w.put_string(&desc.name);
    w.put_string(&desc.version);
    w.put_string(&desc.comment);
    crate::record::write_table_record(w, &desc.keywords).expect("write table keyword record");
    crate::record::write_table_record(w, &desc.private_keywords)
        .expect("write private keyword record");
    w.put_u32(desc.columns.len() as u32);
    for col in &desc.columns {
        write_column_desc(w, col);
    }
    w.put_object_end();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aipsio::Writer;
    use crate::record::{ArrayData, ArrayValue, RecordValue};

    fn empty_record() -> TableRecord {
        TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        }
    }

    fn kw_record() -> TableRecord {
        let mut r = empty_record();
        r.set(
            "Units",
            RecordValue::Array(ArrayValue {
                shape: vec![1],
                data: ArrayData::String(vec!["Jy".into()]),
            }),
        );
        // `set` records array fields with shape None, but the codec always
        // serializes an IPosition, so a write/read round trip yields
        // Some([]) for "no fixed shape". Normalize to the canonical form so
        // the round-trip assertion below is an identity check.
        for f in &mut r.desc.fields {
            if f.data_type.is_array() && f.shape.is_none() {
                f.shape = Some(Vec::new());
            }
        }
        let mut nested = empty_record();
        nested.set("scale", RecordValue::Double(1.0e-3));
        r.set("Nested", RecordValue::Record(nested));
        r
    }

    fn col(
        name: &str,
        dt: DataType,
        kind: ColumnKind,
        ndim: i32,
        shape: Option<Vec<i64>>,
        keywords: TableRecord,
    ) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: format!("comment on {name}"),
            data_type: dt,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 2,
            ndim,
            shape,
            max_length: 0,
            keywords,
            kind,
            tile_shape: None,
        }
    }

    /// A scalar column default must survive the binary round trip.
    fn scalar_col(name: &str, dt: DataType, default: RecordValue) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: 0,
            keywords: empty_record(),
            kind: ColumnKind::Scalar(default),
            tile_shape: None,
        }
    }

    #[test]
    fn write_read_round_trips_all_column_kinds() {
        let desc = TableDesc {
            name: "T".into(),
            version: "1.0".into(),
            comment: "a table".into(),
            keywords: {
                let mut k = kw_record();
                k.set("MS_VERSION", RecordValue::Double(2.0));
                k
            },
            private_keywords: {
                let mut k = empty_record();
                k.set("_private", RecordValue::Int(7));
                k
            },
            columns: vec![
                scalar_col("COL_B", DataType::Bool, RecordValue::Bool(true)),
                scalar_col("COL_D", DataType::Double, RecordValue::Double(1.25e300)),
                scalar_col("COL_S", DataType::String, RecordValue::String("hi".into())),
                ColumnDesc {
                    name: "ARR".into(),
                    comment: "fixed 2x3".into(),
                    data_type: DataType::Complex,
                    data_manager_type: "TiledColumnStMan".into(),
                    data_manager_group: "TiledData_GROUP".into(),
                    options: 0,
                    ndim: 2,
                    shape: Some(vec![3, 2]),
                    max_length: 0,
                    keywords: kw_record(),
                    kind: ColumnKind::Array,
                    tile_shape: None,
                },
                ColumnDesc {
                    name: "REC".into(),
                    comment: "a record column".into(),
                    data_type: DataType::Record,
                    data_manager_type: "StandardStMan".into(),
                    data_manager_group: "StandardStMan".into(),
                    options: 0,
                    ndim: -1,
                    shape: None,
                    max_length: 0,
                    keywords: empty_record(),
                    kind: ColumnKind::Record,
                    tile_shape: None,
                },
            ],
        };

        let mut w = Writer::new();
        write_table_desc(&mut w, &desc);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        let parsed = parse_table_desc(&mut r).unwrap();
        assert_eq!(parsed, desc);
    }

    #[test]
    fn data_type_ids_are_eight_char_padded() {
        assert_eq!(data_type_id(DataType::Int), "Int     ");
        assert_eq!(data_type_id(DataType::UChar), "uChar   ");
        assert_eq!(data_type_id(DataType::DComplex), "DComplex");
        assert_eq!(data_type_id(DataType::String), "String  ");
        for dt in [
            DataType::Bool,
            DataType::UChar,
            DataType::Short,
            DataType::Int,
            DataType::Double,
            DataType::DComplex,
        ] {
            assert_eq!(data_type_id(dt).len(), 8, "{dt:?}");
        }
    }

    /// The 8-char type suffix inside `ScalarColumnDesc<...>` /
    /// `ArrayColumnDesc<...>` class names. casacore's ColumnDesc registry is
    /// keyed by this exact string (ColumnDesc.cc `initRegisterMap`), so a
    /// spelling mismatch makes casacore unable to open the table. The
    /// canonical spellings are taken from casacore-written table.dat files
    /// (casacore `dataTypeId`): `Int64` is capital-I (NOT `int64`).
    #[test]
    fn column_class_name_suffixes_match_casacore_spelling() {
        for (dt, id) in [
            (DataType::Bool, "Bool"),
            (DataType::UChar, "uChar"),
            (DataType::Short, "Short"),
            (DataType::UShort, "uShort"),
            (DataType::Int, "Int"),
            (DataType::UInt, "uInt"),
            (DataType::Int64, "Int64"),
            (DataType::Float, "float"),
            (DataType::Double, "double"),
            (DataType::Complex, "Complex"),
            (DataType::DComplex, "DComplex"),
            (DataType::String, "String"),
        ] {
            let want = format!("{id:8}");
            assert_eq!(data_type_id(dt), want, "type id for {dt:?}");
        }
    }

    #[test]
    fn from_desc_json_builds_scalar_array_and_record_columns() {
        let json = r#"{
          "INT": {"valueType":"int","comment":"an int","keywords":{}},
          "ARR": {"valueType":"double","ndim":1,"shape":[3,2],"keywords":{"Units":["Jy"]}},
          "REC": {"valueType":"record","keywords":{}}
        }"#;
        let desc = TableDesc::from_desc_json(json).unwrap();
        assert_eq!(desc.columns.len(), 3);

        let int = desc.column("INT").unwrap();
        assert_eq!(int.name, "INT");
        assert_eq!(int.data_type, DataType::Int);
        assert_eq!(int.kind, ColumnKind::Scalar(RecordValue::Int(0)));
        // Empty dataManagerType/group fall back to StandardStMan.
        assert_eq!(int.data_manager_type, "StandardStMan");
        assert_eq!(int.data_manager_group, "StandardStMan");

        let arr = desc.column("ARR").unwrap();
        assert_eq!(arr.data_type, DataType::Double);
        assert_eq!(arr.kind, ColumnKind::Array);
        assert_eq!(arr.ndim, 1);
        // Logical [3,2] is stored in CASA (reversed) order.
        assert_eq!(arr.shape, Some(vec![2, 3]));
        assert_eq!(
            arr.keywords.get("Units"),
            Some(&RecordValue::Array(ArrayValue {
                shape: vec![1],
                data: ArrayData::String(vec!["Jy".into()]),
            }))
        );

        let rec = desc.column("REC").unwrap();
        assert_eq!(rec.data_type, DataType::Record);
        assert_eq!(rec.kind, ColumnKind::Record);
    }

    #[test]
    fn from_desc_json_captures_table_keywords() {
        let json = r#"{
          "A": {"valueType":"int","keywords":{}},
          "_define_hypercolumn_":{},
          "_keywords_":{"MS_VERSION":2.0,"NAME":"x"},
          "_private_keywords_":{"P":1}
        }"#;
        let desc = TableDesc::from_desc_json(json).unwrap();
        assert_eq!(desc.columns.len(), 1);
        assert_eq!(
            desc.keywords.get("MS_VERSION"),
            Some(&RecordValue::Double(2.0))
        );
        assert_eq!(
            desc.keywords.get("NAME"),
            Some(&RecordValue::String("x".into()))
        );
        assert_eq!(desc.private_keywords.get("P"), Some(&RecordValue::Int(1)));
    }

    #[test]
    fn column_desc_normalizes_unsupported_data_managers() {
        // Empty type/group default to StandardStMan; unsupported tiled
        // Variable-shape managers and shape-less TiledShapeStMan are
        // downgraded; TiledColumnStMan is kept.
        let base = |dmt: &str, dmg: &str| {
            format!(
                r#"{{"C":{{"valueType":"int","dataManagerType":"{dmt}","dataManagerGroup":"{dmg}","keywords":{{}}}}}}"#
            )
        };
        for (dmt, dmg, want_type, want_group) in [
            ("", "", "StandardStMan", "StandardStMan"),
            ("", "MYGRP", "StandardStMan", "MYGRP"),
            ("IncrementalStMan", "", "IncrementalStMan", "StandardStMan"),
            ("TiledShapeStMan", "G", "StandardStMan", "G"),
            ("TiledCellStMan", "G", "StandardStMan", "G"),
            ("TiledColumnStMan", "G", "TiledColumnStMan", "G"),
        ] {
            let desc = TableDesc::from_desc_json(&base(dmt, dmg)).unwrap();
            let c = desc.column("C").unwrap();
            assert_eq!(c.data_manager_type, want_type, "{dmt}/{dmg}");
            assert_eq!(c.data_manager_group, want_group, "{dmt}/{dmg}");
        }
    }

    #[test]
    fn from_desc_json_rejects_missing_value_type() {
        let json = r#"{"C":{"comment":"no valueType","keywords":{}}}"#;
        assert!(matches!(
            TableDesc::from_desc_json(json),
            Err(TableDescError::UnknownColumnClass(_))
        ));
    }

    #[test]
    fn parse_rejects_unknown_column_class() {
        // A framed TableDesc with a nonsense column class name.
        let mut w = Writer::new();
        w.put_object_start("TableDesc", 2);
        w.put_string("T");
        w.put_string("1");
        w.put_string("");
        crate::record::write_table_record(&mut w, &empty_record()).unwrap();
        crate::record::write_table_record(&mut w, &empty_record()).unwrap();
        w.put_u32(1);
        w.put_u32(1); // class version
        w.put_string("BogusColumnDesc<Int     ");
        w.put_object_end();
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert!(matches!(
            parse_table_desc(&mut r),
            Err(TableDescError::UnknownColumnClass(_))
        ));
    }

    #[test]
    fn array_shape_round_trips_in_casa_order() {
        // The descriptor stores the CASA (reversed logical) shape on disk
        // unchanged: logical [3,2] -> Some([2,3]) -> read back as [2,3].
        let desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![col(
                "ARR",
                DataType::Double,
                ColumnKind::Array,
                2,
                Some(vec![2, 3]),
                empty_record(),
            )],
        };
        let mut w = Writer::new();
        write_table_desc(&mut w, &desc);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        let parsed = parse_table_desc(&mut r).unwrap();
        assert_eq!(parsed, desc);
    }
}

#[cfg(test)]
mod tsm_shape_tests {
    use super::TableDesc;

    #[test]
    fn tiled_shape_stman_with_fixed_shape_is_kept() {
        let desc = TableDesc::from_desc_json(
            r#"{"DATA":{"valueType":"dcomplex","dataManagerType":"TiledShapeStMan","dataManagerGroup":"TiledData","ndim":2,"shape":[4,2],"_c_order":true}}"#,
        )
        .unwrap();
        assert_eq!(
            desc.column("DATA").unwrap().data_manager_type,
            "TiledShapeStMan"
        );
    }
}
