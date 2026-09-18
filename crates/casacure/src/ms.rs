//! Measurement Set schema knowledge and creation.
//!
//! `default_ms` creates the main MS table together with its standard subtable
//! tree and keyword linkage, mirroring casacore's `default_ms`. The canonical
//! descriptors come from the vendored `ms_schema` module (generated from
//! casacore `required_ms_desc`/`complete_ms_desc`).

use std::path::Path;

use thiserror::Error;

use crate::ms_schema;
use crate::record::{RecordValue, TableRecord};
use crate::tabledesc::TableDesc;
use crate::WritableTable;

/// The subtables casacore's `default_ms` creates and links (the MS 2.0
/// standard set; FREQ_OFFSET / SOURCE / SYSCAL / WEATHER / DOPPLER are
/// optional MS 2.1 tables that `default_ms_subtable` can add).
pub const STANDARD_SUBTABLES: &[&str] = &[
    "ANTENNA",
    "DATA_DESCRIPTION",
    "FEED",
    "FIELD",
    "FLAG_CMD",
    "HISTORY",
    "OBSERVATION",
    "POINTING",
    "POLARIZATION",
    "PROCESSOR",
    "SPECTRAL_WINDOW",
    "STATE",
];

/// Errors from building Measurement Sets.
#[derive(Debug, Error)]
pub enum MsError {
    #[error("unknown MS subtable: {0}")]
    UnknownSubtable(String),
    #[error("descriptor for {0} could not be read from the vendored schema")]
    BadSchema(String),
    #[error(transparent)]
    Create(#[from] crate::WriteTableError),
    #[error(transparent)]
    Desc(#[from] crate::tabledesc::TableDescError),
    #[error("storage error: {0}")]
    Storage(String),
    #[error(transparent)]
    Record(#[from] crate::record::RecordError),
}

fn schema_desc(table: &str, complete: bool) -> Result<TableDesc, MsError> {
    let json = ms_schema::SCHEMAS
        .iter()
        .find(|(t, c, _)| *t == table && *c == complete)
        .map(|(_, _, j)| *j)
        .ok_or_else(|| MsError::BadSchema(table.to_string()))?;
    TableDesc::from_desc_json(json).map_err(MsError::Desc)
}

/// `required_ms_desc()` / `required_ms_desc(subtable)` — the vendored
/// canonical required descriptors.
pub fn required_ms_desc(table: Option<&str>) -> Result<TableDesc, MsError> {
    schema_desc(table.unwrap_or("MS"), false)
}

/// `complete_ms_desc()` / `complete_ms_desc(subtable)`.
pub fn complete_ms_desc(table: Option<&str>) -> Result<TableDesc, MsError> {
    schema_desc(table.unwrap_or("MS"), true)
}

/// Merge extra columns (a python-casacore desc dict) into a `TableDesc`,
/// adding or replacing by name. Used for dask-ms custom columns such as
/// `DATA`, `MODEL_DATA`, `IMAGING_WEIGHT`.
fn merge_extra_columns(desc: &mut TableDesc, extra: &str) -> Result<(), MsError> {
    let record = crate::record::parse_json_record(extra)?;
    for (field, value) in record.desc.fields.iter().zip(record.values.iter()) {
        if field.name.starts_with('_') {
            continue;
        }
        let col = crate::tabledesc::column_from_desc_dict(field.name.as_str(), value)?;
        if let Some(existing) = desc.columns.iter_mut().find(|c| c.name == col.name) {
            *existing = col;
        } else {
            desc.columns.push(col);
        }
    }
    Ok(())
}

/// `default_ms(path, tabdesc=...)`: create the main MS table from the
/// required columns plus any extra columns in `extra_desc`, and the standard
/// subtable tree (each subtable in `<path>/<NAME>`), linked from the main
/// table by `TpTable` keywords.
pub fn default_ms(path: &Path, extra_desc: Option<&str>) -> Result<(), MsError> {
    let mut desc = required_ms_desc(None)?;
    if let Some(extra) = extra_desc {
        merge_extra_columns(&mut desc, extra)?;
    }
    // The main-table keyword record already carries MS_VERSION from the
    // vendored `_keywords_`; link the standard subtables.
    let mut wt = WritableTable::create(path.to_path_buf(), desc);
    for sub in STANDARD_SUBTABLES {
        wt.putkeyword(
            sub,
            RecordValue::Table(path.join(sub).display().to_string()),
        );
    }
    wt.flush()?;

    for sub in STANDARD_SUBTABLES {
        let sub_desc = required_ms_desc(Some(sub))?;
        let mut swt = WritableTable::create(path.join(sub), sub_desc);
        swt.flush()?;
    }
    Ok(())
}

/// `default_ms_subtable(name, path)`: create a single (possibly optional)
/// subtable table from its vendored required descriptor.
pub fn default_ms_subtable(name: &str, path: &Path) -> Result<(), MsError> {
    let sub_desc = required_ms_desc(Some(name))?;
    let mut swt = WritableTable::create(path.to_path_buf(), sub_desc);
    swt.flush()?;
    Ok(())
}

/// A `maketabdesc(column_descs...)`-equivalent: build a name from a list of
/// desc-dict JSON strings. Kept for test/fixture symmetry with casacore.
pub fn maketabdesc_from_json(columns: &[&str]) -> Result<TableDesc, MsError> {
    let mut desc = TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        private_keywords: TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        columns: Vec::new(),
    };
    for json in columns {
        let record = crate::record::parse_json_record(json)?;
        for (field, value) in record.desc.fields.iter().zip(record.values.iter()) {
            if field.name.starts_with('_') {
                continue;
            }
            let col = crate::tabledesc::column_from_desc_dict(field.name.as_str(), value)?;
            desc.columns.push(col);
        }
    }
    Ok(desc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Table;

    #[test]
    fn default_ms_creates_main_and_subtables() {
        let base = std::env::temp_dir().join(format!(
            "casacure-ms-default-{}-{n}",
            std::process::id(),
            n = std::sync::atomic::AtomicUsize::new(0)
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&base);
        let path = base.join("test.ms");
        default_ms(&path, None).unwrap();

        // Main table files + 12 subtable directories.
        assert!(path.join("table.dat").exists());
        for sub in STANDARD_SUBTABLES {
            assert!(
                path.join(sub).join("table.dat").exists(),
                "missing subtable {sub}"
            );
        }

        let t = Table::open(&path, true).unwrap();
        // The 21 required MS columns.
        let expect_cols = [
            "UVW",
            "FLAG",
            "FLAG_CATEGORY",
            "WEIGHT",
            "SIGMA",
            "ANTENNA1",
            "ANTENNA2",
            "ARRAY_ID",
            "DATA_DESC_ID",
            "EXPOSURE",
            "FEED1",
            "FEED2",
            "FIELD_ID",
            "FLAG_ROW",
            "INTERVAL",
            "OBSERVATION_ID",
            "PROCESSOR_ID",
            "SCAN_NUMBER",
            "STATE_ID",
            "TIME",
            "TIME_CENTROID",
        ];
        let cols = t.colnames();
        for c in expect_cols {
            assert!(
                cols.contains(&c.to_string()),
                "missing column {c}: {cols:?}"
            );
        }
        assert_eq!(t.nrows(), 0);

        // Keyword links: MS_VERSION (float) + each subtable as Table path.
        let kw = t.getkeywords();
        let abs = path.canonicalize().unwrap();
        for sub in STANDARD_SUBTABLES {
            assert!(
                kw.contains(&format!("Table: {}/{}", abs.display(), sub)),
                "{sub} not linked in {kw}"
            );
        }
        assert!(kw.starts_with(r#"{"MS_VERSION":2"#), "kw starts: {kw}");

        // Columns with keywords survive (e.g. TIME QuantumUnits/MEASINFO).
        let time_idx = t.colnames().iter().position(|c| c == "TIME").unwrap();
        let cd = t.getcoldesc(time_idx).unwrap();
        assert!(
            cd.contains("\"QuantumUnits\""),
            "TIME coldesc should carry QuantumUnits: {cd}"
        );

        // A subtable opens and has its required columns.
        let ant = Table::open(path.join("ANTENNA"), true).unwrap();
        assert_eq!(
            ant.colnames(),
            [
                "OFFSET",
                "POSITION",
                "TYPE",
                "DISH_DIAMETER",
                "FLAG_ROW",
                "MOUNT",
                "NAME",
                "STATION"
            ]
        );
    }

    #[test]
    fn default_ms_subtable_and_maketabdesc() {
        // default_ms_subtable: any of the 17 known subtables on its own.
        for (name, ncols) in [
            ("ANTENNA", 8),
            ("SPECTRAL_WINDOW", 14),
            ("SYSCAL", 5),
            ("WEATHER", 3),
        ] {
            let path = std::env::temp_dir().join(format!(
                "casacure-mssub-{}-{}",
                std::process::id(),
                name
            ));
            let _ = std::fs::remove_dir_all(&path);
            default_ms_subtable(name, &path).unwrap();
            let t = Table::open(&path, true).unwrap();
            assert_eq!(t.colnames().len(), ncols, "{name} column count");
        }

        // maketabdesc: build a TableDesc from column desc dicts.
        let coldesc = r#"{"FOO":{"valueType":"int","dataManagerType":"StandardStMan","dataManagerGroup":"StandardStMan","option":0,"maxlen":0,"comment":"","keywords":{}}}"#;
        let subtable = r#"{"BAR":{"valueType":"double","dataManagerType":"StandardStMan","dataManagerGroup":"StandardStMan","option":0,"maxlen":0,"comment":"","ndim":1,"shape":[2],"keywords":{}}}"#;
        let desc = maketabdesc_from_json(&[coldesc, subtable]).unwrap();
        assert_eq!(desc.columns.len(), 2);
        assert_eq!(desc.columns[0].data_type, crate::record::DataType::Int);
        assert_eq!(desc.columns[1].data_type, crate::record::DataType::Double);
        assert_eq!(desc.columns[1].shape.as_deref(), Some(&[2i64][..]));
    }

    #[test]
    fn default_ms_merges_extra_columns() {
        let path = std::env::temp_dir().join(format!("casacure-ms-extra-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let extra = r#"{"DATA":{"_c_order":true,"comment":"DATA column","dataManagerGroup":"StandardStMan","dataManagerType":"StandardStMan","keywords":{},"maxlen":0,"ndim":2,"option":0,"valueType":"COMPLEX"}}"#;
        default_ms(&path, Some(extra)).unwrap();
        let t = Table::open(&path, true).unwrap();
        let cols = t.colnames();
        assert!(cols.contains(&"DATA".to_string()), "DATA missing: {cols:?}");
        let di = cols.iter().position(|c| c == "DATA").unwrap();
        assert!(
            t.getcoldesc(di)
                .unwrap()
                .contains("\"valueType\":\"complex\""),
            "DATA coldesc: {}",
            t.getcoldesc(di).unwrap()
        );
    }
}
