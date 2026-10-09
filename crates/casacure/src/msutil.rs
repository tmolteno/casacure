//! MS utility functions (`casacore/tables/msutil.py::addImagingColumns` /
//! `removeImagingColumns`).
//!
//! These are the only module-level table helpers DDFacet and killMS call that
//! casacore provides outside `casacore.tables` proper: killMS's
//! `ClassMS.PutCasaCols` runs `pyrap.tables.addImagingColumns(self.MSName)`
//! before it writes `CORRECTED_DATA`/`MODEL_DATA`/`IMAGING_WEIGHT`
//! (`killMS/killMS/Data/ClassMS.py:1213`).
//!
//! Semantics are pinned to python-casacore 3.8.1's `msutil.py`, verified
//! against the real module on this machine:
//!
//! * `addImagingColumns(msname, ack=True)` opens the MS for update, and adds
//!   `MODEL_DATA`, `CORRECTED_DATA` (both clones of `DATA`'s descriptor with
//!   a new comment, tiled when `DATA` is tiled, `TiledShapeStMan` otherwise)
//!   and `IMAGING_WEIGHT` (1-dim float, shape `[nchan]`) — skipping any that
//!   already exist. It then defines `MODEL_DATA`'s `CHANNEL_SELECTION`
//!   column keyword as `int32 [[0, nch], ...]`, one pair per spectral
//!   window, and flushes.
//! * `removeImagingColumns(msname)` removes those three columns (whichever
//!   exist) and flushes.
//!
//! Both operate through a transient [`WritableTable`], never a live Python
//! handle, so they can be called from Rust (`killms-core`) as well as from
//! the Python binding.

use std::path::Path;

use crate::record::{ArrayData, ArrayValue, RecordValue};
use crate::tabledesc::ColumnDesc;
use crate::{Table, WritableTable};

/// Errors from the `msutil` helpers.
#[derive(Debug)]
pub enum MsUtilError {
    /// The MS directory does not exist or is not a table.
    NotATable(String),
    /// The MS exists but has no `DATA` column (`addImagingColumns`
    /// raises `ValueError('Column DATA does not exist')` in python-casacore).
    NoDataColumn,
    /// A storage-layer failure (read, write, flush).
    Storage(String),
}

impl std::fmt::Display for MsUtilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MsUtilError::NotATable(p) => write!(f, "{p}: not a CASA table"),
            MsUtilError::NoDataColumn => write!(f, "Column DATA does not exist"),
            MsUtilError::Storage(e) => write!(f, "storage error: {e}"),
        }
    }
}

impl std::error::Error for MsUtilError {}

fn storage(e: impl std::fmt::Display) -> MsUtilError {
    MsUtilError::Storage(e.to_string())
}

/// Parse one column's `getcoldesc` JSON dict back into a `ColumnDesc`, by
/// wrapping it as a one-column table-desc record (the same deserializer
/// `addcols` uses, so the clone goes in with the same field mapping).
fn parse_coldesc(json: &str) -> Result<ColumnDesc, MsUtilError> {
    let wrapped = format!("{{\"__col__\":{json}}}");
    let desc = crate::tabledesc::TableDesc::from_desc_json(&wrapped)
        .map_err(|e| MsUtilError::Storage(e.to_string()))?;
    desc.columns
        .into_iter()
        .next()
        .ok_or_else(|| MsUtilError::Storage("empty descriptor".into()))
}

/// Clone one column's descriptor under a new name, comment and
/// data-manager group (dropping its keywords, as python-casacore's
/// `makecoldesc(name, cdesc)` does). Each cloned column gets its own group —
/// matching python-casacore's `msutil.py`, which names them `modeldata`,
/// `correcteddata` and `imagingweight`.
fn clone_col(cd: &ColumnDesc, name: &str, comment: &str, group: &str) -> ColumnDesc {
    let mut c = cd.clone();
    c.name = name.to_string();
    c.comment = comment.to_string();
    c.data_manager_group = group.to_string();
    c.keywords = crate::record::TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    };
    c
}

/// `addImagingColumns(msname, ack=True)` — add `MODEL_DATA`,
/// `CORRECTED_DATA` and `IMAGING_WEIGHT` to `path` (a Measurement Set), and
/// define `MODEL_DATA`'s `CHANNEL_SELECTION` keyword. Returns the names
/// actually added (empty when all three were already present).
pub fn add_imaging_columns(path: &Path) -> Result<Vec<String>, MsUtilError> {
    let dir = crate::table::absolute_dir(path);
    if !dir.join("table.dat").exists() {
        return Err(MsUtilError::NotATable(dir.display().to_string()));
    }
    // Open for update: read the current descriptor + keywords, buffer the new
    // columns, write `CHANNEL_SELECTION`, then flush — same order as the
    // python-casacore implementation (addcols… then putcolkeyword, flush).
    let (read, mut wt) =
        WritableTable::open_for_update(&dir).map_err(|e| MsUtilError::Storage(e.to_string()))?;

    let data_idx = read
        .colnames()
        .iter()
        .position(|n| n == "DATA")
        .ok_or(MsUtilError::NoDataColumn)?;
    let data_desc = read
        .getcoldesc(data_idx)
        .ok_or_else(|| MsUtilError::Storage("DATA descriptor unreadable".into()))?;
    let data_cd = parse_coldesc(&data_desc)?;
    // `hasTiled`: DATA's manager is one of the tiled managers (python-casacore
    // checks `dminfo['TYPE'][:5] == 'Tiled'`).
    let data_tiled = data_cd.data_manager_type.starts_with("Tiled");

    let existing: Vec<String> = wt.desc().columns.iter().map(|c| c.name.clone()).collect();
    let mut added: Vec<String> = Vec::new();

    // MODEL_DATA / CORRECTED_DATA: clones of DATA's descriptor, tiled when
    // DATA is tiled, `TiledShapeStMan` otherwise (python-casacore picks a
    // default tile shape for the not-tiled case; casacure derives it from the
    // column shape, so the manager record stays consistent on disk).
    for (name, comment) in [
        ("MODEL_DATA", "The model data column"),
        ("CORRECTED_DATA", "The corrected data column"),
    ] {
        if existing.iter().any(|n| n == name) {
            continue;
        }
        let group = name.to_ascii_lowercase();
        let mut c = clone_col(&data_cd, name, comment, &group);
        if !data_tiled {
            c.data_manager_type = "TiledShapeStMan".into();
        }
        wt.addcol(c);
        added.push(name.to_string());
    }

    // IMAGING_WEIGHT: 1-dim float, shape [nchan] from DATA's declared shape
    // (or from cell 0's actual data when DATA is variable-shape).
    if !existing.iter().any(|n| n == "IMAGING_WEIGHT") {
        let nchan = imaging_weight_nchan(&read, data_idx, &data_cd)?;
        let c = ColumnDesc {
            name: "IMAGING_WEIGHT".into(),
            comment: String::new(),
            data_type: crate::record::DataType::Float,
            data_manager_type: "TiledShapeStMan".into(),
            data_manager_group: "imagingweight".into(),
            options: data_cd.options,
            ndim: 1,
            shape: Some(vec![nchan as i64]), // stored CASA-order (reversed): 1-d
            max_length: 0,
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind: crate::tabledesc::ColumnKind::Array,
            tile_shape: None,
        };
        wt.addcol(c);
        added.push("IMAGING_WEIGHT".into());
    }

    // CHANNEL_SELECTION on MODEL_DATA: int32 [[0, nch], ...], one pair per
    // spectral window, read from the SPECTRAL_WINDOW subtable's NUM_CHAN.
    if let Some(midx) = wt
        .desc()
        .columns
        .iter()
        .position(|c| c.name == "MODEL_DATA")
    {
        let chans = channel_selection(&dir)?;
        wt.putcolkeyword(
            midx,
            "CHANNEL_SELECTION",
            RecordValue::Array(ArrayValue {
                shape: vec![chans.len() as u32 / 2, 2],
                data: ArrayData::Int(chans),
            }),
        )
        .map_err(storage)?;
    }

    wt.flush().map_err(storage)?;
    Ok(added)
}

/// `removeImagingColumns(msname)` — drop `MODEL_DATA`, `CORRECTED_DATA` and
/// `IMAGING_WEIGHT` (whichever exist) and flush. Returns the names actually
/// removed.
pub fn remove_imaging_columns(path: &Path) -> Result<Vec<String>, MsUtilError> {
    let dir = crate::table::absolute_dir(path);
    if !dir.join("table.dat").exists() {
        return Err(MsUtilError::NotATable(dir.display().to_string()));
    }
    let (_read, mut wt) =
        WritableTable::open_for_update(&dir).map_err(|e| MsUtilError::Storage(e.to_string()))?;
    let mut removed: Vec<String> = Vec::new();
    for name in ["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"] {
        if let Some(idx) = wt.desc().columns.iter().position(|c| c.name == name) {
            wt.removecol(idx);
            removed.push(name.to_string());
        }
    }
    if !removed.is_empty() {
        wt.flush().map_err(storage)?;
    }
    Ok(removed)
}

/// `nchan` for the new `IMAGING_WEIGHT` column: DATA's declared fixed shape
/// first axis, else the first cell's actual first dimension
/// (`t.getcell('DATA', 0).shape[0]` in python-casacore), else 0.
fn imaging_weight_nchan(
    read: &Table,
    data_idx: usize,
    data_cd: &ColumnDesc,
) -> Result<u32, MsUtilError> {
    if let Some(stored) = &data_cd.shape {
        // `shape` is stored CASA-order (reversed); the logical first axis
        // (nchan, per the MS spec) is the last stored one. But casacore's
        // `addImagingColumns` uses `shp[0]` of the *logical* shape, which is
        // the first stored element for a 2-D (nchan, ncorr) column. Take
        // the logical first axis = last stored, which for (nchan, ncorr)
        // stored as [ncorr, nchan] is nchan.
        if let Some(&n) = stored.last() {
            return Ok(n.max(0) as u32);
        }
    }
    // Variable-shape (or shape-less): read cell 0 and take its logical first
    // dimension — `getcell('DATA', 0).shape[0]`.
    if read.nrows() == 0 {
        return Ok(0);
    }
    match read.getcell(data_idx, 0) {
        Ok(RecordValue::Array(a)) if !a.shape.is_empty() => Ok(a.shape[0]),
        Ok(_) => Ok(0),
        Err(e) => Err(MsUtilError::Storage(format!("reading DATA cell 0: {e}"))),
    }
}

/// `MODEL_DATA`'s `CHANNEL_SELECTION`: for each spectral window,
/// `[first_chan, nchan]`, read from `<ms>/SPECTRAL_WINDOW`'s `NUM_CHAN`
/// column. Falls back to `[[0, 0]]` when the subtable or column is absent
/// (python-casacore would raise; casacure degrades the same way its other
/// MS helpers do — the keyword is informational for old imagers).
fn channel_selection(ms_dir: &Path) -> Result<Vec<i32>, MsUtilError> {
    let spw_dir = ms_dir.join("SPECTRAL_WINDOW");
    if !spw_dir.join("table.dat").exists() {
        return Ok(vec![0, 0]);
    }
    let spw = Table::open(&spw_dir, false).map_err(|e| MsUtilError::Storage(e.to_string()))?;
    let names = spw.colnames();
    let Some(idx) = names.iter().position(|n| n == "NUM_CHAN") else {
        return Ok(vec![0, 0]);
    };
    let n = spw.nrows();
    let cells = spw
        .getcol(idx, 0, n)
        .map_err(|e| MsUtilError::Storage(e.to_string()))?;
    let mut out = Vec::with_capacity(cells.len() * 2);
    for c in &cells {
        let nch = match c {
            RecordValue::Int(i) => *i,
            RecordValue::Int64(i) => *i as i32,
            _ => 0,
        };
        out.push(0);
        out.push(nch);
    }
    if out.is_empty() {
        out.extend_from_slice(&[0, 0]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ms;

    /// A tiny MS with a TiledColumnStMan DATA column (real MSes carry DATA
    /// in a unique tiled group; the required schema does not include DATA
    /// at all — see `PORTING_DDFACET_KILLMS.md` §3), a SPECTRAL_WINDOW
    /// subtable with NUM_CHAN, and `nrow` DATA rows.
    fn make_ms(path: &Path, nrow: u64, nchan: u32, ncorr: u32) {
        // MS convention: DATA shape is (nchan, ncorr) — see
        // `PORTING_DDFACET_KILLMS.md` §3.
        let data_desc = format!(
            r#"{{"DATA":{{"valueType":"complex","dataManagerType":"TiledColumnStMan","dataManagerGroup":"DATA_GROUP","option":4,"maxlen":0,"comment":"The data column","ndim":2,"shape":[{nchan},{ncorr}],"_c_order":true,"keywords":{{}}}}}}"#
        );
        ms::default_ms(path, Some(&data_desc), None).expect("default_ms");
        // Give the main table some rows and fill DATA (tiled complex array).
        let (read, mut wt) = WritableTable::open_for_update(path).expect("open update");
        wt.addrows(nrow);
        let _ = read;
        let data_idx = wt
            .desc()
            .columns
            .iter()
            .position(|c| c.name == "DATA")
            .expect("DATA column");
        for r in 0..nrow {
            let mut vals = Vec::with_capacity((nchan * ncorr) as usize);
            for i in 0..(nchan * ncorr) {
                vals.push((r as f32 + 0.5, i as f32));
            }
            wt.putcell(
                data_idx,
                r,
                RecordValue::Array(ArrayValue {
                    shape: vec![nchan, ncorr],
                    data: ArrayData::Complex(vals),
                }),
            )
            .expect("put DATA");
        }
        wt.flush().expect("flush");
        // NUM_CHAN in the SPECTRAL_WINDOW subtable.
        let spw_dir = path.join("SPECTRAL_WINDOW");
        let (sr, mut swt) = WritableTable::open_for_update(&spw_dir).expect("spw");
        swt.addrows(1);
        let _ = sr;
        let idx = swt
            .desc()
            .columns
            .iter()
            .position(|c| c.name == "NUM_CHAN")
            .expect("NUM_CHAN");
        swt.putcell(idx, 0, RecordValue::Int(nchan as i32))
            .expect("put NUM_CHAN");
        swt.flush().expect("spw flush");
    }

    #[test]
    fn adds_three_imaging_columns() {
        let dir = crate::testdir::TestDir::new(format!(
            "msutil-add-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let path = dir.join("t.ms");
        make_ms(&path, 3, 4, 2);
        let added = add_imaging_columns(&path).expect("add");
        assert_eq!(
            added,
            vec!["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"]
        );

        let t = Table::open(&path, false).expect("reopen");
        let names = t.colnames();
        for want in ["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"] {
            assert!(names.iter().any(|n| n == want), "{want} missing: {names:?}");
        }
        // IMAGING_WEIGHT is float, shape [nchan], 1-dim.
        let i = names.iter().position(|n| n == "IMAGING_WEIGHT").unwrap();
        let cd = parse_coldesc(&t.getcoldesc(i).expect("coldesc")).expect("parse");
        assert_eq!(cd.data_type, crate::record::DataType::Float);
        assert_eq!(cd.ndim, 1);
        assert_eq!(cd.shape, Some(vec![4]));
        // CHANNEL_SELECTION = [[0, 4]] (one SPW with NUM_CHAN=4).
        let midx = names.iter().position(|n| n == "MODEL_DATA").unwrap();
        let kw = t.getcolkeywords(midx).expect("colkeywords");
        assert!(kw.contains("CHANNEL_SELECTION"), "kw missing: {kw}");
        // The keyword is an int32 array [[0, nchan]] (one pair per SPW).
        assert!(kw.contains("[0,4]") || kw.contains("[0, 4]"), "kw={kw}");
    }

    #[test]
    fn idempotent() {
        let dir = crate::testdir::TestDir::new(format!(
            "msutil-idem-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let path = dir.join("t.ms");
        make_ms(&path, 2, 4, 2);
        add_imaging_columns(&path).expect("first");
        let added = add_imaging_columns(&path).expect("second");
        assert!(added.is_empty(), "second call re-added: {added:?}");
        let t = Table::open(&path, false).expect("reopen");
        assert_eq!(
            t.colnames()
                .iter()
                .filter(|n| *n == "IMAGING_WEIGHT")
                .count(),
            1
        );
    }

    #[test]
    fn remove_drops_exactly_three() {
        let dir = crate::testdir::TestDir::new(format!(
            "msutil-rm-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let path = dir.join("t.ms");
        make_ms(&path, 2, 4, 2);
        add_imaging_columns(&path).expect("add");
        let removed = remove_imaging_columns(&path).expect("rm");
        assert_eq!(
            removed,
            vec!["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"]
        );
        let t = Table::open(&path, false).expect("reopen");
        for gone in ["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"] {
            assert!(
                !t.colnames().iter().any(|n| n == gone),
                "{gone} still there"
            );
        }
        // DATA survived.
        assert!(t.colnames().iter().any(|n| n == "DATA"));
    }

    #[test]
    fn missing_data_column_errors() {
        let dir = crate::testdir::TestDir::new(format!(
            "msutil-nodata-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let path = dir.join("t.ms");
        // A non-MS table (no DATA column).
        let mut wt = WritableTable::create(
            path.clone(),
            crate::tabledesc::TableDesc {
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
                columns: vec![ColumnDesc {
                    name: "X".into(),
                    comment: String::new(),
                    data_type: crate::record::DataType::Double,
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
                    kind: crate::tabledesc::ColumnKind::Scalar(RecordValue::Double(0.0)),
                    tile_shape: None,
                }],
            },
        );
        wt.flush().expect("flush");
        let err = add_imaging_columns(&path).expect_err("no DATA");
        assert!(matches!(err, MsUtilError::NoDataColumn), "{err:?}");
    }
}
