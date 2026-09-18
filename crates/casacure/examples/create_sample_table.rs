//! Write a small CASA table with casacure, to hand to python-casacore (or
//! CASA) for the byte-level write-interop proof:
//!
//! ```sh
//! cargo run -p casacure --example create_sample_table -- /tmp/sample.tab
//! python -c "from casacore.tables import table; t=table('/tmp/sample.tab', ack=False); print(t.getcol('R4')); print(t.nrows())"
//! ```

use casacure::record::{DataType, RecordValue, TableRecord};
use casacure::tabledesc::{ColumnDesc, ColumnKind, TableDesc};
use std::path::Path;

fn empty_record() -> TableRecord {
    TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    }
}

fn scalar(name: &str, dt: DataType, default: RecordValue) -> ColumnDesc {
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
    }
}

/// A fixed-shape array column: `shape` in CASA dim order (reversed relative
/// to the logical row-major shape), `option = 4` (FixedShape).
fn arr(name: &str, dt: DataType, casa_shape: &[i64]) -> ColumnDesc {
    ColumnDesc {
        name: name.into(),
        comment: String::new(),
        data_type: dt,
        data_manager_type: "StandardStMan".into(),
        data_manager_group: "StandardStMan".into(),
        options: 4,
        ndim: casa_shape.len() as i32,
        shape: Some(casa_shape.to_vec()),
        max_length: 0,
        keywords: empty_record(),
        kind: ColumnKind::Array,
    }
}

fn tsm_arr(name: &str, dt: DataType, casa_shape: &[i64]) -> ColumnDesc {
    let mut d = arr(name, dt, casa_shape);
    d.data_manager_type = "TiledColumnStMan".into();
    d.data_manager_group = "TiledData_GROUP".into();
    d
}

fn ism_scalar(name: &str, dt: DataType, default: RecordValue) -> ColumnDesc {
    let mut d = scalar(name, dt, default);
    d.data_manager_type = "IncrementalStMan".into();
    d.data_manager_group = "IncrementalStMan".into();
    d.options = 1; // Direct
    d
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let out = args.nth(1).unwrap_or_else(|| "sample.tab".into());
    // Optional row count (default 3); exercises multi-bucket layout.
    let nrows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut desc = TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: empty_record(),
        private_keywords: empty_record(),
        columns: vec![
            scalar("I4", DataType::Int, RecordValue::Int(0)),
            scalar("R4", DataType::Float, RecordValue::Float(0.0)),
            scalar("NAME", DataType::String, RecordValue::String(String::new())),
            // Fixed-shape 2x3 complex array (CASA dim order [3,2]).
            arr("ARR", DataType::Complex, &[3, 2]),
            // MS-style index columns stored incrementally (IncrementalStMan).
            ism_scalar("TIME", DataType::Double, RecordValue::Double(0.0)),
            ism_scalar("ANT1", DataType::Int, RecordValue::Int(0)),
            // MS-style visibility data stored in tiles (TiledColumnStMan),
            // fixed-shape 2x3 dcomplex per row.
            tsm_arr("DATA", DataType::DComplex, &[3, 2]),
        ],
    };
    // A few keywords to exercise the keyword write path.
    desc.keywords.set("MS_VERSION", RecordValue::Int(56));
    desc.columns[0]
        .keywords
        .set("UNITS", RecordValue::String("Jy".into()));

    let arr_values = (0..nrows)
        .map(|row| {
            use casacure::record::{ArrayData, ArrayValue};
            RecordValue::Array(ArrayValue {
                shape: vec![2, 3],
                data: ArrayData::Complex((1..=6).map(|k| (row as f32, k as f32)).collect()),
            })
        })
        .collect::<Vec<_>>();

    let values = vec![
        (0..nrows).map(|i| RecordValue::Int(i as i32)).collect(),
        (0..nrows)
            .map(|i| RecordValue::Float(i as f32 + 0.5))
            .collect(),
        (0..nrows)
            // Long enough to exercise the SSM string buckets (> 8 chars).
            .map(|i| {
                RecordValue::String(format!(
                    "row{i}: a label far longer than the eight-character inline limit"
                ))
            })
            .collect(),
        arr_values,
        // MS-style index columns stored incrementally (IncrementalStMan).
        (0..nrows)
            .map(|i| RecordValue::Double((i as f64) / 2.0))
            .collect(),
        (0..nrows)
            .map(|i| RecordValue::Int((i / 3) as i32))
            .collect(),
        // Tiled visibility data.
        (0..nrows)
            .map(|row| {
                use casacure::record::{ArrayData, ArrayValue};
                RecordValue::Array(ArrayValue {
                    shape: vec![2, 3],
                    data: ArrayData::DComplex((1..=6).map(|k| (row as f64, k as f64)).collect()),
                })
            })
            .collect(),
    ];

    let written = casacure::create_table(Path::new(&out), &desc, &values)?;
    println!(
        "wrote {nrows} rows in {} files under {}",
        written.len(),
        out
    );
    Ok(())
}
