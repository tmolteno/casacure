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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let out = args.nth(1).unwrap_or_else(|| "sample.tab".into());
    // Optional row count (default 3); exercises multi-bucket layout.
    let nrows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let desc = TableDesc {
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
        ],
    };

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
            .map(|i| RecordValue::String(format!("row{}", i)))
            .collect(),
        arr_values,
    ];

    let written = casacure::create_table(Path::new(&out), &desc, &values)?;
    println!(
        "wrote {nrows} rows in {} files under {}",
        written.len(),
        out
    );
    Ok(())
}
