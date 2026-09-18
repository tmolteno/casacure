# casacure

A pure-Rust implementation of the CASA table system — the on-disk format used
by casacore for Measurement Sets and general tables — sufficient to replace
`casacore` as the I/O backend of dask-ms.

casacure reads and writes real casacore tables: the `StandardStMan`,
`IncrementalStMan` and `TiledColumnStMan` data managers, table metadata,
keywords, subtable linkage and a TaQL subset. It needs no C++ toolchain and
no dependency on casacore itself, so it builds and runs anywhere Rust builds,
including `aarch64`.

## Features

- StandardStMan / IncrementalStMan / TiledColumnStMan — read and write
- Scalar, fixed-shape and variable-shape array columns, strings, record columns
- Keywords, column keywords, dminfo and `::SUBTABLE` linkage
- TaQL `SELECT` (WHERE / ORDERBY / GROUPBY / UNIQUE) and `CREATE TABLE`
- Measurement Set schema (`required_ms_desc` / `complete_ms_desc`) and
  `default_ms` with the full 17-subtable tree
- Binary-compatible with casacore: tables written by casacure open in real
  casacore, and vice-versa

## Author

Tim Molteno ([tim@elec.ac.nz](mailto:tim@elec.ac.nz))

## Usage

Add the dependency:

```toml
[dependencies]
casacure = "0.1"
```

Create and write a small table in one shot:

```rust
use casacure::record::{DataType, RecordValue};
use casacure::tabledesc::{ColumnDesc, ColumnKind, TableDesc};
use std::path::Path;

let desc = TableDesc {
    name: String::new(),
    version: String::new(),
    comment: String::new(),
    keywords: Default::default(),
    private_keywords: Default::default(),
    columns: vec![ColumnDesc {
        name: "TIME".into(),
        comment: String::new(),
        data_type: DataType::Double,
        data_manager_type: "StandardStMan".into(),
        data_manager_group: "StandardStMan".into(),
        options: 0,
        ndim: -1,
        shape: None,
        max_length: 0,
        keywords: Default::default(),
        kind: ColumnKind::Scalar(RecordValue::Double(0.0)),
    }],
};

let values = vec![vec![
    RecordValue::Double(0.0),
    RecordValue::Double(1.5),
    RecordValue::Double(3.0),
]];
casacure::create_table(Path::new("sample.tab"), &desc, &values)?;
```

Or build incrementally with `WritableTable`, then read it back:

```rust
let mut wt = casacure::WritableTable::create("sample.tab", desc);
wt.addrows(3);
wt.putcell(0, 2, RecordValue::Double(3.0))?;
wt.flush()?;

let t = casacure::Table::open("sample.tab", true)?;   // readonly = true
assert_eq!(t.colnames(), ["TIME"]);
assert_eq!(t.nrows(), 3);
println!("{:?}", t.getcol(0, 0, 3)?);                 // Vec<RecordValue>
```

Run TaQL against an open table:

```rust
use casacure::taql::{execute, TaqlResult};
if let TaqlResult::Query(out) =
    execute("SELECT * FROM $1 WHERE TIME > 1.0", &[&t])?
{
    // `out` has columns in the same shape as `create_table` expects.
}
```

The example binary `create_sample_table` writes a multi-manager table (SSM +
ISM + TSM) that real casacore can open:

```sh
cargo run -p casacure --example create_sample_table -- /tmp/sample.tab
```

## Python bindings

The companion `casacure-python` crate ships the same table engine as the
`pip install casacure` package, exposing a python-casacore-compatible
`casacure.tables` surface. dask-ms then runs on casacure (its entire test
suite passes against it); see the repository `README.md` for the drop-in
integration steps.

## License

Licensed under the LGPL-3.0-or-later.
