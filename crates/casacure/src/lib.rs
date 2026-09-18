//! casacure: a Rust implementation of the CASA table system, sufficient to
//! replace casacore as the I/O backend of dask-ms.
//!
//! See `CASACORE_TO_CASA_RS.md` for the required functionality inventory.

pub mod aipsio;
pub mod record;
pub mod table;
pub mod tabledesc;
pub mod types;

pub use table::{parse_table_dat, parse_table_header, TableDat, TableError, TableHeader};
pub use tabledesc::{parse_table_desc, ColumnDesc, ColumnKind, TableDesc, TableDescError};
pub use types::{UnknownValueType, ValueType};
