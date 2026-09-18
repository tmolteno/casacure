//! casacure: a Rust implementation of the CASA table system, sufficient to
//! replace casacore as the I/O backend of dask-ms.
//!
//! See `CASACORE_TO_CASA_RS.md` for the required functionality inventory.

pub mod aipsio;
pub mod columnset;
pub mod record;
pub mod ssm;
pub mod table;
pub mod tabledesc;
pub mod types;

pub use columnset::{
    parse_column_set, parse_standard_stman, ColumnInfo, ColumnSet, ColumnSetError, DataManager,
    DataManagerBlob, StandardStMan,
};
pub use ssm::{scalar_cell_size, SsmError, SsmIndex, StandardStManFile, StandardStManHeader};
pub use table::{parse_table_dat, parse_table_header, TableDat, TableError, TableHeader};
pub use tabledesc::{parse_table_desc, ColumnDesc, ColumnKind, TableDesc, TableDescError};
pub use types::{UnknownValueType, ValueType};
