//! casacure: a Rust implementation of the CASA table system, sufficient to
//! replace casacore as the I/O backend of dask-ms.
//!
//! See `CASACORE_TO_CASA_RS.md` for the required functionality inventory.

pub mod aipsio;
pub mod columnset;
pub mod ism;
pub mod record;
pub mod ssm;
pub mod table;
pub mod tabledesc;
pub mod types;

pub use columnset::{
    parse_column_set, parse_standard_stman, write_multi_column_set, write_standard_stman,
    ColumnInfo, ColumnSet, ColumnSetError, DataManager, DataManagerBlob, DmBlob, StandardStMan,
};
pub use ism::{write_ism_file, IsmError, IsmFile, IsmHeader, IsmIndex, WriteIsmColumn};
pub use ssm::{
    encode_scalar_cell, layout, read_array_cell, scalar_cell_size, write_standard_stman_file,
    SsmError, SsmIndex, StandardStManFile, StandardStManHeader, StandardStManLayout, WriteColumn,
    ARRAY_REF_SIZE,
};
pub use table::{
    build_table_dat, create_table, parse_table_dat, parse_table_header, TableCreateError, TableDat,
    TableError, TableHeader, ROWS_PER_BUCKET,
};
pub use tabledesc::{
    parse_table_desc, write_table_desc, ColumnDesc, ColumnKind, TableDesc, TableDescError,
};
pub use types::{UnknownValueType, ValueType};
