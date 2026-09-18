//! casacure: a Rust implementation of the CASA table system, sufficient to
//! replace casacore as the I/O backend of dask-ms.
//!
//! See `CASACORE_TO_CASA_RS.md` for the required functionality inventory.

pub mod types;

pub use types::{UnknownValueType, ValueType};
