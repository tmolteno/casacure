# Changelog

All notable changes to this project are documented here. Completed `TODO.md`
subtasks are moved here.

## [Unreleased]

### Added

- Project scaffolding: Cargo workspace with the `casacure` core crate
  (`crates/casacure`) and `casacure-python` pyo3 bindings crate
  (`crates/casacure-python`), pip-installable via maturin (`pyproject.toml`).
- casacore as a git submodule (shallow clone of casacore master, commit
  `56a917a`) for reference and compatibility testing.
- `types` module: `ValueType` enum covering all 11 CASA types with every
  python-casacore alias, canonical/`table.dat` names, numpy dtype mapping, and
  element sizes — 7 unit tests.
- Python bindings: `casacure.numpy_dtype()` / `casacure.casa_type()` type
  mapping functions.
- Comparison test framework (`tests/test_types_compat.py`): creates real CASA
  tables via python-casacore TaQL DDL and verifies casacure's type mapping
  against actual descriptor and `getcol` behaviour — 5 tests. Documented
  casacore quirks: no TaQL `USHORT`, `uchar`→`uint16` promotion in `getcol`,
  `getcolnp` unsupported for uchar/short/uint, 1-D strings as Python lists,
  TaQL boolean code `B`.
- `TODO.md` subtask tracker and `ARE_WE_CURED.md` progress document.
