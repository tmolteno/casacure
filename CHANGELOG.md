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
- Rust-side comparison tests (`crates/casacure/tests/compat_fixtures.rs`):
  `tests/make_fixtures.py` writes real casacore tables plus `manifest.json`
  into `tests/fixtures/` (gitignored), and manifest-driven
  `cargo test -p casacure` integration tests verify `ValueType` parsing and
  the observed `getcol` dtypes (including the uchar→uint16 promotion quirk)
  without needing the pyo3 bindings — 3 tests.
- GitHub issues #1–#7 for each major functionality area (§1 on-disk format,
  §2 lifecycle/locking, §3 column access, §5 metadata, §6 TaQL, §7 MS schema,
  dask-ms integration).
- CI: GitHub Actions workflow (`.github/workflows/ci.yml`) — generates the
  casacore fixtures, runs `cargo test --workspace`, `cargo fmt --check`,
  `cargo clippy -D warnings`, builds the bindings with `pip install .`, and
  runs the pytest comparison suite.
