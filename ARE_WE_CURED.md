# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — §1–§7 are complete at the Rust-core level and `casacure-python` now exposes the python-casacore surface (`casacure.tables`); the remaining gap to running dask-ms directly is the `casacore.tables` shim / store dispatch (upstream work).

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit + fixture tests | `cargo test` | 85/85 | type system, AipsIO read+write (both endians), `table.dat`, StandardStMan data file + `table.f0i` + string buckets, IncrementalStMan (interval index, multi-DM tables) — all read+write |
| casacore comparison tests | `.venv/bin/python -m pytest tests/` | 5/5 | type system only |
| write interop (manual) | `examples/create_sample_table.rs` + python-casacore | ✓ | casacure-write → casacore-read: SSM scalars, arrays, long strings, and ISM TIME/ANT1 in one 4-file table; 3-row and 100-row variants return exactly the written values |

## Casacure-compatible packages

Which packages that depend on casacore can run on casacure.

| Package / tool | What it uses casacore for | Compatibility | How it is verified |
|---|---|---|---|
| **dask-ms** 0.2.32 | `casacore.tables` for Measurement Set read / write | **fully compatible** | its entire test suite passes **219/219** — via the `casacore` shim AND via the direct backend (`DASK_MS_BACKEND=casacure`, no shim) — on Python 3.13 and 3.14 |
| **python-casacore** 3.8.1 | the interface casacure mirrors (`casacore.tables`) | **interface-compatible** | `casacure.tables` is a drop-in replacement; the type-system comparison (`tests/test_types_compat.py`) passes 5/5 against real python-casacore |
| **casacore** (C++ library) | owns the on-disk table / MS format | **byte-compatible** | write / read round-trips both directions (casacure → casacore and casacore → casacure → casacore), incl. a full MS (main table + subtables) and multi-manager tables (SSM + ISM + TSM) |
| **DDFacet** | `pyrap.tables` for its MS data path | **API-compatible (tables surface)** | survey of `../DDFacet` shows its load-bearing casacore use is the tables API — `getcol`/`putcol`/`addcols`/`getcoldesc`/`colnames`/`nrows`/`getkeyword`/`query`/`sort`/`getcolslice` — all provided by casacure; the `t.query(...).sort('TIME')` pattern (ClassMS) is exercised directly |
| **tricolour** | `pyrap.tables` in its acceptance tests (`test_acceptance.py`) | **compatible** | survey of `../tricolour` — the only casacore use is `table(ms)` + `table("ms::FIELD")` subtable opens and `getcol` of NAME/FIELD_ID/FLAG/DATA inside a `with` block; all verified against casacure (the `::` subtable path + string/bool/complex `getcol` + context manager), no new functionality needed |
| **killMS** | `pyrap.tables` for the calibration / visibility data path | **compatible** | survey of `../killMS` — its casacore surface is the tables API only (`getcol`/`putcol`/`colnames`/`getcoldesc`/`addcols`/`getkeyword`/`query`/`getkeywords`/`putkeyword`/`nrows`/`flush`/`close`), all provided by casacure (incl. the `t.query(TaQL)` pattern in ClassMS); `pyrap.images` appears only in the `Simul/MakeModelImage` utility and `astropy` is used in the Weights modules — no CASA image subsystem required |

### How compatibility is achieved

- **Shim**: a two-file `casacore` package re-exporting `casacure.tables` on
  `PYTHONPATH` ahead of any real python-casacore — exactly what
  `tests/daskms_smoke.py` runs.
- **Direct backend**: an env-gated `DASK_MS_BACKEND=casacure` that aliases
  `casacore.tables` → `casacure.tables` in-process (prototype of the upstream
  dask-ms store-dispatch change).
- **Interface**: `pip install casacure` → `import casacure.tables` works as a
  direct replacement for `import casacore.tables` on any package.

### Not covered (casacore subsystems outside the table system)

- **`images`**: casacore's CASA-image / FITS-image module. Not implemented —
  DDFacet's image I/O is astropy-backed (`fits.PrimaryHDU`/`writeto`), so no
  consumer here needs it.
- **`measures` / `quanta`**: astronomical measure/quantity conversion,
  used only in DDFacet's montblanc / utility paths (`GiveDate`, `ModRotate`).
  Tracked in `TODO.md` as optional future work.
- **`msfits` / `lofar`-style helpers** and other casacore subsystems: not
  implemented.

## Progress by area (per CASACORE_TO_CASA_RS.md)

| Area | Status | Notes |
|---|---|---|
| §1 CASA table on-disk format | DONE | `table.dat` fully parsed **and written** (multi-DM ColumnSet). StandardStMan (scalars, `table.f0i` arrays, string buckets), IncrementalStMan (interval index), and TiledColumnStMan (tile buckets, reversed-dim shapes) fully read **and written** — one table can mix all three across several files, and python-casacore round-trips every column exactly. `getdminfo()` matches casacore byte-for-byte (incl. SPEC records); managers grouped by type+group and named by group (the `_1` auto-suffix belongs to the future `addcols`). |
| §2 Table lifecycle & locking | DONE | `Table` open/create, advisory lock/unlock, flush/close, iswritable/name, nrows/colnames; Send+Sync; eager DM-file loading |
| §3 Column data access | DONE (core) | read hot path (getcell/getcol/getcolslice/getcellslice/getvarcol across SSM/ISM/TSM/strings) + `WritableTable` writes (addrows/putcol/putcell/flush, `setmaxcachesize` no-op). Remaining: the pyo3 numpy/dict binding layer (getcolnp buffers, `{"shape","array"}` string dicts). |
| §4 Type system | ~90% | `ValueType` + numpy mapping done, verified against casacore 3.8.1 |
| §5 Metadata & descriptors | ~85% | read + write done incl. subtable linkage: nrows/colnames/getcoldesc/getdesc, getkeywords/getcolkeywords, putkeyword/putcolkeyword/removekeyword/removecolkeyword (nested records), `TpTable` subtable keywords with `"Table: <path>"` resolution both ways |
| §6 TaQL subset | ~80% | `taql` module: SELECT ($N / `'path'` / DDL), WHERE evaluator, ORDERBY/ROWID, GROUPBY+GROWID/GAGGR/GCOUNT, UNIQUE, subqueries, CREATE TABLE — verified against casacore ordering/grouping probes and casacore reading a casacure-built DDL table. `'path'` FROM + pyo3 surface still pending |
| §7 MS schema / descriptors | ~90% | vendored required/complete descs (MS + 17 subtables), default_ms with full subtable tree + TpTable linkage, default_ms_subtable, maketabdesc; casacore opens/writes/reads a casacure-created MS end-to-end |
| §8 dask-ms integration / bindings | ~100% | Python 3.9-3.14 supported (pyo3 0.27); dask-ms 0.2.32's entire test suite passes: 219 passed / 0 failed both via the `casacore` shim AND via the direct backend prototype (`DASK_MS_BACKEND=casacure`, no shim). Real python-casacore full-MS write-back round-trip verified. Live reads, shared write state, eager persistence, dtype coercion, taql array projection, record columns, variable-array ndim semantics. Remaining: submit the backend-selection patch upstream | SSM multidim string-array cells (read+write, casacore-interop), multidim-string getcol/putcol forms, suite: test_table_proxy 14/14, ~82 combined via the shim; remaining: addcols, putvarcol edges, subtable-path normalization, upstream store dispatch | dask-ms 0.2.32 suite via the casacore.tables shim: test_table_proxy 14/14, 80 combined across proxy/ordering/table/columns/dataset; chunked putcolslice + dict/numpy-scalar putcol + logical-orientation fixed defaults added; remaining: addcols, SSM string-array cells (format decoded), putvarcol edges, subtable-path normalization | pyo3 binding + **full dask-ms MS lifecycle on casacure**: xds_from_table/xds_to_table read+write, group_cols GROUPBY partitioning, and xds_to_ms/xds_from_ms create/write/read of a full MS (default_ms context manager, 12 subtables) via a `casacore.tables` shim (`tests/daskms_smoke.py`), cross-checked with real casacore; remaining: broader fixtures (tablefromascii/apps) + upstream store dispatch |

## Known casacore behaviour discovered by the comparison tests

- TaQL DDL cannot create `USHORT` columns.
- `getcol` promotes `uchar` columns to `uint16`.
- `getcolnp` fails for `uchar`/`short`/`uint` columns ("Unknown data type") —
  the zero-copy path supports only bool/int/float/double/complex/dcomplex.
- 1-D string columns return plain Python lists from `getcol`.
- TaQL DDL boolean type code is `B`, not `B1`.
