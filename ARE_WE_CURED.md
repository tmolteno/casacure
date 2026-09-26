# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — §1–§7 are complete at the Rust-core level and `casacure-python` now exposes the python-casacore surface (`casacure.tables`); the remaining gap to running dask-ms directly is the `casacore.tables` shim / store dispatch (upstream work).

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit + fixture tests | `cargo test` | 85/85 | type system, AipsIO read+write (both endians), `table.dat`, StandardStMan data file + `table.f0i` + string buckets, IncrementalStMan (interval index, multi-DM tables) — all read+write |
| casacore comparison tests | `PYTHONPATH=/tmp/cpb:/tmp/shim .venv/bin/python -m pytest tests/` | 14/14 | type system + python-casacore `test_table.py` port (9 tests: datatypes, putdata, addcolumns, keywords, subset, subtables, tableascii, complete/required descs) |
| write interop (manual) | `examples/create_sample_table.rs` + python-casacore | ✓ | casacure-write → casacore-read: SSM scalars, arrays, long strings, and ISM TIME/ANT1 in one 4-file table; 3-row and 100-row variants return exactly the written values |

## Speed and memory relative to casacore (3.8.8)

This is casacure / python-casacore 3.8.1 on the same machine and the same
data, so **< 1 means casacure is faster or lighter**.  The method and raw
numbers are in `BENCHMARK.md`.

| workload | time | peak RSS |
|---|---|---|
| dask-ms chunked read of a 977 MiB DATA column, 25 000-row chunks | 1.03 | 1.04 |
| same, 1000-row chunks | **0.40** | 1.35 |
| dask-ms write of a new MS (256k rows, 2000-row chunks) | **0.60** | **0.80** |
| skarabina flag + 32x average + `--msout`, MeerKAT scan (11 GB) | **0.80** | **0.77** |
| skarabina flag, `--write-changed-only`, same scan | **0.62** | **0.54** |
| `casacure-bench`: whole-column putcol / getcol / taql on a 20k-row cached table | 3.0 / 3.5 / 4.8 | — |

**On MS-shaped work through dask-ms, casacure is at parity or faster.**
Chunked scans and new MS writes are I/O-bound, and both flagging pipelines
measured run 20-40 % faster, in 23-46 % less memory.  **On small, fully
cached tables it is 3-5x slower per call**: the Python bridging and cell
packaging dominate there, not I/O.  Before 3.8.8, writing a new table was
the exception: memory grew with the table and time grew quadratically (57.6 s
and 1.9 GiB for 256k rows, against casacore's 2.4 s and 234 MiB).

Direct arrays: before 3.8.9, casacure could not read StandardStMan Direct
fixed-shape array columns (column option 5, e.g. an MS's ANTENNA POSITION).
It now reads and writes them in casacore's inline layout.  On a MeerKAT
scan, all 132 columns of the MS and its subtables read as in casacore.  An
MS written through skarabina, read back by casacore, is identical to the one
python-casacore writes.

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
| **DDFacet** (recheck) | `pyrap.tables` for its MS data path; `pyrap.images`/`measures`/`quanta` in SkyModel/Imager/Data | **API-compatible (tables surface)** | re-audit of the github.com/saopicc/DDFacet tarball confirms the earlier survey: the MS/ClassMS tables path (`getcol`/`putcol`/`addcols`/`query`/`sort`/…) is fully covered by casacure; the image/measure subsystems remain outside the table system (already documented) |
| **CubiCal** | `casacore.tables` (incl. `getcolslice`/`taql`/`table.taql`/`putkeyword`/`addcols`); `pyrap.measures`/`quanta` in its parallactic/degridder machines | **compatible (tables surface)** | audit — 25 distinct table APIs, all provided by casacure; `pyrap.measures`/`quanta` (parallactic_machine, DDFacetSim, MBTiggerSim) is outside the table system |
| **skarabina** | `dask-ms` + `casacore.tables` for 1GC flagging | **compatible (352/355 tests)** | runs end-to-end on casacure via the tmolteno/dask-ms fork (`DASK_MS_BACKEND=casacure`, selected automatically by skarabina's `__init__`). Its own suite: 352/355 pass; the 3 failures all test the `--write-changed-only` hardlink block-sharing, which needs casacore's exact per-column storage layout (out of scope; skarabina falls back to a full write there). |
| **backend fixes (this session)** | dask-ms fork + casacure table surface | — | dask-ms: `DASK_MS_BACKEND=casacure` + `casacure`/`backend` extras. casacure: `default_ms_subtable` builds the standard schema; 1-element array cells stay `Array`; numpy `<U` string ndarrays store into scalar/array columns; `getsubtables()` returns absolute paths; `removecol(s)`; TiledShapeStMan created as StandardStMan; `getcell` keeps `(1,1)` cells 2-D. |
| **meqtrees-cattery** | `casacore.tables` for MS reading; `pyrap.measures` in Calico refraction | **compatible (tables surface)** | audit — `getcol`/`putcol`/`putkeyword`/`query`/`tableexists`/`tabledelete`/`tablecopy` all provided by casacure; `pyrap.measures` (Calico `solvable_refraction`) is outside the table system |
| **ska-sdp-wflow-low-selfcal** | no casacore imports in package code | **compatible (n/a)** | workflow glue; casacore use (if any) lives in its xradio/dask-ms deps |
| **solarkat** | `casacore.tables` for solar quick-look | **compatible** | audit — tables-only (`getcol`/`query`), all provided by casacure |
| **radiotools** | `casacore.tables` (`getcol`) | **compatible** | audited tarball's only casacore use is `getcol`, provided by casacure |
| **spinifex** | `casacore.tables` for LOFAR beamformer/transient data | **compatible** | audit — tables-only (`getcol`/`getcell`/`colnames`), all provided by casacure |
| **jiveplot** | `casacore.tables` (incl. `table.taql`) plus `pyrap.quanta` for VLBI unit conversions | **compatible (tables surface)** | audit — `table`/`taql`/`query`/`getcol`/`colnames` all provided by casacure; `pyrap.quanta` (parsers/plotiterator/ms2util) is outside the table system |
| **jolly-roger** | `casacore.tables` for observation selection | **compatible** | audit — tables-only (`getcol`/`putcol`/`addcols`/`getcoldesc`/`colnames`), all provided by casacure |
| **flint** | `casacore.tables` for calibration/imaging data | **compatible** | audit — tables-only (`getcol`/`putcol`/`addcols`/`getcoldesc`/`getdminfo`/`colnames`/`query`), all provided by casacure |
| **bipp** | `casacore.tables` (incl. the `table.taql` method) | **compatible** | audit — `taql`/`table`/`getcell`/`getcol`/`getcolslice`/`colnames`, all provided by casacure (the `table.taql()` method was added) |
| **P-AIRCARS** | `casacore.tables` for imaging-pipeline MS I/O | **compatible** | audit — tables-only (`addrows`/`putcell` + the `make*desc` builders), all provided by casacure |
| **xradio** | `casacore.tables` for the MS (measurement_set) path; `casacore.images` in its image submodule | **compatible (tables surface)** | audit — 29 distinct table APIs incl. `default_ms`/`complete_ms_desc`/`taql`/`table.taql`/the `make*desc`+`makedminfo` builders, all provided by casacure; `casacore.images` is confined to `image/_util/_casacore/xds_from_casacore` (image I/O is outside the table system) |
| **graphviper** | no casacore imports in package code | **compatible (n/a)** | `casacore` use sits in its astroviper/xradio deps, not this repo |
| **astroviper** | no load-bearing casacore use (MS handling is delegated to xradio/dask-ms) | **compatible (n/a)** | audited tarball has no casacore imports outside docstrings |
| **RFInder** | `casacore.tables` for RFI flagging | **compatible** | audit — tables-only (`table`/`taql`/`getcol`/`addcols`/`getdminfo`/`getcell`/`putcol`/`query`/`maketabdesc`), all provided by casacure |
| **b4r** | `casacore.tables` for VLBI correlation-sandbox tables | **compatible** | audit — tables-only (`table`/`maketabdesc`/`makescacoldesc`/`putcol`), all provided by casacure |
| **avica** | `casacore.tables` for catalog/calibrator pipelines | **compatible** | audit — tables-only (`getcol`/`putcol`/`getcell`/`putcell`/`query`/`nrows` + the `make*desc` builders), all provided by casacure |
| **dstools** | no casacore imports in package code | **compatible (n/a)** | audited tarball has zero `casacore`/`pyrap` API use |
| **polynomial_preprocessing** | no casacore imports in package code | **compatible (n/a)** | audited tarball mentions `casacore` only in prose; zero API use |
| **solar-viewer** | no casacore imports in package code | **compatible (n/a)** | audited tarball has zero `casacore`/`pyrap` imports |
| **ALMASim** | `casacore.tables` for MS product I/O | **compatible** | audit — tables-only (`getcol`/`putcell`/`putcol`/`getcell`/`colnames`/`nrows`/`table`); the heavy lifting is astropy, all table APIs provided by casacure |
| **sgains** | no casacore imports in package code | **compatible (n/a)** | listed among GitHub's PACKAGE dependents; audited tarball has zero `casacore`/`pyrap` imports — nothing for casacure to satisfy |
| **FixMS** | `casacore.tables` for MS repair/edit | **compatible** | audit of the github.com/AlecThomson/FixMS tarball — tables-only (`getcol`/`putcol`/`getcell`/`putcell`/`getcoldesc`/`addcols`/`colnames`/`nrows`/`addrows`/`tableexists`/`tablecopy`), all provided by casacure |

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
- **`quanta` (the quantity/unit core)**: **shipped** — `casacure.quanta` is a
  drop-in for `casacore.quanta` (`Quantity`, unit parsing/conversion, SI
  canonical form, arithmetic, `near`/`nearabs`, the physical-constant and
  unit/prefix tables, and the MJD/time/angle value classes). Verified
  byte-for-byte against python-casacore 3.8.1 (see `tests/test_quanta.py`).
  This covers the `pyrap.quanta` uses the surveys found (jiveplot unit
  conversions, CubiCal's degridder machines, DDFacet `GiveDate`).
- **`measures` (the M* transforms)**: astronomical measure conversion
  (`direction`/`epoch`/`uvw`/`doppler` and the parallactic-angle machinery,
  which needs the ephemeris-backed `measures` engine). Not implemented —
  used only in DDFacet's montblanc / utility paths (`GiveDate`, `ModRotate`,
  CubiCal's parallactic_machine); tracked in `TODO.md` as optional future
  work.
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
| §6 TaQL subset | DONE for Tiers A+B (dask-ms dialect since 0.2.2) | `taql` module: SELECT ($N / `'path'` / DDL), WHERE evaluator, ORDERBY/ROWID, GROUPBY+GROWID/GAGGR/GCOUNT, UNIQUE, subqueries, CREATE TABLE — plus the full Tier A function library, `LIKE`/`IN`, `HAVING`/`OFFSET`/`COUNT`/`g*` aggregates/array-cell aggregates/presence tests, and the Tier B statements (UPDATE/DELETE/INSERT/DROPTABLE/ALTER/SHOW/HELP/CALC) — all shipped through the `taql`/`table.taql()` pyo3 surface, verified against casacore ordering/grouping probes and casacore reading a casacure-built DDL table. The dialect dask-ms and the surveyed packages generate is complete; the remaining surface (Tier C) is assessed in the "TaQL surface beyond the dask-ms dialect" section below and tracked in TODO.md §5. |
| §7 MS schema / descriptors | ~90% | vendored required/complete descs (MS + 17 subtables), default_ms with full subtable tree + TpTable linkage, default_ms_subtable, maketabdesc; casacore opens/writes/reads a casacure-created MS end-to-end |
| §8 dask-ms integration / bindings | ~100% | Python 3.9-3.14 supported (pyo3 0.27); dask-ms 0.2.32's entire test suite passes: 219 passed / 0 failed both via the `casacore` shim AND via the direct backend prototype (`DASK_MS_BACKEND=casacure`, no shim). Real python-casacore full-MS write-back round-trip verified. Live reads, shared write state, eager persistence, dtype coercion, taql array projection, record columns, variable-array ndim semantics. Remaining: submit the backend-selection patch upstream | SSM multidim string-array cells (read+write, casacore-interop), multidim-string getcol/putcol forms, suite: test_table_proxy 14/14, ~82 combined via the shim; remaining: addcols, putvarcol edges, subtable-path normalization, upstream store dispatch | dask-ms 0.2.32 suite via the casacore.tables shim: test_table_proxy 14/14, 80 combined across proxy/ordering/table/columns/dataset; chunked putcolslice + dict/numpy-scalar putcol + logical-orientation fixed defaults added; remaining: addcols, SSM string-array cells (format decoded), putvarcol edges, subtable-path normalization | pyo3 binding + **full dask-ms MS lifecycle on casacure**: xds_from_table/xds_to_table read+write, group_cols GROUPBY partitioning, and xds_to_ms/xds_from_ms create/write/read of a full MS (default_ms context manager, 12 subtables) via a `casacore.tables` shim (`tests/daskms_smoke.py`), cross-checked with real casacore; remaining: broader fixtures (tablefromascii/apps) + upstream store dispatch |

### TaQL: the full casacore surface vs the implemented subset

§6 implements the dialect dask-ms and the surveyed dependent packages generate
plus a DDL subset for fixtures — a deliberate scope, not an accident (the
`TODO.md` §8 surveys exercise only that dialect). `'path'` FROM and the
`taql` / `table.taql()` pyo3 surface are shipped; the earlier row note calling
them "pending" was stale.

Tier A and Tier B below are **implemented** (shipped through the same
`taql`/`table.taql()` surface); the table keeps their portability notes as the
record of how they were built. Only Tier C remains out of scope.

The remaining casacore TaQL surface beyond the shipped subset (inventoried
from `casacore/tables/TaQL/TableGram.yy` + `TableParseFunc.cc`; casacore
ships 50 TaQL test files as the reference) splits into three tiers:

| Tier | What | Status / effort |
|---|---|---|
| **A — query-language completion** | scalar/string/stats/array-function library (the ~215 dispatched names → ~120 implemented incl. the `s`/`running`/`boxed` statistic families), `IN` set operator, `LIKE`/`ILIKE`/regex/`sqlpattern`, `HAVING`, `OFFSET`, `COUNT`, GROUPBY-specific `g*` aggregates, array-cell aggregates in SELECT, presence tests (`isnull`/`isdefined`/`iscolumn`/…) | **DONE** — pure `taql.rs` work on the existing read/write primitives (a small internal regex engine covers the `~` operators, so no dependency was needed) |
| **B — new statements** | `UPDATE ... SET`, `INSERT INTO`, `SELECT ... INTO TABLE`, `DELETE FROM`, `DROPTABLE`, `ALTER TABLE` (ADD/DROP/RENAME COLUMN, SET/REMOVE keyword), `SHOW TABLE`/`HELP`, `CALC` | **DONE** — sits on the existing write primitives (addrows / putcol / addcols / `removerows` / `renamecol` as cell-store operations), all exposed through the pyo3 `taql()`/`table.taql()` surface |
| **C — other subsystems** | units & quantities (`5*deg`), spherical-angle geometry (`angdist`/`cones`/`findcone`), masked arrays, the `VirtualTaQLColumn` data manager, `derivedmscal.*`/`mscal.*` UDFs | needs measures-quanta-class machinery or a masked-array value type — separate projects, outside the "replace casacore as the dask-ms I/O backend" goal |

Bottom line: **Tiers A + B are shipped**; the remaining TaQL language is
Tier C, which is out of scope for the current goal. Tracked in TODO.md §5.

## Known casacore behaviour discovered by the comparison tests

- TaQL DDL cannot create `USHORT` columns.
- `getcol` promotes `uchar` columns to `uint16`.
- `getcolnp` fails for `uchar`/`short`/`uint` columns ("Unknown data type") —
  the zero-copy path supports only bool/int/float/double/complex/dcomplex.
- 1-D string columns return plain Python lists from `getcol`.
- TaQL DDL boolean type code is `B`, not `B1`.
