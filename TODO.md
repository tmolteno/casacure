# TODO

Needed next steps, broken into small subtasks. Add subtasks here **before**
starting them; remove each when completed and log it in `CHANGELOG.md`.

Work areas follow `CASACORE_TO_CASA_RS.md` (tracked as GitHub issues).

## 1. CASA table on-disk format (core prerequisite)

- [X] StandardStMan column storage: read
- [X] StandardStMan column storage: write
- [X] Scalar + fixed-shape array columns
- [X] Byte-level interop proof: read a casacore-written table
- [X] SSMStringHandler string buckets (variable strings > 8 chars)
- [X] Byte-level interop proof: write a table casacore can read
- [X] IncrementalStMan (Direct option) support
- [X] TiledColumnStMan (`{column}_GROUP`, reversed dim order DEFAULTTILESHAPE)
- [X] dminfo dict round-trip fidelity (incl. `_1` auto-suffix behaviour)

## 2. Table lifecycle and locking API

- [X] `table()` open + create
- [X] `lock()`/`unlock()` (advisory), `flush()`, `close()`
- [X] `iswritable()`, `name()`

## 3. Column data access (hot path)

- [X] `getcolnp` / `putcol` into preallocated numpy buffers (Rust `getcol` reads cells into a Vec; the numpy-buffer binding is the pyo3 layer)
- [X] `getcolslicenp` / `putcolslice` (inclusive `blc`/`trc` ends)
- [X] `getcol` / `getcolslice` / `getcell` / `getcellslice`
- [X] `getvarcol` / `putvarcol` (`{"rN": arr}` dicts) — getvarcol read; putvarcol via `WritableTable::putcol` (per-row array cells)
- [X] Multidim string columns as `{"shape", "array"}` dicts (strings returned as `RecordValue::String`; the dict shape is the pyo3 layer)
- [X] Accept numpy object arrays directly in `putcol` (no segfault wart) — the Rust layer takes `RecordValue`s; the wart is a python-casacore segfault, absent by construction; binding-side conversion is pyo3-layer work
- [X] `addrows`, `setmaxcachesize` (no-op) — via `WritableTable` (addrows + putcol + flush); `setmaxcachesize` is a no-op

## 4. Metadata and descriptors

- [X] `nrows`, `colnames`, `getcoldesc`
- [X] Table keywords: `putkeywords`/`removekeyword`
- [X] Column keywords: `putcolkeyword`/`removecolkeyword`
- [X] Public table-description dict API (`Table::getdesc`; replaces private `_getdesc`)
- [X] Subtable linkage via `TpTable` keywords (read: `"Table: <resolved path>"` strings; write: `./relative` storage + byte-level `TpTable` fields; dask-ms `is_subtable` discovery verified)

## 5. TaQL subset

- [X] `SELECT ... FROM $N` with `$N` table references
- [X] `WHERE` expression evaluator (user `taql_where` passthrough)
- [X] `ORDERBY`, `ROWID()`, and casacore's spaced `ORDER BY <expr>` form
      (python-casacore `test_table.py` port uses `order by ... desc`; covered
      by `order_by_spaced_form_matches_orderby`)
- [X] `GROUPBY` + `GROWID()`/`GAGGR()`/`GCOUNT()`
- [X] `SELECT UNIQUE col`, scalar subqueries
- [X] DDL: `CREATE TABLE ... LIMIT n` for test fixtures

## 6. MS schema / descriptors

- [X] Vendor canonical `required_ms_desc`/`complete_ms_desc` dicts as data (MS + 17 subtables)
- [X] `default_ms(path, tabdesc, dminfo)` with full subtable tree + keyword linkage
- [X] `default_ms_subtable`, `maketabdesc` (tests)

## 7. dask-ms integration

- [X] Cross-implementation round-trip gate (Rust-write → casacore-read and vice versa)
- [~] Full dask-ms test suite passing against casacure — dask-ms 0.2.32's own tests via the `casacore.tables` shim: **test_table_proxy 14/14; ~82 combined across proxy/ordering/table/columns/dataset**, incl. `test_dataset_multidim_string_column`; MS lifecycle + example_ms work end-to-end (`tests/daskms_smoke.py`). Added: chunked `putcolslice`, dict-form putcol (incl. numpy scalars and `{"shape","array"}` multidim strings split per row), logical-orientation fixed defaults, SSM multidim **string-array** cells (read+write via string buckets). Added: `addcols` (append columns to writable tables, incl. TiledColumnStMan layouts), TSM Bool tiles, Bool scalar storage verified as byte-per-row (bucket layout was wrongly bit-packing — fixed). Upstream store dispatch — VALIDATED by a prototype: an env-gated
`DASK_MS_BACKEND=casacure` alias in dask-ms's `__init__` routes `casacore.tables`
to `casacure.tables`, and the full dask-ms 0.2.32 suite passes 219/219 with
**no shim** (only `casacure` on the path). Remaining: submit that small backend
selection patch to dask-ms upstream (real python-casacore, when installed,
still wins by default).
- [ ] Register as a store type in `fsspec_store.py`/`dask_ms.py` dispatch (upstream, later)

## 8. python-casacore dependent-package surveys

Reverse dependents of python-casacore (sources: GitHub PACKAGE-dependents for
`casacore/python-casacore`, PyPI `requires_dist`, and the previous
ARE_WE_CURED.md survey). One entry per package: survey which casacore APIs it
uses and check casacure compatibility. Survey method/result → ARE_WE_CURED.md.

### GitHub PACKAGE dependents (casacore/python-casacore)

- [X] AlecThomson/FixMS — MS repair/edit tool
- [X] KrasnitzLab/sgains — gain-table reader/writer
- [X] MicheleDelliVeneri/ALMASim — ALMA observation simulator
- [X] RobertJaro/solar-viewer — solar image viewer
- [X] StephanSilvaS/polynomial_preprocessing — visibility preprocessing
- [X] askap-vast/dstools — ASKAP VAST data tools
- [X] avikhagol/avica — VLA calibrator/catalog pipeline
- [X] b4r-dev/b4r — VLBI correlation-sandbox tools
- [X] caracal-pipeline/RFInder — RFI flagging (caracal)
- [X] casangi/astroviper — MS fetch/convert library
- [X] casangi/graphviper — data-model/imaging graph library
- [X] casangi/xradio — MS/other radio data ↔ xarray interop
- [X] devojyoti96/P-AIRCARS — EHT imaging pipeline
- [X] epfl-radio-astro/bipp — EPFL radio interferometry pipeline
- [X] flint-crew/flint — FLINT calibration/imaging framework
- [X] flint-crew/jolly-roger — observation selection tool
- [X] haavee/jiveplot — VLBI data plotting
- [X] lofar-astron/spinifex — LOFAR beamformer/transient tools
- [X] radionets-project/radiotools — radio-astronomy utility functions
- [X] ratt-ru/solarkat — solar quick-look pipeline
- [X] ratt-ru/tricolour — surveyed (ARE_WE_CURED: uses table/table//getcol in tests only; works)
- [X] ska-telescope/ska-sdp-wflow-low-selfcal — SKA low self-calibration workflow

### PyPI `requires_dist` dependents (and prior survey)

- [X] dask-ms (ratt-ru/dask-ms) — surveyed: 219/219 suite green via shim + direct backend
- [X] DDFacet (saopicc/DDFacet) — surveyed (ARE_WE_CURED): tables API only; query/select/sort/getkeyword added
- [X] killMS — surveyed (ARE_WE_CURED)
- [X] meqtrees-cattery (ska-sa/meqtrees-cattery) — MeqTrees simulation/cattery
- [ ] cubical (ratt-ru/CubiCal) — direction-dependent calibration

Python 3.14: supported (pyo3 0.27 / numpy 0.27; full dask-ms suite green on 3.14; requires-python >=3.9,<3.15).

DDFacet survey (crates/casacure): DDFacet's MS data path uses only the tables
API, now covered — added `table.query()`, `table.select()`, `table.sort()`,
`table.getkeyword(name)`. Optional future work: `pyrap.measures`/`quanta` for
DDFacet's montblanc/GiveDate utilities.
