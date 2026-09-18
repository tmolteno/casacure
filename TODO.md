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
- [X] `ORDERBY`, `ROWID()`
- [X] `GROUPBY` + `GROWID()`/`GAGGR()`/`GCOUNT()`
- [X] `SELECT UNIQUE col`, scalar subqueries
- [X] DDL: `CREATE TABLE ... LIMIT n` for test fixtures

## 6. MS schema / descriptors

- [X] Vendor canonical `required_ms_desc`/`complete_ms_desc` dicts as data (MS + 17 subtables)
- [X] `default_ms(path, tabdesc, dminfo)` with full subtable tree + keyword linkage
- [X] `default_ms_subtable`, `maketabdesc` (tests)

## 7. dask-ms integration

- [X] Cross-implementation round-trip gate (Rust-write → casacore-read and vice versa)
- [~] Full dask-ms test suite passing against casacure — dask-ms 0.2.32's own tests run via the `casacore.tables` shim: **test_table_proxy 14/14, and 96 passed across the core proxy/ordering/table/columns/utils/executor files**; the MS lifecycle (xds_to_ms/xds_from_ms create+write+read, GROUPBY partitioning, example_ms) works end-to-end (`tests/daskms_smoke.py`). Remaining feature gaps: `addcols`, chunked `putcolslice` (chan/corr chunked writes), SSM string-**array** cell writes, `putvarcol` dict-ordering, subtable-path normalization — plus upstream store dispatch.
- [ ] Register as a store type in `fsspec_store.py`/`dask_ms.py` dispatch (upstream, later)
