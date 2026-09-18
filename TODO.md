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
- [ ] dminfo dict round-trip fidelity (incl. `_1` auto-suffix behaviour)

## 2. Table lifecycle and locking API

- [ ] `table()` open + create
- [ ] `lock()`/`unlock()` (advisory), `flush()`, `close()`
- [ ] `iswritable()`, `name()`

## 3. Column data access (hot path)

- [ ] `getcolnp` / `putcol` into preallocated numpy buffers
- [ ] `getcolslicenp` / `putcolslice` (inclusive `blc`/`trc` ends)
- [ ] `getcol` / `getcolslice` / `getcell` / `getcellslice`
- [ ] `getvarcol` / `putvarcol` (`{"rN": arr}` dicts)
- [ ] Multidim string columns as `{"shape", "array"}` dicts
- [ ] Accept numpy object arrays directly in `putcol` (no segfault wart)
- [ ] `addrows`, `setmaxcachesize` (no-op)

## 4. Metadata and descriptors

- [ ] `nrows`, `colnames`, `getcoldesc`
- [ ] Table keywords: `getkeywords`/`putkeywords`/`removekeyword` (nested records)
- [ ] Column keywords: `getcolkeywords`/`putcolkeyword`/`removecolkeyword`
- [ ] Public table-description dict API (replaces private `_getdesc`)
- [ ] Subtable linkage via `"Table: <path>"` keywords

## 5. TaQL subset

- [ ] `SELECT ... FROM $N` with `$N` table references
- [ ] `WHERE` expression evaluator (user `taql_where` passthrough)
- [ ] `ORDERBY`, `ROWID()`
- [ ] `GROUPBY` + `GROWID()`/`GAGGR()`/`GCOUNT()`
- [ ] `SELECT UNIQUE col`, scalar subqueries
- [ ] DDL: `CREATE TABLE ... LIMIT n` for test fixtures

## 6. MS schema / descriptors

- [ ] Vendor canonical `required_ms_desc`/`complete_ms_desc` dicts as data (MS + 18 subtables)
- [ ] `default_ms(path, tabdesc, dminfo)` with full subtable tree + keyword linkage
- [ ] `default_ms_subtable`, `maketabdesc` (tests)

## 7. dask-ms integration

- [ ] Register as a store type in `fsspec_store.py`/`dask_ms.py` dispatch (upstream, later)
- [ ] Full dask-ms test suite passing against casacure
- [ ] Cross-implementation round-trip gate (Rust-write → casacore-read and vice versa)
