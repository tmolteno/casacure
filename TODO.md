# TODO

Needed next steps, broken into small subtasks. Add subtasks here **before**
starting them; remove each when completed and log it in `CHANGELOG.md`.

Work areas follow `CASACORE_TO_CASA_RS.md` (tracked as GitHub issues).

## Project scaffolding (in progress)

- [ ] Add casacore as a git submodule (track master, never modify)
- [ ] Create Rust workspace: `casacure` core crate + `casacure-python` bindings crate
- [ ] Set up maturin/`pyproject.toml` so the bindings are pip installable
- [ ] Implement the CASA type system module (`ValueType` enum + numpy mapping) with unit tests
- [ ] Set up the casacore comparison test framework (python-casacore vs casacure round-trips)
- [ ] Create `ARE_WE_CURED.md` progress tracker
- [ ] Create GitHub issues for each major functionality area
- [ ] Move the casacore comparison framework into the Rust core: fixture
      generator script + manifest-driven `cargo test -p casacure` integration
      tests (no pyo3 rebuild needed)
- [ ] CI: GitHub Actions workflow running `cargo test` (and comparison tests where possible)

## 1. CASA table on-disk format (core prerequisite)

- [ ] Parse `table.dat` header (magic, version, endianness)
- [ ] StandardStMan column storage: read
- [ ] StandardStMan column storage: write
- [ ] Scalar + fixed-shape array columns
- [ ] Byte-level interop proof: read a casacore-written table
- [ ] Byte-level interop proof: write a table casacore can read
- [ ] IncrementalStMan (Direct option) support
- [ ] TiledColumnStMan (`{column}_GROUP`, reversed dim order DEFAULTTILESHAPE)
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
