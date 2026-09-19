# Changelog

All notable changes to this project are documented here. Completed `TODO.md`
subtasks are moved here.

## [0.2.3] - 2026-09-19

### Added

- `Table.getsubtables()` — the subtable references from `Table:` keywords,
  resolved to absolute paths (matching python-casacore); a string keyword of
  the form `"Table: <path>"` is stored as a TpTable value (containers write
  subtable links exactly this way).
- `Table.copy(newtablename, deep=False, ...)` — on-disk table copy; `deep`
  also copies subtable directories (with a same-parent guard so a subtable is
  never copied onto itself).
- `Table.toascii(filename)` (0.2.2 work) plus `removecol(s)` — drop columns
  and their data from a writable table (accepts a single name or a sequence).
- `default_ms_subtable(name, path)` builds the standard subtable schema when
  no `tabdesc` is given (python-casacore behaviour); previously it created a
  zero-column table (unsupported).
- Descriptor builders and lifecycle helpers `makescacoldesc`, `makearrcoldesc`,
  `makecoldesc`, `maketabdesc`, `makedminfo`, `tableexists`, `tabledelete`,
  `tablecopy` (0.2.2 work) — the python-casacore `casacore.tables` helper
  surface that dependent packages import.
- `DASK_MS_BACKEND=casacure` support (tmolteno/dask-ms fork) makes dask-ms run
  on casacure with no casacore shim; skarabina (352/355 tests) runs
  end-to-end on casacure.

### Fixed

- Array-column `putcol`/`putvarcol` with Python list/tuple values (0.2.2),
  and 1-element array cells now stay `RecordValue::Array` (an ncorr=1 CORR_TYPE
  writes `[[9]]` instead of a scalar the storage managers reject).
- numpy `<U`/`<S` string ndarrays now store into scalar and array string
  columns (dask-ms writes subtable strings this way).
- `getcell` keeps `(1,1)` cells 2-D — an nchan=1/ncorr=1 MS's FLAG/DATA cells
  no longer read back as 1-D, so dask-ms's exemplar inference matches the
  descriptor.
- Columns declared `TiledShapeStMan`/`TiledCellStMan`/`TSM*Bounded*` are
  created with StandardStMan (no variable-shape tiled manager; values
  round-trip, only the on-disk layout differs) — skarabina's flag-version
  tables.
- `ci.yml` runs the ported/type suites against casacure via the in-repo
  `tests/shim` (they assert casacure's strict unset-cell contract).
- scalar dtype coercion for list/tuple/pure-Python-complex puts (0.2.2);
  `getcol` dtype fidelity incl. uchar→uint16 promotion and complex ndarrays
  (0.2.2); `ORDER BY` spaced form (0.2.2); Path support (0.2.2).

## [0.2.2] - 2026-09-19

### Added

- **`tests/test_casacore_ported.py`**: 9 tests ported from python-casacore's
  `tests/test_table.py` (`test_check_datatypes`, `test_check_putdata`,
  `test_addcolumns`, `test_keywords`, `test_subset`, `test_subtables`,
  `test_tableascii`, `test_complete_desc`, `test_required_desc`) with the
  python-casacore builder helpers (`makescacoldesc`, `makearrcoldesc`,
  `maketabdesc`, `makecoldesc`, `makedminfo`) inlined. Tests run through the
  `casacore` shim; porting them surfaced and fixed the real gaps below.
- **`Table.toascii(filename, columnnames=None)`** — writes the
  `tablefromascii`-compatible ascii format (whitespace-separated name/type
  lines, one data row per line).
- **`DIFFERENCES.md`** — documents deliberate behavioural contracts where
  casacure diverges from casacore/python-casacore (unset cells raise instead
  of reading the type default), and why.

### Fixed

- **Scalar `putcol`/`putcell` element types**: Python lists/tuples (and pure
  python complex values) now cast each element to the column's declared type,
  matching casacore — `uchar`/`short`/`uint`/`float`/`complex` columns no
  longer store value as int/int/double/… (`cast_scalar_to` in convert.rs;
  `value_to_cells` in table.rs).
- **`getcol` dtype fidelity**: `scalars_cells_to_array` now covers
  Short/UShort/UInt/Complex/DComplex cells and promotes `uchar` columns to
  `uint16` on read (python-casacore quirk), so casacure-crafted tables round-
  trip at the declared precision.
- **`os.PathLike` in constructors**: `table()`, `default_ms()`,
  `default_ms_subtable()`, `tablefromascii()` accept `pathlib.Path` via a
  shared `__fspath__` coercion (python-casacore's `test_path_support`).
- **`complete_ms_desc("MAIN")` / `required_ms_desc("MAIN")`** now alias the
  main MS (schema key `"MS"`), matching python-casacore.
- **Empty `dataManagerType`/`dataManagerGroup`** in a column descriptor now
  means `StandardStMan` (python-casacore's `makescacoldesc(x, val)` writes
  `''`), instead of failing with "unsupported data-manager type".
- **`putkeyword`/`putcolkeyword` accept a table object** — stored as a
  `TpTable` reference (like python-casacore) and read back as `Table: <path>`.
- **`pyobject_to_record` handles `PyComplex`** for complex keyword/cell
  values instead of stringifying them.
- **TaQL accepts `ORDER BY <expr>`** (casacore's spaced form) as an alias of
  `ORDERBY` — the ported `test_subset` uses the real syntax; covered by a new
  Rust parser test (`order_by_spaced_form_matches_orderby`).

## [0.2.1] - 2026-09-19

### Added

- `casacure-test` / `casacure-bench` console scripts (wheel entry points,
  run outside pytest), and the scalar-column dtype-coercion fix they surfaced
  (see the Unreleased section).

## [Unreleased]

### Added

- **`casacure-test` and `casacure-bench` console scripts** (wheel entry
  points `casacure:run_tests` / `casacure:run_benchmark`, implemented in the
  extension so they run from the installed wheel):
  - `casacure-test`: 14 self-tests over the public `casacure.tables` API —
    scalar/array/string round-trips, fixed-shape arrays, dcomplex/int16 dtype
    coercion, variable-array putvarcol/getvarcol, record columns, taql SELECT,
    `table.query()/sort()` (the DDFacet/killMS pattern), getkeyword,
    default_ms subtables + TpTable keyword, multidim strings, addrows
    persistence. Prints a per-check summary; exit code = number of failures.
  - `casacure-bench`: times putcol / getcol / taql WHERE+ORDERBY on an n-row
    table, and compares against real python-casacore when it is importable as
    a distinct module (the `casacore` shim and a missing casacore are both
    detected and reported as casacure-only).
- **Fixed a real bug the self-test exposed**: `putcol` of int64/int16 (and any
  non-column-dtype) numpy arrays into *scalar* columns silently wrote zeros —
  dtype coercion now applies to scalar columns too (it already covered array
  columns). Verified int32/int64/int16 all round-trip.

## [Unreleased]

### Added

- Table selection API driven by DDFacet's casacore usage (DDFacet's MS data
  path uses only `casacore.tables`: `getcol`/`putcol`/`addcols`/
  `getcoldesc`/`colnames`/`nrows`/`getkeyword`/`query`/`sort`/`getcolslice`):
  - `table.query(taql)` — rows matching a TaQL selection expression, as a
    new table (DDFacet: `t.query("FIELD_ID==1")`);
  - `table.sort(column)` — table sorted ascending by a column (DDFacet:
    `t.query(...).sort("TIME")` in `ClassMS.GiveMainTable`);
  - `table.select(taql)` — alias of `query` (casacore returns a
    TableIterator; a filtered table suffices for DDFacet);
  - `table.getkeyword(name)` — single keyword value (None when absent).
  DDFacet's `pyrap.images.image` imports are astropy-backed (the `image`
  module is only used via `image(...).getdata()` in a few test/mask paths and
  `KeepCasa` is unimplemented), so no CASA-image subsystem is required; the
  `pyrap.measures`/`quanta` usage appears only in the montblanc/utilities
  paths (`GiveDate`), tracked as optional future work.

## [Unreleased]

## [0.2.0] - 2026-09-19

### Changed

- Version bump 0.1.0 -> 0.2.0 (workspace package + workspace dependency, in
  lockstep); the release is tagged `v0.2.0` to drive CI and the crates.io /
  PyPI trusted-publishing workflows.
- `crates/casacure` README shipped in the crate (description, features, Rust
  usage examples, author: Tim Molteno <tim@elec.ac.nz>); `readme`, `authors`
  and `keywords` metadata added to the crate manifest; pyproject author email
  corrected to tim@elec.ac.nz.
- Top-level README retitled "casacure and python-casacure", framing the two
  user-facing packages (the `casacure` Rust crate and the python-casacure
  PyPI package, module `casacure.tables`); the internal `casacure-python`
  crate is noted as build-only, not published to crates.io.
- Publishing setup: PyPI via GitHub trusted publishing (OIDC,
  `publish-python.yml`), crates.io via trusted publishing
  (`rust-lang/crates-io-auth-action`, `publish-rust.yml`, no API token);
  `casacure-python` marked `publish = false` so the crates.io package is the
  `casacure` crate.
- Python 3.14 support: pyo3 0.27 / numpy 0.27; `requires-python >=3.9,<3.15`;
  macOS/Windows publish wheels for CPython 3.10-3.14 (Linux manylinux covers
  3.9-3.14).
- Python package metadata completed for PyPI (PEP 639 SPDX license,
  authors, keywords, classifiers, project.urls, sdist hygiene via
  MANIFEST.in).


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
- `aipsio` module: cursor-based reader for casacore's canonical AipsIO byte
  format — magic check, u32/u64, length-prefixed strings, and typed object
  headers (root vs nested), now endian-aware (`new_le` for the StandardStMan
  data files) — 5 unit tests.
- `ssm` module: StandardStMan data-file (`table.f0`) reader — the
  `"StandardStMan"` header (bucket size, bucket/index locations, endian flag
  with mismatch validation), the index assembled from the chained index
  buckets, `SSMIndex` decoding (used buckets via `lastRow`/`bucketNumber`
  blocks, v1 uInt / v2 u64 rows, `SimpleOrderedMap` free-space), and scalar
  cell reads for every numeric type, bit-packed Bool, fixed-length strings,
  and inline short (≤8 char) variable strings — the bin-bucket
  (SSMStringHandler) path for longer strings is a known gap. `parse` takes
  the table's data-file endianness; `read_scalar_cell` ties the spec offsets
  from `table.dat` to bucket cells. 10 unit tests incl. rows spanning
  multiple data buckets and an index spread over a bucket chain; fixture
  test verifies all 10 values of the casacore-written `typed.tab` read back
  exactly (bool, uchar, int16/int32/uint32, float, double, complex,
  dcomplex, inline "hello" string) from its little-endian `table.f0`.
- Writing: `aipsio::Writer` serializes canonical AipsIO (big or little
  endian) with length-patched framed objects; `record.rs` gained
  `write_scalar_value` and `write_table_record` (empty + scalar/string
  nested records), `tabledesc.rs` `write_table_desc`/`write_column_desc`
  (scalar + array class names verified byte-exact against the casacore
  fixture), `columnset.rs` `write_column_set` + `write_standard_stman`
  (SSM spec blob), and `ssm.rs` the full data-file writer
  (`write_standard_stman_file`: header, data buckets, SSMIndex stream,
  index-bucket chain) with `encode_scalar_cell`.
- `create_table(path, desc, values)` writes a complete scalar-column table
  (`table.dat` + `table.f0`), packing columns into a bucket tile whose
  offsets match casacore's `getFree`/`addColumn` best-fit layout exactly
  (unit test reproduces the real fixture's offsets
  `[0,4,36,100,228,356,484,740,996,1508]` and bucket size 1892).
- `examples/create_sample_table.rs` writes a sample table for interop
  checks. Byte-level write-interop proven: python-casacore reads the
  casacure-written table — 3 rows (int/float/string) and 100 rows spanning
  multiple buckets — returning exactly the written values.
- 11 new writer unit tests (layout vs fixture, table.dat round-trip,
  create→read-back of all 10 typed values, multi-bucket row distribution,
  long-string rejection).
- Fixed-shape array columns (StandardStMan): the array sub-format is now
  read **and** written. Bucket cells hold an `Int64` byte reference into
  `table.f0i` (StManArrayFile / StManAipsIO): files start with
  `[u32 version][1-byte length]` and hold one record per row of
  `[ndim][CASA-order dims][element data]` — dims are reversed relative to
  the logical row-major shape (discovered from the casacore fixture: a 2x3
  complex array stores `[3,2]`). `ColumnSet` array bindings carry the same
  CASA-order `IPosition` shape (`columnset::parse_column_set` previously
  misread this as a string). Element types: all numerics + bit-packed Bool
  arrays; string arrays not yet.
- `read_array_cell` / `encode_array_record` + `create_table` array-column
  support (bucket region = 8-byte refs, fanning out to `table.f0i`).
- `tests/make_fixtures.py` gains an `array.tab` fixture (2x3 complex `ARR`
  + scalar `IDX`, 2 rows). Fixture test verifies the real casacore
  `array.tab` array/index layout (offsets `[0,256]`, CASA shape, complex
  values, the `StManArrayFile` double-index-bucket copy at offset 8 vs 196).
- Array write-interop proven: python-casacore reads casacure-written
  fixed-shape array columns exactly (`getcoldesc` option 4, logical shape
  `[2,3]`, all values; 3 rows and 100 rows across multiple buckets).
- `examples/create_sample_table.rs` now writes an array column too.
- SSM string buckets (`SSMStringHandler`): variable strings longer than 8
  chars are read **and** written. The string bucket has a 16-byte
  **big-endian canonical** header `[unused][usedLength][nDeleted][nextBucket]`
  (independent of the data-file endianness) with raw string data from byte
  16; strings spanning buckets chain via `nextBucket`. New `longstr.tab`
  fixture (52/65-char strings, 2 rows) verifies the real casacore layout
  (bucket 2, refs `[2,0,52]`/`[2,52,65]`, used=117/nDeleted=379), and
  `read_scalar_cell` now resolves long-string cells through the buckets
  instead of erroring. `create_table` grows string buckets (a
  `StringBuckets` writer mirroring `putData`, incl. roll-over chaining) and
  writes them after the index buckets with `last_string_bucket` set.
- Long-string interop proven: python-casacore reads casacure-written strings
  exactly (3-row sample labels, plus 1000-char strings chained across
  multiple string buckets).
- IncrementalStMan support (the MS index-columns storage manager): the
  `table.f0` reader decodes the `"IncrementalStMan"` header (v4/v5, bucket
  size / counts), the `ISMIndex` at the end of the file (bucket boundaries +
  bucket numbers), and the per-bucket interval index
  (`[u32 indexOffset][data][per-col: nr, rownrs, offsets]`) — repeated
  values across consecutive rows share one stored value (incremental
  compression, verified: `[0,0,1,1,1,2]` stored as intervals
  `[0..1],[2..4],[5..5]`). The `option 1`/Direct descriptor flag is a schema
  hint — the layout doesn't depend on it. New `ism.tab` fixture (TIME double
  + ANT1 int in ISM, VAL float in a second StandardStMan DM) verifies
  multi-data-manager tables where `table.f0` is ISM and `table.f1` is SSM.
- `create_table` now supports **multiple data managers** in one table: data
  managers are grouped by type in column order, given sequence numbers, and
  each gets its own `table.f{seq}` file (IncrementalStMan
  `write_ism_file` with interval compression, StandardStMan as before plus
  `table.f{seq}i` for arrays). The ColumnSet writer is generalized to a
  multi-DM form (`write_multi_column_set`, `DmBlob`). Full-MS-pattern
  interop proven: python-casacore reads a casacure-written table combining
  SSM scalars, long strings, fixed-shape arrays, **and** ISM TIME/ANT1 —
  4 files, all values exact.
- TiledColumnStMan support (the MS visibility-data storage manager): the
  `table.f{seq}` reader parses the canonical big-endian `"TiledColumnStMan"`
  header (fixed cell shape), the nested `"TiledStMan"` object (row count,
  column types, hypercolumn name, tile files, hypercubes), and each
  hypercube's `cubeShape`/`tileShape`/file offset. The tile data file
  (`table.f{seq}_TSM{fileSeqNr}`) is a bucket file: tile `t` at
  `fileOffset + t*bucketSize`, and a row's fixed-shape cell spans the full
  non-row tile dims (verified against the real casacore `tsm.tab`: cell
  `[3,2]` dcomplex, tile `[3,2,5461]`, bucket 524256 bytes).
- `write_tsm_file` + `create_table` support for TiledColumnStMan columns
  (one fixed-shape array column per group; the DM ColumnSet blob is empty —
  the spec lives in the header file, which carries the DM sequence number
  that casacore validates on open).
- **Bug fix**: the SSM index-bucket `[checkNr][nextBucket]` header is written
  in **big-endian canonical** regardless of the data-file endianness; both
  the reader and writer now do so (the reader previously followed
  little-endian next-pointers, breaking chained indexes in little-endian
  tables — caught by the `tsm.tab` fixture where the index spills across a
  chain into a reused free bucket).
- Full-MS-pattern interop proven: python-casacore reads a casacure-written
  table combining StandardStMan scalars + long strings + `table.f0i` arrays,
  IncrementalStMan TIME/ANT1, and TiledColumnStMan DATA — 6 files, every
  value exact.
- Data-manager info (`getdminfo`) round-trip: `get_dminfo` builds the
  exact python-casacore `table.getdminfo()` dictionary — `*N` records with
  TYPE/NAME/SEQNR/SPEC/COLUMNS (sorted columns) — with per-manager SPECs
  read from the data-file headers: StandardStMan
  (MaxCacheSize/BUCKETSIZE/PERSCACHESIZE/IndexLength), IncrementalStMan
  (MaxCacheSize/BUCKETSIZE/PERSCACHESIZE), and TiledColumnStMan
  (incl. the HYPERCUBES CubeShape/TileShape/CellShape/BucketSize records).
  Verified byte-for-byte against casacore on the typed/ism/tsm fixtures.
- `create_table` now groups columns by (data-manager type, group) and names
  managers by their group (the `_N` auto-suffix applies to `addcols`,
  tracked for the future column-management step); SSM/ISM/TSM spec names
  follow the group. Full multi-DM interop re-verified (SSM + ISM + TSM in
  one table, all values exact; dminfo NAME now e.g. `TiledData_GROUP`).
- Table lifecycle (§2): a `Table` object tied to a table directory with
  `open(dir, readonly)` / `create(dir, desc, values)`, advisory
  `lock`/`unlock` (internally no-op — no OS locks are held, as the crate
  never needs them), `flush`/`close`, `is_writable()`, `name()`, plus
  `nrows()` and `colnames()`. All content is owned, so a `Table` is
  `Send + Sync` and safe across threads; opening eagerly loads every data
  manager (SSM/ISM/TSM files by sequence number) so later column reads have
  the files ready. `lockoptions="user"` semantics are covered by the
  advisory lock.
- Column data access — the §3 read hot path: `Table` gains `getcell(col,
  row)`, `getcol(col, startrow, nrow)`, `getcolslice(col, blc, trc,
  startrow, nrow)`, `getcellslice(col, row, blc, trc)`, and `getvarcol(col)`
  (all rows). Cells dispatch by data manager (SSM scalar/array/string/inline
  & bucket strings, ISM intervals, TSM tiles) using the per-manager column
  index; `slice_array_value` applies inclusive, logical-dimension `blc`/`trc`
  to array cells (the casacore `getcolslice` semantics). Verified on the real
  fixtures (typed scalars+strings, tsm DATA slices, ism interval columns);
  `setmaxcachesize` is intrinsically a no-op (no caches).
- Column writes (§3): `WritableTable` builds a table incrementally — `create(schema)`, `addrows(n)`, `putcol`/`putcell` batches (the dask-ms MS-writing pattern, incl. per-row arrays for `putvarcol`), `setmaxcachesize` no-op, and `flush()` which assembles the on-disk files via `create_table`, filling missing scalar cells with their defaults. Since the Rust layer takes `RecordValue`s directly, the python-casacore object-array segfault wart is lifted by construction (the binding just converts). Verified: build an SSM+ISM+TSM table via addrows/putcol and read every value back.
- Metadata & descriptors — the §4 read side: `Table::getcoldesc(col)`,
  `getdesc()`, `getkeywords()`, and `getcolkeywords(col)` return exactly the
  dicts python-casacore produces (valueType names like `int`/`dcomplex`,
  logical `shape`/`_c_order` for array columns, keyword records incl. nested
  records). `RecordValue`/`TableRecord` gained JSON serialization. Verified
  byte-for-byte against real casacore on the typed/array fixtures plus a new
  `kw.tab` fixture with table keywords (VER/MAXROWS/NEST nested) and column
  keywords (UNITS/MULTI). Keyword *write* (putkeywords/removekeyword) is the
  next step.
- Keyword *write* side of §4: `WritableTable::putkeyword`/`putcolkeyword`/
  `removekeyword`/`removecolkeyword` (plus `TableRecord::set`/`remove` with
  data-type inference), with `write_record_data_values` now recursing into
  nested records per casacore `putData`. Round-trip test reproduces the
  `kw.tab` keywords (incl. the NEST→HH nested record) and casacore reads a
  casacure-written table's keywords back exactly.
- Subtable linkage (§4): `Table::getkeywords`/`getcolkeywords`/`getcoldesc`/
  `getdesc` expose `TpTable` keyword fields as the `"Table: <resolved path>"`
  strings python-casacore and dask-ms' `CasaFormat.is_subtable` expect —
  relative stored paths are joined against the directory containing the
  parent table and lexically normalized, resolving dynamically on read
  (verified by relocating a fixture tree). `WritableTable::putkeyword`/
  `putcolkeyword` with `RecordValue::Table` store relative `./subtable`
  references like casacore (absolute paths kept when outside the parent's
  directory), recursively into nested records. New `subs` fixture covers
  same-dir / subdir / outside cases; casacore reads casacure-written subtable
  links back exactly (incl. nested) and opens the linked tables (74/74 tests).
- TaQL subset (§5): a `taql` module with a tokenizer, recursive-descent
  parser and expression evaluator covering the dialect dask-ms generates and
  a DDL subset. `SELECT [UNIQUE] expr [AS name] FROM \$N [WHERE ...]
  [ORDERBY ... [DESC]] [GROUPBY ...] [LIMIT n]`, `SELECT *`, `ROWID()`,
  multi-key stable `ORDERBY` (DESC flips equal-key tie order like casacore),
  `WHERE` with word/operator `AND`/`OR`/`NOT` and arithmetic/comparison/
  functions over arbitrary user expressions, `GROUPBY` with `GROWID()`,
  `GROWID()[i]`, `GCOUNT()`, `GAGGR(col)`, `SELECT UNIQUE` (first-occurrence
  order), scalar subqueries `[SELECT ...]` and row lookups
  `[SELECT ...][expr]`, and DDL `CREATE TABLE path [NAME TYPE
  [NDIM=n, SHAPE=[...]] ...] LIMIT n` (casacore's type codes incl. I4/C8/S)
  for fixtures. Semantics verified against real casacore on the ordering
  probe (ORDERBY rowids, GROUPBY groups/aggregates, UNIQUE, subquery lookup)
  and via a DDL-created table that casacore reads back exactly; `ragged`
  GROUPBY groups return logically correct (unpadded) arrays, diverging from
  casacore's fixed-shape zero-padding quirk. 7 unit tests; helpers added:
  `TableRecord`/`ArrayValue::elements` JSON plumbing, `WritableTable::desc`.
- MS schema / descriptors (§6): `ms_schema.rs` vendors the canonical
  `required_ms_desc`/`complete_ms_desc` dicts for the main MS and 17
  subtables (generated from real casacore; `scripts/vendor_ms_schema.py`
  regenerate step in the comments). `TableDesc::from_desc_json` and a
  column-dict converter build descriptors from the python-casacore dict
  format. `ms::default_ms(path, extra_desc)` creates the main table (21
  required columns, MS_VERSION keyword, optional extra columns) plus the 12
  standard subtable directories inside it, each linked from the main table
  by `TpTable` keywords; `default_ms_subtable(name, path)` and
  `maketabdesc_from_json` round out the API. A JSON parser
  (`record::parse_json_record`) reconstructs typed `TableRecord`s from dict
  JSON (also serving the bindings layer), and keyword writes now support
  array fields (`write_array_value`, e.g. `QuantumUnits`). `table.f0i` is
  created for array columns even at zero rows. Verified end-to-end: casacore
  opens a casacure-created MS (MS_VERSION 2.0, 12 `Table:` links, TIME
  `QuantumUnits: ['s']`), writes variable `DATA` and ANTENNA NAME/POSITION
  into it and reads them back exactly. 6 new unit tests (70/70 unit + 15
  fixture in this release).
- casacure-python bindings (§7): a `casacure.tables` submodule exposing the
  python-casacore surface dask-ms uses — `table(...)` (create/open, numpy
  `getcol`/`getcolnp`/`getcolslice`/`getcolslicenp`, `getcell`/
  `getcellslice`/`getvarcol`, `putcol`/`putcolnp`/`putcolslice`/
  `putvarcol`/`putcell`, `addrows`, keyword get/set/remove incl. nested
  records and arrays, `getcoldesc`/`getdminfo`/`nrows`/`colnames`/
  `iswritable`, advisory `lock`/`unlock`, no-op `setmaxcachesize`), plus
  `taql(...)`, `default_ms`, `default_ms_subtable`, `required_ms_desc` /
  `complete_ms_desc` and the type-mapping helpers. Array shape conventions
  match casacore exactly: fixed-shape columns read/write in the logical
  (c-order) orientation with the descriptor shape reversed; variable-shape
  columns as given; strings as lists / `{"shape","array"}` dicts. `table.f0i`
  created for array columns at zero rows; `TableDesc::from_desc_json`
  reverses fixed `shape` keys. Fixed a keyword-record write bug: nested
  keyword records are framed when the *field's* sub-descriptor is empty
  (casacore frames them even for non-empty value descs) — verified by
  re-reading casacore-written keyword tables through the write-back path.
  Interop-gated: casacure-created fixed/variable tables read identically in
  casacore and vice versa (written with pyo3/numpy 0.26; install with
  maturin).
- dask-ms integration run (§7): `tests/daskms_smoke.py` drives dask-ms's
  `xds_from_table`/`xds_to_table` against the casacure backend through a
  `casacore.tables` shim — read (`SELECT ROWID() … ORDERBY` taql + `getcolnp`)
  and write-back verified against real casacore, including the `group_cols`
  GROUPBY partitioning path (GROWID/GCOUNT/GAGGR). Fixes surfaced: `getcell`
  returns just the cell shape (no leading row singleton, matching casacore);
  `getcolnp` maps scalar cells by the buffer dtype (was zero-filling them).
- Full dask-ms MS lifecycle on casacure (§7): `xds_to_ms`/`xds_from_ms`
  now work end-to-end — dask-ms creates an MS from scratch (via our
  `default_ms`, which now returns a context-manager table and accepts the
  `tabdesc=`/`dminfo=` kwargs dask-ms passes; `required/complete_ms_desc`
  take an optional name), writes the dataset columns, and reads them back
  through `xds_from_ms` (MS-schema dims + FIELD_ID/DATA_DESC_ID grouping),
  cross-checked against real casacore (12 subtable dirs, correct values).
  `WritableTable::flush` now fills unwritten array cells with casacore's
  defaults (zeros for fixed shape, empty for variable) instead of failing —
  required for dask-ms's addrows-before-putcol write pattern (was
  `NoDefault`); 85/85 tests still pass.
- dask-ms's own `example_ms()` fixture factory now runs entirely on casacure
  (§7): the shimmed `casacore.tables` builds the main MS plus the
  ANTENNA/POLARIZATION/SPECTRAL_WINDOW/FIELD/DATA_DESCRIPTION subtables,
  and `xds_from_ms` reads the result back (10 rows, DATA (10,16,4)
  complex64). Bindings gained `ms::SUBTABLE` path syntax, int64/UInt array
  `putcol` support, and `default_ms` now returns a writable context-manager
  main table.
- dask-ms chunked/sliced and varcol writes (§7): `putcolslice` now overlays
  a chan/corr-chunked array into the fixed cell at logical blc..trc (the
  multi-call pattern dask-ms uses for data written in slices) —
  `test_dataset_create_table` passes; default cells for fixed-shape array
  columns are built in the logical (as-given) orientation to match real
  cells; `putcol` accepts the `{"rN": value}` dict form and numpy numeric
  scalars (np.int64 etc.) so dask-ms's SPW/row-grouping setup code runs.
- SSM multidim string-**array** cells (§1/§3, previously unsupported): these
  columns store the whole cell in the string buckets — a 12-byte
  (bucket, offset, len) reference in the data file (not the f0i array
  index), with bucket content `[ndim][CASA dims][filled flag][len-prefixed
  strings]` per `SSMStringHandler::put(Array<String>&)` (decoded from the
  casacore source + real files). Reader + writer implemented; casacore
  multidim-string tables read byte-identically and our written tables read
  in casacore; dask-ms's `test_dataset_multidim_string_column` passes
  (incl. `getcol` dicts with the row dim and per-row splitting of the
  `{"shape","array"}` write form).
- `addcols` (§3/§7): `WritableTable::addcol` + binding `table.addcols(dict,
  dminfo=...)` — dask-ms's `test_dataset_add_column` suite passes (8 passed +
  2 xpassed), including array, string, boolean, int16/uint32 columns added to
  an existing table and read back by casacore. Supporting fixes:
  - **TiledColumnStMan Bool tiles** written (1 byte/element, matching the
    tile reader) — previously rejected;
  - **Bool scalar storage** confirmed as byte-per-row (a 100-row casacore
    table reads aligned); the StandardStMan bucket layout was wrongly
    bit-packing Bool (bits=1) which overflowed wide bool tables — fixed;
  - **empty array cells** (null/offset-0 references casacore writes) read as
    empty arrays instead of erroring — unblocked write-back of real MS files.
- **dask-ms suite green**: the full dask-ms 0.2.32 test run via the
  `casacore.tables` shim now passes **104 combined** (test_table_proxy 14/14,
  test_dataset 33/33 incl. `test_write_dict_data`/`test_row_grouping`,
  test_dataset_keywords 10/10, ordering/table/columns). Fixes that closed the
  last gaps:
  - **`putvarcol` varcol semantics**: dict values are full per-row cells, so
    the leading dimension is kept (putcol still drops the row dim) — matches
    casacore's getvarcol `(1,n,...)` returning cells and the `(1,n)` write
    requirement; value-mismatch in `test_write_dict_data` gone.
  - **`taql CREATE ... [NDIM=n]`** creates a *variable* array column
    (previously only `SHAPE=[...]` did; bare `[NDIM=1]` wrongly made a scalar);
    the table.dat desciption now records the declared `ndim` for variable
    columns (was 0). Matches real casacore's getcoldesc (`ndim`, `_c_order`).
  - **`getcell` on a variable array column strips the leading row singleton**
    (casacore returns the exemplar without the while-storage `1`); `getvarcol`
    keeps it — fixed the ndim/exemplar mismatch that dropped CHAN_FREQ in
    `test_row_grouping`.
  - **keyword dict stability**: `getkeywords`/`getcolkeywords`/`_getdesc`
    build the dict directly from the `TableRecord` (no JSON float round-trip,
    so `MS_VERSION` stays a float) and resolve `TpTable` fields to
    `"Table: <abs-path>"` like python-casacore (`test_dataset_keywords` 10/10).

- **dask-ms's entire test suite passes**: **219 passed / 0 failed**
  (all of `daskms/tests/` via the `casacore.tables` shim: dataset, keywords,
  ordering, table, columns, table_proxy, ms_creation, ms_read_and_update,
  dataset_schema, columns, array_api_utils, storage, stress, patterns,
  optimisation, optional, multiton, utils, table_schemas). Major changes that
  closed the last gaps:
  - **live data-manager reads**: read-only `table()` handles re-parse the disk
    on every read instead of serving a stale snapshot (a long-lived read
    handle now sees later writes, like casacore) — `test_ms_update`'s
    write-then-read cycles pass.
  - **process-wide shared writable state**: all writable handles of a table
    directory share one materialised cell store + read snapshot (strong refs
    in a global registry), so dask-ms's parallel per-chunk column writes merge
    instead of one flush clobbering another's — `test_ms_update`/`ms_creation`
    pass.
  - **eager persistence**: every mutating binding call (putcell/putcol/
    putvarcol/putcolslice/addrows/addcols/keywords) flushes immediately, so
    writes survive without an explicit `close()` (python-casacore proxy
    `putcol` does not flush) — `test_ms_update` passes.
  - **numeric dtype coercion in putcol**: incoming ndarrays are cast to the
    column element type (complex64 -> dcomplex etc., like casacore); complex
    arrays previously fell into the string write path and corrupted the SSM —
    `test_array_protocol_write`/`test_fake_{cupy,torch}_gpu_write` pass.
  - **taql SELECT projections** copy array/record cells verbatim (TqValue
    could not round-trip complex/array cells — it flattened shapes and erased
    values) — `test_ms_read` passes; empty taql results no longer panic.
  - **record columns** (`SOURCE_MODEL`, valueType "record"): descriptor
    parsing, `default_ms_subtable` creating from the provided desc and
    returning a context manager, and SSM storage of serialised record cells —
    `test_ms_create` passes.
  - **`ndim: -1` = variable array column** (casacore semantics: any `ndim`
    key, including -1, marks an array column); `getcell` strips the leading
    row singleton when it restores the declared rank; the vendored `ms_schema`
    regenerated from real casacore with correct `ndim`/`_c_order` on all
    array columns.

- **upstream store-dispatch prototype validated**: an env-gated
  `DASK_MS_BACKEND=casacure` alias in dask-ms (`casacore.tables` ->
  `casacure.tables` via `sys.modules`) lets dask-ms use casacure **directly,
  without the `casacore` shim**; the full dask-ms 0.2.32 suite passes
  219/219 that way. The change is ~10 lines in `daskms/__init__.py`; the
  remaining work is submitting it upstream.
- **real-casacore interop verified end-to-end**: python-casacore creates an MS
  (with `addcols` DATA); dask-ms reads it through the shim, doubles DATA /
  bumps TIME, writes back; real python-casacore re-opens the result and reads
  back the exact modified values with all 22 main-table columns intact.

- **Python 3.14 support** (pyo3 0.27.2 + numpy 0.27):
  - the extension builds and imports on Python 3.14.7; the full dask-ms 0.2.32
    suite passes 219/219 on 3.14 (direct backend, no shim) as well as 3.13;
  - `downcast` -> `Bound::cast` across the bindings (downcast is deprecated in
    pyo3 0.27), zero build warnings;
  - `requires-python` widened to `>=3.9,<3.15` with the 3.14 classifier;
  - publish workflow: macOS/Windows now build one wheel per CPython version
    (3.10-3.14) and Linux manylinux covers 3.9-3.14; verified a
    `casacure-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl` installs and
    round-trips a table.

- **crates/casacure README**: added a crate-level `README.md` (description,
  features, Rust usage examples, license) and wired the author metadata
  through the workspace (`authors = ["Tim Molteno <tim@elec.ac.nz>"]` in
  `[workspace.package]`, `authors.workspace = true` + `readme` + `keywords`
  on the crate); pyproject author email corrected to tim@elec.ac.nz. Serves
  the crates.io package listing (next release after 0.1.0, which is live).

- `table` module: `parse_table_header` parses the `table.dat` root object
- `table` module: `parse_table_header` parses the `table.dat` root object
  (`Table` v2/v3: row count, data-file endianness flag, table kind) — 5 unit
  tests plus a manifest-driven fixture test asserting the header of the real
  casacore-written `typed.tab` (nrows and host byte order now recorded in
  `tests/fixtures/manifest.json` by `tests/make_fixtures.py`).
- `record` module: full recursive `TableRecord`/`RecordDesc` parsing —
  all 31 `DataType` codes, scalar values, framed `"Array<...>"` v3 arrays
  (incl. bit-packed Bool arrays), nested records (framed and bare),
  `TpTable` references — 4 unit tests.
- `tabledesc` module: `parse_table_desc` parses the `TableDesc` v2 object
  (name/version/comment, public + private keyword records, ColumnDescSet)
  and each `ColumnDesc` (name, comment, data manager type/group, dtype,
  options, ndim, IPosition shape, keywords, scalar default value / array
  flag). `parse_table_dat` parses a whole `table.dat`. Fixture test verifies
  all 10 columns of the casacore-written `typed.tab` (names, value types,
  StandardStMan data manager, scalar kind).
- `columnset` module: parses the data-manager info following `TableDesc` in
  `table.dat` — `ColumnSet` version (negative version scheme: -2 reads a
  `u32` row count, -3 a `u64` row count plus storage option + block size),
  data-manager list (type + sequence number), per-column `PlainColumn`
  bindings (scalar/record vs array with optional shape column), and the
  per-manager opaque spec blobs. StandardStMan's framed `"SSM"` object is
  decoded into the manager name and the `"Block"`-framed column-offset /
  column-index-map tables; other data-manager types keep their raw blob.
  `parse_table_dat` now returns the full `TableDat` (header, desc,
  column_set). `aipsio` gained a raw-opaque-block reader. 7 unit tests;
  fixture test asserts the real casacore `typed.tab` ColumnSet (version 2,
  one StandardStMan DM at seq 0, all 10 scalar bindings, SSM offsets
  `[0,4,36,100,228,356,484,740,996,1508]`, zero index map).
