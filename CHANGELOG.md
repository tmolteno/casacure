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
