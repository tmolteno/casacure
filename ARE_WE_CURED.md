# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — §1–§3 are complete at the Rust-core level (format, lifecycle, column read **and** write); `casacure-python` (numpy/dict bindings) and §4–§7 remain.

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit + fixture tests | `cargo test` | 71/71 | type system, AipsIO read+write (both endians), `table.dat`, StandardStMan data file + `table.f0i` + string buckets, IncrementalStMan (interval index, multi-DM tables) — all read+write |
| casacore comparison tests | `.venv/bin/python -m pytest tests/` | 5/5 | type system only |
| write interop (manual) | `examples/create_sample_table.rs` + python-casacore | ✓ | casacure-write → casacore-read: SSM scalars, arrays, long strings, and ISM TIME/ANT1 in one 4-file table; 3-row and 100-row variants return exactly the written values |

## Progress by area (per CASACORE_TO_CASA_RS.md)

| Area | Status | Notes |
|---|---|---|
| §1 CASA table on-disk format | DONE | `table.dat` fully parsed **and written** (multi-DM ColumnSet). StandardStMan (scalars, `table.f0i` arrays, string buckets), IncrementalStMan (interval index), and TiledColumnStMan (tile buckets, reversed-dim shapes) fully read **and written** — one table can mix all three across several files, and python-casacore round-trips every column exactly. `getdminfo()` matches casacore byte-for-byte (incl. SPEC records); managers grouped by type+group and named by group (the `_1` auto-suffix belongs to the future `addcols`). |
| §2 Table lifecycle & locking | DONE | `Table` open/create, advisory lock/unlock, flush/close, iswritable/name, nrows/colnames; Send+Sync; eager DM-file loading |
| §3 Column data access | DONE (core) | read hot path (getcell/getcol/getcolslice/getcellslice/getvarcol across SSM/ISM/TSM/strings) + `WritableTable` writes (addrows/putcol/putcell/flush, `setmaxcachesize` no-op). Remaining: the pyo3 numpy/dict binding layer (getcolnp buffers, `{"shape","array"}` string dicts). |
| §4 Type system | ~90% | `ValueType` + numpy mapping done, verified against casacore 3.8.1 |
| §5 Metadata & descriptors | ~40% | read side done: nrows/colnames/getcoldesc/getdesc (public table-desc API) + getkeywords/getcolkeywords matching casacore; keyword writes and subtable linkage pending |
| §6 TaQL subset | 0% | not started |
| §7 MS schema / descriptors | 0% | not started |
| dask-ms integration | 0% | not started |

## Known casacore behaviour discovered by the comparison tests

- TaQL DDL cannot create `USHORT` columns.
- `getcol` promotes `uchar` columns to `uint16`.
- `getcolnp` fails for `uchar`/`short`/`uint` columns ("Unknown data type") —
  the zero-copy path supports only bool/int/float/double/complex/dcomplex.
- 1-D string columns return plain Python lists from `getcol`.
- TaQL DDL boolean type code is `B`, not `B1`.
