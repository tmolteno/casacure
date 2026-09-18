# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — StandardStMan (scalar, array, long-string) **and** IncrementalStMan columns read+write with proven two-way interop, in **multi-data-manager tables**; MS schema creation not yet.

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit + fixture tests | `cargo test` | 62/62 | type system, AipsIO read+write (both endians), `table.dat`, StandardStMan data file + `table.f0i` + string buckets, IncrementalStMan (interval index, multi-DM tables) — all read+write |
| casacore comparison tests | `.venv/bin/python -m pytest tests/` | 5/5 | type system only |
| write interop (manual) | `examples/create_sample_table.rs` + python-casacore | ✓ | casacure-write → casacore-read: SSM scalars, arrays, long strings, and ISM TIME/ANT1 in one 4-file table; 3-row and 100-row variants return exactly the written values |

## Progress by area (per CASACORE_TO_CASA_RS.md)

| Area | Status | Notes |
|---|---|---|
| §1 CASA table on-disk format | ~95% | `table.dat` fully parsed **and written** (multi-DM ColumnSet). StandardStMan data files (scalars, fixed-shape arrays via `table.f0i`, variable strings via string buckets) and IncrementalStMan data files (interval index, Direct flag no-op) fully read **and written** — including tables that mix both storage managers across several files. Read+write interop proven against real casacore both ways. Gap: TiledColumnStMan and dminfo round-trips. |
| §2 Table lifecycle & locking | 0% | not started |
| §3 Column data access | 0% | not started |
| §4 Type system | ~90% | `ValueType` + numpy mapping done, verified against casacore 3.8.1 |
| §5 Metadata & descriptors | 0% | not started |
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
