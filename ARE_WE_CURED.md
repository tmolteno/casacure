# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — StandardStMan scalar, fixed-shape array, **and long-string** columns read+write with proven two-way interop; MS creation not yet.

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit + fixture tests | `cargo test` | 58/58 | type system, AipsIO read+write (both endians), `table.dat`, StandardStMan data file + array index file + **string buckets** read+write (scalars, arrays, long strings incl. chaining, multi-bucket, values) |
| casacore comparison tests | `.venv/bin/python -m pytest tests/` | 5/5 | type system only |
| write interop (manual) | `examples/create_sample_table.rs` + python-casacore | ✓ | casacure-write → casacore-read: scalar, array, and long-string (3 rows, 100 rows, and 1000-char chained) tables return exactly the written values |

## Progress by area (per CASACORE_TO_CASA_RS.md)

| Area | Status | Notes |
|---|---|---|
| §1 CASA table on-disk format | ~85% | `table.dat` fully parsed **and written**. StandardStMan data files fully read **and written** for scalar columns, fixed-shape arrays (with `table.f0i`), and variable strings of any length (SSM string buckets with big-endian canonical headers and next-bucket chaining). Read and write interop proven against real casacore both ways. Gaps: Incremental/Tiled managers, dminfo round-trips. |
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
