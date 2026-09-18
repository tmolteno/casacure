# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — scaffolding only; no table files can be read or written yet.

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit tests | `cargo test` | 7/7 | type system only |
| casacore comparison tests | `.venv/bin/python -m pytest tests/` | 5/5 | type system only |

## Progress by area (per CASACORE_TO_CASA_RS.md)

| Area | Status | Notes |
|---|---|---|
| §1 CASA table on-disk format | 0% | not started |
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
