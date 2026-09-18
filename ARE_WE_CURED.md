# ARE WE CURED?

Overall progress towards replacing casacore as the dask-ms I/O backend.
Work areas are tracked as GitHub issues; subtasks live in `TODO.md`.

**Verdict: NOT YET CURED** — reading works for StandardStMan scalar columns; writing not started.

## Test status

| Suite | Command | Passing | Coverage |
|---|---|---|---|
| Rust unit + fixture tests | `cargo test` | 45/45 | type system, AipsIO reader (both endians), `table.dat` (header, TableDesc, records, ColumnSet + StandardStMan spec), StandardStMan data-file read (header, index, scalar cell values) |
| casacore comparison tests | `.venv/bin/python -m pytest tests/` | 5/5 | type system only |

## Progress by area (per CASACORE_TO_CASA_RS.md)

| Area | Status | Notes |
|---|---|---|
| §1 CASA table on-disk format | ~45% | `table.dat` fully parsed (header, TableDesc, column descriptors, keyword records, ColumnSet + StandardStMan spec); StandardStMan data file readable — header, chained index buckets, `SSMIndex`, all scalar cell types (incl. bit-packed Bool, inline short strings) with values verified against the real casacore `typed.tab`. Gaps: string buckets (>8 chars), arrays, all writing. |
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
