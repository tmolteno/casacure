# DIFFERENCES.md

Deliberate behavioural contracts where casacure diverges from casacore /
python-casacore. This document lists only *intentional* design decisions —
not every unimplemented feature or gap (those are tracked in
`ARE_WE_CURED.md`). Each entry is asserted by the ported python-casacore
tests.

Ground truth for every entry below was probed against real
python-casacore 3.8.1 on this machine.

## Out-of-range `getcell` raises `ValueError`

**casacore behaviour.** `t.getcell("C", 99)` on a 3-row table fails inside
the storage manager with `RuntimeError: TableProxy::getCell: no such row`.

**casacure behaviour.** The row bounds are checked before the storage
manager is consulted, raising
`ValueError: row 99 is out of range (table has 3 rows)`.

### Why

`ValueError` is the contract used throughout the binding for *invalid
arguments* (a bad shape, a read-only handle, a row that does not exist),
while `RuntimeError` is reserved for storage-layer failures the caller
cannot have caused. A negative or out-of-range row index is an argument
error, and it is the same class of mistake regardless of which storage
manager the column happens to use — routing it through the manager made the
exception type depend on the column's data manager. Raising up front also
avoids the misleading `row 0 not covered by any indexed bucket` message an
unwritten-but-valid row used to produce.

Arithmetic on the result is unaffected: both `ValueError` and
`RuntimeError` derive from `Exception`, and dask-ms never reads a row it did
not just write.

## Reads of rows that are not on disk return defaults (parity)

Not a divergence, but recorded here because it used to be one: a read of a
row the table does not have yet — `addrows` on a freshly created table, or a
column added this session — returns the column's default value (`0`, `0.0`,
`""`, `False`; a zeroed array of the declared shape), exactly like
casacore's `ColumnSet` defaults. `test_check_putdata` asserts
`getcol("coli") == [0, 0]` after `addrows(2)`, matching python-casacore.

A writable handle merges three sources for every read: the pending
(unflushed) writes, the on-disk values for rows the snapshot actually has,
and the buffered `addrows` default for everything newer. The earlier
"unset cells raise" contract (documented here until 2026-09-24, commit
0ea3db6) was superseded by ad0b510 and is gone: no read path raises for an
unwritten cell anymore.

## Other intentional divergences

None currently. TaQL's `ORDER BY` is an unimplemented feature (a gap), not a
deliberate divergence — unimplemented features and compatibility gaps are
tracked in `ARE_WE_CURED.md`.
