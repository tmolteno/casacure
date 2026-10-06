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

## `AZEL` is the astropy direction, not casacore's `AZEL`

**casacore behaviour.** Its two horizontal references are not the same
transform. `measure(d, 'AZEL')` builds the local horizon from the
observer's **geocentric** latitude, while `measure(d, 'AZELGEO')` uses the
**geodetic (WGS84)** latitude. At MeerKAT those two horizons differ by
0.169°, so for a field 3° from the zenith casacore's `AZEL` and `AZELGEO`
azimuths differ by 2.36° — the parallactic-angle gap reported in issue #15
(131.192182° vs 133.550707°).

**casacure behaviour.** Both references return the topocentric observed
place built from the **geodetic** latitude: the physically correct vertical
(plumb line / ellipsoid normal), the one astropy's `AltAz` frame uses, and
the one casacore itself uses for `AZELGEO`. For the issue #15 case casacure
returns az −47.701503°, alt +86.855931° and PA 133.551306°, which matches
astropy to 0.005″/0.09″ and casacore's `AZELGEO` to 0.55″/2.2″.

### Why

The local zenith is defined by the observer's vertical, and for an observer
on the rotating, flattened Earth that vertical is the ellipsoid normal, not
the geocentric radius. casacore's `AZELGEO` — its "geocentric" reference —
is in fact the geodetic one, so casacure reproduces that and treats
`AZEL` as a synonym, the way astropy has only one `AltAz`. Following
casacore's `AZEL` quirk would have meant *adding* a known 0.169° error to
the horizon pole to match a bug.

### How it is asserted

`tests/test_measures.py::test_casacore_azel_uses_the_geocentric_latitude`
proves the identification rather than just tolerating a difference: it
hands casacure a WGS84 site whose latitude *is* MeerKAT's geocentric
latitude and reproduces real casacore's `AZEL` to under 1″.
`test_casacore_azelgeo_parity_over_the_grid` holds the astropy-compatible
`AZELGEO` path to casacore over 1926–2126, and `MEASURES_ACCURACY.md`
records the full accuracy contract, including the one IERS-prediction
window where the bundled and astropy data disagree.

## Other intentional divergences

None currently. TaQL's `ORDER BY` is an unimplemented feature (a gap), not a
deliberate divergence — unimplemented features and compatibility gaps are
tracked in `ARE_WE_CURED.md`.

## Locking: casacore's protocol, with the yield checked at operation entry

Not a divergence from the on-disk protocol — casacore's locking is
implemented byte- and behaviour-compatibly (fcntl record locks on
`table.lock`, request list, `sync` record, all eight `lockoptions`), and
real python-casacore and casacure exclude each other in both directions
(`tests/test_locking.py`). Recorded here because two behaviours are
casacore's *conventions* rather than its exact mechanics:

- **`AutoLocking` yield points.** casacore checks for a waiting process
  inside its column cache on every operation; casacure reads are immutable
  snapshots, so the check runs at the entry of the data-access methods
  (`nrows`/`getcol`/`getcell`/…, same 25-call + interval throttle). A
  reader that is *between* such operations still holds its read lock,
  where casacore might release marginally earlier within one operation.
- **Missing `table.lock`.** A byte-level directory copy has no lock file;
  as with casacore's `mustExist=False`, every lock request then succeeds
  without actual locking until a table rewrite creates the file. Writers
  that need exclusion against copies should re-create the table rather
  than copy it.
