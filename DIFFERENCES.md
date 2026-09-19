# DIFFERENCES.md

Deliberate behavioural contracts where casacure diverges from casacore /
python-casacore. This document lists only *intentional* design decisions —
not every unimplemented feature or gap (those are tracked in
`ARE_WE_CURED.md`). Each entry is asserted by the ported python-casacore
tests.

## Unset cells: casacure raises instead of returning a default

**casacore behaviour.** Reading a cell of a freshly created column that was
never written returns the column's *default value*: `0` for integer columns,
`0.0` for float/double/complex, `""` for strings, `False` for booleans.
python-casacore inherits this: `t.addrows(2)` followed by `t.getcol("cold")`
yields `[0.0, 0.0]`.

**casacure behaviour.** `getcol` (and cell reads generally) raise
`ValueError: column <i> row <r> has not been set` for any cell never written.

### Why

1. **Silent zeros are a bug-masker.** A fresh table that reads like it
   contains real zero-valued data destroys the difference between
   "the column genuinely holds zeros" and "the column was never touched".
   The most common cause of touching unwritten cells is a read-before-write
   bug (wrong table, wrong column, wrong rows). Returning plausible data
   makes that bug undetectable, and the fabricated zeros then flow unchanged
   into downstream analysis, silently corrupting results — worst possible
   failure mode in a science stack.
2. **Absence is cheap to check, costly to fake.** casacure's write backends
   store cells as vectors that are *empty* where nothing was written;
   honoring a default would allocate and invent a value on every read. The
   cheap, honest representation is a missing cell, and the honest reaction
   to a read of a missing cell is an error.
3. **Correctness first.** casacure's stance throughout is strictness over
   convenience: fail loudly at the first read of an unwritten cell rather
   than turn a latent bug into a plausible wrong answer.

### Impact on ported tests

The python-casacore test `test_check_putdata` reads unset cells expecting
zeros; the casacure port asserts the exception instead
(`with pytest.raises(ValueError): t.getcol("coli")`). Everything else in
that test (put/get roundtrips) is unchanged. Impenetrable to normal use:
dask-ms writes every cell before reading it (its 219-test suite is green
against casacure), so the strict contract costs nothing there.

## Other intentional divergences

None currently. TaQL's `ORDER BY` is an unimplemented feature (a gap), not a
deliberate divergence — unimplemented features and compatibility gaps are
tracked in `ARE_WE_CURED.md`.
