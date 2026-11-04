"""Slice arguments match python-casacore (issue #36).

python-casacore's binding converts a bare integer `blc`/`trc`/`inc` into a
length-1 `IPosition`: it selects a whole 1-D cell and, on any other rank,
fails inside the `Slicer` with a `RuntimeError` — never a `TypeError`.
casacure used to reject the scalar before any slicing happened
(`TypeError: argument 'blc': 'int' object cannot be cast as 'Sequence'`),
which breaks 1-D scalar slices outright and hides the failure from every
caller that catches `RuntimeError` (the casacore idiom).

Issue #36 also covered two gaps found while pinning that down:

* `getcellslice` had no `inc` parameter at all (python-casacore's
  signature has one), and `putcolslice` had neither `inc` nor `rowincr`,
  so python-casacore's own documented forms failed on arity alone;
* `putcolslice` had no rank check, so a wrong-rank slice panicked across
  the FFI boundary (`pyo3_runtime.PanicException: index out of bounds`)
  where casacore raises `Slicer error:Shape IPosition-lengths differ from
  ndim()`.  The last test below pins that it now raises.

Every expectation was verified against python-casacore 3.8.1 in the same
environment; each test notes what casacore does with the same call.
"""

import numpy as np
import pytest

from casacore.tables import (
    makearrcoldesc,
    maketabdesc,
    table,
)


@pytest.fixture
def slice_table(tmp_path):
    """V: variable 1-D with rows of shape 1, 2, 3 (ragged, like the issue's
    repro); A: fixed 1-D [4]; F: fixed 2-D [4, 6]."""
    p = str(tmp_path / "slice.tab")
    t = table(
        p,
        maketabdesc(
            [
                makearrcoldesc("V", 0.0, ndim=1),
                makearrcoldesc("A", 0.0, shape=[4]),
                makearrcoldesc("F", 0.0, shape=[4, 6]),
            ]
        ),
        nrow=3,
        ack=False,
    )
    for r in range(3):
        t.putcell("V", r, np.arange(1.0, r + 2.0))
        t.putcell("A", r, np.arange(1.0 + 10 * r, 5.0 + 10 * r))
        t.putcell("F", r, np.arange(24.0).reshape(4, 6))
    t.close()
    return p


def test_getcolslice_scalar_equals_the_list_form(slice_table):
    """A bare int on a 1-D column selects the same elements as the list
    form — python-casacore returns `[[1.], [1.], [1.]]` for each of these
    (int, numpy integer, 0-d integer array: all length-1 IPositions)."""
    t = table(slice_table, ack=False)
    want = [[1.0], [1.0], [1.0]]
    np.testing.assert_array_equal(t.getcolslice("V", 0, 0), want)
    np.testing.assert_array_equal(t.getcolslice("V", np.int64(0), 0), want)
    np.testing.assert_array_equal(t.getcolslice("V", np.array(0), 0), want)
    np.testing.assert_array_equal(t.getcolslice("V", [0], [0]), want)
    t.close()


def test_getcolslice_scalar_past_the_shortest_row_is_a_runtime_error(slice_table):
    """Out-of-range corners fail in the slicer: casacore raises
    `RuntimeError: Slicer error:infer: startResult>=shape` for
    `getcolslice("V", True, 1)` (bool is an int) — a RuntimeError, which
    `pytest.raises(TypeError)` would not catch either way."""
    t = table(slice_table, ack=False)
    with pytest.raises(RuntimeError):
        t.getcolslice("V", True, 1)
    with pytest.raises(RuntimeError):
        t.getcolslice("V", 1, 1)
    t.close()


def test_getcolslice_rejects_types_casacore_rejects_as_argument_errors(slice_table):
    """casacore refuses float/str corners with its ArgumentError — whose MRO
    is (ArgumentError, TypeError, ...) — so TypeError is the matching type
    here."""
    t = table(slice_table, ack=False)
    with pytest.raises(TypeError):
        t.getcolslice("V", 0.5, 1)
    with pytest.raises(TypeError):
        t.getcolslice("V", "0", 1)
    t.close()


def test_getcolslice_scalar_on_a_2d_column_is_a_runtime_error(slice_table):
    """A length-1 IPosition against a 2-D cell is a *slicer* failure in both
    libraries: casacore raises `Slicer error:Shape IPosition-lengths differ
    from ndim()`. Before issue #36 casacure raised TypeError here — the
    failure callers catching RuntimeError never see."""
    t = table(slice_table, ack=False)
    with pytest.raises(RuntimeError):
        t.getcolslice("F", 0, 1)
    with pytest.raises(RuntimeError):
        t.getcolslice("F", [0], [1])  # the list form already behaved so
    t.close()


def test_getcolslicenp_scalar_corners_fill_the_buffer(slice_table):
    """Same normalization on the buffer form (casacore: fills on 1-D,
    Slicer RuntimeError on 2-D)."""
    t = table(slice_table, ack=False)
    buf = np.zeros((3, 1))
    t.getcolslicenp("V", buf, 0, 0)
    np.testing.assert_array_equal(buf, [[1.0], [1.0], [1.0]])
    with pytest.raises(RuntimeError):
        t.getcolslicenp("F", np.zeros((3, 4, 6)), 0, 1)
    t.close()


def test_getcellslice_scalar_corners_and_inc(slice_table):
    """python-casacore's `getcellslice(columnname, rownr, blc, trc, inc=[])`
    — scalar corners work on a 1-D cell (`[1., 2.]` from row 2), the 5th
    argument exists (casacure used to raise `TypeError: table.getcellslice()
    takes 4 positional arguments but 5 were given`), an integer `inc` is a
    stride like casacore's, and a 2-D cell refuses the length-1 corner with
    a RuntimeError (`Slicer error:Shape IPosition-lengths differ from
    ndim()`)."""
    t = table(slice_table, ack=False)
    np.testing.assert_array_equal(t.getcellslice("V", 2, 0, 1), [1.0, 2.0])
    np.testing.assert_array_equal(t.getcellslice("V", 2, 0, 1, 1), [1.0, 2.0])
    np.testing.assert_array_equal(t.getcellslice("V", 2, [0], [1], [1]), [1.0, 2.0])
    with pytest.raises(RuntimeError):
        t.getcellslice("F", 0, 0, 1)
    t.close()


def test_putcolslice_scalar_corners_write_the_cell(slice_table):
    """casacore's putcolslice reaches the writer with scalar corners
    (`Column ... is not writable` was the only complaint on a read-only
    handle); on a writable table it stores, and so must casacure."""
    t = table(slice_table, ack=False, readonly=False)
    t.putcolslice("A", np.full((1, 4), 5.0), 0, 3, startrow=2, nrow=1)
    t.close()

    t = table(slice_table, ack=False)
    got = t.getcol("A")
    np.testing.assert_array_equal(got[2], [5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(got[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(got[1], [11.0, 12.0, 13.0, 14.0])
    t.close()


def test_putcolslice_strided_inc_matches_casacore(slice_table):
    """casacore honours `inc` when writing: `inc=[2]` over `blc=[0],
    trc=[3]` puts value columns 0 and 1 at cell indices 0 and 2 of every
    row — python-casacore's readback is
    `[[9, 2, 9, 4], [9, 12, 9, 14], [9, 22, 9, 24]]`."""
    t = table(slice_table, ack=False, readonly=False)
    t.putcolslice("A", np.full((3, 2), 9.0), [0], [3], [2])
    t.close()

    t = table(slice_table, ack=False)
    np.testing.assert_array_equal(
        t.getcol("A"),
        [[9.0, 2.0, 9.0, 4.0], [9.0, 12.0, 9.0, 14.0], [9.0, 22.0, 9.0, 24.0]],
    )
    t.close()


def test_putcolslice_legacy_and_casacore_forms_both_map_the_rows(slice_table):
    """The two positional forms must coexist (issue #36): python-casacore's
    `putcolslice(col, value, blc, trc, inc, startrow, nrow, rowincr)` and
    casacure's legacy `(col, value, blc, trc, startrow, nrow)`, where an
    integer in the `inc` slot means `startrow` — the rule
    `tests/test_subcell_tiles.py` relies on. This pins that neither form
    silently lands on the other's rows."""
    t = table(slice_table, ack=False, readonly=False)
    # Legacy: int in the inc slot = startrow 1, nrow 1 -> only row 1.
    t.putcolslice("A", np.full((1, 4), 1.0), [0], [3], 1, 1)
    # python-casacore form: sequence inc, then startrow 2, nrow 1 -> row 2.
    t.putcolslice("A", np.full((1, 4), 2.0), [0], [3], [1], 2, 1, 1)
    t.close()

    t = table(slice_table, ack=False)
    np.testing.assert_array_equal(t.getcol("A")[1], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(t.getcol("A")[2], [2.0, 2.0, 2.0, 2.0])
    t.close()


def test_putcolslice_wrong_rank_raises_instead_of_panicking(slice_table):
    """casacore refuses a corner rank that differs from the cell's
    (`Slicer error:Shape IPosition-lengths differ from ndim()` for the
    first case, `IPosition-lengths differ` for the second). casacure used
    to panic across the FFI boundary — `PanicException: index out of
    bounds`, which is a BaseException and would escape
    `pytest.raises(RuntimeError)`, so this test fails loudly if the rank
    check regresses."""
    t = table(slice_table, ack=False, readonly=False)
    with pytest.raises(RuntimeError, match="rank"):
        t.putcolslice("F", np.full((3, 4, 2), 1.0), [0], [1])
    with pytest.raises(RuntimeError, match="lengths differ"):
        t.putcolslice("F", np.full((3, 4, 6), 1.0), [0, 0], [1])
    t.close()

    # Nothing was written through either refusal.
    t = table(slice_table, ack=False)
    np.testing.assert_array_equal(
        t.getcell("F", 0), np.arange(24.0).reshape(4, 6)
    )
    t.close()
