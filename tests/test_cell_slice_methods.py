"""`getcellslicenp` and `putcellslice` (issue #37).

python-casacore's `table` has both per-cell slice methods; casacure's had
neither, so code ported from casacore died with
`AttributeError: 'builtins.table' object has no attribute ...`.

Every expectation was pinned against python-casacore 3.8.1 in the same
environment — see issue #37 for the full contract table — including the
readback values:

* the buffer of `getcellslicenp` carries the *slice's* shape (no leading
  row axis, unlike `getcolslicenp`), must match it exactly, and must be
  C-contiguous (`ValueError`, as `getcolslicenp` already raises);
* `putcellslice`'s value must be exactly the slice shape — no leading
  row axis, no broadcast — and `inc` is the per-dimension stride;
* both refuse a scalar column and take issue #36's scalar-corner rules.
"""

import numpy as np
import pytest

from casacore.tables import (
    makearrcoldesc,
    maketabdesc,
    makescacoldesc,
    table,
)


@pytest.fixture
def cell_slice_table(tmp_path):
    """S: scalar; V: variable 1-D with rows of shape 1/2/3; A: fixed 1-D
    [4]; F: fixed 2-D [4, 6] — the issue's fixture."""
    p = str(tmp_path / "cell_slice.tab")
    t = table(
        p,
        maketabdesc(
            [
                makescacoldesc("S", 3.0),
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
        t.putcell("S", r, 3.0 + r)
    t.close()
    return p


def test_getcellslicenp_fills_the_slice_and_returns_none(cell_slice_table):
    """The buffer holds the slice (2 elements of A row 1: `[11., 12.]` ->
    `[70., 71.]` after the write below), with no leading row axis, and the
    call returns `None` — python-casacore does exactly so."""
    t = table(cell_slice_table, ack=False)
    buf = np.zeros(2)
    assert t.getcellslicenp("A", buf, 1, 1, 2) is None
    # corners are 0-based inclusive: [1..2] is indices 1 and 2 of
    # [11, 12, 13, 14].
    np.testing.assert_array_equal(buf, [12.0, 13.0])
    # ...the same elements getcellslice returns for that slice.
    np.testing.assert_array_equal(
        buf, np.asarray(t.getcellslice("A", 1, 1, 2))
    )
    t.close()


def test_getcellslicenp_scalar_corners(cell_slice_table):
    """#36's rules carry over: a bare int selects the whole 1-D cell (as
    python-casacore's `[0]..[1]` does: `[1., 2.]` from row 2 of V) and a
    2-D cell refuses the length-1 corner with a `RuntimeError`, never a
    `TypeError` (casacore: `Slicer error:Shape IPosition-lengths differ
    from ndim()`)."""
    t = table(cell_slice_table, ack=False)
    buf = np.zeros(2)
    t.getcellslicenp("V", buf, 2, 0, 1)
    np.testing.assert_array_equal(buf, [1.0, 2.0])
    with pytest.raises(RuntimeError):
        t.getcellslicenp("F", np.zeros((4, 6)), 0, 0, 1)
    t.close()


def test_getcellslicenp_buffer_shape_must_match_the_slice(cell_slice_table):
    """casacore's `ArrayColumn::getSlice` conformance: too big and too
    small buffers both raise; only the exact shape fills."""
    t = table(cell_slice_table, ack=False)
    # casacore: `RuntimeError: ... Table array conformance error
    # (shape=[3], expected [2])` for the same three buffers.
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcellslicenp("A", np.zeros(3), 1, 1, 2)
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcellslicenp("A", np.zeros(1), 1, 1, 2)
    buf = np.zeros(2)
    t.getcellslicenp("A", buf, 1, 1, 2)
    np.testing.assert_array_equal(buf, [12.0, 13.0])
    t.close()


def test_getcellslicenp_non_contiguous_buffer_is_a_value_error(cell_slice_table):
    """python-casacore: `ValueError: Argument 'nparray' has to be a
    contiguous numpy array` — a ValueError, not a RuntimeError (the
    shape check passes first: this buffer is the right shape)."""
    t = table(cell_slice_table, ack=False)
    with pytest.raises(ValueError):
        t.getcellslicenp("A", np.zeros(4)[::2], 1, 1, 2)
    t.close()


def test_getcellslicenp_scalar_column_is_refused(cell_slice_table):
    """casacore: `TableProxy::getColumnSlice: column S is not an array
    column` (RuntimeError) — every slice entry point, this one included."""
    t = table(cell_slice_table, ack=False)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.getcellslicenp("S", np.zeros(3), 0, 0, 3)
    t.close()


def test_putcellslice_writes_exactly_the_slice(cell_slice_table):
    """`putcellslice("A", 1, [70., 71.], 1, 2)` turns row 1
    `[11, 12, 13, 14]` into `[11, 70, 71, 14]` — python-casacore's
    readback is identical."""
    t = table(cell_slice_table, ack=False, readonly=False)
    t.putcellslice("A", 1, np.array([70.0, 71.0]), 1, 2)
    t.close()

    t = table(cell_slice_table, ack=False)
    np.testing.assert_array_equal(t.getcol("A")[1], [11.0, 70.0, 71.0, 14.0])
    np.testing.assert_array_equal(t.getcol("A")[0], [1.0, 2.0, 3.0, 4.0])
    t.close()


def test_putcellslice_inc_is_the_stride(cell_slice_table):
    """`inc=[2]` over corners `0..3` puts value elements 0 and 1 at cell
    indices 0 and 2 — casacore's readback for the same call:
    row 0 becomes `[30, 2, 31, 4]`."""
    t = table(cell_slice_table, ack=False, readonly=False)
    t.putcellslice("A", 0, np.array([30.0, 31.0]), 0, 3, [2])
    t.close()

    t = table(cell_slice_table, ack=False)
    np.testing.assert_array_equal(t.getcol("A")[0], [30.0, 2.0, 31.0, 4.0])
    t.close()


def test_putcellslice_value_must_match_the_slice_exactly(cell_slice_table):
    """casacore refuses, with a conformance `RuntimeError`, every value
    that is not exactly the slice: a scalar (no broadcast —
    `(shape=[1], expected [2])`), a too-long one, a wrong-rank one, and
    one that ignores that a `-1` corner means `begin` — corners
    `(-1, 2)` slice 3 elements, so a 2-element value is refused too."""
    t = table(cell_slice_table, ack=False, readonly=False)
    with pytest.raises(RuntimeError):
        t.putcellslice("A", 1, 9.0, 1, 2)          # scalar value
    with pytest.raises(RuntimeError):
        t.putcellslice("A", 1, np.full(6, 1.0), 1, 2)  # too long
    with pytest.raises(RuntimeError):
        t.putcellslice("A", 0, np.full((1, 4), 1.0), 0, 3)  # wrong rank
    with pytest.raises(RuntimeError):
        t.putcellslice("A", 2, np.array([80.0, 81.0]), -1, 2)  # -1 is begin
    # Nothing was written through any refusal.
    t.close()
    t = table(cell_slice_table, ack=False)
    np.testing.assert_array_equal(t.getcol("A")[1], [11.0, 12.0, 13.0, 14.0])
    np.testing.assert_array_equal(t.getcol("A")[2], [21.0, 22.0, 23.0, 24.0])
    t.close()


def test_putcellslice_two_dimensional_cell(cell_slice_table):
    """Full-rank corners and value on F: casacore writes the (2, 6) block
    into rows 0..1 of the cell (its readback after the same call: rows 0
    and 1 become 5.0)."""
    t = table(cell_slice_table, ack=False, readonly=False)
    t.putcellslice("F", 0, np.full((2, 6), 5.0), [0, 0], [1, 5])
    t.close()

    t = table(cell_slice_table, ack=False)
    cell = t.getcell("F", 0)
    np.testing.assert_array_equal(cell[0], np.full(6, 5.0))
    np.testing.assert_array_equal(cell[1], np.full(6, 5.0))
    np.testing.assert_array_equal(
        cell[2], np.arange(12.0, 18.0)
    )  # untouched
    t.close()


def test_putcellslice_scalar_column_is_refused(cell_slice_table):
    """casacore: `Invalid Table data type ... ArrayColumn ctor for column
    S` (RuntimeError) on the write side too."""
    t = table(cell_slice_table, ack=False, readonly=False)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.putcellslice("S", 0, np.array([9.0]), 0, 0)
    t.close()


def test_getcellslicenp_inc_matches_getcellslice(cell_slice_table):
    """The `inc` argument reaches the same slice machinery as
    `getcellslice` (whose #36 semantics casacore pins): the filled buffer
    equals what getcellslice returns for the same request."""
    t = table(cell_slice_table, ack=False)
    want = np.asarray(t.getcellslice("A", 1, 0, 3, [2]))
    buf = np.zeros(want.shape)
    t.getcellslicenp("A", buf, 1, 0, 3, [2])
    np.testing.assert_array_equal(buf, want)
    t.close()
