"""casacore's slice validation, which casacure used to skip (issue #38).

Four gaps, all pinned against python-casacore 3.8.1 in the same
environment (issue #38 has the full comparison tables):

1. **Empty corners** (`blc=[], trc=[]`) mean "the whole cell" in
   casacure's core — an internal idiom the `-1` corner conversion feeds
   and dask-ms's `(-1, ..)` whole-cell idiom relies on — but casacore's
   `Slicer` compares the `IPosition` length against `ndim()` and always
   raises. Only the Python-visible empty corners are refused now.
2. **Scalar columns**: every casacore slice entry point refuses a
   non-array column (`column S is not an array column`, RuntimeError);
   casacure sliced them happily.
3. **Buffer conformance** (`getcolslicenp`): casacore raises for a
   buffer too big, too small, or with the wrong row count; casacure
   silently filled whatever overlapped (`fill_flat` bounds-checks, so no
   memory was at risk — the caller just got partial data).
4. **Value conformance** (`putcolslice`): a value whose rows or slice
   extent differ from what would be written, or a window past the last
   row, is refused *before* anything is written — casacure used to write
   a prefix silently and, for a window past the end, raise only after
   the rows before it were already written. `nrow <= 0` now means the
   rest of the column (casacore's `-1`), with the value required to
   cover exactly those rows.
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
def validated_table(tmp_path):
    """S: scalar; V: variable 1-D rows 1/2/3; A: fixed 1-D [4]; F: fixed
    2-D [4, 6] — the issue's fixture."""
    p = str(tmp_path / "validated.tab")
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


# --------------------------------------------------------------------------
# 1. Empty corners
# --------------------------------------------------------------------------


def test_empty_corners_raise_on_every_read_entry(validated_table):
    """casacore raises `Slicer error:Shape IPosition-lengths differ from
    ndim()` for empty corners on any array column — from getcolslice,
    getcolslicenp, getcellslice and getcellslicenp alike — where casacure
    used to return the whole column/cell. Both are `RuntimeError`."""
    t = table(validated_table, ack=False)
    with pytest.raises(RuntimeError, match="empty blc/trc"):
        t.getcolslice("F", [], [])
    with pytest.raises(RuntimeError, match="empty blc/trc"):
        t.getcolslicenp("F", np.zeros((3, 4, 6)), [], [])
    with pytest.raises(RuntimeError, match="empty blc/trc"):
        t.getcellslice("A", 1, [], [])
    with pytest.raises(RuntimeError, match="empty blc/trc"):
        t.getcellslicenp("A", np.zeros(4), 1, [], [])
    # The -1 whole-cell idiom (the dask-ms one) still works: casacore
    # agrees that -1 corners mean begin/end.
    np.testing.assert_array_equal(
        np.asarray(t.getcellslice("A", 1, -1, -1)), [11.0, 12.0, 13.0, 14.0]
    )
    t.close()


def test_empty_corners_raise_on_putcolslice(validated_table):
    """The write side: casacore raises the same Slicer rank error where
    casacure used to take the full-cell fast path and write everything."""
    t = table(validated_table, ack=False, readonly=False)
    with pytest.raises(RuntimeError, match="empty blc/trc"):
        t.putcolslice("A", np.full((3, 4), 1.0), [], [])
    t.close()

    t = table(validated_table, ack=False)
    np.testing.assert_array_equal(t.getcol("A")[0], [1.0, 2.0, 3.0, 4.0])
    t.close()


# --------------------------------------------------------------------------
# 2. Scalar columns
# --------------------------------------------------------------------------


def test_scalar_column_is_refused_by_every_slice_api(validated_table):
    """casacore refuses a non-array column from every slice entry point
    with a RuntimeError (its message differs per entry point; casacore's
    own `getcolslicenp` even reports the Slicer error instead) — where
    casacure used to slice right through. The plain getcol/getcell paths
    are untouched: a scalar column still reads normally."""
    t = table(validated_table, ack=False)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.getcolslice("S", 0, 3)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.getcolslicenp("S", np.zeros(3), 0, 0, 3)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.getcellslice("S", 0, 0, 3)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.getcellslicenp("S", np.zeros(3), 0, 0, 3)
    # Not a slice: scalar reads keep working.
    np.testing.assert_array_equal(t.getcol("S"), [3.0, 4.0, 5.0])
    assert t.getcell("S", 0) == 3.0
    t.close()

    t = table(validated_table, ack=False, readonly=False)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.putcolslice("S", np.full((3, 1), 9.0), 0, 0)
    with pytest.raises(RuntimeError, match="not an array column"):
        t.putcellslice("S", 0, 9.0, 0, 0)
    t.close()


# --------------------------------------------------------------------------
# 3. Buffer conformance (getcolslicenp)
# --------------------------------------------------------------------------


def test_getcolslicenp_buffer_shape_must_match(validated_table):
    """Slice `[1..2]` of A over 3 rows is `(3, 2)`: casacore raises
    `ArrayColumn::getColumn ... Table array conformance error` for a
    buffer too big, too small, or with the wrong row count; casacure used
    to fill whatever overlapped, silently."""
    t = table(validated_table, ack=False)
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcolslicenp("A", np.zeros((3, 7)), [1], [2])  # too big
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcolslicenp("A", np.zeros((3, 1)), [1], [2])  # too small
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcolslicenp("A", np.zeros((2, 2)), [1], [2])  # wrong rows
    buf = np.zeros((3, 2))
    t.getcolslicenp("A", buf, [1], [2])
    # 0-based inclusive: [1..2] is indices 1 and 2 of each row
    # ([1,2,3,4] -> [2,3], ...).
    np.testing.assert_array_equal(
        buf, [[2.0, 3.0], [12.0, 13.0], [22.0, 23.0]]
    )
    t.close()


def test_getcolslicenp_whole_cell_fast_path_shape_must_match(validated_table):
    """The whole-cell fast path (DDFacet's chunk reader) gets the same
    conformance check: corners `0..3` select A's whole cell, so the
    buffer must be `(3, 4)` — casacore raises for `(3, 7)` and `(3, 1)`
    there too."""
    t = table(validated_table, ack=False)
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcolslicenp("A", np.zeros((3, 7)), 0, 3)
    with pytest.raises(RuntimeError, match="conformance"):
        t.getcolslicenp("A", np.zeros((3, 1)), 0, 3)
    buf = np.zeros((3, 4))
    t.getcolslicenp("A", buf, 0, 3)
    np.testing.assert_array_equal(
        buf,
        [[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0]],
    )
    t.close()


# --------------------------------------------------------------------------
# 4. Value conformance and atomicity (putcolslice)
# --------------------------------------------------------------------------


def test_putcolslice_refuses_a_value_that_does_not_match(validated_table):
    """casacore's put-side conformance, all refused before a single cell
    is written (casacore's messages in parentheses):

    * an extent mismatch — value `(3, 3)` for corners `0..3`
      (`putColumn(slicer) ... (shape=[3], expected [4])`);
    * a row-count mismatch — value `(2, 4)` with `nrow=1`
      (`putColumnCells ... conformance`);
    * a default-nrow value shorter than the rest of the column
      (`column A has 3, but array has 1 rows`);
    * a window past the last row — value `(3, 4)` from row 1, where only
      two rows remain (casacore refuses up front; casacure used to write
      rows 1 and 2 *then* raise `row 3 is out of range`)."""
    t = table(validated_table, ack=False, readonly=False)
    with pytest.raises(RuntimeError, match="conformance"):
        t.putcolslice("A", np.full((3, 3), 2.0), [0], [3])
    with pytest.raises(RuntimeError, match="conformance"):
        t.putcolslice("A", np.full((2, 4), 3.0), [0], [3], startrow=0, nrow=1)
    with pytest.raises(RuntimeError, match="conformance"):
        t.putcolslice("A", np.full((1, 4), 6.0), [0], [3], startrow=0)
    with pytest.raises(RuntimeError, match="conformance"):
        t.putcolslice("A", np.full((3, 4), 4.0), [0], [3], startrow=1)
    t.close()

    # Atomicity: none of the four refusals wrote anything.
    t = table(validated_table, ack=False)
    np.testing.assert_array_equal(
        t.getcol("A"),
        [[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0]],
    )
    t.close()


def test_putcolslice_default_nrow_is_the_rest_of_the_column(validated_table):
    """casacore's `nrow=-1` (and casacure's `nrow=0` default) means *the
    rest of the column*: a 3-row value from row 0 covers all 3 rows and
    is accepted; a 1-row value covers only row 0 and is refused."""
    t = table(validated_table, ack=False, readonly=False)
    t.putcolslice("A", np.full((3, 4), 7.0), [0], [3])  # full length: OK
    with pytest.raises(RuntimeError, match="conformance"):
        t.putcolslice("A", np.full((1, 4), 8.0), [0], [3])  # prefix: refused
    t.close()

    t = table(validated_table, ack=False)
    np.testing.assert_array_equal(
        t.getcol("A"),
        [[7.0, 7.0, 7.0, 7.0], [7.0, 7.0, 7.0, 7.0], [7.0, 7.0, 7.0, 7.0]],
    )
    t.close()


def test_putcolslice_explicit_window_still_writes_exactly_it(validated_table):
    """The strictness is about conformance, not about refusing valid
    writes: an explicit `startrow`/`nrow` with a matching value writes
    only that window (casacore-verified behaviour, unchanged)."""
    t = table(validated_table, ack=False, readonly=False)
    t.putcolslice("A", np.full((1, 4), 9.0), [0], [3], startrow=2, nrow=1)
    t.close()

    t = table(validated_table, ack=False)
    np.testing.assert_array_equal(
        t.getcol("A"),
        [[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [9.0, 9.0, 9.0, 9.0]],
    )
    t.close()
