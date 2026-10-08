"""`table.query()` / `table.sort()` results are casacore-style reference
tables (issue #16).

A reference table carries the source table's columns with a **row order**
(`Vec<u64>`), not a copy of the selected cells: on the ssd0000.MS killMS case
the materialised form held ~2.05 GiB per distinct selection for the lifetime
of the handle (2.48 GiB peak measured on this checkout before #16); the row
order costs kilobytes.

These tests hold the reference table against the materialising TaQL path
(`taql()`, which still copies) — same statement, row for row, column for
column — plus the API surface DDFacet's `GiveMainTable` exercises
(`Data/ClassMS.py:227-234`).

Run:
    PYTHONPATH=tests/shim python -m pytest tests/test_query_sort_views.py -q
"""

import glob
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

from casacore.tables import (
    default_ms,
    maketabdesc,
    makearrcoldesc,
    makescacoldesc,
    table,
    taql,
)

NROW = 24
NCHAN = 4
NCORR = 2
# Shuffled on purpose: row i has TIME[i], so the sorted order is a real
# permutation (and rows 4/5/13 tie, which pins the tie-breaking rule).
TIME = np.array(
    [3.0, 1.0, 2.0, 1.0, 5.0, 5.0, 0.5, 4.0, 2.5, 6.0, 0.5, 7.0,
     1.5, 2.0, 3.5, 8.0, 4.5, 0.0, 6.5, 2.5, 9.0, 7.5, 5.5, 8.5]
)
SEL = "FIELD_ID==0 && DATA_DESC_ID==0"


@pytest.fixture
def ms(tmp_path):
    """A small MS with the columns DDFacet/killMS read, on the storage
    managers a real MS uses (TIME on IncrementalStMan, DATA tiled)."""
    p = str(tmp_path / "t.ms")
    data_desc = {
        "DATA": {
            "valueType": "complex",
            "dataManagerType": "TiledColumnStMan",
            "dataManagerGroup": "TiledData",
            "option": 4,
            "maxlen": 0,
            "comment": "The data column",
            "ndim": 2,
            "shape": [NCHAN, NCORR],
            "_c_order": True,
            "keywords": {},
        }
    }
    default_ms(p, tabdesc=data_desc)

    rng = np.random.default_rng(11)
    data = (
        rng.normal(size=(NROW, NCHAN, NCORR)) + 1j * rng.normal(size=(NROW, NCHAN, NCORR))
    ).astype(np.complex64)
    t = table(p, readonly=False, ack=False)
    t.addrows(NROW)
    t.putcol("TIME", TIME)
    t.putcol("TIME_CENTROID", TIME)
    t.putcol("ANTENNA1", np.arange(NROW) % 3, dtype=np.int32) if False else t.putcol(
        "ANTENNA1", (np.arange(NROW) % 3).astype(np.int32)
    )
    t.putcol("ANTENNA2", ((np.arange(NROW) + 1) % 3).astype(np.int32))
    t.putcol("UVW", rng.normal(size=(NROW, 3)))
    t.putcol("WEIGHT", np.ones((NROW, NCORR), dtype=np.float32))
    t.putcol("SIGMA", np.ones((NROW, NCORR), dtype=np.float32))
    t.putcol("FLAG", np.zeros((NROW, NCHAN, NCORR), dtype=bool))
    t.putcol("DATA", data)
    t.putcol("FIELD_ID", np.zeros(NROW, dtype=np.int32))
    t.putcol("DATA_DESC_ID", np.zeros(NROW, dtype=np.int32))
    t.close()
    return p


def _expected_order(time, mask=None):
    """The rows `sort("TIME")` must produce: selected, ascending, stable."""
    rows = np.arange(len(time))
    if mask is not None:
        rows = rows[mask]
    return rows[np.argsort(time[rows], kind="stable")]


def _assert_same_column(got, want):
    if isinstance(want, dict):
        assert set(got) == set(want)
        for k in want:
            np.testing.assert_array_equal(got[k], want[k])
    else:
        got = np.asarray(got)
        want = np.asarray(want)
        assert got.shape == want.shape, (got.shape, want.shape)
        assert got.dtype == want.dtype, (got.dtype, want.dtype)
        np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Correctness: the reference table is the source's rows, reordered.
# ---------------------------------------------------------------------------


def test_give_main_table_chain_returns_the_selected_sorted_rows(ms):
    """DDFacet Data/ClassMS.py:227-234 — `t.query(TaQL)` then `t.sort("TIME")`."""
    src = table(ms, ack=False)
    t = table(ms, ack=False).query(SEL)
    s = t.sort("TIME")

    assert s.nrows() == NROW
    assert np.all(np.diff(s.getcol("TIME")) >= 0)

    order = _expected_order(TIME)
    for name in s.colnames():
        _assert_same_column(s.getcol(name), src.getcol(name)[order])
    # The descriptor travels with it.
    assert s.colnames() == src.colnames()
    for name in s.colnames():
        assert s.getcoldesc(name) == src.getcoldesc(name), name
    # `name()` is the source directory (the subtable/dminfo paths resolve
    # against it); casacore's reference tables report a notional temp path
    # instead — documented in DIFFERENCES.md.
    assert s.name() == ms


def test_a_filtered_selection_is_a_subset_in_source_order(ms):
    t = table(ms, ack=False)
    mask = TIME < 5.0
    q = t.query("TIME < 5.0")
    got = q.getcol("TIME")
    np.testing.assert_array_equal(got, TIME[mask])
    # ... and sorted on top of the selection.
    order = _expected_order(TIME, mask)
    q2 = q.sort("TIME")
    np.testing.assert_array_equal(q2.getcol("TIME"), TIME[order])
    np.testing.assert_array_equal(
        q2.getcol("ANTENNA1"), (np.arange(NROW) % 3)[order].astype(np.int32)
    )


def test_chained_query_query_sort_composes(ms):
    """A chain of reference tables still addresses the source rows."""
    src = table(ms, ack=False)
    t = table(ms, ack=False)
    a = t.query("TIME < 8.0")
    b = a.query("ANTENNA1 != 2")
    c = b.sort("TIME")

    mask = (TIME < 8.0) & ((np.arange(NROW) % 3) != 2)
    order = _expected_order(TIME, mask)
    assert c.nrows() == len(order)
    for name in ("TIME", "ANTENNA1", "DATA", "FLAG", "UVW", "WEIGHT"):
        _assert_same_column(c.getcol(name), src.getcol(name)[order])


def test_every_read_api_maps_the_row_order(ms):
    src = table(ms, ack=False)
    s = table(ms, ack=False).query(SEL).sort("TIME")
    order = _expected_order(TIME)
    n = len(order)

    # getcol / getcolnp / sub-ranges.
    for name in s.colnames():
        want = src.getcol(name)[order]
        _assert_same_column(s.getcol(name), want)
        _assert_same_column(s.getcol(name, 1, 5), want[1:6])
        if want.ndim == 1 and want.dtype != object:
            buf = np.zeros_like(want)
            s.getcolnp(name, buf, 0, n)
            _assert_same_column(buf, want)

    # Whole-cell slices: DDFacet's chunk reader (Data/ClassMS.py:1022-1032).
    for name, shape in (("DATA", (n, NCHAN, NCORR)), ("FLAG", (n, NCHAN, NCORR))):
        want = src.getcol(name)[order]
        got = s.getcolslice(name, [0, 0], [NCHAN - 1, NCORR - 1], [1, 1], 0, n)
        _assert_same_column(got, want)
        buf = np.full(shape, np.nan if name == "DATA" else False, dtype=want.dtype)
        if name == "DATA":
            buf = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)
        s.getcolslicenp(name, buf, (0, 0), (NCHAN - 1, NCORR - 1), (1, 1), 0, n)
        # Bit-identical to getcol: the raw bulk-fill fast path is intact.
        np.testing.assert_array_equal(buf, want)
        # A partial cell slice too.
        part = s.getcolslice(name, [0, 0], [1, NCORR - 1], [1, 1], 2, 3)
        _assert_same_column(part, want[2:5, 0:2, :])

    # Single cells, rows, cell slices.
    for i in (0, 1, n - 1):
        for name in ("TIME", "DATA", "FLAG", "ANTENNA1"):
            np.testing.assert_array_equal(
                s.getcell(name, i), src.getcell(name, order[i])
            )
    _assert_same_column(
        s.getcellslice("DATA", 3, [0, 0], [NCHAN - 1, NCORR - 1]),
        src.getcellslice("DATA", order[3], [0, 0], [NCHAN - 1, NCORR - 1]),
    )
    row = s[2]
    assert row["TIME"] == TIME[order[2]]
    assert len(s) == n

    # varcol reads and metadata.
    varcol = s.getvarcol("DATA")
    assert len(varcol) == n
    assert s.getkeywords() == src.getkeywords()
    assert s.getdminfo().keys() == src.getdminfo().keys()
    assert set(s.getsubtables()) == set(src.getsubtables())


def test_empty_selection_is_an_empty_table(ms):
    q = table(ms, ack=False).query("FIELD_ID==99")
    assert q.nrows() == 0
    assert len(q) == 0
    assert q.getcol("TIME").shape == (0,)
    # An empty array column reads back as a flat empty array — the same as
    # python-casacore (probed) and the same as a materialised result here.
    assert q.getcol("DATA").shape == (0,)
    assert q.getcolslice(
        "DATA", [0, 0], [NCHAN - 1, NCORR - 1], [1, 1], 0, 0
    ).shape == (0,)


def test_unchanged_result_matches_the_materialising_taql_path(ms):
    """`taql()` still materialises — the two paths must agree row for row."""
    t = table(ms, ack=False)
    q = t.query(SEL).sort("TIME")
    m = taql(
        f"SELECT * FROM $1 WHERE {SEL} ORDERBY TIME",
        tables=[t],
    )
    assert q.nrows() == m.nrows()
    assert q.colnames() == m.colnames()
    for name in q.colnames():
        _assert_same_column(q.getcol(name), m.getcol(name))


def test_select_is_query(ms):
    t = table(ms, ack=False)
    q = t.select("TIME < 5.0")
    np.testing.assert_array_equal(q.getcol("TIME"), TIME[TIME < 5.0])
    assert q.nrows() == int((TIME < 5.0).sum())


# ---------------------------------------------------------------------------
# Reference tables are read-only snapshots (casacore parity).
# ---------------------------------------------------------------------------


def test_reference_tables_are_read_only(ms):
    """casacore's query/sort results report `iswritable() is False` and
    refuse writes when their source is read-only; a casacure reference table
    is always read-only (writes through it are not routed by the row order
    yet — see DIFFERENCES.md)."""
    t = table(ms, ack=False)
    s = t.sort("TIME")
    assert s.iswritable() is False
    with pytest.raises(ValueError, match="reference table"):
        s.putcol("TIME", np.zeros(s.nrows()))
    with pytest.raises(ValueError, match="reference table"):
        s.putcell("TIME", 0, 1.0)
    with pytest.raises(ValueError, match="reference table"):
        s.putkeyword("X", 1.0)
    with pytest.raises(ValueError, match="reference table"):
        s.addrows(1)
    with pytest.raises(ValueError, match="reference table"):
        s.addcols(maketabdesc(makescacoldesc("EXTRA", 0.0)))
    # The ordinary read-only message is unchanged for a normal handle
    # (casacure's `table()` defaults to `readonly=False`, like python-casacore
    # defaults the other way — pass `readonly=True` for a read handle).
    ro = table(ms, ack=False, readonly=True)
    assert ro.iswritable() is False
    with pytest.raises(ValueError, match="table is not writable"):
        ro.putcol("TIME", np.zeros(NROW))


def test_close_lock_and_flush_leave_the_row_order_intact(ms):
    s = table(ms, ack=False).query(SEL).sort("TIME")
    before = s.getcol("TIME").copy()
    s.lock()
    s.close()
    s.flush()
    assert s.haslock() is False
    assert s.ismultiused() in (True, False)
    # A close/lock must not silently replace the reference table with the
    # whole source (that is what auto-reopen would do to an unlocked view).
    assert s.nrows() == NROW
    np.testing.assert_array_equal(s.getcol("TIME"), before)


def test_taql_on_a_reference_table_still_works(ms):
    s = table(ms, ack=False).query(SEL).sort("TIME")
    r = taql("SELECT TIME FROM $1 WHERE TIME > 8.0", tables=[s])
    np.testing.assert_array_equal(r.getcol("TIME"), [8.5, 9.0])
    # The generic select_run path is the same row order for a plain
    # `SELECT *` — the row-mode fast path is the whole point of #16. A
    # projection that needs values still materialises into a temp table
    # (the `taql()` shape, see test_taql_scratch.py).
    m = s.select_run("SELECT * FROM $1 LIMIT 3")
    np.testing.assert_array_equal(m.getcol("TIME"), s.getcol("TIME")[:3])
    assert s.nrows() == NROW
    assert m.iswritable() is False
    computed = s.select_run("SELECT TIME * 2 AS double_time FROM $1 LIMIT 3")
    np.testing.assert_array_equal(
        computed.getcol("double_time"), s.getcol("TIME")[:3] * 2
    )
    assert "casacure-taql-" in os.path.basename(computed.name())
    assert computed.iswritable() is True  # a materialised scratch table
    computed.close()


def test_copy_copies_the_selection_not_the_source(ms, tmp_path):
    s = table(ms, ack=False).query(SEL).sort("TIME")
    out = str(tmp_path / "copy.tab")
    s.copy(out)
    c = table(out, ack=False)
    assert c.nrows() == s.nrows()
    for name in s.colnames():
        _assert_same_column(c.getcol(name), s.getcol(name))
    c.close()


def test_writes_through_a_query_result_are_refused_not_lost(ms, tmp_path):
    """Before #16 a write through a `query()` result went into a discarded
    temp copy of the selection (silent data loss — killMS's
    `GiveMainTable(readonly=False)` + `putcol` path); casacore routes such
    writes through the row order. casacure refuses them, loudly."""
    p = str(tmp_path / "w.tab")
    t = table(p, maketabdesc([makescacoldesc("a", 0.0), makescacoldesc("b", 0.0)]), nrow=5,
              ack=False)
    t.putcol("a", [3.0, 1.0, 2.0, 1.0, 5.0])
    t.putcol("b", np.zeros(5))
    t.close()

    w = table(p, readonly=False, ack=False)
    q = w.query("a > 1.5")
    with pytest.raises(ValueError, match="reference table"):
        q.putcol("b", np.array([9.0, 9.0, 9.0]))
    q.close()
    w.close()
    again = table(p, ack=False)
    np.testing.assert_array_equal(again.getcol("b"), np.zeros(5))


# ---------------------------------------------------------------------------
# Memory: no materialisation (issue #16's repro, scaled for CI).
# ---------------------------------------------------------------------------

_VMHWM_WORKER = r"""
import sys

import numpy as np
import casacore
from casacore.tables import table

want_shim = sys.argv[1] == "shim"
if bool(getattr(casacore, "__casacure_shim__", False)) != want_shim:
    raise SystemExit("backend mismatch: %%s" %% casacore.__file__)
path, nselects = sys.argv[2], int(sys.argv[3])
nchan, ncorr = %(nchan)d, %(ncorr)d

def hwm_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0

t = table(path, ack=False)
baseline = hwm_mib()
held = []
for i in range(nselects):
    # A distinct selection each time: no result is shared with another.
    sel = "TIME >= %%(i)d && ANTENNA1 >= %%(i)d" %% {"i": i}
    q = t.query(sel).sort("TIME")
    held.append(q)
# Read through them (the DDFacet chunk-reader shape), so a lazy result would
# have to materialise here if it were ever going to.
chunk = 7
for q in held:
    n = q.nrows()
    for r0 in range(0, n, chunk):
        nrow = min(chunk, n - r0)
        buf = np.zeros((nrow, nchan, ncorr), dtype=np.complex64)
        q.getcolslicenp("DATA", buf, (0, 0), (nchan - 1, ncorr - 1),
                        (1, 1), r0, nrow)
        q.getcol("TIME", r0, nrow)
print("%%.1f %%.1f %%.1f" %% (baseline, hwm_mib(), hwm_mib() - baseline))
""" % {"nchan": NCHAN, "ncorr": NCORR}

# The control for the memory test: the same selections run through `taql()`,
# which still materialises every selected cell as a `RecordValue` tree (the
# pre-#16 `query()`/`sort()` path, kept for generic TaQL). Its peak growth
# must dwarf the reference table's — otherwise the benchmark cannot tell the
# two paths apart.
_CONTROL_WORKER = r"""
import sys

import casacore
from casacore.tables import taql, table

if not bool(getattr(casacore, "__casacure_shim__", False)):
    raise SystemExit("backend mismatch: %s" % casacore.__file__)
path = sys.argv[1]

def hwm_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0

t = table(path, ack=False)
baseline = hwm_mib()
held = []
for i in range(4):
    sel = "TIME >= %(i)d && ANTENNA1 >= %(i)d" % {"i": i}
    held.append(taql("SELECT * FROM $1 WHERE %s ORDERBY TIME" % sel, tables=[t]))
print("%.1f" % (hwm_mib() - baseline))
"""


def _run_vmhwm(script, args):
    """Run `script` in a fresh subprocess of this interpreter and return the
    growth the *worker itself* reported (its last stdout field).

    The parent deliberately does not poll `/proc/<pid>/status` the way
    `test_memory_chunking._run` does: a child inherits the parent's VmHWM
    across `fork` (`dup_mm` seeds the high-water with the current RSS) and
    only resets it at `execve`, so an early poll can record the pytest
    parent's resident set — measured at 371 MiB for a trivial worker under a
    300 MiB-ballast parent.  Each worker takes its own `baseline` and `hwm`
    from `/proc/self/status`, so the growth is exact without any polling.
    """
    out = tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False)
    name = out.name
    out.close()
    pid = os.fork()
    if pid == 0:
        with open(name, "w") as f:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.execve(sys.executable, [sys.executable, "-c", script, *args], dict(os.environ))
    _, state = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(state)
    with open(name) as f:
        text = f.read()
    os.unlink(name)
    assert code == 0, text
    return float(text.split()[-1])


@pytest.fixture(scope="module")
def big_ms(tmp_path_factory):
    """A table whose *selection* is large enough that materialising it is
    unmistakable in the peak RSS (50k rows x 4x4 complex64 = 50 MiB of
    cells; the `RecordValue` trees the old path built were several times
    that per selection)."""
    p = str(tmp_path_factory.mktemp("bigms") / "big.ms")
    n = 50_000
    rng = np.random.default_rng(3)
    t = table(
        p,
        maketabdesc(
            [
                makescacoldesc("TIME", 0.0),
                makescacoldesc("ANTENNA1", 0),
                makearrcoldesc(
                    "DATA", 0j, 0, [NCHAN, NCORR],
                    "TiledColumnStMan", "TiledData", 0, valuetype="complex",
                ),
            ]
        ),
        nrow=n,
        ack=False,
    )
    t.putcol("TIME", np.arange(n, dtype=np.float64) % 1000.0)
    t.putcol("ANTENNA1", (np.arange(n) % 7).astype(np.int32))
    t.putcol(
        "DATA",
        (rng.normal(size=(n, NCHAN, NCORR)) + 1j).astype(np.complex64),
    )
    t.close()
    return p


def test_distinct_query_and_sort_handles_do_not_materialise(big_ms):
    """Issue #16's repro, scaled: N distinct `query().sort()` handles, read
    through, must stay within a bounded growth of the process's peak — the
    materialising form added the selection's heap (hundreds of MiB here,
    ~2 GiB on ssd0000.MS) for each one.

    Peak VmHWM is measured in a fresh subprocess: it survives nothing else
    and cannot inherit the pytest parent's high-water mark."""
    env = dict(os.environ)
    shim = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "shim"
    )
    env["PYTHONPATH"] = shim + os.pathsep + env.get("PYTHONPATH", "")
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    try:
        growth = _run_vmhwm(_VMHWM_WORKER, ["shim", big_ms, "4"])
        # The control: the same selections through the materialising path
        # (taql() still copies), so this test fails if the *measurement*
        # stops being able to see materialisation — not only if casacure
        # regresses.
        materialised = _run_vmhwm(_CONTROL_WORKER, [big_ms])
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old
    assert growth < 100.0, (
        f"4 distinct query().sort() handles grew the peak by {growth:.0f} MiB; "
        "a reference table copies no cells (the materialising path added the "
        "whole selection per handle)"
    )
    assert 4 * growth < materialised, (
        f"the materialising control reached {materialised:.0f} MiB over a "
        f"{growth:.0f} MiB reference-table growth — the benchmark can no "
        "longer tell the two paths apart, so its bound proves nothing"
    )
