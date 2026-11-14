"""Never-written cells: shapes and dtypes (issue #45).

Three read/write disagreements with the column description, all pinned
against python-casacore 3.8.1's behavior:

1. a never-written variable-shape array column keeps its declared rank:
   `getcol` returns an empty `(nrow, 0, ..., 0)` array (the cell default
   with zero extents per dimension), not a `(nrow,)` scalar-shaped result
   that breaks `[:, :, :, k]` indexing;
2. a 0-row table reads every column as an empty array of the column's
   dtype (not `int32` for everything), and a string column as an empty
   list;
3. `addcols` of a variable-shape array column leaves its cells UNDEFINED
   (a null "no array" reference, `iscelldefined` False) rather than
   storing defined, empty 1-D cells that contradict the column's `ndim` —
   casacore refuses to iterate the latter (`ArrayIterator<T>: at the
   moment cannot iterate by scalars`).

Real python-casacore runs in subprocesses without ``tests/shim`` on the
path (the shim stands in for it); those tests skip when it is not
installed.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import casacure.tables as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return env


def _real_casacore_available():
    r = subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_real_casacore_env(),
        capture_output=True,
    )
    return r.returncode == 0


needs_casacore = pytest.mark.skipif(
    not _real_casacore_available(), reason="real python-casacore not installed"
)


def _real(code, *args):
    """Run `code` under real python-casacore; return its stdout."""
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _desc():
    return ct.maketabdesc(
        [
            ct.makescacoldesc("D", 0.0),
            ct.makescacoldesc("B", False),
            ct.makearrcoldesc("A", 0.0, ndim=1),
            ct.makearrcoldesc("FLAG_CATEGORY", False, ndim=3),
        ]
    )


def test_never_written_variable_shape_keeps_declared_rank(tmp_path):
    t = ct.table(str(tmp_path / "u.tab"), _desc(), nrow=3, ack=False)
    t.putcol("D", np.arange(3.0))
    t.close()

    t = ct.table(str(tmp_path / "u.tab"), ack=False)
    v = t.getcol("FLAG_CATEGORY")
    assert v.shape == (3, 0, 0, 0)
    assert v.dtype == np.bool_
    assert t.getcell("FLAG_CATEGORY", 0).shape == (0, 0, 0)
    # Undefined on disk, and still reported so.
    assert not t.iscelldefined("FLAG_CATEGORY", 0)
    t.close()


def test_zero_row_table_reads_column_dtypes(tmp_path):
    ct.table(str(tmp_path / "z.tab"), _desc(), nrow=0, ack=False).close()
    t = ct.table(str(tmp_path / "z.tab"), ack=False)
    dtypes = {c: np.asarray(t.getcol(c)).dtype.str for c in ("D", "B", "A", "FLAG_CATEGORY")}
    assert dtypes == {"D": "<f8", "B": "|b1", "A": "<f8", "FLAG_CATEGORY": "|b1"}
    t.close()


def test_zero_row_string_column_reads_empty_list(tmp_path):
    d = ct.maketabdesc([ct.makescacoldesc("NAME", "")])
    ct.table(str(tmp_path / "s.tab"), d, nrow=0, ack=False).close()
    t = ct.table(str(tmp_path / "s.tab"), ack=False)
    assert t.getcol("NAME") == []
    t.close()


def _add_variable_column(path):
    t = ct.table(str(path), ct.maketabdesc([ct.makescacoldesc("TIME", 0.0)]), nrow=3, ack=False)
    t.putcol("TIME", np.arange(3.0))
    t.addcols(ct.makearrcoldesc("CORRECTED_DATA", 0j, ndim=2, valuetype="complex"))
    t.close()


def test_addcols_variable_shape_cells_stay_undefined(tmp_path):
    path = tmp_path / "ac.tab"
    _add_variable_column(path)
    t = ct.table(str(path), ack=False)
    assert not t.iscelldefined("CORRECTED_DATA", 0)
    assert t.getcol("CORRECTED_DATA").shape == (3, 0, 0)
    t.close()


@needs_casacore
def test_casacore_reads_addcols_variable_column_as_undefined(tmp_path):
    # The on-disk contract: casacore must see no array in the row (the
    # pre-#45 files held defined empty 1-D cells and casacore's
    # ArrayIterator refused to iterate them).
    path = tmp_path / "ac.tab"
    _add_variable_column(path)
    out = _real(
        "import casacore.tables as pt, sys\n"
        "t = pt.table(sys.argv[1], ack=False)\n"
        "print(t.iscelldefined('CORRECTED_DATA', 0))\n"
        "try:\n"
        "    t.getcol('CORRECTED_DATA')\n"
        "    print('getcol ok')\n"
        "except RuntimeError as e:\n"
        "    print('getcol raises:', e)\n",
        str(path),
    )
    assert "False" in out  # undefined, like a casacore-added column
