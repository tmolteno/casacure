"""`iscelldefined` (issue #32): whether a cell holds a value.

python-casacore's `table.iscelldefined(columnname, rownr)` is casacore's
`TableColumn::isDefined`:

* a scalar column is defined in every row (the `Undefined` option is not
  modelled — casacure reads undefined scalars as their default, issue #8);
* a fixed-shape array column is defined in every row too (casacore's
  `ColumnDesc::isFixedShape` implies a value per row — an unwritten
  fixed-shape string cell reads as the shape filled with "");
* a **variable-shape** array cell is defined only when the storage
  manager holds an array for the row (`ArrayColumnData::isDefined` ->
  the storage `isShapeDefined`): SSM probes the cell reference (a null
  offset / zero-length string reference is "no array"), TSM the tile
  presence.

This is the only way a caller can distinguish an undefined cell from one
holding the default value (#8's read-as-defaults convention keeps the
read paths uniform).

Real python-casacore runs in subprocesses without ``tests/shim`` on the
path (the shim stands in for it); tests needing it are skipped when it
is not installed.
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
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_variable_shape_written_vs_unwritten(tmp_path):
    """A written variable-shape cell is defined; a fresh or unwritten one
    is not — through the live handle and on reopen."""
    path = str(tmp_path / "var.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makearrcoldesc("V", 0.0, ndim=1)]),
        nrow=3,
        ack=False,
    )
    t.putcell("V", 0, np.array([1.0, 2.0]))
    assert t.iscelldefined("V", 0)
    assert not t.iscelldefined("V", 1)
    assert not t.iscelldefined("V", 2)
    t.close()
    r = ct.table(path, ack=False)
    assert r.iscelldefined("V", 0)
    assert not r.iscelldefined("V", 1)
    assert not r.iscelldefined("V", 2)
    r.close()


def test_variable_shape_string_column(tmp_path):
    """The string variant probes the 12-byte (bucket, offset, len)
    reference: len 0 is "no array"."""
    path = str(tmp_path / "str.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makearrcoldesc("S", "", ndim=1,
                                          valuetype="string")]),
        nrow=2,
        ack=False,
    )
    t.putcell("S", 0, ["a", "bb"])
    assert t.iscelldefined("S", 0)
    assert not t.iscelldefined("S", 1)
    t.close()
    r = ct.table(path, ack=False)
    assert r.iscelldefined("S", 0)
    assert not r.iscelldefined("S", 1)
    r.close()


def test_fixed_shape_and_scalar_always_defined(tmp_path):
    """Fixed-shape arrays (including a never-written fixed-shape string
    cell, which reads as ""-filled) and scalars are defined in every
    row."""
    path = str(tmp_path / "fixed.tab")
    t = ct.table(
        path,
        ct.maketabdesc(
            [
                ct.makearrcoldesc("F", 0.0, shape=[2]),
                ct.makearrcoldesc("S", "", shape=[2], options=5,
                                  valuetype="string"),
                ct.makescacoldesc("D", 0.0, valuetype="float"),
            ]
        ),
        nrow=2,
        ack=False,
    )
    t.putcol("F", np.ones((2, 2), dtype=np.float32))
    t.putcell("S", 0, ["a", "bb"])  # row 1 never written
    t.putcol("D", np.ones(2, dtype=np.float32))
    for row in (0, 1):
        for col in ("F", "S", "D"):
            assert t.iscelldefined(col, row), (col, row)
    t.close()


def test_pending_write_and_added_rows(tmp_path):
    """Through a writable handle: a pending buffered write is defined; a
    row added but never written reads as the schema default — defined for
    a scalar, "no array" for a variable-shape column."""
    path = str(tmp_path / "w.tab")
    t = ct.table(
        path,
        ct.maketabdesc(
            [
                ct.makescacoldesc("D", 0.0, valuetype="float"),
                ct.makearrcoldesc("V", 0.0, ndim=1),
            ]
        ),
        nrow=1,
        ack=False,
    )
    t.putcol("D", np.ones(1, dtype=np.float32))
    t.putcell("V", 0, np.array([1.0]))
    t.flush()
    t.addrows(1)
    t.putcell("D", 1, 42.0)  # pending
    assert t.iscelldefined("D", 1)
    assert not t.iscelldefined("V", 1)  # added, never written
    t.flush()
    t.close()


def test_out_of_range_row(tmp_path):
    """The read-path contract: ValueError, not a storage error."""
    path = str(tmp_path / "b.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("D", 0.0, valuetype="float")]),
        nrow=1,
        ack=False,
    )
    with pytest.raises(ValueError, match="out of range"):
        t.iscelldefined("D", 5)
    t.close()


def test_tiled_shape_stman(tmp_path):
    """A TiledShapeStMan column probes tile presence: written rows are
    defined."""
    path = str(tmp_path / "t.tab")
    t = ct.table(
        path,
        ct.maketabdesc(
            [
                ct.makearrcoldesc("DATA", 0.0 + 0.0j, shape=[4, 2],
                                  valuetype="dcomplex",
                                  datamanagertype="TiledShapeStMan"),
            ]
        ),
        nrow=3,
        ack=False,
    )
    t.putcol(
        "DATA",
        np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2) * (1 + 1j),
    )
    for row in range(3):
        assert t.iscelldefined("DATA", row)
    t.close()


@needs_casacore
def test_matches_python_casacore(tmp_path):
    """The answers match real python-casacore 3.8.1 on the same table —
    written/unwritten variable-shape rows, fixed-shape and scalar rows."""
    path = str(tmp_path / "p.tab")
    t = ct.table(
        path,
        ct.maketabdesc(
            [
                ct.makescacoldesc("D", 0.0, valuetype="float"),
                ct.makearrcoldesc("V", 0.0, ndim=1),
                ct.makearrcoldesc("F", 0.0, shape=[2]),
            ]
        ),
        nrow=3,
        ack=False,
    )
    t.putcell("V", 0, np.array([1.0, 2.0]))
    t.putcol("F", np.ones((3, 2), dtype=np.float32))
    t.putcol("D", np.ones(3, dtype=np.float32))
    t.close()
    out = _real(
        "import sys; import casacore.tables as pt; "
        "t = pt.table(sys.argv[1], ack=False); "
        "print([t.iscelldefined(c, r) for c in ('D', 'V', 'F') "
        "for r in range(3)])",
        path,
    )
    want = [
        True, True, True,          # scalar D
        True, False, False,        # variable V: row 0 written
        True, True, True,          # fixed F
    ]
    assert out.strip() == str(want), out
    # And casacure agrees, cell for cell.
    r = ct.table(path, ack=False)
    got = [r.iscelldefined(c, row) for c in ("D", "V", "F")
           for row in range(3)]
    assert got == want
    r.close()
