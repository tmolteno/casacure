"""StandardStMan Direct array columns (column option 1 + FixedShape).

casacore's ``SSMDirColumn`` stores the cells of a Direct fixed-shape array
column inline in the bucket, like a scalar of ``nelem`` elements, rather than
as a reference into the array file (``table.f<n>i``).  The MS schema uses the
layout for ANTENNA POSITION/OFFSET, FEED POSITION and (in tables made by
python-casacore's ``default_ms``) the main table's UVW.  casacure <= 3.8.8
read those inline cells as references, so it could not read the POSITION of
a real MeerKAT MS's ANTENNA table.  It also wrote references for Direct
columns, which casacore misreads.

These tests exchange tables both ways with real python-casacore:
- casacore writes and casacure reads;
- casacure writes, in one go and by growing chunk by chunk, and casacore
  reads, extends and rewrites.

They also check that a table written by casacure 3.8.8 in the old layout
(``tests/data/legacy_direct_388.tab``) still reads correctly and is converted
on its first write.
"""
import os
import shutil

import pytest

from test_memory_chunking import (  # noqa: E402
    CASACORE_AVAILABLE,
    CASACURE_AVAILABLE,
    _casacore_env,
    _casacure_env,
    _run,
)

pytestmark = pytest.mark.skipif(not CASACURE_AVAILABLE, reason="casacure not built")
needs_casacore = pytest.mark.skipif(not CASACORE_AVAILABLE, reason="real python-casacore not installed")

LEGACY = os.path.join(os.path.dirname(__file__), "data", "legacy_direct_388.tab")

# The columns and their values, shared by every script: a function of the
# row.  POS/PAIR/CPLX/FLAGS/INTS are Direct; IND is an ordinary (indirect)
# array in the same StandardStMan, so the array file exists too, and PAIR has
# 8-byte cells (the size of an array-file reference).
_COMMON = r"""
import sys
import numpy as np
from casacore.tables import table, maketabdesc, makearrcoldesc, makescacoldesc

def desc():
    return maketabdesc([
        makearrcoldesc("POS", 0.0, 1, [3], options=5),
        makearrcoldesc("PAIR", 0.0, 1, [2], valuetype="float", options=5),
        makearrcoldesc("CPLX", 0j, 2, [2, 3], valuetype="complex", options=5),
        makearrcoldesc("FLAGS", False, 1, [5], options=5),
        makearrcoldesc("INTS", 0, 1, [4], options=5),
        makearrcoldesc("IND", 0.0, 1, [2]),
        makescacoldesc("ID", 0),
    ])

def want(name, r, gen=0):
    r = np.asarray(r)
    if name == "POS":
        return np.stack([r * 1.5 + gen, r * 2.5 + 1, -r * 1.0], axis=-1)
    if name == "PAIR":
        return np.stack([r + 0.25, r + 0.5 + gen], axis=-1).astype(np.float32)
    if name == "CPLX":
        k = np.arange(6).reshape(2, 3)
        return (r[..., None, None] + 1j * (k + gen)).astype(np.complex64)
    if name == "FLAGS":
        return (r[..., None] + np.arange(5) + gen) % 3 == 0
    if name == "INTS":
        return (r[..., None] * 4 + np.arange(4) + gen).astype(np.int32)
    if name == "IND":
        return np.stack([r * 10.0, r * 10.0 + 1 + gen], axis=-1)
    return (r + gen).astype(np.int32)

NAMES = ["POS", "PAIR", "CPLX", "FLAGS", "INTS", "IND", "ID"]

def check(t, nrow, gens=None):
    assert t.nrows() == nrow, (t.nrows(), nrow)
    rows = np.arange(nrow)
    for name in NAMES:
        g = np.zeros(nrow, int) if gens is None else gens
        exp = np.stack([want(name, r, gi) for r, gi in zip(rows, g)]) if nrow else None
        got = np.asarray(t.getcol(name))
        assert got.shape == exp.shape, (name, got.shape, exp.shape)
        assert np.array_equal(got, exp), (name, np.flatnonzero(
            (got != exp).reshape(nrow, -1).any(axis=1))[:5])
        cell = np.asarray(t.getcell(name, nrow - 1))
        assert np.array_equal(cell, exp[-1]), (name, "getcell")
"""

_CASACORE_WRITES = _COMMON + r"""
path, nrow = sys.argv[1], int(sys.argv[2])
t = table(path, desc(), nrow=nrow, ack=False)
rows = np.arange(nrow)
for name in NAMES:
    t.putcol(name, want(name, rows))
t.close()
print("WRITTEN")
"""

_CASACURE_GROWS = _COMMON + r"""
import casacore
assert "casacure" in casacore.__file__ or "shim" in casacore.__file__, casacore.__file__
path, nrow, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = table(path, desc(), nrow=0, ack=False)
for start in range(0, nrow, chunk):
    n = min(chunk, nrow - start)
    t.addrows(n)
    rows = np.arange(start, start + n)
    for name in NAMES:
        t.putcol(name, want(name, rows), start, n)
        t.flush()
t.close()
print("WRITTEN")
"""

_CHECK = _COMMON + r"""
path, nrow = sys.argv[1], int(sys.argv[2])
check(table(path, ack=False), nrow)
print("CHECKED")
"""

# casacore extends the table and rewrites some cells: rows 3 and 40 become
# generation 1, and `extra` rows are appended.
_CASACORE_EDITS = _COMMON + r"""
path, nrow, extra = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = table(path, readonly=False, ack=False)
t.addrows(extra)
rows = np.arange(nrow, nrow + extra)
for name in NAMES:
    t.putcol(name, want(name, rows), nrow, extra)
    for r in (3, 40):
        t.putcell(name, r, want(name, r, 1))
t.close()
print("EDITED")
"""

_CHECK_EDITED = _COMMON + r"""
path, nrow = sys.argv[1], int(sys.argv[2])
gens = np.zeros(nrow, int)
gens[[3, 40]] = 1
check(table(path, ack=False), nrow, gens)
print("CHECKED")
"""


def _ok(script, args, env, token):
    rc, out, _ = _run(script, [str(a) for a in args], env)
    assert rc == 0 and token in out, out


@needs_casacore
def test_casacure_reads_casacore_direct_arrays(tmp_path):
    path = str(tmp_path / "core.tab")
    _ok(_CASACORE_WRITES, [path, 100], _casacore_env(), "WRITTEN")
    _ok(_CHECK, [path, 100], _casacure_env(), "CHECKED")


@needs_casacore
@pytest.mark.parametrize("nrow,chunk", [(100, 100), (1000, 37)])
def test_casacore_reads_what_casacure_writes(tmp_path, nrow, chunk):
    """Written in one go, and grown chunk by chunk (dask-ms's pattern, which
    takes the in-place growth path); casacore then extends and rewrites it,
    and both read the result."""
    path = str(tmp_path / "cure.tab")
    _ok(_CASACURE_GROWS, [path, nrow, chunk], _casacure_env(), "WRITTEN")
    for env in (_casacore_env(), _casacure_env()):
        _ok(_CHECK, [path, nrow], env, "CHECKED")
    _ok(_CASACORE_EDITS, [path, nrow, 25], _casacore_env(), "EDITED")
    for env in (_casacore_env(), _casacure_env()):
        _ok(_CHECK_EDITED, [path, nrow + 25], env, "CHECKED")


# The legacy fixture: 70 rows of POS (Direct double [3]), PAIR (Direct float
# [2], 8-byte cells), IND (indirect double [2]) and ID, written by casacure
# 3.8.8 with array-file references for the Direct columns.
_LEGACY_COMMON = r"""
import sys
import numpy as np
from casacore.tables import table
r = np.arange(70)
WANT = {
    "POS": np.stack([r * 1.5, r * 2.5 + 1, -r * 1.0], axis=1),
    "PAIR": np.stack([r + 0.25, r + 0.5], axis=1).astype(np.float32),
    "IND": np.stack([r * 10.0, r * 10.0 + 1], axis=1),
    "ID": r.astype(np.int32),
}
def check(t, edited):
    want = {k: v.copy() for k, v in WANT.items()}
    if edited:
        want["POS"][5] = [-7.0, -8.0, -9.0]
    for name, exp in want.items():
        got = np.asarray(t.getcol(name))
        assert np.array_equal(got, exp), (name, got[:3], exp[:3])
"""

_LEGACY_READ = _LEGACY_COMMON + r"""
check(table(sys.argv[1], ack=False), edited=sys.argv[2] == "1")
print("CHECKED")
"""

_LEGACY_WRITE = _LEGACY_COMMON + r"""
t = table(sys.argv[1], readonly=False, ack=False)
t.putcell("POS", 5, np.array([-7.0, -8.0, -9.0]))
t.close()
print("WRITTEN")
"""


def test_a_casacure_388_table_still_reads(tmp_path):
    path = str(tmp_path / "legacy.tab")
    shutil.copytree(LEGACY, path)
    _ok(_LEGACY_READ, [path, 0], _casacure_env(), "CHECKED")


@needs_casacore
def test_writing_a_casacure_388_table_converts_it(tmp_path):
    """The first write rewrites the table in casacore's layout, so real
    casacore then reads it (it misread the old references as values)."""
    path = str(tmp_path / "legacy.tab")
    shutil.copytree(LEGACY, path)
    _ok(_LEGACY_WRITE, [path], _casacure_env(), "WRITTEN")
    for env in (_casacure_env(), _casacore_env()):
        _ok(_LEGACY_READ, [path, 1], env, "CHECKED")


_DEFAULT_MS = r"""
import sys
import numpy as np
from casacore.tables import default_ms, table
path = sys.argv[1]
default_ms(path)
t = table(path + "/ANTENNA", readonly=False, ack=False)
t.addrows(4)
t.putcol("POSITION", np.arange(12.0).reshape(4, 3) + 5.0e6)
t.putcol("OFFSET", np.arange(12.0).reshape(4, 3) * 0.5)
t.putcol("NAME", ["m000", "m001", "m002", "m003"])
t.close()
print("WRITTEN")
"""

_ANTENNA = r"""
import sys
import numpy as np
from casacore.tables import table
t = table(sys.argv[1] + "/ANTENNA", ack=False)
assert np.array_equal(t.getcol("POSITION"), np.arange(12.0).reshape(4, 3) + 5.0e6)
assert np.array_equal(t.getcol("OFFSET"), np.arange(12.0).reshape(4, 3) * 0.5)
assert list(t.getcol("NAME")) == ["m000", "m001", "m002", "m003"]
print("CHECKED")
"""


@needs_casacore
def test_an_ms_antenna_table_from_casacore(tmp_path):
    """ANTENNA POSITION/OFFSET are Direct in the MS schema: every
    casacore-written MS has them."""
    path = str(tmp_path / "x.ms")
    _ok(_DEFAULT_MS, [path], _casacore_env(), "WRITTEN")
    _ok(_ANTENNA, [path], _casacure_env(), "CHECKED")
