"""casacure#41: casacore and CASA must be able to modify a casacure-written table.

casacure wrote every keyword record (the table, column and private keyword
sets) as a *Fixed* record (record type 0), and every nested record with its
description in its parent's.  casacore makes all of these Variable.  Real
python-casacore then refused to add a keyword, a column keyword or a tiled
column (``Record cannot be changed (fixed structure)``: a tiled column
registers a hypercolumn in the private keyword set), and adding a field to a
nested record such as ``MEASINFO`` left a table it could not reopen
(``AipsIO::getend: part of object not read``).  CASA's ``clearcal`` adds
``MODEL_DATA`` by cloning DATA's storage manager, so it failed on every
skarabina 1.0.22+ averaged MS.

casacure also wrote an empty TiledColumnStMan tile shape in the header,
which casacore reports as ``SPEC.DEFAULTTILESHAPE``.

python-casacore runs in a subprocess without ``tests/shim`` on the path (the
shim stands in for it); the tests are skipped when it is not installed.
"""

import json
import os
import subprocess
import sys

import pytest

import casacure.tables as ct


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


def _casacure(code, *args):
    """Run `code` under casacure in a fresh process; return its stdout."""
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _write_casacure_table(path):
    td = ct.maketabdesc(
        [
            ct.makescacoldesc(
                "TIME",
                0.0,
                keywords={"QuantumUnits": ["s"], "MEASINFO": {"type": "epoch", "Ref": "UTC"}},
            ),
            ct.makearrcoldesc("DATA", 0j, shape=[4, 2], keywords={"UNIT": "Jy"}),
        ]
    )
    dminfo = {
        "*1": {
            "TYPE": "TiledColumnStMan",
            "NAME": "DATA_GROUP",
            "SPEC": {"DEFAULTTILESHAPE": [2, 4, 64]},
            "COLUMNS": ["DATA"],
        }
    }
    t = ct.table(path, td, nrow=10, ack=False, readonly=False, dminfo=dminfo)
    t.putkeyword("TELESCOPE", "MeerKAT")
    t.putkeyword("SUB", {"a": 1, "b": "x"})
    t.close()


# Each operation runs on its own copy of the table, then the copy is reopened:
# a modification that writes an unreadable table fails at the reopen.
MODIFY = r"""
import json, shutil, sys
import casacore.tables as pt
src = sys.argv[1]
ops = {
    "putkeyword": lambda t: t.putkeyword("NEWKW", 1),
    "putcolkeyword": lambda t: t.putcolkeyword("DATA", "NEWKW", 1),
    "nested table keyword: add field": lambda t: t.putkeyword("SUB.c", 2.0),
    "MEASINFO: add field": lambda t: t.putcolkeyword("TIME", "MEASINFO.TabRefTypes", ["UTC"]),
    "MEASINFO: change field": lambda t: t.putcolkeyword("TIME", "MEASINFO.Ref", "TAI"),
    "addcols TiledShapeStMan": lambda t: t.addcols(
        pt.maketabdesc(pt.makearrcoldesc("MODEL_DATA", 0j, shape=[4, 2])),
        {"TYPE": "TiledShapeStMan", "NAME": "TiledModel", "SPEC": {"DEFAULTTILESHAPE": [2, 4, 64]}}),
    "addcols clone of DATA's storage manager": lambda t: t.addcols(
        pt.maketabdesc(pt.makearrcoldesc("CORRECTED_DATA", 0j, shape=[4, 2])),
        dict([v for v in t.getdminfo().values() if "DATA" in v["COLUMNS"]][0],
             NAME="TiledCorrected", COLUMNS=["CORRECTED_DATA"])),
}
out = {}
t = pt.table(src, ack=False)
out["DEFAULTTILESHAPE"] = [list(map(int, v["SPEC"]["DEFAULTTILESHAPE"]))
                           for v in t.getdminfo().values() if v["TYPE"] == "TiledColumnStMan"]
t.close()
for name, fn in ops.items():
    w = src + "." + str(len(out))
    shutil.copytree(src, w)
    try:
        t = pt.table(w, readonly=False, ack=False)
        fn(t)
        t.close()
        t = pt.table(w, ack=False)
        t.getkeywords(); t.getcolkeywords("TIME"); t.getcolkeywords("DATA")
        t.close()
        out[name] = "ok"
    except Exception as e:
        out[name] = str(e)
print(json.dumps(out))
"""


@needs_casacore
def test_python_casacore_can_modify_casacure_table(tmp_path):
    path = str(tmp_path / "cc.tab")
    _write_casacure_table(path)
    out = json.loads(_real(MODIFY, path).strip().splitlines()[-1])
    assert out.pop("DEFAULTTILESHAPE") == [[2, 4, 64]]
    assert out == {name: "ok" for name in out}, out


@needs_casacore
def test_casacure_reads_back_casacore_modifications(tmp_path):
    """The records casacore writes back (framed Variable sub-records with
    added fields) read correctly through casacure.

    Written and read back in fresh casacure processes, as a pipeline does:
    a reopen in the process that wrote the table keeps that process's
    keyword sets, which is the write backing's caching (#30), not the
    on-disk format under test here."""
    path = str(tmp_path / "cc.tab")
    _casacure(
        r"""
import sys
sys.path.insert(0, sys.argv[2])
from test_record_type import _write_casacure_table
_write_casacure_table(sys.argv[1])
""",
        path,
        os.path.dirname(os.path.abspath(__file__)),
    )
    _real(
        r"""
import sys
import casacore.tables as pt
t = pt.table(sys.argv[1], readonly=False, ack=False)
t.putkeyword("NEWKW", 1)
t.putkeyword("SUB.c", 2.0)
t.putcolkeyword("TIME", "MEASINFO.TabRefTypes", ["UTC"])
t.close()
""",
        path,
    )
    out = _casacure(
        r"""
import json, sys
import casacure.tables as ct
t = ct.table(sys.argv[1], ack=False)
mi = t.getcolkeywords("TIME")["MEASINFO"]
print(json.dumps({"kw": t.getkeywords(), "mi": {k: (list(v) if isinstance(v, list) else v) for k, v in mi.items()}}))
""",
        path,
    )
    got = json.loads(out.strip().splitlines()[-1])
    assert got["kw"] == {"TELESCOPE": "MeerKAT", "SUB": {"a": 1, "b": "x", "c": 2.0}, "NEWKW": 1}
    assert got["mi"] == {"type": "epoch", "Ref": "UTC", "TabRefTypes": ["UTC"]}


@needs_casacore
def test_casacure_rewrite_keeps_casacore_table_extendable(tmp_path):
    """A casacore-written table that casacure adds keywords to (and so
    rewrites its keyword sets) stays extendable by casacore."""
    path = str(tmp_path / "pc.tab")
    _real(
        r"""
import sys
import casacore.tables as pt
td = pt.maketabdesc([pt.makescacoldesc("TIME", 0.0,
                     keywords={"MEASINFO": {"type": "epoch", "Ref": "UTC"}})])
t = pt.table(sys.argv[1], td, nrow=3, ack=False)
t.putkeyword("SUB", {"a": 1})
t.close()
""",
        path,
    )
    t = ct.table(path, readonly=False, ack=False)
    t.putkeyword("FROM_CASACURE", {"x": 1.5})
    t.putcolkeyword("TIME", "UNIT", "s")
    t.close()
    out = _real(
        r"""
import sys
import casacore.tables as pt
t = pt.table(sys.argv[1], readonly=False, ack=False)
t.putkeyword("SUB.b", 2)
t.putkeyword("FROM_CASACURE.y", 2)
t.putcolkeyword("TIME", "MEASINFO.TabRefTypes", ["UTC"])
t.close()
t = pt.table(sys.argv[1], ack=False)
print(sorted(t.getkeyword("SUB")), sorted(t.getkeyword("FROM_CASACURE")),
      sorted(t.getcolkeywords("TIME")["MEASINFO"]))
""",
        path,
    )
    assert out.strip().splitlines()[-1] == (
        "['a', 'b'] ['x', 'y'] ['Ref', 'TabRefTypes', 'type']"
    )
