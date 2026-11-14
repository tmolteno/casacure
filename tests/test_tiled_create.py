"""Tiled storage managers on create (issue #42).

python-casacore 3.8.1's contract, probed from it:

1. a variable-shape column is refused for TiledColumnStMan at creation
   (``TiledColumnStMan needs array column FLAG to be FixedShape``), writing
   nothing — a table casacure accepted here was unreadable by casacore;
2. creating with rows (``nrow > 0``) works for a valid TiledShapeStMan /
   fixed-shape TiledColumnStMan request, and a refusal leaves no table
   directory behind;
3. ``getcoldesc(col)`` (``actual=True``, casacore's default) reports the
   manager the column is bound to — for a table *casacore* wrote the stored
   column description says ``StandardStMan`` while the dminfo binds
   IncrementalStMan/TiledShapeStMan;
4. a requested ``SPEC.DEFAULTTILESHAPE`` survives the ``nrow=0`` creation
   deferral: it is applied once the cell shape is known (was stored as
   ``[2, 4, 8192]`` — 512 KiB tiles for 512 bytes of data).

The ``needs_casacore`` tests rerun the same probes against real
python-casacore for parity.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from casacore.tables import table, maketabdesc, makearrcoldesc, makescacoldesc


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return env


def _real(code, *args):
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _tcsm_dminfo():
    return {
        "*1": {
            "TYPE": "TiledColumnStMan",
            "NAME": "TiledFlag",
            "SPEC": {"DEFAULTTILESHAPE": np.array([4, 2, 8], dtype=np.int32)},
            "COLUMNS": ["FLAG"],
        }
    }


def test_tcsm_variable_shape_refused_without_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = maketabdesc([makearrcoldesc("FLAG", False, ndim=2)])
    for nrow in (0, 4):
        name = f"tcsm{nrow}.tab"
        with pytest.raises((RuntimeError, ValueError), match="needs array column FLAG to be FixedShape"):
            table(name, d, nrow=nrow, dminfo=_tcsm_dminfo(), ack=False)
        assert not os.path.exists(name), f"{name} left on disk"


def test_tsm_nrow_positive_creates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = maketabdesc([makearrcoldesc("DATA", 0.0, ndim=2)])
    dm = {
        "*1": {
            "TYPE": "TiledShapeStMan",
            "NAME": "TiledDATA",
            "SPEC": {"DEFAULTTILESHAPE": np.array([2, 4, 8], dtype=np.int32)},
            "COLUMNS": ["DATA"],
        }
    }
    t = table("tsm8.tab", d, nrow=8, dminfo=dm, ack=False)
    t.putcol("DATA", np.ones((8, 4, 2)))
    t.close()
    t = table("tsm8.tab", ack=False)
    assert t.nrows() == 8
    assert np.array_equal(t.getcol("DATA"), np.ones((8, 4, 2)))
    t.close()


def test_tcsm_fixed_shape_nrow_positive_creates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = maketabdesc(
        [makearrcoldesc("FLAG", False, ndim=2, shape=np.array([4, 2], dtype=np.int32))]
    )
    t = table("fixed.tab", d, nrow=4, dminfo=_tcsm_dminfo(), ack=False)
    t.putcol("FLAG", np.zeros((4, 4, 2), bool))
    t.close()
    t = table("fixed.tab", ack=False)
    assert t.iscelldefined("FLAG", 0)
    assert t.getcell("FLAG", 0).shape == (4, 2)
    t.close()


def _needs_casacore():
    return pytest.mark.skipif(
        subprocess.run(
            [sys.executable, "-c", "import casacore.tables"],
            env=_real_casacore_env(),
            capture_output=True,
        ).returncode
        != 0,
        reason="real python-casacore not installed",
    )


needs_casacore = _needs_casacore()


@needs_casacore
def test_casacore_reads_casacure_tiled_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = maketabdesc(
        [makearrcoldesc("FLAG", False, ndim=2, shape=np.array([4, 2], dtype=np.int32))]
    )
    t = table("fixed.tab", d, nrow=4, dminfo=_tcsm_dminfo(), ack=False)
    t.putcol("FLAG", np.zeros((4, 4, 2), bool))
    t.close()
    out = _real(
        "import casacore.tables as pt, sys\n"
        "t = pt.table(sys.argv[1], ack=False)\n"
        "print(t.nrows(), t.iscelldefined('FLAG', 0), t.getcell('FLAG', 0).shape)\n",
        str(tmp_path / "fixed.tab"),
    )
    assert out.split() == ["4", "True", "(4,", "2)"]


def _write_mixed(tmp_path):
    """The issue's part-3/4 table: ISM scalar + TiledShapeStMan array."""
    d = maketabdesc([makescacoldesc("TIME", 0.0), makearrcoldesc("DATA", 0.0, ndim=2)])
    dm = {
        "*1": {"TYPE": "IncrementalStMan", "NAME": "TimeISM", "SPEC": {}, "COLUMNS": ["TIME"]},
        "*2": {
            "TYPE": "TiledShapeStMan",
            "NAME": "TiledDATA",
            "SPEC": {"DEFAULTTILESHAPE": np.array([2, 4, 8], dtype=np.int32)},
            "COLUMNS": ["DATA"],
        },
    }
    t = table(str(tmp_path / "desc.tab"), d, nrow=0, dminfo=dm, ack=False)
    t.addrows(8)
    t.putcol("TIME", np.repeat([1.0, 2.0], 4))
    t.putcol("DATA", np.ones((8, 4, 2)))
    t.close()


def test_getcoldesc_actual_reports_bound_manager(tmp_path):
    _write_mixed(tmp_path)
    t = table(str(tmp_path / "desc.tab"), ack=False)
    actual = {c: t.getcoldesc(c) for c in ("TIME", "DATA")}
    t.close()
    assert (actual["TIME"]["dataManagerType"], actual["TIME"]["dataManagerGroup"]) == (
        "IncrementalStMan",
        "TimeISM",
    )
    assert (actual["DATA"]["dataManagerType"], actual["DATA"]["dataManagerGroup"]) == (
        "TiledShapeStMan",
        "TiledDATA",
    )


def test_defaulttileshape_applied_after_nrow0_deferral(tmp_path):
    _write_mixed(tmp_path)
    t = table(str(tmp_path / "desc.tab"), ack=False)
    spec = t.getdminfo()["*2"]["SPEC"]
    t.close()
    assert list(spec["DEFAULTTILESHAPE"]) == [2, 4, 8]
    # The real hypercube carries the geometry, like casacore's dminfo.
    cubes = {
        k: {kk: vv for kk, vv in v.items() if kk != "ID"}
        for k, v in spec["HYPERCUBES"].items()
    }
    real = [c for c in cubes.values() if c["CubeShape"]]
    assert real, f"no hypercube with a cube shape: {cubes}"
    assert list(real[0]["CubeShape"]) == [2, 4, 8]
    assert list(real[0]["TileShape"]) == [2, 4, 8]
    assert list(real[0]["CellShape"]) == [2, 4]
    assert real[0]["BucketSize"] == 512


@needs_casacore
def test_casacore_getcoldesc_matches(tmp_path):
    # casacore writes StandardStMan into the stored column description of
    # a dminfo-bound column; getcoldesc(actual=True) must still report the
    # bound manager — both for a casacure-written table (above) and for a
    # table casacore itself wrote.
    out = _real(
        "import numpy as np\n"
        "import casacore.tables as pt, sys\n"
        "d = pt.maketabdesc([pt.makescacoldesc('TIME', 0.0),"
        " pt.makearrcoldesc('DATA', 0.0, ndim=2)])\n"
        "dm = {'*1': {'TYPE': 'IncrementalStMan', 'NAME': 'TimeISM', 'SPEC': {},"
        " 'COLUMNS': ['TIME']},\n"
        "      '*2': {'TYPE': 'TiledShapeStMan', 'NAME': 'TiledDATA',"
        " 'SPEC': {'DEFAULTTILESHAPE': np.array([2, 4, 8], dtype=np.int32)},"
        " 'COLUMNS': ['DATA']}}\n"
        "t = pt.table(sys.argv[1], d, nrow=0, dminfo=dm, ack=False)\n"
        "t.addrows(8)\n"
        "t.putcol('TIME', np.repeat([1.0, 2.0], 4))\n"
        "t.putcol('DATA', np.ones((8, 4, 2)))\n"
        "t.close()\n"
        "t = pt.table(sys.argv[1], ack=False)\n"
        "for c in ('TIME', 'DATA'):\n"
        "    cd = t.getcoldesc(c)\n"
        "    print(cd['dataManagerType'], cd['dataManagerGroup'])\n"
        "cd = t.getcoldesc('TIME', actual=False)\n"
        "print('stored:', cd['dataManagerType'], cd['dataManagerGroup'])\n",
        str(tmp_path / "cc.tab"),
    )
    assert out.splitlines()[:2] == [
        "IncrementalStMan TimeISM",
        "TiledShapeStMan TiledDATA",
    ]
    assert out.splitlines()[2] == "stored: StandardStMan StandardStMan"
    # casacure reads the casacore-written table the same way.
    t = table(str(tmp_path / "cc.tab"), ack=False)
    actual = {c: t.getcoldesc(c) for c in ("TIME", "DATA")}
    stored = t.getcoldesc("TIME", actual=False)
    t.close()
    assert (actual["TIME"]["dataManagerType"], actual["TIME"]["dataManagerGroup"]) == (
        "IncrementalStMan",
        "TimeISM",
    )
    assert (actual["DATA"]["dataManagerType"], actual["DATA"]["dataManagerGroup"]) == (
        "TiledShapeStMan",
        "TiledDATA",
    )
    assert stored["dataManagerType"] == "StandardStMan"
