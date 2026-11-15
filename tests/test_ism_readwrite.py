"""IncrementalStMan non-scalar columns read and write like casacore
(issues #47/#48).

The `ism_arrays.tab` / `ism_buckets.tab` fixtures are written by
python-casacore (see `tests/make_fixtures.py`): one IncrementalStMan group
binding fixed- and variable-shape arrays (stored by casacore as 8-byte
references into `table.f0i`), scalar strings (stored inline, u32-prefixed)
and scalar controls. casacure decodes all of them; only an *array of
strings* keeps the issue-#47 named error (its elements live in the string
buckets of the index file, which no fixture covers yet).

The write direction is pinned by handing casacure-written tables to real
python-casacore in a subprocess (skipped where it is not installed).

Run as part of the project suite (the `casacore` import must resolve to the
casacure shim):
    PYTHONPATH=tests/shim python -m pytest tests/test_ism_readwrite.py -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from casacore.tables import table, maketabdesc, makearrcoldesc, makescacoldesc

CASACORE_FIXTURES = ("ism_arrays.tab", "ism_buckets.tab", "manifest.json")

FIXTURES = Path(__file__).parent / "fixtures"
ARRAYS = FIXTURES / "ism_arrays.tab"
BUCKETS = FIXTURES / "ism_buckets.tab"


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return env


needs_casacore = pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_real_casacore_env(),
        capture_output=True,
    ).returncode
    != 0,
    reason="real python-casacore not installed",
)


def _real(code, *args):
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _ism_dminfo(cols):
    return {
        "*1": {
            "TYPE": "IncrementalStMan",
            "NAME": "ism",
            "SPEC": {},
            "COLUMNS": list(cols),
        }
    }


@pytest.mark.skipif(not ARRAYS.exists(), reason="ism_arrays.tab fixture not generated")
def test_ism_array_and_string_columns_read_like_casacore():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    values = manifest["tables"]["ism_arrays"]["values"]
    t = table(str(ARRAYS), ack=False)
    np.testing.assert_array_equal(t.getcol("WEIGHT"), np.array(values["WEIGHT"], np.float32))
    for r in range(t.nrows()):
        np.testing.assert_array_equal(t.getcell("SIGMA", r), values["SIGMA"][r])
        np.testing.assert_array_equal(t.getcell("ARR2", r), np.array(values["ARR2"][r], np.int32))
        np.testing.assert_array_equal(t.getcell("FLAG", r), values["FLAG"][r])
        assert t.getcell("NAME", r) == values["NAME"][r]
        assert t.iscelldefined("SIGMA", r)
    assert t.getcol("NAME") == values["NAME"]
    np.testing.assert_array_equal(t.getcol("TIME"), np.arange(10, dtype=float))
    # Whole-column read of a fixed-shape array column stacks (nrow, 4).
    assert t.getcol("WEIGHT").shape == (10, 4)
    # A windowed read through the interval walk.
    np.testing.assert_array_equal(t.getcol("WEIGHT", 4, 3), np.array(values["WEIGHT"][4:7], np.float32))
    t.close()


@pytest.mark.skipif(not BUCKETS.exists(), reason="ism_buckets.tab fixture not generated")
def test_ism_multi_bucket_reads_cross_boundaries():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    t = table(str(BUCKETS), ack=False)
    assert t.nrows() == 5000
    # Rows either side of where bucket boundaries must fall, plus the ends.
    for r in (0, 1, 255, 256, 511, 512, 1023, 2500, 4998, 4999):
        np.testing.assert_array_equal(t.getcell("VEC", r), [float(r % 5)] * 3)
        assert t.getcell("NAME", r) == f"row-{r % 13}"
        assert t.getcell("TIME", r) == r * 1.25
        assert t.getcell("ANT1", r) == r % 7
    time = t.getcol("TIME")
    assert len(time) == 5000
    np.testing.assert_array_equal(time[::1000], [0.0, 1250.0, 2500.0, 3750.0, 5000.0])
    assert time[-1] == 4999 * 1.25
    t.close()


@needs_casacore
def test_casacore_reads_casacure_written_ism_table(tmp_path):
    """Round-trip (issue #48's write scope): casacure creates, fills and
    flushes an ISM group of arrays and strings; real python-casacore must
    read every cell back."""
    import shutil

    p = str(tmp_path / "rt.ms")
    shutil.rmtree(p, ignore_errors=True)
    td = maketabdesc(
        [
            makearrcoldesc("WEIGHT", 0.0, ndim=1, shape=[4], valuetype="float"),
            makearrcoldesc("SIGMA", 0.0, ndim=1, valuetype="double"),
            makearrcoldesc("ARR2", 0, ndim=2, valuetype="integer"),
            makescacoldesc("TIME", 0.0),
            makescacoldesc("NAME", ""),
        ]
    )
    nrow = 10
    dm = _ism_dminfo(["WEIGHT", "SIGMA", "ARR2", "TIME", "NAME"])
    with table(p, td, nrow=nrow, dminfo=dm, ack=False) as t:
        t.putcol("WEIGHT", np.ones((nrow, 4), np.float32))
        for r in range(nrow):
            t.putcell("SIGMA", r, np.arange(r + 1) * 1.5)
            t.putcell(
                "ARR2",
                r,
                (np.arange(2 * (r % 3 + 1)) + 100 * r).reshape(2, r % 3 + 1).astype(np.int32),
            )
        t.putcol("TIME", np.arange(nrow, dtype=float))
        t.putcol("NAME", ["", "a", "abc", "abcde", "a much longer string", "xy", "", "end", "", "a"])
    # Patch cells through a write handle + flush (rebuild path), then let
    # casacore verify the result.
    with table(p, readonly=False, ack=False) as t:
        t.putcell("NAME", 0, "rewritten-name")
        t.putcell("SIGMA", 0, np.array([9.0, 8.0]))
        t.flush()
    out = _real(
        "import numpy as np, sys\n"
        "import casacore.tables as pt\n"
        "t = pt.table(sys.argv[1], ack=False)\n"
        "np.testing.assert_array_equal(t.getcell('SIGMA', 7), np.arange(8) * 1.5)\n"
        "np.testing.assert_array_equal(t.getcell('SIGMA', 0), [9.0, 8.0])\n"
        "assert t.getcell('NAME', 0) == 'rewritten-name'\n"
        "assert t.getcell('NAME', 4) == 'a much longer string'\n"
        "assert t.getcell('ARR2', 2).shape == (2, 3)\n"
        "np.testing.assert_array_equal(t.getcol('WEIGHT'), np.ones((10, 4), np.float32))\n"
        "np.testing.assert_array_equal(t.getcol('TIME'), np.arange(10.0))\n"
        "assert t.iscelldefined('SIGMA', 3)\n"
        "print('oracle OK')\n",
        p,
    )
    assert "oracle OK" in out


@needs_casacore
def test_casacore_reads_casacure_ism_after_addrows_and_addcols(tmp_path):
    """Row growth and a later addcols both rebuild the ISM file; casacore
    must see the grown rows, defaults and the new column."""
    import shutil

    p = str(tmp_path / "grow.ms")
    shutil.rmtree(p, ignore_errors=True)
    td = maketabdesc(
        [
            makearrcoldesc("VEC", 0.0, ndim=1, shape=[3]),
            makescacoldesc("NAME", ""),
            makescacoldesc("TIME", 0.0),
        ]
    )
    with table(p, td, nrow=2, dminfo=_ism_dminfo(["VEC", "NAME", "TIME"]), ack=False) as t:
        t.putcol("VEC", np.stack([np.full(3, 1.0), np.full(3, 2.0)]))
        t.putcol("NAME", ["r0", "r1"])
        t.putcol("TIME", [0.0, 1.0])
    with table(p, readonly=False, ack=False) as t:
        t.addrows(2)
        t.putcell("VEC", 2, np.full(3, 5.0))
        t.putcell("NAME", 3, "r3")
        t.putcol("TIME", [0.0, 1.0, 2.0, 3.0])
        t.addcols(
            makearrcoldesc("EXTRA", 0j, ndim=1, valuetype="complex"),
            dminfo={"TYPE": "IncrementalStMan", "NAME": "ism", "SPEC": {}, "COLUMNS": ["EXTRA"]},
        )
        t.putcell("EXTRA", 1, np.array([1j, 2j]))
        t.flush()
    out = _real(
        "import numpy as np, sys\n"
        "import casacore.tables as pt\n"
        "t = pt.table(sys.argv[1], ack=False)\n"
        "assert t.nrows() == 4\n"
        "np.testing.assert_array_equal(t.getcell('VEC', 2), [5.0]*3)\n"
        "assert t.getcell('NAME', 2) == ''\n"
        "assert t.getcell('NAME', 3) == 'r3'\n"
        "np.testing.assert_array_equal(t.getcell('EXTRA', 1), [1j, 2j])\n"
        "assert not t.iscelldefined('EXTRA', 0)\n"
        "print('oracle OK')\n",
        p,
    )
    assert "oracle OK" in out
