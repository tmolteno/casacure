"""TaQL forms and rownumbers(other) (issue #44).

python-casacore 3.8.1 accepts four query forms casacure 3.8.41 rejected:

* ``WHERE expr BETWEEN lo AND hi`` (and ``NOT BETWEEN``);
* ``SELECT FROM t ...`` with no column list (the whole row, like ``*``);
* ``ORDERBY DESC col`` — the direction flag before the key (the
  ``ORDERBY col DESC`` form already worked);
* ``sel.rownumbers(t)`` / ``t.rownumbers(sel)`` — this table's rows as
  row numbers of another table sharing the base (-1 where absent).

The expected values are probed from python-casacore 3.8.1 (the
``needs_casacore`` parity test reruns the rownumbers half against it).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from casacore.tables import table, taql, maketabdesc, makescacoldesc


@pytest.fixture()
def tq(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = table(
        "tq.tab",
        maketabdesc([makescacoldesc("TIME", 0.0), makescacoldesc("ANT", 0)]),
        nrow=6,
        ack=False,
    )
    t.putcol("TIME", np.arange(6.0))
    t.putcol("ANT", np.array([3, 1, 2, 3, 1, 2]))
    t.flush()
    return t


def test_between(tq):
    q = taql("SELECT TIME FROM tq.tab WHERE TIME BETWEEN 1 AND 3")
    assert q.getcol("TIME").tolist() == [1.0, 2.0, 3.0]
    q = taql("SELECT TIME FROM tq.tab WHERE TIME NOT BETWEEN 1 AND 3")
    assert q.getcol("TIME").tolist() == [0.0, 4.0, 5.0]


def test_select_without_column_list(tq):
    q = taql("SELECT FROM tq.tab WHERE ANT == 3")
    assert q.getcol("TIME").tolist() == [0.0, 3.0]
    assert q.getcol("ANT").tolist() == [3, 3]


def test_orderby_desc_before_key(tq):
    q = taql("SELECT TIME FROM tq.tab ORDERBY DESC TIME")
    assert q.getcol("TIME").tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    # The spaced form, with the flag in both positions.
    q = taql("SELECT TIME FROM tq.tab ORDER BY DESC TIME")
    assert q.getcol("TIME").tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    q = taql("SELECT TIME FROM tq.tab ORDER BY TIME DESC")
    assert q.getcol("TIME").tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]


def test_rownumbers_other_table(tq):
    sel = tq.query("ANT == 3")
    assert [int(x) for x in sel.rownumbers(tq)] == [0, 3]
    assert [int(x) for x in tq.rownumbers(sel)] == [0, -1, -1, 1, -1, -1]
    # A chained view (selection then sort) roots at the same base.
    srt = tq.query("ANT == 3").sort("TIME")
    assert [int(x) for x in srt.rownumbers(tq)] == [0, 3]
    assert [int(x) for x in tq.rownumbers(srt)] == [0, -1, -1, 1, -1, -1]
    # Unrelated tables refuse.
    other = table(
        os.path.join(os.path.dirname(tq.name()), "other.tab"),
        maketabdesc([makescacoldesc("X", 0)]),
        nrow=1,
        ack=False,
    )
    with pytest.raises(ValueError, match="base table"):
        sel.rownumbers(other)


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def test_rownumbers_matches_casacore(tmp_path):
    t = table(
        str(tmp_path / "tq.tab"),
        maketabdesc([makescacoldesc("TIME", 0.0), makescacoldesc("ANT", 0)]),
        nrow=6,
        ack=False,
    )
    t.putcol("ANT", np.array([3, 1, 2, 3, 1, 2]))
    t.flush()
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import casacore.tables as pt, sys\n"
            "t = pt.table(sys.argv[1], ack=False)\n"
            "sel = t.query('ANT == 3')\n"
            "print([int(x) for x in sel.rownumbers(t)])\n"
            "print([int(x) for x in t.rownumbers(sel)])\n",
            str(tmp_path / "tq.tab"),
        ],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"real python-casacore not usable: {r.stderr.strip()}")
    sel_rows, base_rows = (eval(line) for line in r.stdout.splitlines())
    assert sel_rows == [0, 3]
    assert base_rows == [0, -1, -1, 1, -1, -1]
    # And casacure reads the same table the same way.
    assert [int(x) for x in t.query("ANT == 3").rownumbers(t)] == sel_rows
    assert [int(x) for x in t.rownumbers(t.query("ANT == 3"))] == base_rows
