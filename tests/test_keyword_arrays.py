"""casacure#40: string-array keywords must be written as arrays.

A list keyword in a table description (``QuantumUnits``, ``MEASINFO``'s
neighbours, ``FLAG_CATEGORY``'s ``CATEGORY``) used to land on disk as a
``{shape, array}`` *sub-record* where casacore declares ``Array<String>``:
CASA then refuses the table outright (``TableRecordRep::get_pointer -
incorrect data type used for field QuantumUnits``) and the description's
fixed structure makes it unrepairable afterwards.  ``putcolkeyword`` with a
numpy string array was stored as its repr (``"['s']"``).

The suite runs with ``tests/shim`` shadowing ``casacore``, so every test
asserts through casacure's own reader first — a sub-record reads back as a
dict, an array as a list, and that distinction *is* the bug — and then, where
real python-casacore is importable, through it too: that is the CASA-side
oracle the issue's comparison table used.
"""

import os
import subprocess
import sys

import numpy as np

from casacure.tables import (
    makearrcoldesc,
    makescacoldesc,
    maketabdesc,
    table,
)

_ORACLE_CODE = """\
import sys
import casacore.tables as ct

kw = ct.table(sys.argv[1], ack=False).getcolkeywords(sys.argv[2])
v = kw[sys.argv[3]]
if isinstance(v, dict):
    print("DICT:" + repr(sorted(v.keys())))
else:
    print("LIST:" + ",".join(map(str, v)))
"""


def _real_casacore_keyword(path, column, keyword):
    """The keyword as *real* python-casacore reads it, or None when no
    interpreter here can import it (the suite itself runs under the shim, so
    reach a clean environment; CI's pip-installed python-casacore answers
    under ``sys.executable``, a dev box's system python under ``python3``)."""
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")
    }
    for exe in (sys.executable, "python3"):
        try:
            out = subprocess.run(
                [exe, "-c", _ORACLE_CODE, str(path), column, keyword],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip().startswith(("LIST:", "DICT:")):
            kind, _, payload = out.stdout.strip().partition(":")
            return kind, payload.split(",") if kind == "LIST" else payload
    return None


def _assert_array_like(value, expected):
    """An array keyword: a list (never a ``{shape, array}`` sub-record)."""
    assert not isinstance(value, dict), f"keyword written as a sub-record: {value!r}"
    assert list(value) == list(expected), (value, expected)


def _assert_oracle_array(path, column, keyword, expected):
    got = _real_casacore_keyword(path, column, keyword)
    if got is None:
        return  # no real python-casacore on this box; casacure read above
    kind, payload = got
    assert kind == "LIST", (
        f"real python-casacore reads {keyword} as a {kind}: {payload} "
        "(CASA refuses the table)"
    )
    assert payload == [str(e) for e in expected], (payload, expected)


def test_desc_string_list_keywords_are_arrays(tmp_path):
    """casacure#40 case 1: list keywords in a description survive table
    creation as Array<String> on disk."""
    path = str(tmp_path / "desclist.tab")
    cds = [
        makescacoldesc(
            "TIME",
            0.0,
            keywords={
                "QuantumUnits": ["s"],
                "MEASINFO": {"type": "epoch", "Ref": "UTC"},
            },
        ),
        makearrcoldesc(
            "UVW", 0.0, shape=[3], keywords={"QuantumUnits": ["m", "m", "m"]}
        ),
    ]
    t = table(path, maketabdesc(cds), nrow=1, ack=False)
    t.close()

    kw = table(path, ack=False).getcolkeywords("UVW")
    _assert_array_like(kw["QuantumUnits"], ["m", "m", "m"])
    time_kw = table(path, ack=False).getcolkeywords("TIME")
    _assert_array_like(time_kw["QuantumUnits"], ["s"])
    # a genuine record keyword keeps its record type
    assert time_kw["MEASINFO"] == {"type": "epoch", "Ref": "UTC"}

    _assert_oracle_array(path, "UVW", "QuantumUnits", ["m", "m", "m"])
    _assert_oracle_array(path, "TIME", "QuantumUnits", ["s"])


def test_putcolkeyword_numpy_string_array(tmp_path):
    """casacure#40 case 2: a numpy string array is stored as Array<String>,
    not its repr (``"['s']"``)."""
    path = str(tmp_path / "npkw.tab")
    t = table(path, maketabdesc([makescacoldesc("T2", 0.0)]), nrow=2, ack=False)
    t.putcolkeyword("T2", "QuantumUnits", np.array(["s"]))
    t.flush()

    got = table(path, ack=False).getcolkeywords("T2")
    _assert_array_like(got["QuantumUnits"], ["s"])
    _assert_oracle_array(path, "T2", "QuantumUnits", ["s"])


def test_empty_list_keyword_is_an_array(tmp_path):
    """``[]`` (FLAG_CATEGORY's CATEGORY) is an empty array, not a
    ``{shape: [0], array: []}`` sub-record."""
    path = str(tmp_path / "emptykw.tab")
    cds = [makescacoldesc("T3", 0.0, keywords={"CATEGORY": []})]
    t = table(path, maketabdesc(cds), nrow=1, ack=False)
    t.close()

    kw = table(path, ack=False).getcolkeywords("T3")
    assert not isinstance(kw["CATEGORY"], dict), kw
    assert list(kw["CATEGORY"]) == []
    got = _real_casacore_keyword(path, "T3", "CATEGORY")
    if got is not None:
        assert got[0] == "LIST", f"real python-casacore reads CATEGORY as {got}"


def test_shape_array_dict_repairs_on_rewrite(tmp_path):
    """A description read back from a table written by an older casacure
    carries ``{shape, array}`` dicts; re-writing it repairs them into arrays
    (python-casacore turns any such dict into an array too)."""
    path = str(tmp_path / "repair.tab")
    cds = [
        makearrcoldesc(
            "S",
            0.0,
            shape=[2],
            keywords={"QuantumUnits": {"shape": [2], "array": ["a", "b"]}},
        )
    ]
    t = table(path, maketabdesc(cds), nrow=1, ack=False)
    t.close()

    kw = table(path, ack=False).getcolkeywords("S")
    _assert_array_like(kw["QuantumUnits"], ["a", "b"])
    _assert_oracle_array(path, "S", "QuantumUnits", ["a", "b"])


def test_addcols_string_list_keywords(tmp_path):
    """The dask-ms write path (addcols on an existing table) carries the
    keyword arrays through too."""
    path = str(tmp_path / "added.tab")
    t = table(path, maketabdesc([makescacoldesc("A", 0.0)]), nrow=2, ack=False)
    t.addcols(
        makearrcoldesc(
            "UVW2", 0.0, shape=[3], keywords={"QuantumUnits": ["m", "m", "m"]}
        )
    )
    t.close()

    kw = table(path, ack=False).getcolkeywords("UVW2")
    _assert_array_like(kw["QuantumUnits"], ["m", "m", "m"])
    _assert_oracle_array(path, "UVW2", "QuantumUnits", ["m", "m", "m"])
