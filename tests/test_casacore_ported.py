"""Ported table tests from python-casacore's `tests/test_table.py`
(https://github.com/casacore/python-casacore), adapted to run against
`casacure.tables` through the `casacore` shim.

Each test keeps the python-casacore scenario; only the calls that casacure
does not implement are replaced with the equivalent implemented call
(noted in the docstring). Builder helpers (`makescacoldesc` & co.) are
inlined here (they are pure python-casacore helpers).
"""

import collections
import pathlib
import tempfile

import numpy as np
import pytest

from casacore.tables import (
    complete_ms_desc,
    default_ms,
    default_ms_subtable,
    required_ms_desc,
    taql,
    table,
    tablefromascii,
)

subtables = (
    "ANTENNA",
    "DATA_DESCRIPTION",
    "DOPPLER",
    "FEED",
    "FIELD",
    "FLAG_CMD",
    "FREQ_OFFSET",
    "HISTORY",
    "OBSERVATION",
    "POINTING",
    "POLARIZATION",
    "PROCESSOR",
    "SOURCE",
    "SPECTRAL_WINDOW",
    "STATE",
    "SYSCAL",
    "WEATHER",
)


def _value_type_name(value):
    """Map a sample Python value to the casacore valueType name (mirrors
    python-casacore's `_value_type_name`)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, complex):
        return "dcomplex"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "record"
    raise TypeError("unsupported sample value %r" % (value,))


def makescacoldesc(columnname, value, datamanagertype="", datamanagergroup="",
                   options=0, maxlen=0, comment="", valuetype="", keywords={}):
    vtype = valuetype or _value_type_name(value)
    return {
        "name": columnname,
        "desc": {
            "valueType": vtype,
            "dataManagerType": datamanagertype,
            "dataManagerGroup": datamanagergroup,
            "option": options,
            "maxlen": maxlen,
            "comment": comment,
            "keywords": keywords,
        },
    }


def makearrcoldesc(columnname, value, ndim=0, shape=[], datamanagertype="",
                   datamanagergroup="", options=0, maxlen=0, comment="",
                   valuetype="", keywords={}):
    vtype = valuetype or _value_type_name(value)
    if len(shape) > 0 and ndim <= 0:
        ndim = len(shape)
    desc = {
        "valueType": vtype,
        "dataManagerType": datamanagertype,
        "dataManagerGroup": datamanagergroup,
        "ndim": ndim,
        "shape": shape,
        "_c_order": True,
        "option": options,
        "maxlen": maxlen,
        "comment": comment,
        "keywords": keywords,
    }
    return {"name": columnname, "desc": desc}


def maketabdesc(descs=[]):
    if isinstance(descs, dict):
        descs = [descs]
    rec = {}
    for desc in descs:
        colname = desc["name"]
        if colname in rec:
            raise ValueError("Column name %s multiply used in table description" % colname)
        rec[colname] = desc["desc"]
    return rec


def makecoldesc(columnname, desc):
    return {"name": columnname, "desc": desc}


def makedminfo(tabdesc, spec):
    """A minimal `makedminfo` (casacure copies the data-manager grouping from
    the column descriptors)."""
    return spec


@pytest.fixture
def tdir():
    d = tempfile.mkdtemp(prefix="casacure-ported-")
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def _base_descs():
    c1 = makescacoldesc("coli", 0)
    c2 = makescacoldesc("cold", 0.0)
    c3 = makescacoldesc("cols", "")
    c4 = makescacoldesc("colb", True)
    c5 = makescacoldesc("colc", 0.0 + 0j)
    c6 = makearrcoldesc("colarr", 0.0)
    return (c1, c2, c3, c4, c5, c6)


def test_check_datatypes(tdir):
    """python-casacore test_check_datatypes (via getcoldesc)."""
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc(_base_descs()), ack=False)
    self = type("T", (), {})()  # stub for assertEqual-style use
    assert t.getcoldesc("coli")["valueType"] == "int"
    assert t.getcoldesc("cold")["valueType"] == "double"
    assert t.getcoldesc("cols")["valueType"] == "string"
    assert t.getcoldesc("colb")["valueType"] == "boolean"
    assert t.getcoldesc("colc")["valueType"] == "dcomplex"
    assert t.getcoldesc("colarr")["valueType"] == "double"
    t.close()


def test_check_putdata(tdir):
    """python-casacore test_check_putdata (without removerows)."""
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc(_base_descs()), ack=False)
    t.addrows(2)
    assert t.nrows() == 2
    # Unset cells read back as the column default (python-casacore parity).
    np.testing.assert_array_equal(t.getcol("coli"), np.array([0, 0]))
    t.putcol("coli", (1, 2))
    np.testing.assert_array_equal(t.getcol("coli"), np.array([1, 2]))
    t.putcol("cold", t.getcol("coli") + 3)
    np.testing.assert_array_equal(t.getcol("cold"), np.array([4.0, 5.0]))
    t.close()


def test_addcolumns(tdir):
    """python-casacore test_addcolumns (casacure has no renamecol; the
    addcols check is the portable part)."""
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc(_base_descs()), ack=False)
    t.addrows(2)
    cd1 = makecoldesc("col2", t.getcoldesc("coli"))
    t.addcols(maketabdesc(cd1))
    assert len(t.colnames()) == 7
    assert "col2" in t.colnames()
    t.close()


def test_keywords(tdir):
    """python-casacore test_keywords (adapted: keywordnames()/fieldnames()
    from getkeywords())."""
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc(_base_descs()), ack=False)
    t.addrows(2)
    t.putkeyword("key1", "keyval")
    t.putkeyword("keyrec", {"skey1": 1, "skey2": 3.0})
    assert t.getkeyword("key1") == "keyval"
    keys = t.getkeywords()
    assert "key1" in keys
    assert "keyrec" in keys
    assert keys["keyrec"]["skey1"] == 1
    t.putcolkeyword("coli", "keycoli", {"colskey": 1, "colskey2": 3.0})
    assert t.getcolkeywords("coli")["keycoli"]["colskey2"] == 3
    t.removekeyword("key1")
    assert "key1" not in t.getkeywords()
    t.close()


def test_subset(tdir):
    """python-casacore test_subset (query/taql agree on columns)."""
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc(_base_descs()), ack=False)
    t.addrows(3)
    t.putcol("coli", (1, 2, 3))
    t1 = taql("select * from $1 where coli > 1 order by coli desc",
              tables=[t])
    taqlcol = t1.colnames()
    q = t.query("coli > 1")
    querycols = q.colnames()
    assert querycols == taqlcol
    t1.close()
    t.close()


def test_subtables(tdir):
    """python-casacore test_subtables (a table keyword pointing at a
    subtable)."""
    (c1, c2, c3, c4, c5, c6) = _base_descs()
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc((c1, c2, c3)), ack=False)
    sub = table(pathlib.Path(tdir) / "sub", maketabdesc((c1, c2, c3)))
    t.putkeyword("subtablename", sub)
    val = t.getkeyword("subtablename")
    assert "Table:" in val or val.endswith("sub")
    t.close()
    sub.close()


def test_tableascii(tdir):
    """python-casacore test_tableascii."""
    (c1, c2, c3, c4, c5) = [
        makescacoldesc("coli", 0),
        makescacoldesc("cold", 0.0),
        makescacoldesc("cols", ""),
        makescacoldesc("colb", True),
        makescacoldesc("colc", 0.0 + 0j),
    ]
    t = table(pathlib.Path(tdir) / "tab1", maketabdesc((c1, c2, c3, c4, c5)), ack=False)
    t.addrows(5)
    t.putcol("coli", (1, 2, 3, 4, 5))
    t.putcol("cold", (1.5, 2.5, 3.5, 4.5, 5.5))
    t.putcol("cols", ("a", "b", "c", "d", "e"))
    t.putcol("colb", (True, False, True, False, True))
    t.putcol("colc", (1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j))
    ascii_file = pathlib.Path(tdir) / "asciitemp1"
    t.toascii(str(ascii_file), columnnames=t.colnames())
    ta = tablefromascii(pathlib.Path(tdir) / "tablefromascii",
                        str(ascii_file))
    assert t.colnames() == ta.colnames()
    ta.close()
    t.close()


def test_complete_desc(tdir):
    """python-casacore test_complete_desc: create rows for every complete
    MS desc."""
    for i, name in enumerate(("MAIN",) + subtables):
        desc = complete_ms_desc(name)
        assert isinstance(desc, dict)
        assert len(desc) > 0
        with table(pathlib.Path(tdir) / ("complete_%02d.table" % i), desc,
                   ack=False) as T:
            T.addrows(10)


def test_required_desc(tdir):
    """python-casacore test_required_desc: an MS with a TiledColumnStMan UVW
    column + a MODEL_DATA tiled array column."""
    ms1 = default_ms(pathlib.Path(tdir) / "ttable.ms1")
    ms1.close()

    ms2_desc = required_ms_desc("MAIN")
    ms2_desc["UVW"].update(
        options=0, shape=[3], ndim=1,
        dataManagerGroup="UVW", dataManagerType="TiledColumnStMan")
    dmgroup_spec = {"UVW": {"DEFAULTTILESHAPE": [3, 128 * 64]}}

    model_data_desc = makearrcoldesc("MODEL_DATA", 0.0, options=4,
                                     valuetype="complex", shape=[16, 4],
                                     ndim=2,
                                     datamanagertype="TiledColumnStMan",
                                     datamanagergroup="DataGroup")
    dmgroup_spec.update({"DataGroup": {"DEFAULTTILESHAPE": [4, 16, 32]}})
    ms2_desc.update(maketabdesc(model_data_desc))
    ms2_dminfo = makedminfo(ms2_desc, dmgroup_spec)

    with default_ms(pathlib.Path(tdir) / "ttable.ms2",
                    ms2_desc, ms2_dminfo) as ms2:
        desc = ms2.getcoldesc("UVW")
        assert desc["dataManagerType"] == "TiledColumnStMan"
        assert desc["dataManagerGroup"] == "UVW"
        assert desc["valueType"] == "double"
        assert desc["ndim"] == 1
        assert "MODEL_DATA" in ms2.colnames()
        mdesc = ms2.getcoldesc("MODEL_DATA")
        assert mdesc["dataManagerType"] == "TiledColumnStMan"
        assert mdesc["valueType"] == "complex"
        assert mdesc["ndim"] == 2
        assert np.all(mdesc["shape"] == [16, 4])
