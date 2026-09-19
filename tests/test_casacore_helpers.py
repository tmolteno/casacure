"""Unit tests for the `casacore.tables` helper surface that the
python-casacore dependent packages rely on: the column/table-description
builders (`makescacoldesc`, `makearrcoldesc`, `makecoldesc`, `maketabdesc`,
`makedminfo`) and the table lifecycle helpers (`tableexists`, `tabledelete`,
`tablecopy`), plus the `table.taql()` method.

Semantics are pinned to python-casacore 3.8.1 (its `casacore/tables.py`).
"""

import numpy as np
import pytest

from casacore.tables import (
    default_ms_subtable,
    makecoldesc,
    makearrcoldesc,
    makedminfo,
    makescacoldesc,
    maketabdesc,
    table,
    tablecopy,
    tabledelete,
    tableexists,
    taql,
)


def test_makescacoldesc_exact_structure():
    """Output must match python-casacore literally (dict layout + defaults)."""
    d = makescacoldesc("col1", 1)
    assert d == {
        "name": "col1",
        "desc": {
            "valueType": "int",
            "dataManagerType": "",
            "dataManagerGroup": "",
            "option": 0,
            "maxlen": 0,
            "comment": "",
            "keywords": {},
        },
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "int"),
        (1.5, "double"),
        ("x", "string"),
        (True, "boolean"),
        (1 + 2j, "dcomplex"),
        ({"a": 1}, "record"),
    ],
)
def test_makescacoldesc_infers_value_type(value, expected):
    assert makescacoldesc("c", value)["desc"]["valueType"] == expected


def test_makescacoldesc_explicit_valuetype_wins():
    d = makescacoldesc("c", 1, valuetype="double")
    assert d["desc"]["valueType"] == "double"
    d = makescacoldesc("c", 1, datamanagertype="IncrementalStMan",
                       comment="hi", keywords={"k": 5})
    assert d["desc"]["dataManagerType"] == "IncrementalStMan"
    assert d["desc"]["comment"] == "hi"
    assert d["desc"]["keywords"] == {"k": 5}


def test_makescacoldesc_rejects_unknown_value():
    with pytest.raises(TypeError):
        makescacoldesc("c", object())


def test_makearrcoldesc():
    d = makearrcoldesc("a", 1.0, 0, [2, 3])
    desc = d["desc"]
    assert desc["valueType"] == "double"
    assert desc["ndim"] == 2  # inferred from shape
    assert desc["shape"] == [2, 3]
    assert desc["_c_order"] is True
    # explicit ndim beats inference
    desc = makearrcoldesc("a", 1, ndim=1)["desc"]
    assert desc["ndim"] == 1
    assert desc["shape"] == []


def test_makecoldesc():
    src = makescacoldesc("a", 1)
    d = makecoldesc("b", src["desc"])
    assert d["name"] == "b"
    assert d["desc"] == src["desc"]


def test_maketabdesc_merge_and_duplicates():
    c1 = makescacoldesc("coli", 0)
    c2 = makescacoldesc("cold", 0.0)
    td = maketabdesc([c1, c2])
    assert sorted(td) == ["cold", "coli"]
    assert td["coli"] == c1["desc"]
    # a single dict is wrapped into a list
    assert sorted(maketabdesc(c1)) == ["coli"]
    # tuple input
    assert sorted(maketabdesc((c1, c2))) == ["cold", "coli"]
    with pytest.raises(ValueError):
        maketabdesc([c1, c1])
    with pytest.raises(TypeError):
        maketabdesc(7)


def test_builders_create_a_working_table(tmp_path):
    td = maketabdesc([
        makescacoldesc("coli", 0),
        makescacoldesc("colb", True),
        makearrcoldesc("colarr", 0.0, 0, [2]),
    ])
    with table(tmp_path / "t.tab", td, ack=False) as t:
        t.addrows(2)
        t.putcol("coli", (1, 2))
        t.putcol("colb", (True, False))
        t.putvarcol("colarr", {"r0": [1.0, 2.0], "r1": [3.0, 4.0]})
    t = table(tmp_path / "t.tab", ack=False)
    np.testing.assert_array_equal(t.getcol("coli"), [1, 2])
    np.testing.assert_array_equal(t.getcol("colb"), [True, False])
    t.close()


def test_makedminfo_groups_columns():
    td = maketabdesc([
        makescacoldesc("a", 1),
        makescacoldesc("b", 1),
        makescacoldesc("c", 1, datamanagertype="IncrementalStMan",
                       datamanagergroup="ISM"),
    ])
    dmi = makedminfo(td)
    assert len(dmi) == 2
    ssm = dmi["*1"]
    ism = dmi["*2"]
    assert ssm == {
        "COLUMNS": ["a", "b"],
        "TYPE": "StandardStMan",
        "NAME": "StandardStMan",
        "SPEC": {},
        "SEQNR": 0,
    }
    assert ism["COLUMNS"] == ["c"]
    assert ism["TYPE"] == "IncrementalStMan"
    # group spec is threaded through per group
    dmi = makedminfo(td, {"ISM": {"MAXIMUMCACHESIZE": 1000}})
    assert dmi["*2"]["SPEC"] == {"MAXIMUMCACHESIZE": 1000}
    assert dmi["*1"]["SPEC"] == {}


def test_makedminfo_type_mismatch_raises():
    td = maketabdesc([
        makescacoldesc("a", 1),
        makescacoldesc("b", 1, datamanagertype="IncrementalStMan"),
    ])
    with pytest.raises(ValueError):
        makedminfo(td)


def test_makedminfo_empty_type_means_standard(tmp_path):
    # python-casacore: empty dataManagerType/Group -> StandardStMan.
    td = maketabdesc([makescacoldesc("a", 1)])
    dmi = makedminfo(td)
    assert dmi["*1"]["TYPE"] == "StandardStMan"
    assert dmi["*1"]["NAME"] == "StandardStMan"


def test_tableexists(tmp_path):
    with table(tmp_path / "t.tab", maketabdesc(makescacoldesc("a", 1)),
               ack=False):
        pass
    assert tableexists(tmp_path / "t.tab") is True
    assert tableexists(tmp_path / "missing.tab") is False
    assert tableexists(tmp_path / "somefile.txt") is False


def test_tabledelete(tmp_path):
    p = tmp_path / "t.tab"
    with table(p, maketabdesc(makescacoldesc("a", 1)), ack=False):
        pass
    assert tableexists(p)
    tablecopy(str(p), str(tmp_path / "t2.tab"))
    tabledelete(str(tmp_path / "t2.tab"))
    assert not tableexists(tmp_path / "t2.tab")
    assert tableexists(p)  # original untouched
    with pytest.raises(ValueError):
        tabledelete(str(tmp_path / "t2.tab"))  # already gone


def test_tablecopy_preserves_data(tmp_path):
    p = tmp_path / "src.tab"
    with table(p, maketabdesc(makescacoldesc("a", 1)), ack=False) as t:
        t.addrows(2)
        t.putcol("a", (5, 6))
    dst = tmp_path / "copy.tab"
    tablecopy(str(p), str(dst))
    t = table(dst, ack=False)
    np.testing.assert_array_equal(t.getcol("a"), [5, 6])
    t.close()
    # destination must not pre-exist
    with pytest.raises(RuntimeError):
        tablecopy(str(p), str(dst))
    with pytest.raises(ValueError):
        tablecopy(str(tmp_path / "ghost.tab"), str(tmp_path / "x.tab"))


def test_tablecopy_deep_copies_subtables(tmp_path):
    sub = table(tmp_path / "sub.tab", maketabdesc(makescacoldesc("s", 1)),
                ack=False)
    sub.addrows(1)
    sub.putcol("s", (7,))
    parent = table(tmp_path / "parent.tab",
                   maketabdesc(makescacoldesc("p", 1)), ack=False)
    parent.putkeyword("sub_table", sub)
    parent.close()
    sub.close()

    out = tmp_path / "out"
    out.mkdir()
    # shallow copy: subtable dir is NOT copied
    tablecopy(str(tmp_path / "parent.tab"), str(out / "p1.tab"), deep=False)
    assert tableexists(out / "p1.tab")
    assert not tableexists(out / "sub.tab")
    # deep copy: subtable dir lands next to the copy
    tablecopy(str(tmp_path / "parent.tab"), str(out / "p2.tab"), deep=True)
    assert tableexists(out / "p2.tab")
    assert tableexists(out / "sub.tab")
    t = table(out / "sub.tab", ack=False)
    np.testing.assert_array_equal(t.getcol("s"), [7])
    t.close()


def test_default_ms_subtable_uses_standard_schema(tmp_path):
    """default_ms_subtable(name, path) creates the standard subtable schema
    (like python-casacore) when no tabdesc is given; the subtable is
    writable and reads back."""
    ms = tmp_path / "ms"
    from casacore.tables import default_ms

    default_ms(str(ms))
    sub = ms / "SOURCE"
    with default_ms_subtable("SOURCE", str(sub)) as s:
        assert "SOURCE_ID" in s.colnames()
        assert "NAME" in s.colnames()
        s.addrows(2)
        s.putcol("SOURCE_ID", [0, 1])
        s.putcol("NAME", ["T1", "T2"])
    from casacore.tables import table

    t = table(str(sub), ack=False)
    np.testing.assert_array_equal(t.getcol("SOURCE_ID"), [0, 1])
    assert t.getcol("NAME") == ["T1", "T2"]
    t.close()
    # a caller-supplied tabdesc still wins
    custom = maketabdesc(makescacoldesc("x", 1))
    with default_ms_subtable("WEATHER", str(ms / "W"), custom) as s:
        assert sorted(s.colnames()) == ["x"]
    s.close()


def test_putcol_array_column_single_element_cell(tmp_path):
    """A 1-element array cell on an array column stays an Array value (not a
    scalar): an MS with ncorr=1 stores and reads back [[9]]."""
    td = maketabdesc([makearrcoldesc("arr", 1, 0, [1])])
    with table(tmp_path / "t.tab", td, ack=False) as t:
        t.addrows(1)
        t.putcol("arr", np.array([[7]], dtype=np.int64))
    t = table(tmp_path / "t.tab", ack=False)
    np.testing.assert_array_equal(t.getcol("arr"), [[7]])
    t.close()


def test_ms_polarization_corr_type_single_corr(tmp_path):
    """The skarabina MS fixture path: write CORR_TYPE (ncorr=1) into the
    standard POLARIZATION subtable and read it back."""
    from casacore.tables import complete_ms_desc, default_ms

    p = tmp_path / "ms"
    default_ms(str(p))
    with table(str(p / "POLARIZATION"), readonly=False) as t:
        t.addrows(1)
        t.putcol("CORR_TYPE", np.array([[9]], dtype=np.int32))
    t = table(str(p / "POLARIZATION"), ack=False)
    np.testing.assert_array_equal(t.getcol("CORR_TYPE"), [[9]])
    t.close()


def test_putcol_numpy_unicode_strings(tmp_path):
    """numpy '<U'/'<S' string arrays store into scalar and array string
    columns (dask-ms writes subtable string columns this way)."""
    td = maketabdesc([makescacoldesc("name", "")])
    with table(tmp_path / "s.tab", td, ack=False) as t:
        t.addrows(2)
        t.putcol("name", np.array(["a", "b"], dtype="<U4"))
    t = table(tmp_path / "s.tab", ack=False)
    assert t.getcol("name") == ["a", "b"]
    t.close()
    # string array column
    td = maketabdesc([makearrcoldesc("arr", "", 0, [2])])
    with table(tmp_path / "a.tab", td, ack=False) as t:
        t.addrows(1)
        t.putcol("arr", np.array([["x", "y"]], dtype="<U1"))
    t = table(tmp_path / "a.tab", ack=False)
    got = t.getcol("arr")
    assert got["shape"] == [1, 2]
    assert got["array"] == ["x", "y"]
    t.close()


def test_removecols(tmp_path):
    """removecols drops columns and their data; remaining columns survive."""
    with table(tmp_path / "t.tab",
               maketabdesc([makescacoldesc("a", 1), makescacoldesc("b", 0.0),
                            makescacoldesc("c", "")]), ack=False) as t:
        t.addrows(1)
        t.putcol("a", [1])
        t.putcol("b", [2.0])
        t.putcol("c", ["x"])
    with table(tmp_path / "t.tab", readonly=False) as t:
        t.removecols(["b"])
    t = table(tmp_path / "t.tab", ack=False)
    assert sorted(t.colnames()) == ["a", "c"]
    np.testing.assert_array_equal(t.getcol("a"), [1])
    assert t.getcol("c") == ["x"]
    t.close()
    with pytest.raises(KeyError):
        with table(tmp_path / "t.tab", readonly=False) as t:
            t.removecols(["z"])


def test_removecol_single(tmp_path):
    with table(tmp_path / "t.tab",
               maketabdesc([makescacoldesc("a", 1), makescacoldesc("b", 1)]),
               ack=False) as t:
        t.addrows(1)
    with table(tmp_path / "t.tab", readonly=False) as t:
        t.removecol("a")
    t = table(tmp_path / "t.tab", ack=False)
    assert sorted(t.colnames()) == ["b"]
    t.close()


def test_tiledshapestman_created_as_standard(tmp_path):
    """A column declared TiledShapeStMan (skarabina's flag versions) is
    created writable, stored via StandardStMan, and round-trips."""
    desc = {
        "FLAG": {"valueType": "boolean", "ndim": 2, "_c_order": True,
                 "dataManagerType": "TiledShapeStMan",
                 "dataManagerGroup": "TiledFlag"}
    }
    with table(tmp_path / "fv.tab", desc, ack=False) as t:
        assert t.getcoldesc("FLAG")["dataManagerType"] == "StandardStMan"
        t.addrows(1)
        t.putcol("FLAG", np.zeros((1, 8), dtype=bool))
    t = table(tmp_path / "fv.tab", ack=False)
    np.testing.assert_array_equal(t.getcol("FLAG"), np.zeros((1, 8), bool))
    t.close()


def test_getsubtables_via_table_keyword(tmp_path):
    """A "Table: <path>" string keyword becomes a TpTable and is listed by
    getsubtables() (mirrors casacore's subtable linkage)."""
    sub = table(tmp_path / "sub.tab", maketabdesc(makescacoldesc("s", 1)),
                ack=False)
    sub.close()
    with table(tmp_path / "parent", maketabdesc(makescacoldesc("a", 1)),
               ack=False) as t:
        t.putkeyword("K", "Table: ./sub.tab")
    t = table(tmp_path / "parent", ack=False)
    assert t.getsubtables() == [str((tmp_path / "sub.tab").resolve())]
    # a table-object keyword also lists it
    sub = table(tmp_path / "sub2.tab", maketabdesc(makescacoldesc("s", 1)),
                ack=False)
    with table(tmp_path / "parent2", maketabdesc(makescacoldesc("a", 1)),
               ack=False) as t:
        t.putkeyword("K", sub)
    sub.close()
    t = table(tmp_path / "parent2", ack=False)
    assert t.getsubtables() == [str((tmp_path / "sub2.tab").resolve())]
    t.close()
    # plain string keyword is not a subtable
    with table(tmp_path / "parent3", maketabdesc(makescacoldesc("a", 1)),
               ack=False) as t:
        t.putkeyword("K", "not a table reference")
    t = table(tmp_path / "parent3", ack=False)
    assert t.getsubtables() == []
    t.close()


def test_table_copy_shallow_and_deep(tmp_path):
    """table.copy(dest) copies the table; deep=True also copies subtables."""
    sub = table(tmp_path / "sub.tab", maketabdesc(makescacoldesc("s", 1)),
                ack=False)
    sub.addrows(1)
    sub.putcol("s", (9,))
    sub.close()
    with table(tmp_path / "parent", maketabdesc(makescacoldesc("a", 1)),
               ack=False) as t:
        t.addrows(2)
        t.putcol("a", (1, 2))
        t.putkeyword("K", "Table: ./sub.tab")

    out = tmp_path / "out"
    out.mkdir()
    # shallow: subtable dir is not copied into the new parent (reference
    # resolves to a path that does not exist yet)
    t = table(tmp_path / "parent", ack=False)
    t.copy(str(out / "shallow.tab"), deep=False)
    t.close()
    assert tableexists(out / "shallow.tab")
    assert not tableexists(out / "sub.tab")

    # deep: subtable lands next to the copy and is re-linked
    t = table(tmp_path / "parent", ack=False)
    t.copy(str(out / "deep.tab"), deep=True)
    t.close()
    assert tableexists(out / "deep.tab")
    assert tableexists(out / "sub.tab")
    c = table(out / "deep.tab", ack=False)
    assert c.getsubtables() == [str((out / "sub.tab").resolve())]
    np.testing.assert_array_equal(c.getcol("a"), [1, 2])
    c.close()
    s = table(out / "sub.tab", ack=False)
    np.testing.assert_array_equal(s.getcol("s"), [9])
    s.close()

    # deep copy into the SAME parent shares the subtable (no self-copy, and
    # the original subtable is left intact)
    t = table(tmp_path / "parent", ack=False)
    t.copy(str(tmp_path / "same.tab"), deep=True)
    t.close()
    assert tableexists(tmp_path / "same.tab")
    s = table(tmp_path / "sub.tab", ack=False)
    np.testing.assert_array_equal(s.getcol("s"), [9])
    s.close()

    # destination must not pre-exist
    t = table(tmp_path / "parent", ack=False)
    with pytest.raises(RuntimeError):
        t.copy(str(out / "shallow.tab"))
    t.close()


def test_table_taql_method(tmp_path):
    with table(tmp_path / "t.tab", maketabdesc(makescacoldesc("a", 1)),
               ack=False) as t:
        t.addrows(3)
        t.putcol("a", [3, 1, 2])
    t = table(tmp_path / "t.tab", ack=False)
    r1 = t.taql("select a from $1 where a > 1 order by a")
    r2 = taql("select a from $1 where a > 1 order by a", tables=[t])
    np.testing.assert_array_equal(r1.getcol("a"), [2, 3])
    np.testing.assert_array_equal(r2.getcol("a"), [2, 3])
    assert r1.colnames() == r2.colnames()
    r1.close()
    r2.close()
    t.close()
