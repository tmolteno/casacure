"""dminfo on the creation paths (issue #20).

`table()`, `default_ms()`, `default_ms_subtable()`, `tablecopy()`/`copy()`
and `addcols()` honour their `dminfo` argument: the named columns land in
the requested storage manager with the requested `SPEC.DEFAULTTILESHAPE`
(CASA order — the stored cell dims plus the rows per tile), an unsupported
manager or SPEC field raises instead of silently falling back, and the
tiling survives a reopen-and-rewrite. Real python-casacore builds the same
fixtures through its own `test_required_desc` port (test_casacore_ported.py)
and dask-ms through the write-scaling suite.
"""

import numpy as np
import pytest

from casacore.tables import (
    default_ms,
    default_ms_subtable,
    makearrcoldesc,
    makecoldesc,
    maketabdesc,
    makedminfo,
    makescacoldesc,
    table,
    tablecopy,
)


def test_issue_reproducer_two_tiled_managers(tmp_path):
    """The issue's reproducer: two TiledColumnStMan records with awkward
    tile shapes (neither divides nrow) — both land, and the data reads
    back."""
    d = str(tmp_path / "x.tab")
    cd = [
        makearrcoldesc("DATA", 0j, valuetype="complex", shape=[6, 4]),
        makearrcoldesc("FLAG", False, shape=[6, 4]),
    ]
    dm = {
        "*1": {"NAME": "tiled", "TYPE": "TiledColumnStMan",
               "SPEC": {"DEFAULTTILESHAPE": [4, 6, 5]}, "COLUMNS": ["DATA"]},
        "*2": {"NAME": "tiledf", "TYPE": "TiledColumnStMan",
               "SPEC": {"DEFAULTTILESHAPE": [4, 6, 7]}, "COLUMNS": ["FLAG"]},
    }
    t = table(d, maketabdesc(cd), nrow=23, dminfo=dm, ack=False)
    info = t.getdminfo()
    assert info["*1"]["TYPE"] == "TiledColumnStMan"
    assert info["*1"]["COLUMNS"] == ["DATA"]
    assert info["*1"]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 5]
    assert info["*2"]["TYPE"] == "TiledColumnStMan"
    assert info["*2"]["COLUMNS"] == ["FLAG"]
    assert info["*2"]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 7]

    data = np.arange(23 * 6 * 4, dtype=np.float32).reshape(23, 6, 4) + 1j
    t.putcol("DATA", data)
    flag = np.zeros((23, 6, 4), dtype=bool)
    flag[::3] = True
    t.putcol("FLAG", flag)
    t.flush()
    t.close()

    r = table(d, readonly=True, ack=False)
    np.testing.assert_array_equal(r.getcol("DATA"), data)
    np.testing.assert_array_equal(r.getcol("FLAG"), flag)
    r.close()


def test_table_dminfo_cell_splitting_tile(tmp_path):
    """dask-ms's `_fit_tile_shape` caps cell dims at 4, so wide rows get a
    tile whose cell part splits the cell — the layout real casacore writes
    natively. casacure writes and reads it too."""
    d = str(tmp_path / "straddle.tab")
    cd = [makearrcoldesc("FLAG", False, shape=[32, 4])]
    dm = {"*1": {"NAME": "tiledf", "TYPE": "TiledColumnStMan",
                 "SPEC": {"DEFAULTTILESHAPE": [4, 4, 64]}, "COLUMNS": ["FLAG"]}}
    t = table(d, maketabdesc(cd), nrow=100, dminfo=dm, ack=False)
    flag = np.arange(100 * 32 * 4).reshape(100, 32, 4) % 2 == 0
    t.putcol("FLAG", flag)
    t.flush()
    t.close()
    r = table(d, readonly=True, ack=False)
    assert r.getdminfo()["*1"]["SPEC"]["DEFAULTTILESHAPE"] == [4, 4, 64]
    np.testing.assert_array_equal(r.getcol("FLAG"), flag)
    r.close()


def test_default_ms_applies_dminfo_to_the_main_table(tmp_path):
    """`default_ms(dminfo=...)` tiles the main table's named columns; the
    schema's other columns keep StandardStMan."""
    p = str(tmp_path / "t.ms")
    dm = {"*1": {"NAME": "TiledData", "TYPE": "TiledColumnStMan",
                 "SPEC": {"DEFAULTTILESHAPE": [4, 6, 5]}, "COLUMNS": ["DATA"]}}
    data_desc = makearrcoldesc("DATA", 0j, valuetype="complex", shape=[6, 4])
    with default_ms(p, tabdesc=maketabdesc([data_desc]), dminfo=dm) as ms:
        info = ms.getdminfo()
        tiled = [v for v in info.values() if v["TYPE"] == "TiledColumnStMan"]
        assert len(tiled) == 1
        assert tiled[0]["NAME"] == "TiledData"
        assert tiled[0]["COLUMNS"] == ["DATA"]
        assert tiled[0]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 5]
        assert any(v["TYPE"] == "StandardStMan" for v in info.values())


def test_default_ms_subtable_applies_dminfo(tmp_path):
    """`default_ms_subtable` forwards `dminfo` to `table()` (it did before;
    the argument used to be dropped)."""
    p = str(tmp_path / "sub.tab")
    dm = {"*1": {"NAME": "TiledChanFreq", "TYPE": "TiledColumnStMan",
                 "SPEC": {"DEFAULTTILESHAPE": [8, 4]}, "COLUMNS": ["CHAN_FREQ"]}}
    desc = makearrcoldesc("CHAN_FREQ", 0.0, valuetype="double", shape=[8])
    with default_ms_subtable("SPECTRAL_WINDOW", p, tabdesc=maketabdesc([desc]),
                             dminfo=dm) as t:
        info = t.getdminfo()
        assert info["*1"]["TYPE"] == "TiledColumnStMan"
        assert info["*1"]["COLUMNS"] == ["CHAN_FREQ"]


def test_tablecopy_deep_converts_storage(tmp_path):
    """`tablecopy(deep=True, dminfo=...)` is casacore's storage conversion:
    the copy's named columns are rewritten into the requested managers with
    their values intact."""
    src = str(tmp_path / "src.tab")
    dst = str(tmp_path / "dst.tab")
    t = table(src, maketabdesc([
        makescacoldesc("TIME", 0.0),
        makearrcoldesc("DATA", 0j, valuetype="complex", shape=[6, 4]),
    ]), 7, ack=False)
    data = np.arange(7 * 6 * 4, dtype=np.float32).reshape(7, 6, 4) + 2j
    t.putcol("TIME", np.arange(7, dtype=np.float64))
    t.putcol("DATA", data)
    t.flush()
    t.close()

    dm = {"*1": {"NAME": "tiled", "TYPE": "TiledColumnStMan",
                 "SPEC": {"DEFAULTTILESHAPE": [4, 6, 3]}, "COLUMNS": ["DATA"]}}
    tablecopy(src, dst, deep=True, dminfo=dm)

    r = table(dst, readonly=True, ack=False)
    tiled = [v for v in r.getdminfo().values() if v["TYPE"] == "TiledColumnStMan"]
    assert len(tiled) == 1
    assert tiled[0]["NAME"] == "tiled"
    assert tiled[0]["COLUMNS"] == ["DATA"]
    assert tiled[0]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 3]
    np.testing.assert_array_equal(r.getcol("TIME"), np.arange(7, dtype=np.float64))
    np.testing.assert_array_equal(r.getcol("DATA"), data)
    r.close()


def test_table_copy_method_with_dminfo(tmp_path):
    """`t.copy(newname, deep=True, dminfo=...)` — the method form of the
    same conversion."""
    src = str(tmp_path / "src.tab")
    dst = str(tmp_path / "dst.tab")
    t = table(src, maketabdesc([
        makearrcoldesc("FLAG", False, shape=[4, 6]),
    ]), 5, ack=False)
    flag = np.zeros((5, 4, 6), dtype=bool)
    flag[1] = True
    t.putcol("FLAG", flag)
    t.flush()
    t.copy(dst, deep=True, dminfo={
        "*1": {"NAME": "tiledf", "TYPE": "TiledShapeStMan",
               "SPEC": {"DEFAULTTILESHAPE": [4, 6, 2]}, "COLUMNS": ["FLAG"]}})
    t.close()
    r = table(dst, readonly=True, ack=False)
    info = r.getdminfo()
    assert info["*1"]["TYPE"] == "TiledShapeStMan"
    assert info["*1"]["COLUMNS"] == ["FLAG"]
    np.testing.assert_array_equal(r.getcol("FLAG"), flag)
    r.close()


def test_makedminfo_output_round_trips_through_creation(tmp_path):
    """`makedminfo(tabdesc, group_spec)` — the dask-ms shape, records with
    COLUMNS — applied at creation reports the same managers back."""
    d = str(tmp_path / "rt.tab")
    cd = [
        makearrcoldesc("DATA", 0j, valuetype="complex", shape=[6, 4]),
        makearrcoldesc("FLAG", False, shape=[6, 4]),
    ]
    desc = maketabdesc(cd)
    desc["DATA"]["dataManagerType"] = "TiledColumnStMan"
    desc["DATA"]["dataManagerGroup"] = "DataGroup"
    group_spec = {"DataGroup": {"DEFAULTTILESHAPE": [4, 6, 4]}}
    dm = makedminfo(desc, group_spec)
    t = table(d, desc, nrow=9, dminfo=dm, ack=False)
    info = t.getdminfo()
    tiled = [v for v in info.values() if v["TYPE"] == "TiledColumnStMan"]
    assert len(tiled) == 1
    assert tiled[0]["NAME"] == "DataGroup"
    assert tiled[0]["COLUMNS"] == ["DATA"]
    assert tiled[0]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 4]
    t.close()


def test_requested_tile_survives_reopen_and_rewrite(tmp_path):
    """A reopened writable handle regenerates a written column with the
    tiling it was created with, not the derived shape (the descriptor
    re-derives its tile shape from the stored TSM header on open)."""
    p = str(tmp_path / "t.tab")
    dm = {"*1": {"NAME": "tiled", "TYPE": "TiledColumnStMan",
                 "SPEC": {"DEFAULTTILESHAPE": [4, 6, 3]}, "COLUMNS": ["DATA"]}}
    t = table(p, maketabdesc([
        makearrcoldesc("DATA", 0j, valuetype="complex", shape=[6, 4]),
    ]), 12, dminfo=dm, ack=False)
    t.flush()
    t.close()

    t = table(p, readonly=False, ack=False)
    data = np.arange(12 * 6 * 4, dtype=np.float32).reshape(12, 6, 4) + 3j
    t.putcol("DATA", data)
    t.flush()
    assert t.getdminfo()["*1"]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 3]
    t.close()
    r = table(p, readonly=True, ack=False)
    assert r.getdminfo()["*1"]["SPEC"]["DEFAULTTILESHAPE"] == [4, 6, 3]
    np.testing.assert_array_equal(r.getcol("DATA"), data)
    r.close()


def test_dminfo_fails_loudly(tmp_path):
    """An unsupported manager, a SPEC field creation cannot honour, and a
    named column the description lacks all raise — silence is the bug."""
    desc = maketabdesc([makearrcoldesc("DATA", 0j, valuetype="complex", shape=[6, 4])])

    with pytest.raises(ValueError, match="unsupported data-manager type"):
        table(str(tmp_path / "a.tab"), desc, 0,
              dminfo={"TYPE": "TiledDataStMan"}, ack=False)

    with pytest.raises(ValueError, match="HYPERCUBES"):
        table(str(tmp_path / "b.tab"), desc, 0,
              dminfo={"TYPE": "TiledShapeStMan",
                      "SPEC": {"HYPERCUBES": {"*1": {}}}}, ack=False)

    with pytest.raises(ValueError, match="DEFAULTTILESHAPE"):
        table(str(tmp_path / "c.tab"), desc, 0,
              dminfo={"TYPE": "StandardStMan",
                      "SPEC": {"DEFAULTTILESHAPE": [4, 6, 5]}}, ack=False)

    with pytest.raises(KeyError, match="NOSUCH"):
        table(str(tmp_path / "d.tab"), desc, 0,
              dminfo={"*1": {"TYPE": "TiledShapeStMan", "COLUMNS": ["NOSUCH"]}},
              ack=False)

    with pytest.raises(ValueError, match="bad SPEC.DEFAULTTILESHAPE"):
        table(str(tmp_path / "e.tab"), desc, 0,
              dminfo={"*1": {"TYPE": "TiledShapeStMan", "COLUMNS": ["DATA"],
                             "SPEC": {"DEFAULTTILESHAPE": [0, 4, 5]}}}, ack=False)


def test_getcell_conforms_to_declared_ndim(tmp_path):
    """python-casacore's `getcell` conforms a cell to the column's declared
    ndim by dropping leading length-1 axes: a (1, N) cell stored in an
    NDIM=1 column reads back as (N,), while `getvarcol` keeps the stored
    (1, N) shape. dask-ms's exemplar check compares `getcell` against the
    descriptor's ndim — the mismatch dropped CHAN_FREQ from reads (issue
    #19)."""
    d = str(tmp_path / "sw.tab")
    q = """
    CREATE TABLE %s
    [NUM_CHAN I4,
     CHAN_FREQ R8 [NDIM=1]]
    LIMIT 3
    """ % d
    from casacore.tables import taql

    freqs = [np.arange(8, dtype=np.float64), np.arange(16, dtype=np.float64),
             np.arange(32, dtype=np.float64)]
    with taql(q) as spw:
        spw.putvarcol("NUM_CHAN", {f"r{i}": s.shape[0] for i, s in enumerate(freqs)})
        spw.putvarcol("CHAN_FREQ", {f"r{i}": s[None, :] for i, s in enumerate(freqs)})

    t = table(d, readonly=True, ack=False)
    for r, want in enumerate(freqs):
        got = t.getcell("CHAN_FREQ", r)
        assert got.shape == want.shape, f"getcell row {r}"
        np.testing.assert_array_equal(got, want)
        stored = t.getvarcol("CHAN_FREQ")[f"r{r + 1}"]
        assert stored.shape == (1, want.shape[0]), f"getvarcol row {r}"
        np.testing.assert_array_equal(stored[0], want)
    # A conforming cell (its ndim already equals the declaration) is untouched.
    np.testing.assert_array_equal(t.getcell("NUM_CHAN", 0), 8)
    t.close()


def test_tile_shape_deferred_on_a_zero_row_variable_shape_column(tmp_path):
    """A DEFAULTTILESHAPE on a variable-shape column created at 0 rows
    cannot be laid out against a cell shape that does not exist yet —
    skarabina's flag-version tables are exactly this (a `TiledShapeStMan`
    FLAG column declared `ndim: 2` with no shape, a dminfo whose tile
    shape is shorter than the eventual cell's, `nrow=0`), and the create
    used to fail. casacore defers the hypercube until the first
    `setShape`; the create now writes the derived layout the same way and
    the first write lands through it."""
    d = str(tmp_path / "flagversion.tab")
    nchan = 64
    flag_desc = {"valueType": "boolean", "ndim": 2, "_c_order": True,
                 "dataManagerType": "TiledShapeStMan",
                 "dataManagerGroup": "TiledFlag"}
    tabdesc = maketabdesc([makecoldesc("FLAG", flag_desc),
                           makecoldesc("FLAG_ROW", {"valueType": "boolean"})])
    dminfo = {"TiledFlag": {"TYPE": "TiledShapeStMan", "NAME": "TiledFlag",
                            "SEQNR": 0,
                            "SPEC": {"DEFAULTTILESHAPE": np.array([nchan, 1], dtype=np.int32)},
                            "COLUMNS": ["FLAG"]}}
    t = table(d, tabdesc, nrow=0, readonly=False, dminfo=dminfo, ack=False)
    t.addrows(37)
    flag = np.arange(37 * nchan * 2).reshape(37, nchan, 2) % 3 == 0
    t.putcol("FLAG", flag)
    t.putcol("FLAG_ROW", np.arange(37) % 5 == 0)
    t.flush()
    t.close()

    r = table(d, readonly=True, ack=False)
    assert r.nrows() == 37
    np.testing.assert_array_equal(r.getcol("FLAG"), flag)
    np.testing.assert_array_equal(r.getcol("FLAG_ROW"), np.arange(37) % 5 == 0)
    r.close()


def test_addcols_cell_shaped_tile_without_row_axis(tmp_path):
    """casacure#21: dask-ms's add-columns path asks `addcols` for a
    TiledShapeStMan with `DEFAULTTILESHAPE` = the cell shape and no rows
    axis.  casacore treats the request as a hint (`adjustTileShape`: missing
    axes default to 1, each axis is clipped to the cube), so the write and the
    flush must succeed and round-trip the data -- it used to fail at flush with
    `unsupported tile shape [4, 2] for cells of shape [2, 4]`."""
    p = str(tmp_path / "t.tab")
    t = table(p, maketabdesc([
        makearrcoldesc("DATA", 0j, valuetype="complex", shape=[4, 2]),
    ]), 24, ack=False)
    values = (np.arange(24 * 4 * 2, dtype=np.float32).reshape(24, 4, 2) + 1j).astype(np.complex64)
    t.putcol("DATA", values)
    cell = t.getcell("DATA", 0)
    t.removecols("DATA")
    t.addcols(
        maketabdesc(makearrcoldesc("DATA", [], ndim=cell.ndim,
                                   shape=list(cell.shape), valuetype="complex")),
        {"TYPE": "TiledShapeStMan", "NAME": "TiledData",
         "SPEC": {"DEFAULTTILESHAPE": np.array(cell.shape, dtype=np.int32)}},
    )
    t.putcol("DATA", values)
    t.close()
    r = table(p, readonly=True, ack=False)
    assert list(r.getcoldesc("DATA")["shape"]) == [4, 2]
    np.testing.assert_array_equal(r.getcol("DATA"), values)
    r.close()
