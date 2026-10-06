"""DDFacet + killMS usage replay against casacure.

This suite exercises the exact call patterns `../DDFacet` and `../killMS`
subject casacore to, with each test citing the source file:line it mirrors.
See `PORTING_DDFACET_KILLMS.md` for the audit that produced it.

Run:
    PYTHONPATH=tests/shim python -m pytest tests/test_ddfacet_killms_usage.py -q

The suite builds a small Measurement Set with a `DATA` column (the MS
convention: DATA shape is `(nchan, ncorr)`), populates the subtables
DDFacet/killMS read (`SPECTRAL_WINDOW`, `FIELD`, `ANTENNA`, `POLARIZATION`,
`DATA_DESCRIPTION`), and replays:

  A. `GiveMainTable` — `query(...)` + `sort("TIME")`
     (DDFacet Data/ClassMS.py:230-234)
  B. chunked `getcol` / `getcolslicenp` / `getcolslice` with the
     `cs_tlc/cs_brc/cs_inc` tuple-slice convention
     (DDFacet Data/ClassMS.py:1022-1032, 612-668)
  C. column management — `getcoldesc` -> `addcols` (IMAGING_WEIGHT-shaped
     ColDesc), `removecols` (killMS Data/ClassMS.py:1102-1130,
     DDFacet Data/ClassMS.py:1457-1477)
  D. `addImagingColumns` / `removeImagingColumns`
     (killMS Data/ClassMS.py:1211-1213 `PutCasaCols`)
  E. subtable linkage — `getkeyword` + `::SUBTABLE` and `/SUBTABLE` opens
     (killMS Data/ClassMS.py:745-797, DDFacet Data/ClassMS.py:899-918)
  F. weight/flag write path — `putcol` partial windows
     (killMS kMS.py:1108-1140, ClipCal.py)
  G. `query("FIELD_ID==%d")` and chained `sort`
     (DDFacet Imager/ClassWeighting.py:49)
  H. quanta call forms incl. the ISO8601 date-string risk item
     (DDFacet Data/ClassMS.py:1178-1183, Data/PointingProvider.py:108)
  I. measures call forms (ClassFITSBeam.evaluateBeam/getBeamSampleTimes,
     GiveDate) — pinned to the values probed from real casacore
"""

import numpy as np
import pytest

from casacore.tables import (
    addImagingColumns,
    default_ms,
    makearrcoldesc,
    maketabdesc,
    removeImagingColumns,
    table,
)

# ---------------------------------------------------------------------------
# Fixture: a small MS with a DATA column and populated subtables.
# ---------------------------------------------------------------------------

NROW = 6          # 3 times x 2 baselines
NCHAN = 4
NCORR = 2
TIME = np.repeat([1000.0, 1001.0, 1002.0], 2)


@pytest.fixture
def ms(tmp_path):
    """A minimal MS: DATA (nchan, ncorr), SPECTRAL_WINDOW, FIELD, ANTENNA,
    POLARIZATION, DATA_DESCRIPTION, and `NROW` main-table rows."""
    p = str(tmp_path / "t.ms")
    # MS convention: DATA shape is (nchan, ncorr) — see
    # PORTING_DDFACET_KILLMS.md §3.
    data_desc = {
        "DATA": {
            "valueType": "complex",
            "dataManagerType": "TiledColumnStMan",
            "dataManagerGroup": "DATA_GROUP",
            "option": 4,
            "maxlen": 0,
            "comment": "The data column",
            "ndim": 2,
            "shape": [NCHAN, NCORR],
            "_c_order": True,
            "keywords": {},
        }
    }
    default_ms(p, tabdesc=data_desc)

    spw = table(p + "/SPECTRAL_WINDOW", readonly=False, ack=False)
    spw.addrows(1)
    spw.putcol("NUM_CHAN", [NCHAN])
    spw.putcol("CHAN_FREQ", np.arange(NCHAN).reshape(1, NCHAN) * 1e6)
    spw.putcol("CHAN_WIDTH", np.ones((1, NCHAN)))
    spw.close()

    fld = table(p + "/FIELD", readonly=False, ack=False)
    fld.addrows(1)
    fld.putcol("PHASE_DIR", np.array([[[0.3, -0.5]]]))
    fld.putcol("DELAY_DIR", np.array([[[0.3, -0.5]]]))
    fld.putcol("REFERENCE_DIR", np.array([[[0.3, -0.5]]]))
    fld.putcol("NAME", ["src"])
    fld.close()

    ant = table(p + "/ANTENNA", readonly=False, ack=False)
    ant.addrows(2)
    ant.putcol("NAME", ["a0", "a1"])
    ant.putcol("POSITION", np.zeros((2, 3)))
    ant.close()

    pol = table(p + "/POLARIZATION", readonly=False, ack=False)
    pol.addrows(1)
    pol.putcol("NUM_CORR", [NCORR])
    pol.putcol("CORR_TYPE", np.array([[9, 12]]))
    pol.putcol("CORR_PRODUCT", np.array([[[0, 0], [1, 1]]]))
    pol.close()

    ddid = table(p + "/DATA_DESCRIPTION", readonly=False, ack=False)
    ddid.addrows(1)
    ddid.putcol("SPECTRAL_WINDOW_ID", [0])
    ddid.putcol("POLARIZATION_ID", [0])
    ddid.close()

    rng = np.random.default_rng(3)
    t = table(p, readonly=False, ack=False)
    t.addrows(NROW)
    t.putcol("TIME", TIME)
    t.putcol("TIME_CENTROID", TIME)
    t.putcol("ANTENNA1", np.zeros(NROW, dtype=np.int32))
    t.putcol("ANTENNA2", np.ones(NROW, dtype=np.int32))
    t.putcol("UVW", rng.normal(size=(NROW, 3)))
    t.putcol("WEIGHT", np.ones((NROW, NCORR), dtype=np.float32))
    t.putcol("SIGMA", np.ones((NROW, NCORR), dtype=np.float32))
    t.putcol("FLAG", np.zeros((NROW, NCHAN, NCORR), dtype=bool))
    t.putcol(
        "DATA",
        (rng.normal(size=(NROW, NCHAN, NCORR))
         + 1j * rng.normal(size=(NROW, NCHAN, NCORR))).astype(np.complex64),
    )
    t.putcol("FIELD_ID", np.zeros(NROW, dtype=np.int32))
    t.putcol("DATA_DESC_ID", np.zeros(NROW, dtype=np.int32))
    t.close()
    return p


# ---------------------------------------------------------------------------
# A. GiveMainTable: query + sort("TIME")
#    (DDFacet Data/ClassMS.py:230-234)
# ---------------------------------------------------------------------------

def test_give_main_table_query_and_sort(ms):
    t = table(ms, ack=False)
    # DDFacet Data/ClassMS.py:233 — t = t.query(self.TaQL)
    q = t.query("FIELD_ID==0 && DATA_DESC_ID==0")
    # DDFacet Data/ClassMS.py:234 — return t.sort("TIME")
    s = q.sort("TIME")
    got = s.getcol("TIME")
    assert np.all(np.diff(got) >= 0), "sort('TIME') did not order"
    assert s.nrows() == NROW
    t.close()


def test_give_main_table_sort_preserves_rows(ms):
    # A shuffled TIME column must come back ordered with the same multiset.
    t = table(ms, readonly=False, ack=False)
    shuffled = np.array([1002.0, 1000.0, 1001.0, 1000.0, 1002.0, 1001.0])
    t.putcol("TIME", shuffled)
    t.close()
    t = table(ms, ack=False)
    s = t.sort("TIME")
    got = s.getcol("TIME")
    assert np.all(np.diff(got) >= 0)
    assert sorted(got.tolist()) == sorted(shuffled.tolist())
    t.close()


# ---------------------------------------------------------------------------
# B. Chunked column reads (DDFacet Data/ClassMS.py:612-668, 1022-1032)
# ---------------------------------------------------------------------------

def test_getcol_chunked_reads(ms):
    t = table(ms, ack=False)
    # DDFacet Data/ClassMS.py:563-569 — getcol(name, row0, nRowRead)
    for name, shape in [
        ("ANTENNA1", (NROW,)),
        ("ANTENNA2", (NROW,)),
        ("TIME", (NROW,)),
        ("UVW", (NROW, 3)),
        ("WEIGHT", (NROW, NCORR)),
        ("FLAG", (NROW, NCHAN, NCORR)),
        ("DATA", (NROW, NCHAN, NCORR)),
    ]:
        full = t.getcol(name)
        assert full.shape == shape, name
        # Chunked read concatenates to the whole.
        chunks = []
        for row0 in range(0, NROW, 2):
            chunks.append(t.getcol(name, row0, 2))
        assert np.allclose(np.concatenate(chunks), full), name
    t.close()


def test_getcolslicenp_tuple_slice(ms):
    """DDFacet Data/ClassMS.py:1022-1032 builds cs_tlc/cs_brc/cs_inc as
    tuples and calls getcolslicenp(col, buf, cs_tlc, cs_brc, cs_inc, row0,
    nrow) (Data/ClassMS.py:612-668)."""
    t = table(ms, ack=False)
    data = t.getcol("DATA")
    # cs_tlc=(chan0, 0), cs_brc=(chan1, ncorr-1), cs_inc=(step, 1).
    # blc/trc are inclusive corners; inc is the per-axis step.
    chan0, chan1, step = 0, 3, 2  # channels 0, 2
    nsel = len(range(chan0, chan1 + 1, step))
    buf = np.zeros((NROW, nsel, NCORR), dtype=np.complex64)
    t.getcolslicenp("DATA", buf, (chan0, 0), (chan1, NCORR - 1), (step, 1), 0, NROW)
    expected = data[:, chan0 : chan1 + 1 : step, :]
    assert buf.shape == expected.shape
    assert np.allclose(buf, expected)
    t.close()


def test_getcolslicenp_with_row_window(ms):
    t = table(ms, ack=False)
    data = t.getcol("DATA")
    buf = np.zeros((3, 2, NCORR), dtype=np.complex64)
    t.getcolslicenp("DATA", buf, (0, 0), (1, NCORR - 1), (1, 1), 0, 3)
    assert np.allclose(buf, data[0:3, 0:2, :])
    t.close()


def test_getcolslice_return_array(ms):
    # DDFacet Data/ClassMS.py:661 — getcolslice("FLAG", blc, trc, ...)
    t = table(ms, ack=False)
    got = t.getcolslice("FLAG", [0, 0], [NCHAN - 1, NCORR - 1], [1, 1], 0, NROW)
    assert got.shape == (NROW, NCHAN, NCORR)
    assert np.array_equal(got, t.getcol("FLAG"))
    t.close()


# ---------------------------------------------------------------------------
# C. Column management (killMS Data/ClassMS.py:1102-1130, DDFacet 1457-1477)
# ---------------------------------------------------------------------------

def test_addcol_clone_of_data(ms):
    """killMS Data/ClassMS.py:1115-1119 — desc = t.getcoldesc(LikeCol);
    desc['name'] = ColName; t.addcols(desc)."""
    t = table(ms, readonly=False, ack=False)
    desc = t.getcoldesc("DATA")
    desc["name"] = "CORRECTED_DATA"
    desc["comment"] = desc["comment"].replace(" ", "_")
    t.addcols(desc)
    t.close()

    t = table(ms, ack=False)
    assert "CORRECTED_DATA" in t.colnames()
    d = t.getcoldesc("CORRECTED_DATA")
    assert d["valueType"] == "complex"
    assert list(d["shape"]) == [NCHAN, NCORR]
    # Round-trip a value.
    t2 = table(ms, readonly=False, ack=False)
    v = (np.arange(NROW * NCHAN * NCORR).reshape(NROW, NCHAN, NCORR)
         + 0j).astype(np.complex64)
    t2.putcol("CORRECTED_DATA", v)
    t2.close()
    t3 = table(ms, ack=False)
    assert np.allclose(t3.getcol("CORRECTED_DATA"), v)
    t3.close()
    t.close()


def test_addcol_imaging_weight_shaped_desc(ms):
    """killMS Data/ClassMS.py:1116-1127 — the IMAGING_WEIGHT-shaped ColDesc
    (option 4, shape [nchan], float) passed to addcols."""
    t = table(ms, readonly=False, ack=False)
    col_desc = {
        "_c_order": True,
        "comment": "",
        "name": "IMAGING_WEIGHT",
        "dataManagerGroup": "imagingweight",
        "dataManagerType": "TiledShapeStMan",
        "maxlen": 0,
        "ndim": 1,
        "option": 4,
        "shape": np.array([NCHAN], dtype=np.int32),
        "valueType": "float",
    }
    t.addcols(col_desc)
    t.close()
    t = table(ms, ack=False)
    assert "IMAGING_WEIGHT" in t.colnames()
    d = t.getcoldesc("IMAGING_WEIGHT")
    assert d["valueType"] == "float"
    assert d["ndim"] == 1
    t.close()


def test_removecols(ms):
    """DDFacet Data/ClassMS.py:1457-1477 — AddCol, then the commented
    removecols overwrite path."""
    t = table(ms, readonly=False, ack=False)
    desc = t.getcoldesc("DATA")
    desc["name"] = "TEMP"
    t.addcols(desc)
    t.close()
    t = table(ms, readonly=False, ack=False)
    assert "TEMP" in t.colnames()
    t.removecols(["TEMP"])
    t.close()
    t = table(ms, ack=False)
    assert "TEMP" not in t.colnames()
    assert "DATA" in t.colnames()
    t.close()


# ---------------------------------------------------------------------------
# D. addImagingColumns / removeImagingColumns
#    (killMS Data/ClassMS.py:1211-1213 PutCasaCols)
# ---------------------------------------------------------------------------

def test_add_imaging_columns(ms):
    t = table(ms, ack=False)
    assert "MODEL_DATA" not in t.colnames()
    t.close()
    added = addImagingColumns(ms, ack=False)
    assert set(added) == {"MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"}
    t = table(ms, ack=False)
    for name in added:
        assert name in t.colnames(), name
    # IMAGING_WEIGHT is float, shape [nchan].
    d = t.getcoldesc("IMAGING_WEIGHT")
    assert d["valueType"] == "float"
    assert list(d["shape"]) == [NCHAN]
    # MODEL_DATA's CHANNEL_SELECTION = [[0, nchan]] (one SPW).
    cs = t.getcolkeywords("MODEL_DATA")["CHANNEL_SELECTION"]
    assert np.array_equal(np.asarray(cs), np.array([[0, NCHAN]], dtype=np.int32))
    t.close()


def test_add_imaging_columns_idempotent(ms):
    addImagingColumns(ms, ack=False)
    added = addImagingColumns(ms, ack=False)
    assert added == [], "second call re-added columns"
    t = table(ms, ack=False)
    assert sum(1 for n in t.colnames() if n == "IMAGING_WEIGHT") == 1
    t.close()


def test_remove_imaging_columns(ms):
    addImagingColumns(ms, ack=False)
    removed = removeImagingColumns(ms)
    assert set(removed) == {"MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"}
    t = table(ms, ack=False)
    for name in removed:
        assert name not in t.colnames(), name
    assert "DATA" in t.colnames()
    t.close()


# ---------------------------------------------------------------------------
# E. Subtable linkage (killMS Data/ClassMS.py:745-797, DDFacet 899-918)
# ---------------------------------------------------------------------------

def test_subtable_keywords_open_subtables(ms):
    t = table(ms, ack=False)
    for sub, col in [
        ("SPECTRAL_WINDOW", "NUM_CHAN"),
        ("FIELD", "PHASE_DIR"),
        ("ANTENNA", "NAME"),
        ("POLARIZATION", "CORR_TYPE"),
        ("DATA_DESCRIPTION", "SPECTRAL_WINDOW_ID"),
    ]:
        kw = t.getkeyword(sub)
        assert kw.startswith("Table:"), (sub, kw)
        st = table(kw, ack=False)
        assert col in st.colnames(), (sub, col)
        st.close()
    t.close()


def test_subtable_colon_opens(ms):
    """killMS Weights/W_ImagCov.py:157 — table('%s::SPECTRAL_WINDOW' % ms)."""
    t = table(ms + "::SPECTRAL_WINDOW", ack=False)
    assert "NUM_CHAN" in t.colnames()
    t.close()


def test_subtable_slash_opens(ms):
    """DDFacet Data/ClassMS.py:176 — table(ms + '/OBSERVATION')."""
    t = table(ms + "/OBSERVATION", ack=False)
    assert "TELESCOPE_NAME" in t.colnames()
    t.close()


# ---------------------------------------------------------------------------
# F. Weight/flag write path (killMS kMS.py:1108-1140, ClipCal.py)
# ---------------------------------------------------------------------------

def test_putcol_partial_window(ms):
    """killMS kMS.py:1108-1111 — t.putcol('IMAGING_WEIGHT', W, row0, nrow)."""
    addImagingColumns(ms, ack=False)
    t = table(ms, readonly=False, ack=False)
    w = np.ones((2, NCHAN), dtype=np.float32) * 7.0
    t.putcol("IMAGING_WEIGHT", w, 0, 2)
    t.close()
    t = table(ms, ack=False)
    got = t.getcol("IMAGING_WEIGHT")
    assert got.shape == (NROW, NCHAN)
    assert np.allclose(got[0:2], w)
    assert np.allclose(got[2:], 0.0)
    t.close()


def test_flag_roundtrip(ms):
    """killMS Simul/DoSimul.py:643-651 — f = t.getcol('FLAG'); t.putcol('FLAG', f)."""
    t = table(ms, readonly=False, ack=False)
    f = t.getcol("FLAG")
    f[0, 0, 0] = True
    f[3, 2, 1] = True
    t.putcol("FLAG", f)
    t.close()
    t = table(ms, ack=False)
    got = t.getcol("FLAG")
    assert got[0, 0, 0] and got[3, 2, 1]
    assert got.sum() == 2
    t.close()


# ---------------------------------------------------------------------------
# G. Query + sort (DDFacet Imager/ClassWeighting.py:49)
# ---------------------------------------------------------------------------

def test_query_field_id(ms):
    """DDFacet Imager/ClassWeighting.py:49 —
    t = table(ms, ack=False).query('FIELD_ID==%d' % field)"""
    t = table(ms, ack=False)
    q = t.query("FIELD_ID==%d" % 0)
    assert q.nrows() == NROW
    assert np.all(q.getcol("FIELD_ID") == 0)
    q2 = t.query("FIELD_ID==%d" % 1)
    assert q2.nrows() == 0
    t.close()


# ---------------------------------------------------------------------------
# H. Quanta call forms (DDFacet Data/ClassMS.py:1178-1183, PointingProvider)
# ---------------------------------------------------------------------------

def test_quantity_unix_time_ms_time_range():
    """DDFacet Data/ClassMS.py:1178-1183 —
    dt.utcfromtimestamp(qa.quantity(x).to_unix_time())
    qa.quantity('{}s'.format(x)).to_unix_time()"""
    from casacore import quanta as qa
    # Plain float seconds -> unix time (the TimeRange selection).
    x = 1000.0
    assert qa.quantity("%ss" % x).to_unix_time() == qa.quantity(x, "s").to_unix_time()
    # A quantity in days -> unix time.
    assert qa.quantity(51544.0, "d").to_unix_time() == 946684800.0


def test_quantity_iso_date_string():
    """DDFacet Parset TimeRange: ISO8601 date strings —
    qa.quantity('2017-04-01T12:00:00') -> an MJD-day quantity."""
    from casacore import quanta as qa
    q = qa.quantity("2017-04-01T12:00:00")
    assert q.get_unit() == "d"
    # 2017-04-01 12:00 UTC = MJD 57844.5.
    assert abs(q.get_value() - 57844.5) < 1e-9
    # Date-only form.
    q2 = qa.quantity("2017-04-01")
    assert abs(q2.get_value() - 57844.0) < 1e-9


# ---------------------------------------------------------------------------
# I. Measures call forms (ClassFITSBeam.evaluateBeam, GiveDate)
# ---------------------------------------------------------------------------

def test_measures_give_date_pattern():
    """ClassMS.GiveDate (both repos):
    time_start = qa.quantity(tt, 's'); me = pm.measures()
    dict_time_start_MDJ = me.epoch('utc', time_start)
    JD = dict_time_start_MDJ['m0']['value'] + 2400000.5 - 2415020"""
    from casacore import measures as pm
    from casacore import quanta as qa
    tt = 1000.0
    me = pm.measures()
    e = me.epoch("utc", qa.quantity(tt, "s"))
    assert e["type"] == "epoch"
    assert e["refer"] == "UTC"
    # casacore: epoch('utc', q) treats q as days since MJD 0.
    mjd_days = e["m0"]["value"]
    assert abs(mjd_days - tt / 86400.0) < 1e-12
    assert e["m0"]["unit"] == "d"
    # The GiveDate JD computation.
    jd = mjd_days + 2400000.5 - 2415020
    assert abs(jd - (tt / 86400.0 + 2400000.5 - 2415020)) < 1e-9


def test_measures_direction_and_posangle():
    """ClassFITSBeam.evaluateBeam (DDFacet Data/ClassFITSBeam.py:431-433):
    dm.do_frame(self.pos0) and dm.do_frame(dm.epoch('UTC', dq.quantity(t0,'s')))
    dm.posangle(self.pointing_centre, self.zenith).get_value('deg')"""
    from casacore import measures as pm
    from casacore import quanta as qa
    me = pm.measures()
    # ITRF position (metres) and J2000 direction.
    me.do_frame(
        me.position("itrf", *[qa.quantity(x, "m") for x in (6100000.0, 100000.0, 100000.0)])
    )
    me.do_frame(me.epoch("UTC", qa.quantity(57844.5, "d")))
    src = me.direction("J2000", qa.quantity(0.3, "rad"), qa.quantity(-0.5, "rad"))
    zen = me.direction("AZELGEO", qa.quantity(0, "deg"), qa.quantity(90, "deg"))
    pa = me.posangle(src, zen)
    deg = pa.get_value("deg")
    # Pinned to real casacore at this position/epoch (probed live).
    # casacure 3.8.25 gives -12.876001978, 0.55 arcsec away; the tolerance is
    # 1.8 arcsec (MEASURES_ACCURACY.md holds the model to <1" of astropy).
    assert abs(deg - (-12.875850547)) < 5e-4, deg


def test_measures_direction_to_azelgeo():
    """ClassFITSBeam.evaluateBeam:500-502 —
    dir_j2000 = dm.direction('J2000', ...)
    dir_azel = dm.measure(dir_j2000, 'AZELGEO')
    dir_azel_val = dm.get_value(dir_azel)"""
    from casacore import measures as pm
    from casacore import quanta as qa
    me = pm.measures()
    me.do_frame(
        me.position("itrf", *[qa.quantity(x, "m") for x in (6100000.0, 100000.0, 100000.0)])
    )
    me.do_frame(me.epoch("UTC", qa.quantity(57844.5, "d")))
    d = me.direction("J2000", qa.quantity(0.3, "rad"), qa.quantity(-0.5, "rad"))
    azel = me.measure(d, "AZELGEO")
    assert azel["type"] == "direction"
    assert azel["refer"] == "AZELGEO"
    vals = me.get_value(azel)
    assert len(vals) == 2
    # Pinned to real casacore at this position/epoch (probed live).
    # casacure 3.8.25 gives 2.945111616 / 1.044795251: 0.55" and 0.31" away.
    az = vals[0].get_value()
    alt = vals[1].get_value()
    assert abs(az - 2.945114291) < 2e-5, az
    assert abs(alt - 1.044796770) < 2e-5, alt


def test_measures_missing_frame_errors():
    """casacore raises 'Cannot convert due to missing frame information'
    when the frame is not set."""
    from casacore import measures as pm
    from casacore import quanta as qa
    me = pm.measures()
    d = me.direction("J2000", qa.quantity(0.3, "rad"), qa.quantity(-0.5, "rad"))
    with pytest.raises(Exception):
        me.measure(d, "AZELGEO")
