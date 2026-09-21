"""Extensive end-to-end tests of the core `casacore.tables` surface running on
casacure.

This suite drives the full table lifecycle through the public API — every
scalar dtype through the Standard and Incremental storage managers, fixed-
and variable-shape array columns (incl. the Tiled storage manager), string
length boundaries, row/column lifecycle, keyword round-trips, persistence
across reopen (the buffered-write contract), slices, and the documented
error paths. It mirrors python-casacore 3.8.1 semantics.

Run as part of the project suite (the `casacore` import must resolve to the
casacure shim):
    PYTHONPATH=tests/shim python -m pytest tests/test_core_tables_e2e.py -q
"""

import numpy as np
import pytest

from casacore.tables import (
    makearrcoldesc,
    makescacoldesc,
    maketabdesc,
    table,
)

# valueType -> (numpy dtype of the stored value, getcol dtype, sample values)
# USHORT is deliberately absent: casacore's storage managers reject it
# ("unknown data type 4"), so it is covered only at the mapping layer.
SCALAR_TYPES = [
    ("boolean", np.dtype(np.bool_), np.array([True, False, True])),
    ("uchar", np.dtype(np.uint8), np.array([0, 255, 7], dtype=np.uint8)),
    # uchar reads back promoted to uint16 (python-casacore quirk).
    ("short", np.dtype(np.int16), np.array([-32768, 32767, -300], dtype=np.int16)),
    ("int", np.dtype(np.int32), np.array([-2**31, 2**31 - 1, -70000], dtype=np.int32)),
    ("uint", np.dtype(np.uint32), np.array([0, 2**32 - 1, 7], dtype=np.uint32)),
    ("int64", np.dtype(np.int64), np.array([-2**63, 2**63 - 1, -9_000_000_000_000], dtype=np.int64)),
    ("float", np.dtype(np.float32), np.array([1.5, -1.5e38, 0.0], dtype=np.float32)),
    ("double", np.dtype(np.float64), np.array([1.5e300, -1.5e300, 0.0])),
    ("complex", np.dtype(np.complex64), np.array([1.5 + 2.5j, -3e38 - 2e38j, 0j], dtype=np.complex64)),
    ("dcomplex", np.dtype(np.complex128), np.array([1.5e300 + 2.5e300j, 0j, -1j])),
    ("string", None, np.array(["hello", "a very long string > 8 chars", ""])),
]
SCALAR_IDS = [vt for vt, _, _ in SCALAR_TYPES]


def _assert_eq(arr, expected):
    expected = np.asarray(expected)
    if expected.dtype.kind in "fc":
        assert arr.dtype == np.asarray(arr).dtype
        assert np.array_equal(np.asarray(arr), expected, equal_nan=True), (
            f"{np.asarray(arr)} != {expected}"
        )
    else:
        assert np.array_equal(np.asarray(arr), expected)


def _make_scalar_desc(name, vt, dmt=""):
    cd = makescacoldesc(name, 0.0, valuetype=vt)
    cd["desc"]["dataManagerType"] = dmt
    return maketabdesc([cd])


def _array_desc(name, vt, shape, dmt=""):
    # valuetype is explicit: makearrcoldesc cannot infer every type from a
    # python scalar sample value (e.g. np scalar values are not mapped).
    cd = makearrcoldesc(name, 0, 0, shape, datamanagertype=dmt, valuetype=vt)
    return maketabdesc([cd])


# ---------------------------------------------------------------------------
# Scalar dtype matrix: every dtype through StandardStMan and
# IncrementalStMan, with write -> same-handle read -> flush -> reopen read.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vt,dtype,values", SCALAR_TYPES, ids=SCALAR_IDS)
def test_scalar_roundtrip_standard(tmp_path, vt, dtype, values):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", vt), len(values))
    t.putcol("C", values)
    got = t.getcol("C")
    if vt == "uchar":
        # uchar columns read back promoted to uint16 (python-casacore parity).
        assert np.asarray(got).dtype == np.dtype(np.uint16)
        _assert_eq(got, values.astype(np.uint16))
    elif dtype is not None:
        assert np.asarray(got).dtype == dtype, f"dtype {np.asarray(got).dtype} != {dtype}"
        _assert_eq(got, values)
    else:
        assert list(got) == list(values)
    t.close()
    # Persistence across reopen (buffered write flushed on close).
    t2 = table(p)
    again = t2.getcol("C")
    if vt == "uchar":
        _assert_eq(again, values.astype(np.uint16))
    elif dtype is not None:
        _assert_eq(again, values)
    else:
        assert list(again) == list(values)
    t2.close()


# IncrementalStMan round-trips the numeric dtypes; STRING is excluded —
# casacure raises on a string ISM write (casacore's IncrementalStMan is
# numeric-only), asserted separately.
ISM_TYPES = [(vt, dt, v) for vt, dt, v in SCALAR_TYPES if vt != "string"]


@pytest.mark.parametrize("vt,dtype,values", ISM_TYPES, ids=[t[0] for t in ISM_TYPES])
def test_scalar_roundtrip_incremental(tmp_path, vt, dtype, values):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", vt, "IncrementalStMan"), len(values))
    t.putcol("C", values)
    got = t.getcol("C")
    if vt == "uchar":
        _assert_eq(got, values.astype(np.uint16))
    else:
        assert np.asarray(got).dtype == dtype
        _assert_eq(got, values)
    t.close()
    t2 = table(p)
    _assert_eq(t2.getcol("C"), got)
    t2.close()


def test_incremental_rejects_string(tmp_path):
    p = str(tmp_path / "t.tab")
    with pytest.raises(RuntimeError):
        table(p, _make_scalar_desc("C", "string", "IncrementalStMan"), 2)


# ---------------------------------------------------------------------------
# NaN / infinity / signed-zero fidelity through the storage layer.
# ---------------------------------------------------------------------------


def test_double_special_values_roundtrip(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "double"), 5)
    vals = np.array([np.nan, np.inf, -np.inf, -0.0, 1.5e300])
    t.putcol("C", vals)
    got = np.asarray(t.getcol("C"))
    assert np.isnan(got[0])
    assert got[1] == np.inf and got[2] == -np.inf
    assert np.signbit(got[3]) and got[3] == 0.0  # -0.0 preserved
    assert got[4] == 1.5e300
    t.close()
    got2 = np.asarray(table(p).getcol("C"))
    assert np.isnan(got2[0]) and got2[1] == np.inf and np.signbit(got2[3])
    table(p).close()


def test_int64_extreme_values_roundtrip(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "int64"), 3)
    vals = np.array([-2**63, 2**63 - 1, 123456789012345], dtype=np.int64)
    t.putcol("C", vals)
    assert np.array_equal(np.asarray(t.getcol("C")), vals)
    t.close()


# ---------------------------------------------------------------------------
# Array columns: fixed-shape (SSM + TSM) and variable-shape.
# ---------------------------------------------------------------------------

ARRAY_NUMERIC = [
    ("boolean", np.dtype(np.bool_), np.bool_),
    ("short", np.dtype(np.int16), np.int16),
    ("int", np.dtype(np.int32), np.int32),
    ("uint", np.dtype(np.uint32), np.uint32),
    ("int64", np.dtype(np.int64), np.int64),
    ("float", np.dtype(np.float32), np.float32),
    ("double", np.dtype(np.float64), np.float64),
    ("complex", np.dtype(np.complex64), np.complex64),
    ("dcomplex", np.dtype(np.complex128), np.complex128),
]


@pytest.mark.parametrize("vt,dtype,_", ARRAY_NUMERIC, ids=[v[0] for v in ARRAY_NUMERIC])
def test_fixed_shape_array_roundtrip(tmp_path, vt, dtype, _):
    """(n, 2, 3) int arrays write and read back shape- and value-exact."""
    p = str(tmp_path / "t.tab")
    n = 4
    t = table(p, _array_desc("A", vt, [2, 3]), n)
    data = _array_data(n * 6, dtype).reshape(n, 2, 3)
    t.putcol("A", data)
    got = np.asarray(t.getcol("A"))
    assert got.shape == (n, 2, 3)
    assert got.dtype == dtype
    assert np.array_equal(got, data)
    assert np.asarray(t.getcell("A", 1)).shape == (2, 3)
    t.close()
    # Reopen: persisted cells keep shape + values.
    t2 = table(p)
    assert np.array_equal(np.asarray(t2.getcol("A")), data)
    t2.close()


@pytest.mark.parametrize("vt,dtype,_", ARRAY_NUMERIC, ids=[v[0] for v in ARRAY_NUMERIC])
def test_tiled_array_roundtrip(tmp_path, vt, dtype, _):
    """TiledColumnStMan array columns (the dask-ms MS storage manager)."""
    p = str(tmp_path / "t.tab")
    n = 4
    t = table(p, _array_desc("A", vt, [2, 2], "TiledColumnStMan"), n)
    data = _array_data(n * 4, dtype).reshape(n, 2, 2)
    t.putcol("A", data)
    assert np.array_equal(np.asarray(t.getcol("A")), data)
    assert np.asarray(t.getcell("A", 2)).shape == (2, 2)
    t.close()
    t2 = table(p)
    assert np.array_equal(np.asarray(t2.getcol("A")), data)
    t2.close()


def test_variable_shape_arrays_roundtrip(tmp_path):
    """Variable-shape array column: per-row sizes differ and round-trip."""
    p = str(tmp_path / "t.tab")
    n = 3
    t = table(p, maketabdesc([makearrcoldesc("A", 0.0, 0, [])]), n)
    t.putvarcol("A", {"r0": [1.0, 2.0], "r1": [3.0, 4.0, 5.0], "r2": [6.0]})
    assert np.asarray(t.getcell("A", 0)).shape == (2,)
    assert np.asarray(t.getcell("A", 1)).shape == (3,)
    assert np.asarray(t.getcell("A", 2)).shape == (1,)
    assert np.asarray(t.getcell("A", 0)).tolist() == [1.0, 2.0]
    t.close()
    t2 = table(p)
    assert np.asarray(t2.getcell("A", 1)).shape == (3,)
    assert np.asarray(t2.getcell("A", 1)).tolist() == [3.0, 4.0, 5.0]
    t2.close()


def test_3d_array_and_varcol_keys(tmp_path):
    """3-D fixed arrays round-trip; getvarcol keys follow the r1.. convention."""
    p = str(tmp_path / "t.tab")
    t = table(p, maketabdesc([makearrcoldesc("A", 0, 0, [2, 2, 2])]), 2)
    data = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    t.putcol("A", data)
    assert np.asarray(t.getcol("A")).shape == (2, 2, 2, 2)
    rows = t.getvarcol("A")
    assert set(rows.keys()) == {"r1", "r2"}
    assert np.asarray(rows["r1"]).tolist() == data[0].tolist()
    t.close()


# ---------------------------------------------------------------------------
# String columns: length boundaries, fixed maxlen, array strings, unicode.
# ---------------------------------------------------------------------------


def test_string_length_boundaries(tmp_path):
    """0 / 8 / 9 / 20 / 300-byte and multi-byte UTF-8 strings survive."""
    p = str(tmp_path / "t.tab")
    vals = np.array(
        ["", "a", "1234567", "12345678", "123456789", "x" * 300, "日本語テキスト"],
        dtype=object,
    )
    t = table(p, _make_scalar_desc("S", "string"), len(vals))
    t.putcol("S", list(vals))
    got = np.asarray(t.getcol("S"), dtype=object)
    assert list(got) == list(vals)
    t.close()
    t2 = table(p)
    assert np.asarray(t2.getcol("S"), dtype=object).tolist() == list(vals)
    t2.close()


def _array_data(n, dtype):
    """A distinct n-value array of `dtype` (arange is illegal for bool > 2)."""
    if np.dtype(dtype) == np.dtype(np.bool_):
        return (np.arange(n) % 2 == 0).astype(bool)
    return np.arange(n, dtype=dtype)


def test_string_array_column(tmp_path):
    """String array cells come back in the casacore dict form
    `{'shape': [n, *cell], 'array': [...]}` (python-casacore parity)."""
    p = str(tmp_path / "t.tab")
    t = table(p, maketabdesc([makearrcoldesc("A", "x", 0, [2])]), 3)
    data = np.array([["ab", "cd"], ["", "a very long string"], ["日本語", "xy"]], dtype=object)
    t.putcol("A", _as_unicode(data))
    got = t.getcol("A")
    assert list(got["shape"]) == [3, 2]
    assert list(got["array"]) == data.ravel().tolist()
    assert np.asarray(t.getcell("A", 1)).shape == (2,)
    t.close()


def _as_unicode(a):
    out = np.empty(a.shape, dtype=object)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = a[i, j]
    return out


# ---------------------------------------------------------------------------
# Row ranges, single cells, buffers (getcolnp).
# ---------------------------------------------------------------------------


def test_row_ranges_and_cells(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "int"), 5)
    t.putcol("C", [10, 20, 30, 40, 50])
    assert np.asarray(t.getcol("C", startrow=1, nrow=3)).tolist() == [20, 30, 40]
    assert np.asarray(t.getcol("C", startrow=-2)).tolist() == [40, 50]
    assert t.getcell("C", 3) == 40
    t.putcell("C", 0, 99)
    assert t.getcell("C", 0) == 99
    # putcol with a row offset writes in place.
    t.putcol("C", [7, 8], startrow=2)
    assert np.asarray(t.getcol("C")).tolist() == [99, 20, 7, 8, 50]
    t.close()


def test_getcolnp_fills_buffer(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "double"), 3)
    t.putcol("C", [1.0, 2.0, 3.0])
    buf = np.empty(3)
    t.getcolnp("C", buf)
    assert buf.tolist() == [1.0, 2.0, 3.0]
    t.close()


@pytest.mark.parametrize(
    "vt,dtype", [("complex", "complex64"), ("dcomplex", "complex128"),
                 ("double", "float64"), ("float", "float32"),
                 ("int", "int32"), ("int64", "int64"),
                 ("bool", "bool")], ids=["complex", "dcomplex", "double", "float", "int", "int64", "bool"])
def test_getcolnp_array_matches_getcol(tmp_path, vt, dtype):
    """The typed-buffer getcolnp path (StandardStMan array columns) must
    agree exactly with getcol for the same rows."""
    p = str(tmp_path / "t.tab")
    t = table(p, _array_desc("A", vt, [2, 3]), 4)
    data = _array_data(24, dtype).reshape(4, 2, 3)
    t.putcol("A", data)
    expect = np.asarray(t.getcol("A"))
    buf = np.empty((4, 2, 3), dtype=dtype)
    t.getcolnp("A", buf)
    assert buf.dtype == expect.dtype
    assert np.array_equal(buf, expect), f"{vt}: typed getcolnp != getcol"
    # a row-range chunk through the typed path too
    sub = np.empty((2, 2, 3), dtype=dtype)
    t.getcolnp("A", sub, startrow=1, nrow=2)
    assert np.array_equal(sub, expect[1:3]), f"{vt}: range mismatch"
    t.close()


# ---------------------------------------------------------------------------
# Table and column keywords round-trip, including after reopen.
# ---------------------------------------------------------------------------


def test_table_keywords_roundtrip(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "int"), 1)
    t.putkeyword("VERSION", 1.5)
    t.putkeyword("NAME", "test")
    t.putkeyword("FLAG", True)
    assert t.getkeyword("VERSION") == 1.5
    assert t.getkeyword("NAME") == "test"
    assert t.getkeyword("FLAG") is True
    t.close()
    t2 = table(p)
    assert t2.getkeyword("VERSION") == 1.5
    assert t2.getkeyword("NAME") == "test"
    t2.close()


def test_column_keywords_roundtrip(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "double"), 2)
    t.putcolkeyword("C", "UNITS", "Jy")
    t.putcolkeyword("C", "SCALE", 3)
    kw = t.getcolkeywords("C")
    assert kw["UNITS"] == "Jy" and kw["SCALE"] == 3
    t.close()
    t2 = table(p)
    assert t2.getcolkeywords("C")["UNITS"] == "Jy"
    t2.close()


# ---------------------------------------------------------------------------
# Row / column lifecycle through reopen.
# ---------------------------------------------------------------------------


def test_addrows_then_persist(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "int"), 2)
    t.putcol("C", [1, 2])
    t.addrows(3)
    assert t.nrows() == 5
    t.putcol("C", [3, 4, 5], startrow=2)
    t.close()
    t2 = table(p)
    assert t2.nrows() == 5
    assert np.asarray(t2.getcol("C")).tolist() == [1, 2, 3, 4, 5]
    t2.close()


def test_addcols_removecols(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, maketabdesc([makescacoldesc("A", 0), makescacoldesc("B", 0.0)]), 2)
    t.putcol("A", [1, 2])
    t.putcol("B", [1.5, 2.5])
    assert t.colnames() == ["A", "B"]
    t.removecols(["A"])
    assert t.colnames() == ["B"]
    assert np.asarray(t.getcol("B")).tolist() == [1.5, 2.5]
    t.close()
    t2 = table(p)
    assert t2.colnames() == ["B"]
    assert np.asarray(t2.getcol("B")).tolist() == [1.5, 2.5]
    t2.close()


def test_removecol_then_reopen_preserves_others(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, maketabdesc([makescacoldesc("A", 0), makescacoldesc("B", 0)]), 2)
    t.putcol("A", [7, 8])
    t.putcol("B", [9, 10])
    t.removecol("A")
    t.close()
    t2 = table(p)
    assert t2.colnames() == ["B"]
    assert np.asarray(t2.getcol("B")).tolist() == [9, 10]
    t2.close()


# ---------------------------------------------------------------------------
# Documented error paths.
# ---------------------------------------------------------------------------

ERROR_CASES = [
    ("readonly putcol", "ro", ValueError, lambda t: t.putcol("C", [1.0, 2.0, 3.0])),
    ("out-of-range getcell", "w", ValueError, lambda t: t.getcell("C", 99)),
    ("wrong-dtype into double", "w", TypeError, lambda t: t.putcol("C", ["a", "b", "c"])),
    ("unknown column", "w", KeyError, lambda t: t.getcol("NOPE")),
]


@pytest.mark.parametrize(
    "label,mode,exc,op",
    ERROR_CASES,
    ids=[c[0].replace(" ", "_") for c in ERROR_CASES],
)
def test_error_paths(tmp_path, label, mode, exc, op):
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "double"), 3)
    t.putcol("C", [1.0, 2.0, 3.0])
    t.close()
    handle = table(p, readonly=(mode == "ro"))
    with pytest.raises(exc):
        op(handle)
    handle.close()


def test_fixed_shape_conformance_error(tmp_path):
    p = str(tmp_path / "t.tab")
    t = table(p, _array_desc("A", "int", [2, 3]), 2)
    with pytest.raises(ValueError):
        t.putcol("A", np.zeros((2, 3, 3), dtype=np.int32))  # wrong cell shape
    t.close()


def test_partial_putcol_leaves_defaults(tmp_path):
    """A putcol shorter than the row count fills the given rows; the rest
    hold the column's default (0.0), matching casacore's cell defaults."""
    p = str(tmp_path / "t.tab")
    t = table(p, _make_scalar_desc("C", "double"), 3)
    t.putcol("C", [1.0, 2.0])
    assert np.asarray(t.getcol("C")).tolist() == [1.0, 2.0, 0.0]
    t.close()
    t2 = table(p)
    assert np.asarray(t2.getcol("C")).tolist() == [1.0, 2.0, 0.0]
    t2.close()


def test_tiled_shape_stman_round_trip(tmp_path):
    """A TiledShapeStMan array column writes real shape-stman blocks and
    reads back (the manager real casacore MSs use for DATA/FLAG)."""
    desc = maketabdesc([
        makearrcoldesc("DATA", 0.0 + 0.0j, shape=[4, 2],
                       valuetype="dcomplex", datamanagertype="TiledShapeStMan"),
    ])
    t = table(str(tmp_path / "t.tab"), desc, nrow=3, ack=False)
    data = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2)
    data = data * (1.0 + 1.0j)
    t.putcol("DATA", data)
    t.flush()
    t.close()

    t = table(str(tmp_path / "t.tab"), readonly=True, ack=False)
    assert t.getdminfo()["*1"]["TYPE"] == "TiledShapeStMan"
    np.testing.assert_array_equal(t.getcol("DATA"), data)
    t.close()
