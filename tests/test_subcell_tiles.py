"""Tiled columns whose tiles are smaller than the cell, cross-checked with
real python-casacore.

casacore tiles a hypercube in ``tileShape`` pieces over every axis, so a
row's cell can span several tiles (numbered first-axis-fastest over the tile
grid, each a full bucket even at the cube edge).  dask-ms tiles channel axes
at <= 64 channels, so every MS it writes stores DATA/FLAG/... that way; this
module builds such tables with real casacore, reads them with casacure
(getcol / getcell / getcolslice / getcolnp), writes them in place with
casacure (putcol / putcell / putcolslice), and verifies the result with real
casacore -- including that the tile shape survives (an in-place patch, not a
rebuild).  A dask-ms round trip does the same through
``DASK_MS_BACKEND=casacure``.

Real casacore runs in subprocesses with ``tests/shim`` stripped from
PYTHONPATH (the shim redirects ``casacore`` to casacure); the module is
skipped when python-casacore is not installed.
"""

import json
import os
import subprocess
import sys
import textwrap
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(ROOT, "tests", "shim")

ct = pytest.importorskip("casacure.tables")


def _env(casacure_backend):
    env = dict(os.environ)
    shim_real = os.path.realpath(SHIM)
    pp = [
        p
        for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.realpath(p) != shim_real
    ]
    if casacure_backend:
        pp.insert(0, SHIM)
        env["DASK_MS_BACKEND"] = "casacure"
    else:
        env["DASK_MS_BACKEND"] = ""
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


_BACKEND_CHECK = """
import casacore
_is_cure = ("casacure" in casacore.__file__) or ("shim" in casacore.__file__)
assert _is_cure == {want}, casacore.__file__
"""


def _run(script, *args, casacure_backend=False):
    code = _BACKEND_CHECK.format(want=casacure_backend) + textwrap.dedent(script)
    p = subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        env=_env(casacure_backend),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    return p.stdout


def _have_real_casacore():
    p = subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_env(False),
        capture_output=True,
    )
    return p.returncode == 0


pytestmark = pytest.mark.skipif(
    not _have_real_casacore(), reason="real python-casacore not installed"
)

NROW = 150  # three 64-row tile layers, the last partial

# name -> (valuetype, logical cell shape or None (variable), stman, tile
# shape in casacore (CASA) order incl. rows).
COLUMNS = {
    "FLAG": ("bool", (79, 2), "TiledColumnStMan", [2, 8, 64]),
    "DATA": ("complex", (79, 2), "TiledColumnStMan", [2, 8, 64]),
    "SDATA": ("complex", (79, 2), "TiledShapeStMan", [2, 8, 64]),
    "SFLAG": ("bool", (79, 2), "TiledShapeStMan", [2, 8, 64]),
    # padding on both cell axes (3 -> 2 tiles, 5 -> 3 tiles)
    "WS": ("float", (5, 3), "TiledShapeStMan", [2, 2, 3]),
    "DD": ("double", (5, 3), "TiledColumnStMan", [2, 2, 3]),
    # variable shape: alternating cell shapes -> two cubes in two tile files
    "VAR": ("int", None, "TiledShapeStMan", [2, 3, 4]),
}

_NP = {
    "bool": np.bool_,
    "complex": np.complex64,
    "float": np.float32,
    "double": np.float64,
    "int": np.int32,
}


def _var_shape(r):
    return (5, 2) if r % 2 == 0 else (7, 3)


def _values(name, gen):
    """Deterministic column values (a list of per-row arrays for VAR)."""
    vt, shape, _, _ = COLUMNS[name]
    rng = np.random.default_rng(zlib.crc32(f"{name}/{gen}".encode()))
    if shape is None:
        return [
            rng.integers(-1000, 1000, _var_shape(r)).astype(np.int32)
            for r in range(NROW)
        ]
    full = (NROW,) + shape
    if vt == "bool":
        return rng.random(full) < 0.4
    if vt == "complex":
        return (rng.standard_normal(full) + 1j * rng.standard_normal(full)).astype(
            np.complex64
        )
    return rng.standard_normal(full).astype(_NP[vt])


def _save(path, cols):
    flat = {}
    for name, v in cols.items():
        if isinstance(v, list):
            for r, a in enumerate(v):
                flat[f"{name}__{r}"] = a
        else:
            flat[name] = v
    np.savez(path, **flat)


_BUILD = r"""
import json, sys
import numpy as np
from casacore.tables import table, maketabdesc, makearrcoldesc, makescacoldesc
path, spec, npz, endian = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4]
vals = np.load(npz)
nrow = int(spec["nrow"])
descs, dminfo = [makescacoldesc("ID", 0)], {}
for i, (name, (vt, shape, stman, tile)) in enumerate(spec["columns"].items()):
    sample = {"bool": False, "complex": 0j, "float": 0.0, "double": 0.0, "int": 0}[vt]
    if shape is None:
        descs.append(makearrcoldesc(name, sample, ndim=2, valuetype=vt,
                                    datamanagertype=stman, datamanagergroup="G" + name))
    else:
        descs.append(makearrcoldesc(name, sample, shape=list(shape), valuetype=vt,
                                    datamanagertype=stman, datamanagergroup="G" + name))
    dminfo["*%d" % (i + 1)] = {"TYPE": stman, "NAME": "G" + name,
                               "SPEC": {"DEFAULTTILESHAPE": tile}, "COLUMNS": [name]}
t = table(path, maketabdesc(descs), nrow=nrow, dminfo=dminfo, endian=endian, ack=False)
t.putcol("ID", np.arange(nrow, dtype=np.int32))
for name, (vt, shape, stman, tile) in spec["columns"].items():
    if shape is None:
        for r in range(nrow):
            t.putcell(name, r, vals[f"{name}__{r}"])
    else:
        t.putcol(name, vals[name])
for name, (vt, shape, stman, tile) in spec["columns"].items():
    cubes = t.getdminfo(name)["SPEC"]["HYPERCUBES"]
    for c in cubes.values():
        assert list(c["TileShape"]) == tile, (name, c)
        # the point of the test: tiles smaller than (or padding) the cell
        assert list(c["TileShape"][:-1]) != list(c["CellShape"]), (name, c)
    if shape is None:
        assert len(cubes) == 2, cubes
t.close()
print("built")
"""

_VERIFY = r"""
import json, sys
import numpy as np
from casacore.tables import table
path, spec, npz = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
vals = np.load(npz)
t = table(path, ack=False)
nrow = int(spec["nrow"])
for name, (vt, shape, stman, tile) in spec["columns"].items():
    for c in t.getdminfo(name)["SPEC"]["HYPERCUBES"].values():
        assert list(c["TileShape"]) == tile, ("tile shape changed", name, c)
    if shape is None:
        for r in range(nrow):
            got = t.getcell(name, r)
            want = vals[f"{name}__{r}"]
            assert got.shape == want.shape and np.array_equal(got, want), (name, r)
    else:
        got = t.getcol(name)
        want = vals[name]
        assert got.shape == want.shape, (name, got.shape, want.shape)
        bad = np.argwhere(~(got == want).reshape(nrow, -1).all(axis=1))
        assert bad.size == 0, (name, "rows differ", bad[:10].ravel().tolist())
assert np.array_equal(t.getcol("ID"), np.arange(nrow))
t.close()
print("verified")
"""


def _spec():
    return json.dumps({"nrow": NROW, "columns": COLUMNS})


def _assert_cols_equal(t, expected):
    for name, want in expected.items():
        if isinstance(want, list):
            for r in range(NROW):
                got = np.asarray(t.getcell(name, r))
                assert got.shape == want[r].shape, (name, r)
                assert np.array_equal(got, want[r]), (name, r)
            continue
        got = np.asarray(t.getcol(name))
        assert got.dtype == want.dtype and got.shape == want.shape, (name, got.dtype)
        assert np.array_equal(got, want), name
        # chunked reads, across tile-layer boundaries
        for start, n in [(0, 64), (60, 10), (63, 2), (128, 22), (149, 1)]:
            assert np.array_equal(t.getcol(name, start, n), want[start : start + n])
        buf = np.empty_like(want[30:100])
        t.getcolnp(name, buf, 30, 70)
        assert np.array_equal(buf, want[30:100]), name
        for r in (0, 63, 64, 100, 149):
            assert np.array_equal(t.getcell(name, r), want[r]), (name, r)
        # a slice crossing tile boundaries on the channel/first axis
        hi = min(20, want.shape[1] - 1)
        s = t.getcolslice(name, [3, 1], [hi, 1], 10, 100)
        assert np.array_equal(np.asarray(s), want[10:110, 3 : hi + 1, 1:2]), name


@pytest.mark.parametrize("endian", ["little", "big"])
def test_subcell_tiles_read_and_write_in_place(tmp_path, endian):
    path = str(tmp_path / "subtile.tab")
    initial = {name: _values(name, 0) for name in COLUMNS}
    npz0 = str(tmp_path / "v0.npz")
    _save(npz0, initial)
    _run(_BUILD, path, _spec(), npz0, endian)

    # --- read with casacure
    t = ct.table(path, ack=False)
    _assert_cols_equal(t, initial)
    t.close()

    # --- write in place with casacure
    new = {name: (list(v) if isinstance(v, list) else v.copy()) for name, v in initial.items()}
    upd = {name: _values(name, 1) for name in COLUMNS}
    t = ct.table(path, readonly=False, ack=False)
    # putcol over a row range crossing tile layers
    for name in ("FLAG", "DATA", "SFLAG", "WS"):
        t.putcol(name, upd[name][10:100], 10, 90)
        new[name][10:100] = upd[name][10:100]
    # single rows (first/last of a layer, the partial last layer).  Written
    # as one-row putcol: casacure's putcell does not yet convert float32 /
    # complex64 / int32 ndarrays (a separate, pre-existing limitation).
    for name in ("SDATA", "DD"):
        for r in (0, 63, 64, 127, 128, 149):
            t.putcol(name, upd[name][r : r + 1], r, 1)
            new[name][r] = upd[name][r]
    # putcolslice: channels 5..30 of correlation 0, rows 20..80
    blk = upd["SDATA"][20:80, 5:31, 0:1]
    t.putcolslice("SDATA", blk, [5, 0], [30, 0], 20, 60)
    new["SDATA"][20:80, 5:31, 0:1] = blk
    # variable-shape column: rewrite cells of both cubes in place
    for r in (0, 1, 2, 3, 148, 149):
        t.putcol("VAR", upd["VAR"][r][None], r, 1)
        new["VAR"][r] = upd["VAR"][r]
    t.flush()
    t.close()

    # casacure reads its own write
    t = ct.table(path, ack=False)
    _assert_cols_equal(t, new)
    t.close()

    # real casacore agrees, and the tile shapes are unchanged
    npz1 = str(tmp_path / "v1.npz")
    _save(npz1, new)
    assert "verified" in _run(_VERIFY, path, _spec(), npz1)


_DASKMS_WRITE = r"""
import sys
import numpy as np
import dask, dask.array as da, xarray as xr
from daskms import xds_to_table
path, npz = sys.argv[1], sys.argv[2]
v = np.load(npz)
n = v["DATA"].shape[0]
ds = xr.Dataset({
    "TIME": (("row",), da.from_array(np.arange(n, dtype=np.float64), chunks=n)),
    "DATA": (("row", "chan", "corr"), da.from_array(v["DATA"], chunks=(n, -1, -1))),
    "FLAG": (("row", "chan", "corr"), da.from_array(v["FLAG"], chunks=(n, -1, -1))),
})
dask.compute(xds_to_table(ds, path, descriptor="ms"))
from casacore.tables import table
t = table(path, ack=False)
for col in ("DATA", "FLAG"):
    for c in t.getdminfo(col)["SPEC"]["HYPERCUBES"].values():
        print(col, [int(x) for x in c["TileShape"]], [int(x) for x in c["CellShape"]])
        assert c["TileShape"][1] < c["CellShape"][1], "dask-ms did not sub-tile"
t.close()
"""

_DASKMS_CURE = r"""
import sys
import numpy as np
import dask, dask.array as da
from daskms import xds_from_ms, xds_to_table
path, npz = sys.argv[1], sys.argv[2]
v = np.load(npz)
chunks = {"row": 700}
ds = xds_from_ms(path, columns=["DATA", "FLAG"], chunks=chunks)[0]
assert np.array_equal(ds.DATA.values, v["DATA"]), "DATA read mismatch"
assert np.array_equal(ds.FLAG.values, v["FLAG"]), "FLAG read mismatch"
newflag = ds.FLAG.data | (abs(ds.DATA.data) > 1.0)
ds = ds.assign(FLAG=(("row", "chan", "corr"), newflag))
dask.compute(xds_to_table(ds, path, columns=["FLAG"]))
back = xds_from_ms(path, columns=["FLAG"], chunks=chunks)[0].FLAG.values
want = v["FLAG"] | (np.abs(v["DATA"]) > 1.0)
assert np.array_equal(back, want), "FLAG write mismatch (casacure read-back)"
print("casacure ok")
"""

_DASKMS_VERIFY = r"""
import sys
import numpy as np
from casacore.tables import table
path, npz = sys.argv[1], sys.argv[2]
v = np.load(npz)
t = table(path, ack=False)
want = v["FLAG"] | (np.abs(v["DATA"]) > 1.0)
assert np.array_equal(t.getcol("FLAG"), want), "FLAG mismatch (casacore)"
assert np.array_equal(t.getcol("DATA"), v["DATA"]), "DATA changed"
for c in t.getdminfo("FLAG")["SPEC"]["HYPERCUBES"].values():
    print("FLAG", [int(x) for x in c["TileShape"]], [int(x) for x in c["CellShape"]])
t.close()
print("verified")
"""


def test_daskms_subtiled_ms_flag_round_trip(tmp_path):
    """dask-ms (real casacore) writes an MS with its default <= 64-channel
    tiles; dask-ms on the casacure backend reads DATA/FLAG and writes FLAG
    back in place; real casacore verifies."""
    pytest.importorskip("daskms")
    for mod in ("dask", "xarray"):
        pytest.importorskip(mod)
    n, nchan, ncorr = 2000, 79, 2
    rng = np.random.default_rng(11)
    vals = {
        "DATA": (
            rng.standard_normal((n, nchan, ncorr))
            + 1j * rng.standard_normal((n, nchan, ncorr))
        ).astype(np.complex64),
        "FLAG": rng.random((n, nchan, ncorr)) < 0.1,
    }
    npz = str(tmp_path / "v.npz")
    np.savez(npz, **vals)
    path = str(tmp_path / "dm.ms")
    out = _run(_DASKMS_WRITE, path, npz)
    tiles = [line for line in out.splitlines() if line.startswith(("DATA", "FLAG"))]
    assert tiles, out
    assert "casacure ok" in _run(_DASKMS_CURE, path, npz, casacure_backend=True)
    out = _run(_DASKMS_VERIFY, path, npz)
    assert "verified" in out
    # the tile shape dask-ms chose survives casacure's write
    assert [line for line in out.splitlines() if line.startswith("FLAG ")] == [
        line for line in tiles if line.startswith("FLAG ")
    ]
