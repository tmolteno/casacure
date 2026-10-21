"""`addcols` + `flush`/`close` extends an existing table in place.

Adding a column to an existing table used to rewrite the whole table: the
flush read every cell of every column into memory and regenerated every data
file.  On a 15 GB MeerKAT MS, killMS's ``addcols(IMAGING_WEIGHT)`` took 19 s
and 32 GB where python-casacore takes 0.2 s and nothing, and a process dying
mid-rewrite left a half-written MS.  The flush now writes only the new
column's data manager and replaces ``table.dat`` last (``tests/
addcols_in_place.rs`` pins the byte-level contract in Rust).

Here, through the Python binding and against real python-casacore:

* the table casacure extends is read back identically by python-casacore, and
  a python-casacore-written MS (several data managers, like a real MS) is
  extended in place by casacure, its existing columns left alone;
* the columns killMS adds (IMAGING_WEIGHT as a ``TiledShapeStMan`` array, UVWDT
  as a copy of UVW's descriptor) work, written in chunks or never written;
* peak RSS of ``addcols`` + ``close`` tracks the NEW column, not the table;
* a layout the in-place path does not handle still comes out correct.

Real python-casacore runs in subprocesses without ``tests/shim`` on the path
(the shim stands in for it); tests needing it are skipped when it is not
installed.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import casacure.tables as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return env


def _real_casacore_available():
    r = subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_real_casacore_env(),
        capture_output=True,
    )
    return r.returncode == 0


needs_casacore = pytest.mark.skipif(
    not _real_casacore_available(), reason="real python-casacore not installed"
)


def _real(code, *args):
    """Run `code` under real python-casacore; return its stdout."""
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


NCHAN, NCORR = 16, 4


def killms_imaging_weight_desc(nchan=NCHAN):
    """IMAGING_WEIGHT exactly as killMS (ClassMS.AddCol) builds it."""
    return {
        "_c_order": True,
        "comment": "",
        "name": "IMAGING_WEIGHT",
        "dataManagerGroup": "imagingweight",
        "dataManagerType": "TiledShapeStMan",
        "maxlen": 0,
        "ndim": 1,
        "option": 4,
        "shape": np.array([nchan], dtype=np.int32),
        "valueType": "float",
    }


def iw_values(n, nchan=NCHAN):
    return (np.arange(n, dtype=np.float32)[:, None] * 0.5
            + np.arange(nchan, dtype=np.float32)[None, :])


def _files(path):
    """name -> (bytes, inode, mtime_ns) of the table's data files."""
    out = {}
    for name in sorted(os.listdir(path)):
        if name.startswith("table.f"):
            full = os.path.join(path, name)
            st = os.stat(full)
            with open(full, "rb") as f:
                out[name] = (f.read(), st.st_ino, st.st_mtime_ns)
    return out


def _assert_files_untouched(before, path, replaced=()):
    after = _files(path)
    for name, rec in before.items():
        if name in replaced:
            continue
        assert name in after, f"{name} disappeared"
        assert after[name] == rec, f"{name} was rewritten"


def _build_ms_like(path, nrows):
    """A casacure table with several data managers (as a real MS): scalars
    on StandardStMan and IncrementalStMan, UVW, tiled DATA and FLAG."""
    td = ct.maketabdesc([
        ct.makescacoldesc("TIME", 0.0),
        ct.makescacoldesc("ANTENNA1", 0),
        ct.makescacoldesc("ANTENNA2", 0),
        ct.makearrcoldesc("UVW", 0.0, 1, [3]),
        ct.makearrcoldesc("DATA", 0j, 0, [NCHAN, NCORR], "TiledColumnStMan",
                          "TiledData", 4, valuetype="complex"),
        ct.makearrcoldesc("FLAG", False, 0, [NCHAN, NCORR], "TiledShapeStMan",
                          "TiledFlag", 4, valuetype="bool"),
    ])
    t = ct.table(path, td, nrow=nrows)
    rng = np.random.default_rng(3)
    t.putcol("TIME", np.arange(nrows) / 7.0)
    t.putcol("ANTENNA1", (np.arange(nrows) % 5).astype(np.int32))
    t.putcol("ANTENNA2", (np.arange(nrows) % 7).astype(np.int32))
    t.putcol("UVW", rng.standard_normal((nrows, 3)))
    t.putcol("DATA", (rng.standard_normal((nrows, NCHAN, NCORR))
                      + 1j * rng.standard_normal((nrows, NCHAN, NCORR))
                      ).astype(np.complex64))
    t.putcol("FLAG", rng.random((nrows, NCHAN, NCORR)) > 0.7)
    t.close()


def _read_all(path, cols):
    t = ct.table(path, ack=False)
    out = {c: t.getcol(c) for c in cols}
    t.close()
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


EXISTING = ["TIME", "ANTENNA1", "ANTENNA2", "UVW", "DATA", "FLAG"]


# --- the columns killMS adds ---


def test_imaging_weight_added_in_place_and_read_back(tmp_path):
    path = str(tmp_path / "t.ms")
    n = 3000
    _build_ms_like(path, n)
    before = _files(path)
    existing = _read_all(path, EXISTING)

    t = ct.table(path, readonly=False, ack=False)
    t.addcols(killms_imaging_weight_desc())
    t.close()

    _assert_files_untouched(before, path)
    assert "IMAGING_WEIGHT" in ct.table(path, ack=False).colnames()
    got = _read_all(path, EXISTING + ["IMAGING_WEIGHT"])
    np.testing.assert_array_equal(got.pop("IMAGING_WEIGHT"),
                                  np.zeros((n, NCHAN), np.float32))
    _assert_same(got, existing)


def test_imaging_weight_putcol_in_chunks_then_close(tmp_path):
    path = str(tmp_path / "t.ms")
    n = 3000
    _build_ms_like(path, n)
    before = _files(path)
    existing = _read_all(path, EXISTING)
    iw = iw_values(n)

    t = ct.table(path, readonly=False, ack=False)
    t.addcols(killms_imaging_weight_desc())
    for start in range(0, n, 700):
        t.putcol("IMAGING_WEIGHT", iw[start:start + 700], start, 700)
    t.close()

    _assert_files_untouched(before, path)
    got = _read_all(path, EXISTING + ["IMAGING_WEIGHT"])
    np.testing.assert_array_equal(got.pop("IMAGING_WEIGHT"), iw)
    _assert_same(got, existing)


def test_column_copied_from_uvw_then_putcol(tmp_path):
    """killMS's UVWDT: ``desc = getcoldesc("UVW"); desc["name"] = "UVWDT"``."""
    path = str(tmp_path / "t.ms")
    n = 2500
    _build_ms_like(path, n)
    existing = _read_all(path, EXISTING)
    uvwdt = np.random.default_rng(5).standard_normal((n, 3))

    t = ct.table(path, readonly=False, ack=False)
    desc = t.getcoldesc("UVW")
    desc["name"] = "UVWDT"
    desc["comment"] = desc["comment"].replace(" ", "_")
    t.addcols(desc)
    t.putcol("UVWDT", uvwdt)
    t.close()

    got = _read_all(path, EXISTING + ["UVWDT"])
    np.testing.assert_array_equal(got.pop("UVWDT"), uvwdt)
    _assert_same(got, existing)


def test_scalar_standardstman_column_and_default_values(tmp_path):
    path = str(tmp_path / "t.ms")
    n = 1800
    _build_ms_like(path, n)
    before = _files(path)
    existing = _read_all(path, EXISTING)

    t = ct.table(path, readonly=False, ack=False)
    t.addcols(ct.maketabdesc([ct.makescacoldesc("EXTRA", 1.5)]))  # never written
    t.addcols(ct.maketabdesc([ct.makescacoldesc("COUNT", 0)]))
    t.putcol("COUNT", np.arange(n, dtype=np.int32))
    t.close()

    got = _read_all(path, EXISTING + ["EXTRA", "COUNT"])
    np.testing.assert_array_equal(got.pop("COUNT"), np.arange(n))
    np.testing.assert_array_equal(got.pop("EXTRA"), np.zeros(n))
    _assert_same(got, existing)
    # DATA / FLAG tile files are not part of any rebuilt manager.
    _assert_files_untouched(
        before, path,
        replaced={n_ for n_ in before if n_.startswith(("table.f0", "table.f1"))
                  and not n_.endswith("_TSM0")},
    )


def test_several_addcols_one_flush_then_another(tmp_path):
    path = str(tmp_path / "t.ms")
    n = 1500
    _build_ms_like(path, n)
    existing = _read_all(path, EXISTING)
    iw = iw_values(n)

    t = ct.table(path, readonly=False, ack=False)
    t.addcols(killms_imaging_weight_desc())
    t.addcols(ct.maketabdesc([ct.makescacoldesc("A", 0)]))
    t.addcols(ct.maketabdesc([ct.makescacoldesc("B", 0.0)]))
    t.putcol("IMAGING_WEIGHT", iw)
    t.putcol("A", np.arange(n, dtype=np.int32))
    t.flush()
    t.addcols(ct.maketabdesc([ct.makescacoldesc("C", 0)]))
    t.putcol("C", np.arange(n, dtype=np.int32) * 2)
    t.putcol("IMAGING_WEIGHT", iw[:10] + 100, 0, 10)
    t.close()

    got = _read_all(path, EXISTING + ["IMAGING_WEIGHT", "A", "B", "C"])
    want_iw = iw.copy()
    want_iw[:10] += 100
    np.testing.assert_array_equal(got.pop("IMAGING_WEIGHT"), want_iw)
    np.testing.assert_array_equal(got.pop("A"), np.arange(n))
    np.testing.assert_array_equal(got.pop("B"), np.zeros(n))
    np.testing.assert_array_equal(got.pop("C"), np.arange(n) * 2)
    _assert_same(got, existing)


def test_layout_not_handled_in_place_falls_back_to_a_correct_rewrite(tmp_path):
    """A column joining an IncrementalStMan group is not extended in place;
    the whole-table rewrite it falls back to must still be right."""
    path = str(tmp_path / "t.ms")
    n = 1200
    td = ct.maketabdesc([
        ct.makescacoldesc("TIME", 0.0),
        ct.makescacoldesc("ID", 0),
        ct.makearrcoldesc("DATA", 0j, 0, [NCHAN, NCORR], "TiledColumnStMan",
                          "TiledData", 4, valuetype="complex"),
    ])
    t = ct.table(path, td, nrow=n)
    t.putcol("TIME", np.arange(n, dtype=np.float64))
    t.putcol("ID", np.arange(n, dtype=np.int32))
    data = (np.arange(n * NCHAN * NCORR, dtype=np.float32).reshape(n, NCHAN, NCORR)
            * (1 + 1j)).astype(np.complex64)
    t.putcol("DATA", data)
    t.close()
    t = ct.table(path, readonly=False, ack=False)
    desc = t.getcoldesc("TIME")
    desc["name"] = "TIME2"
    desc["dataManagerType"] = "IncrementalStMan"
    desc["dataManagerGroup"] = t.getcoldesc("TIME")["dataManagerGroup"]
    t.addcols(desc)
    t.putcol("TIME2", np.arange(n, dtype=np.float64) * 3)
    t.close()
    got = _read_all(path, ["TIME", "ID", "DATA", "TIME2"])
    np.testing.assert_array_equal(got["TIME"], np.arange(n))
    np.testing.assert_array_equal(got["ID"], np.arange(n))
    np.testing.assert_array_equal(got["DATA"], data)
    np.testing.assert_array_equal(got["TIME2"], np.arange(n) * 3)


# --- interop with python-casacore ---

_REAL_DUMP = """
import sys, numpy as np
from casacore.tables import table
path, out, cols = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
t = table(path, ack=False)
np.savez(out, **{c: t.getcol(c) for c in cols})
print(",".join(t.colnames()))
t.close()
"""


def _real_dump(path, out, cols):
    names = _real(_REAL_DUMP, path, str(out), ",".join(cols)).strip().split(",")
    return names, np.load(str(out))


@needs_casacore
def test_python_casacore_reads_a_table_casacure_extended_in_place(tmp_path):
    path = str(tmp_path / "t.ms")
    n = 2200
    _build_ms_like(path, n)
    existing = _read_all(path, EXISTING)
    iw = iw_values(n)
    uvwdt = np.random.default_rng(1).standard_normal((n, 3))

    t = ct.table(path, readonly=False, ack=False)
    t.addcols(killms_imaging_weight_desc())
    t.putcol("IMAGING_WEIGHT", iw)
    d = t.getcoldesc("UVW")
    d["name"] = "UVWDT"
    t.addcols(d)
    t.putcol("UVWDT", uvwdt)
    t.addcols(ct.maketabdesc([ct.makescacoldesc("COUNT", 0)]))
    t.putcol("COUNT", np.arange(n, dtype=np.int32))
    t.close()

    names, data = _real_dump(path, tmp_path / "d.npz",
                             EXISTING + ["IMAGING_WEIGHT", "UVWDT", "COUNT"])
    assert names == EXISTING + ["IMAGING_WEIGHT", "UVWDT", "COUNT"]
    for c in EXISTING:
        np.testing.assert_array_equal(data[c], existing[c], err_msg=c)
    np.testing.assert_array_equal(data["IMAGING_WEIGHT"], iw)
    np.testing.assert_array_equal(data["UVWDT"], uvwdt)
    np.testing.assert_array_equal(data["COUNT"], np.arange(n))

    # And python-casacore can extend it again, then casacure reads that.
    _real(
        "import sys; from casacore.tables import table, makescacoldesc, "
        "maketabdesc; import numpy as np\n"
        "t = table(sys.argv[1], readonly=False, ack=False)\n"
        "t.addcols(maketabdesc([makescacoldesc('FROM_CASACORE', 0)]))\n"
        "t.putcol('FROM_CASACORE', np.arange(t.nrows(), dtype=np.int32) + 5)\n"
        "t.close()\n",
        path,
    )
    # (read in a fresh process: a casacure handle of this process may still
    # serve its own earlier view of the table to later opens.)
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys, numpy as np, casacure.tables as ct\n"
         "t = ct.table(sys.argv[1], ack=False)\n"
         "np.savez(sys.argv[2], **{c: t.getcol(c) for c in sys.argv[3].split(',')})\n",
         path, str(tmp_path / "f.npz"),
         ",".join(EXISTING + ["IMAGING_WEIGHT", "FROM_CASACORE"])],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    after = np.load(str(tmp_path / "f.npz"))
    np.testing.assert_array_equal(after["FROM_CASACORE"], np.arange(n) + 5)
    np.testing.assert_array_equal(after["IMAGING_WEIGHT"], iw)
    _assert_same({c: after[c] for c in EXISTING}, existing)


_REAL_BUILD_MS = """
import sys, numpy as np
from casacore.tables import (default_ms, makearrcoldesc, maketabdesc, table)
path, nrows, nchan, ncorr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
t = default_ms(path)
t.addcols(maketabdesc([
    makearrcoldesc("DATA", 0j, 2, [nchan, ncorr], "TiledColumnStMan",
                   "TiledData", valuetype="complex"),
    makearrcoldesc("MODEL_DATA", 0j, 2, [nchan, ncorr], "TiledShapeStMan",
                   "TiledModel", valuetype="complex"),
]))
t.addrows(nrows)
rng = np.random.default_rng(11)
t.putcol("TIME", np.arange(nrows) * 8.0)
t.putcol("ANTENNA1", (np.arange(nrows) % 5).astype(np.int32))
t.putcol("ANTENNA2", (np.arange(nrows) % 7).astype(np.int32))
t.putcol("UVW", rng.standard_normal((nrows, 3)))
t.putcol("FLAG", rng.random((nrows, nchan, ncorr)) > 0.8)
t.putcol("DATA", (rng.standard_normal((nrows, nchan, ncorr))
                  + 1j * rng.standard_normal((nrows, nchan, ncorr))).astype(np.complex64))
t.putcol("MODEL_DATA", np.ones((nrows, nchan, ncorr), np.complex64))
t.close()
"""

MS_COLS = ["TIME", "ANTENNA1", "ANTENNA2", "UVW", "FLAG", "DATA", "MODEL_DATA"]


@needs_casacore
def test_casacure_extends_a_python_casacore_ms_in_place(tmp_path):
    """A python-casacore-written MS (one StandardStMan for the scalar and
    small columns, two tiled managers) takes killMS's two new columns."""
    path = str(tmp_path / "t.ms")
    n = 1500
    _real(_REAL_BUILD_MS, path, str(n), str(NCHAN), str(NCORR))
    before = _files(path)
    names0, existing = _real_dump(path, tmp_path / "e.npz", MS_COLS)
    existing = {c: existing[c] for c in MS_COLS}
    iw = iw_values(n)
    uvwdt = np.random.default_rng(9).standard_normal((n, 3))

    t = ct.table(path, readonly=False, ack=False)
    t.addcols(killms_imaging_weight_desc())
    t.putcol("IMAGING_WEIGHT", iw)
    d = t.getcoldesc("UVW")
    d["name"] = "UVWDT"
    d["comment"] = d["comment"].replace(" ", "_")
    t.addcols(d)
    t.putcol("UVWDT", uvwdt)
    t.close()

    # The two tiled managers' files (DATA, MODEL_DATA) were never rewritten;
    # only the StandardStMan UVWDT joined was rebuilt.
    ssm = [n_ for n_ in before if n_ in ("table.f0", "table.f0i")]
    _assert_files_untouched(before, path, replaced=ssm)
    for tsm in [n_ for n_ in before if "_TSM" in n_]:
        assert tsm in _files(path)

    names, data = _real_dump(path, tmp_path / "a.npz",
                             MS_COLS + ["IMAGING_WEIGHT", "UVWDT"])
    assert names == names0 + ["IMAGING_WEIGHT", "UVWDT"]
    for c in MS_COLS:
        np.testing.assert_array_equal(data[c], existing[c], err_msg=c)
    np.testing.assert_array_equal(data["IMAGING_WEIGHT"], iw)
    np.testing.assert_array_equal(data["UVWDT"], uvwdt)
    # python-casacore's view of the data managers includes the new one.
    dm = _real(
        "import sys, json; from casacore.tables import table\n"
        "t = table(sys.argv[1], ack=False)\n"
        "print(json.dumps({v['NAME']: v['COLUMNS'] "
        "for v in t.getdminfo().values()}))\n",
        path,
    )
    dms = json.loads(dm.strip().splitlines()[-1])
    # (the binding names a cloned tiled column's group after the column)
    assert dms["imaging_weight"] == ["IMAGING_WEIGHT"]
    assert any("UVWDT" in c and "UVW" in c for c in dms.values())


# --- memory: O(new column), not O(table) ---

_PEAK_CHILD = r"""
import sys
import numpy as np
import casacure.tables as ct


def vmhwm_kib():
    # Peak resident set of THIS process (ru_maxrss can report a container-wide
    # figure in sandboxes; /proc/self/status stays per process).
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])


path, nchan, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = ct.table(path, readonly=False, ack=False)
n = t.nrows()
desc = {"_c_order": True, "comment": "", "name": "IMAGING_WEIGHT",
        "dataManagerGroup": "imagingweight",
        "dataManagerType": "TiledShapeStMan", "maxlen": 0, "ndim": 1,
        "option": 4, "shape": np.array([nchan], dtype=np.int32),
        "valueType": "float"}
if mode == "uvwdt":
    desc = t.getcoldesc("UVW")
    desc["name"] = "UVWDT"
    name = "UVWDT"
else:
    name = "IMAGING_WEIGHT"
before = vmhwm_kib()
t.addcols(desc)
if mode == "putcol":
    w = np.ones((n, nchan), dtype=np.float32)
    t.putcol(name, w)
t.close()
print(vmhwm_kib() - before)
"""

_BUILD_BIG = r"""
import sys
import numpy as np
import casacure.tables as ct
path, nrows, nchan, ncorr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# DATA, MODEL_DATA and CORRECTED_DATA, as a real MS carries them.
big = ["DATA", "MODEL_DATA", "CORRECTED_DATA"]
td = ct.maketabdesc([
    ct.makescacoldesc("TIME", 0.0),
    ct.makearrcoldesc("UVW", 0.0, 1, [3]),
] + [
    ct.makearrcoldesc(c, 0j, 0, [nchan, ncorr], "TiledColumnStMan",
                      "Tiled" + c, 4, valuetype="complex")
    for c in big
])
t = ct.table(path, td, nrow=nrows)
chunk = 20000
for start in range(0, nrows, chunk):
    m = min(chunk, nrows - start)
    for c in big:
        t.putcol(c, np.full((m, nchan, ncorr), 1 + 2j, np.complex64), start, m)
    t.flush()
t.putcol("TIME", np.arange(nrows, dtype=np.float64))
t.close()
"""

BIG_ROWS, BIG_NCHAN, BIG_NCORR = 120_000, 64, 4
BIG_DATA_MB = 3 * BIG_ROWS * BIG_NCHAN * BIG_NCORR * 8 / 1e6  # 737 MB
NEW_COL_MB = BIG_ROWS * BIG_NCHAN * 4 / 1e6  # 31 MB IMAGING_WEIGHT
# The in-place add peaks at ~4-5x the new column (values + encoded cells +
# tile image); the whole-table rewrite at ~3.4x the table.  Between the two.
GROWTH_LIMIT_MB = 0.35 * BIG_DATA_MB


@pytest.fixture(scope="module")
def big_table(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("big") / "big.ms")
    r = subprocess.run(
        [sys.executable, "-c", _BUILD_BIG, path, str(BIG_ROWS), str(BIG_NCHAN),
         str(BIG_NCORR)],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return path


def _peak_growth_mb(table_path, tmp_path, mode):
    import shutil

    work = str(tmp_path / "work.ms")
    shutil.copytree(table_path, work)
    r = subprocess.run(
        [sys.executable, "-c", _PEAK_CHILD, work, str(BIG_NCHAN), mode],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return int(r.stdout.strip().splitlines()[-1]) / 1024 * 1.048576  # KiB -> MB


@pytest.mark.parametrize("mode", ["default", "putcol"])
def test_addcols_peak_memory_tracks_the_new_column_not_the_table(
        big_table, tmp_path, mode):
    """The table holds ~737 MB of tiled visibilities; the new column is ~31
    MB.  The whole-table rewrite peaked at ~3.4x the table (every cell
    materialised as a value, then encoded; 2.5 GB here); the in-place add
    stays within a small multiple of the new column (~140 MB)."""
    growth = _peak_growth_mb(big_table, tmp_path, mode)
    limit = GROWTH_LIMIT_MB
    assert growth < limit, (
        f"addcols+close ({mode}) grew peak RSS by {growth:.0f} MB on a "
        f"{BIG_DATA_MB:.0f} MB table (new column {NEW_COL_MB:.0f} MB); "
        f"limit {limit:.0f} MB"
    )


def test_uvwdt_addcols_peak_memory_is_independent_of_tiled_data(big_table, tmp_path):
    growth = _peak_growth_mb(big_table, tmp_path, "uvwdt")
    limit = GROWTH_LIMIT_MB
    assert growth < limit, f"UVWDT addcols grew peak RSS by {growth:.0f} MB (limit {limit:.0f})"


# --- removecols(X) then addcols(X) starts X from its default ---


def test_removecols_then_addcols_same_name_does_not_inherit_old_data(tmp_path):
    path = str(tmp_path / "t.ms")
    n = 1000
    _build_ms_like(path, n)
    existing = _read_all(path, EXISTING)
    assert existing["ANTENNA2"].any() and np.abs(existing["DATA"]).max() > 0

    t = ct.table(path, readonly=False, ack=False)
    t.removecols(["ANTENNA2", "DATA"])
    t.addcols(ct.maketabdesc([ct.makescacoldesc("ANTENNA2", 0)]))
    t.addcols(ct.maketabdesc([
        ct.makearrcoldesc("DATA", 0j, 0, [NCHAN, NCORR], "TiledColumnStMan",
                          "TiledData", 4, valuetype="complex")]))
    # Before the flush the new columns already read as their defaults.
    assert not t.getcol("ANTENNA2").any()
    assert not np.abs(t.getcol("DATA")).any()
    t.putcell("ANTENNA2", 3, 9)
    t.close()

    got = _read_all(path, ["TIME", "ANTENNA1", "ANTENNA2", "UVW", "FLAG", "DATA"])
    want_a2 = np.zeros(n, np.int32)
    want_a2[3] = 9
    np.testing.assert_array_equal(got["ANTENNA2"], want_a2)
    assert not np.abs(got["DATA"]).any()
    for c in ["TIME", "ANTENNA1", "UVW", "FLAG"]:
        np.testing.assert_array_equal(got[c], existing[c], err_msg=c)
