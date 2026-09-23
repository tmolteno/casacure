"""Memory-conservation tests: casacure vs casacore under dask-ms chunking.

Mirrors the skarabina flagging workload (see ``handover.md``): an MS whose
DATA (complex64) and FLAG (bool) columns hold visibility data, read through
dask-ms in row chunks, with a flag derived per chunk and written back
changed-only (``xds_to_table`` -- dask-ms's write-lock runner flushes the
table after every write chunk, exactly the skarabina ``--write-changed-only``
path).

Every measured number is a fresh subprocess whose peak RSS the *parent* reads
via ``os.wait4``.  The child's own ``getrusage(RUSAGE_SELF).ru_maxrss`` is
deliberately not trusted: in sandboxed/container environments it has been
observed to report a ~600 MiB container-level peak on a 9 MiB process, while
``wait4``/``/proc/<pid>/status VmHWM`` stay per-process and correct.

The assertions encode the dask-ms contract reported by the numbers in
``MEMORY.md``/``BENCHMARK.md``: for a chunked scan, peak RSS must track the
row chunk size / the rows actually read -- NOT the whole column size -- and
casacure must stay within a bounded factor of python-casacore.

Skip policy
-----------
* The whole module is skipped when dask-ms is not importable (CI).
* A backend is measured only when it is importable in this interpreter:
  casacure (installed / on sys.path) and real python-casacore (probed in a
  clean subprocess so the ``tests/shim`` casacore redirect cannot mask it).
* The flagging-write test skips when real python-casacore is unavailable:
  only it can build the TSM-layout MS.  casacure cannot yet create
  ``TiledShapeStMan`` columns, and the synthetic ``default_ms`` layout puts
  FLAG on ``StandardStMan``, where casacure's per-chunk flush is still
  O(column) -- the known gap in handover "next actions", not the skarabina
  production path (real MSes store FLAG as tiled bit-packed bool).
"""

import os
import shutil
import sys

import pytest

pytest.importorskip("daskms")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(ROOT, "tests", "shim")

NROWS = 100_000
NCHAN = 32
NCORR = 4
COLUMN_MIB = NROWS * NCHAN * NCORR * 8 / 2**20  # DATA complex64 bytes
WINDOW = 15_000


# --- worker scripts (run in a fresh subprocess so peak RSS is isolated) ---

_BUILD_TSM = r"""
import sys
import numpy as np
from casacore.tables import table, maketabdesc, makearrcoldesc, makescacoldesc
path, nrows, nchan, ncorr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
n = int(nrows)
td = maketabdesc([
    makescacoldesc("TIME", 0.0),
    makescacoldesc("ANTENNA1", 0),
    makescacoldesc("ANTENNA2", 0),
    makearrcoldesc("DATA", 0j, 0, [nchan, ncorr],
                   "TiledColumnStMan", "TiledData", 0, valuetype="complex"),
    makearrcoldesc("FLAG", False, 0, [nchan, ncorr],
                   "TiledShapeStMan", "TiledShape", 0, valuetype="bool"),
])
t = table(path, td, nrow=n)
rng = np.random.default_rng(7)
data = (rng.standard_normal((n, nchan, ncorr))
        + 1j * rng.standard_normal((n, nchan, ncorr))).astype(np.complex64)
t.putcol("DATA", data)
t.putcol("TIME", rng.random(n))
t.putcol("ANTENNA1", rng.integers(0, 64, n, dtype=np.int32))
t.putcol("ANTENNA2", rng.integers(0, 64, n, dtype=np.int32))
t.putcol("FLAG", np.zeros((n, nchan, ncorr), dtype=bool))
t.close()
"""

_PROBE = "import casacore.tables  # noqa: F401"

_BUILD_SSM = r"""
import sys

import numpy as np
import casacure.tables as ct
path, nrows, nchan, ncorr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
n = int(nrows)
t = ct.default_ms(path)
for name, sample, vt in [("DATA", 0j, "complex"), ("FLAG", False, "bool")]:
    if name not in t.colnames():
        t.addcols(ct.maketabdesc(
            [ct.makearrcoldesc(name, sample, 0, [nchan, ncorr], valuetype=vt)]))
t.addrows(n)
rng = np.random.default_rng(7)
data = (rng.standard_normal((n, nchan, ncorr))
        + 1j * rng.standard_normal((n, nchan, ncorr))).astype(np.complex64)
t.putcol("DATA", data)
t.putcol("FLAG", np.zeros((n, nchan, ncorr), dtype=bool))
t.close()
"""

_WORKER = r"""
import shutil
import sys
import time

import numpy as np

mode, path, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
window = int(sys.argv[4]) if len(sys.argv) > 4 else 0
want_casacure = sys.argv[5] == "casacure" if len(sys.argv) > 5 else False

import dask
dask.config.set(scheduler="sync")  # one chunk in flight at a time
import daskms  # noqa: F401  (activates the casacure alias when requested)
import casacore  # resolves to the selected backend
from daskms import xds_from_table, xds_to_table

# casacure reaches `casacore` via the tests/shim redirect or via dask-ms's
# DASK_MS_BACKEND=casacure alias; both are identifiable by the root module's
# file path (the `tables` submodule may be a lazy proxy without __file__).
_is_casacure = ("casacure" in casacore.__file__) or ("shim" in casacore.__file__)
if _is_casacure != want_casacure:
    raise SystemExit(
        f"backend mismatch: wanted casacure={want_casacure} but "
        f"casacore resolved to {casacore.__file__}"
    )


t0 = time.perf_counter()
if mode == "full":
    ds = xds_from_table(path, chunks={"row": chunk}, columns=["DATA"])[0]
    value = complex(ds.DATA.sum().compute())
elif mode == "window":
    ds = xds_from_table(path, chunks={"row": chunk}, columns=["DATA"])[0]
    ds = ds.isel(row=slice(0, window))  # read only this many rows
    value = complex(ds.DATA.sum().compute())
elif mode == "flag":
    import xarray as xr
    ds = xds_from_table(path, chunks={"row": chunk},
                        columns=["DATA", "FLAG"])[0]
    # skarabina-style per-chunk flag: |DATA| above a clip threshold.
    flagged = xr.apply_ufunc(lambda a: np.abs(a) > 0.5, ds.DATA,
                             dask="allowed", output_dtypes=[bool])
    ds = ds.assign(FLAG=flagged)
    out = path + ".flagged"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(path, out)  # write-changed-only: patch a copy of the MS
    writes = xds_to_table(ds, out, columns=["FLAG"])
    dask.compute(*writes)
    value = len(writes)
else:
    raise SystemExit(f"bad mode {mode!r}")
print(f"{(time.perf_counter() - t0) * 1e3:.0f} {value!r}")
sys.stdout.flush()
"""


# --- helpers ---


def _run(script, args, env_extra):
    """Run `script` with `args` in a fresh subprocess of this interpreter.

    Returns (exitcode, stdout, peak_rss_mib).  Peak RSS is the max of the
    child's /proc/<pid>/status VmHWM polled by the parent while it runs.
    Neither wait4 ru_maxrss nor the child's own getrusage is reliable here:

    * ``getrusage(RUSAGE_SELF).ru_maxrss`` has been observed to report a
      ~600 MiB container-level peak on a 9 MiB process in sandboxed runs;
    * ``ru_maxrss`` is per-process cumulative and *survives execve*, so a
      worker forked from a heavyweight pytest parent would inherit the
      parent's peak.  VmHWM is tied to the process image (reset by exec),
      which is exactly the peak we want.
    """
    import tempfile
    import time

    env = dict(os.environ)
    env.update(env_extra)
    out = tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False)
    name = out.name
    out.close()
    pid = os.fork()
    if pid == 0:
        with open(name, "w") as f:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
        os.chdir(ROOT)
        os.execve(sys.executable, [sys.executable, "-c", script, *args], env)
    peak_kib = 0
    status_path = f"/proc/{pid}/status"
    while True:
        try:
            with open(status_path) as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak_kib = max(peak_kib, int(line.split()[1]))
                        break
        except FileNotFoundError:
            pass  # not yet visible / already gone; waitpid below is decisive
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid:
            exitcode = os.waitstatus_to_exitcode(status)
            break
        time.sleep(0.01)
    with open(name) as f:
        stdout = f.read()
    os.unlink(name)
    return exitcode, stdout, peak_kib / 1024.0


def _strip_shim(pythonpath):
    return os.pathsep.join(
        p for p in pythonpath.split(os.pathsep) if p and p != SHIM
    )


def _casacore_env():
    """Real python-casacore (no shim forwarding to casacure)."""
    env = {"DASK_MS_BACKEND": ""}
    env["PYTHONPATH"] = _strip_shim(os.environ.get("PYTHONPATH", ""))
    return env


def _casacure_env():
    """casacure backend: the shim re-exports casacure.tables; dask-ms 0.2.32
    also aliases casacore->casacure via DASK_MS_BACKEND."""
    env = {"DASK_MS_BACKEND": "casacure"}
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if SHIM not in pp:
        pp.insert(0, SHIM)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


CASACORE_AVAILABLE = _run(_PROBE, [], _casacore_env())[0] == 0
try:
    import casacure  # noqa: F401

    CASACURE_AVAILABLE = True
except ImportError:
    CASACURE_AVAILABLE = False

BACKENDS = [
    b for b in ("casacore", "casacure") if b == "casacore"
    and CASACORE_AVAILABLE or b == "casacure" and CASACURE_AVAILABLE
]


@pytest.fixture(scope="module")
def ms(tmp_path_factory):
    """An MS with DATA on TiledColumnStMan and FLAG on TiledShapeStMan
    (built with real python-casacore -- the on-disk layout both engines must
    agree on; casacure cannot create TiledShapeStMan columns).

    Returns (path, is_tsm): is_tsm is False for the fallback casacure-built
    MS (FLAG on StandardStMan), which the flagging-write test skips.

    Both builds run in a subprocess: the measurement workers poll the
    child's /proc VmHWM, which is inherited from the parent's high-water
    mark across fork+exec, so the pytest parent must stay lean (no MS-sized
    allocations) for the numbers to be clean.
    """
    tmp = tmp_path_factory.mktemp("memchunk")
    if CASACORE_AVAILABLE:
        path = str(tmp / "ms.tab")
        shutil.rmtree(path, ignore_errors=True)
        rc, out, _ = _run(
            _BUILD_TSM,
            [path, str(NROWS), str(NCHAN), str(NCORR)],
            _casacore_env(),
        )
        assert rc == 0, f"real-casacore MS build failed:\n{out}"
        return path, True
    # Fallback: casacure-built MS (FLAG on StandardStMan).  casacure is
    # importable by construction (CASACURE_AVAILABLE was verified); run the
    # build in a subprocess so the pytest parent's peak RSS stays low.
    path = str(tmp / "fallback.tab")
    shutil.rmtree(path, ignore_errors=True)
    rc, out, _ = _run(
        _BUILD_SSM,
        [path, str(NROWS), str(NCHAN), str(NCORR)],
        _casacure_env(),
    )
    assert rc == 0, f"casacure MS build failed:\n{out}"
    return path, False


def _measure(ms, mode, chunk, backend, window=0):
    env = _casacure_env() if backend == "casacure" else _casacore_env()
    rc, out, rss = _run(
        _WORKER, [mode, ms, str(chunk), str(window), backend], env
    )
    assert rc == 0, f"{backend} {mode} chunk={chunk} failed:\n{out}"
    return rss


# --- tests ---


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_column_read_respects_chunk_size(ms, backend):
    """A full-column pass must scale with the dask-ms row chunk, not the whole
    column: peak RSS grows with the chunk size and a chunked pass stays far
    below a single whole-column read."""
    small = _measure(ms[0], "full", 2000, backend)
    mid = _measure(ms[0], "full", 50_000, backend)
    all_rows = _measure(ms[0], "full", NROWS, backend)

    assert all_rows - small >= 0.5 * COLUMN_MIB, (
        f"{backend}: whole-column read is only {all_rows - small:.1f} MiB "
        f"over a {2000}-row chunk pass ({small:.1f} MiB); chunking is not "
        f"reducing memory (column is {COLUMN_MIB:.0f} MiB)"
    )
    assert small < mid < all_rows, (
        f"{backend}: peak RSS not monotone in chunk size: "
        f"{small:.1f} / {mid:.1f} / {all_rows:.1f} MiB"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounded_read_stays_at_baseline(ms, backend):
    """Reading a 15%-of-rows window at a small chunk must sit at the
    Python/dask stack baseline: peak RSS is independent of the 100 MiB column
    behind it."""
    baseline = _measure(ms[0], "window", 2000, backend, window=WINDOW)
    cap = baseline + 0.5 * COLUMN_MIB
    assert baseline < cap, (
        f"{backend}: {WINDOW}-row window read peaked at {baseline:.1f} MiB, "
        f"above the stack-baseline budget {cap:.1f} MiB; the read is not "
        "limited to the rows actually requested"
    )


def test_memory_parity_with_casacore(ms):
    """casacure must not exceed python-casacore's peak RSS by more than a
    bounded factor on the chunked read paths (MEMORY.md reports parity here).
    """
    if not (CASACORE_AVAILABLE and CASACURE_AVAILABLE):
        pytest.skip("need both backends importable")
    for mode, chunk, window in [
        ("full", 2000, 0),
        ("full", 50_000, 0),
        ("full", NROWS, 0),
        ("window", 2000, WINDOW),
    ]:
        core = _measure(ms[0], mode, chunk, "casacore", window)
        cure = _measure(ms[0], mode, chunk, "casacure", window)
        assert cure <= 1.5 * core + 15, (
            f"casacure {mode} chunk={chunk}: {cure:.1f} MiB vs "
            f"casacore {core:.1f} MiB"
        )


def test_flagging_write_respects_chunk_size(ms):
    """Skarabina-style write-changed-only FLAG patch must be chunk-bounded
    on the production TiledShapeStMan layout skarabina flags: a chunked flag
    write uses less memory than one whole-column read, memory scales with the
    write chunk, and the overhead over the stack baseline stays a small
    fraction of the column (guards against an O(column) per-chunk flush)."""
    if not (CASACORE_AVAILABLE and CASACURE_AVAILABLE):
        pytest.skip("need both backends importable")
    path, is_tsm = ms
    if not is_tsm:
        pytest.skip(
            "no real python-casacore: MS built by casacure puts FLAG on "
            "StandardStMan, where casacure's per-chunk flush is still "
            "O(column) (handover next actions); not the skarabina path"
        )
    for backend in BACKENDS:
        baseline = _measure(path, "window", 2000, backend, window=WINDOW)
        small = _measure(path, "flag", 2000, backend)
        large = _measure(path, "flag", 50_000, backend)
        whole_read = _measure(path, "full", NROWS, backend)

        assert small < whole_read, (
            f"{backend}: chunked flag write {small:.1f} MiB is not below a "
            f"whole-column read ({whole_read:.1f} MiB)"
        )
        assert small < large, (
            f"{backend}: flag write does not scale with chunk size: "
            f"{small:.1f} MiB @2000 vs {large:.1f} MiB @50000"
        )
        assert small - baseline < 2.5 * COLUMN_MIB, (
            f"{backend}: flag-write overhead {small - baseline:.1f} MiB over "
            f"baseline {baseline:.1f} MiB is O(column)-sized; a per-chunk "
            "flush must not rebuild the whole column"
        )
