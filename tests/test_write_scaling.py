"""Writing a NEW table through dask-ms must be bounded by the row chunk.

dask-ms creates the output table and then, per row chunk, ``addrows`` (the
append path, rows without ROWID) or writes into pre-sized rows (the update
path, rows with ROWID), with ``putcol`` + ``flush`` per chunk and column.  A
flush that regrows the whole table from buffered cells makes the peak grow
with the table and the time quadratic in it; these tests pin both, and check
that real python-casacore reads what casacure wrote.

Columns cover every storage manager a dask-ms MS uses: TiledColumnStMan
(complex / bool / float arrays), StandardStMan scalars and fixed-shape arrays
(SIGMA, WEIGHT), and IncrementalStMan scalars (SCAN_NUMBER, FIELD_ID).
"""
import os
import shutil
import sys
import warnings

import pytest

pytest.importorskip("daskms")
pytest.importorskip("xarray")

from test_memory_chunking import (  # noqa: E402
    CASACORE_AVAILABLE,
    CASACURE_AVAILABLE,
    _casacore_env,
    _casacure_env,
    _run,
)

pytestmark = pytest.mark.skipif(not CASACURE_AVAILABLE, reason="casacure not built")

NCHAN, NCORR, CHUNK = 32, 4, 2000
# Large enough that the interpreter/dask start-up (~140 MiB, and a few
# chunks of scheduling slack) is small next to the table.
SMALL, LARGE = 64_000, 256_000

# Every column's values are a function of the row, so a reader can check them.
_WRITE = r"""
import sys, time
import numpy as np, dask, dask.array as da, xarray as xr
import casacore
assert getattr(casacore, "__casacure_shim__", False), casacore.__file__
from daskms import xds_to_table
path, nrow, nchan, ncorr, chunk, mode = sys.argv[1:7]
nrow, nchan, ncorr, chunk = int(nrow), int(nchan), int(ncorr), int(chunk)
rc = (chunk,)
row = da.arange(nrow, chunks=rc)
cube = (nrow, nchan, ncorr)
chan = da.arange(nchan)[None, :, None]
corr = da.arange(ncorr)[None, None, :]
r3 = row[:, None, None]
v = {
    "DATA": ((r3 + 1j * chan + 0 * corr).astype(np.complex64), ("row", "chan", "corr")),
    "FLAG": (((r3 + chan + corr) % 3 == 0), ("row", "chan", "corr")),
    "WEIGHT_SPECTRUM": ((r3 * 0.5 + corr + 0 * chan).astype(np.float32), ("row", "chan", "corr")),
    "SIGMA": ((row[:, None] + da.arange(ncorr)[None, :]).astype(np.float32), ("row", "corr")),
    "WEIGHT": ((row[:, None] * 2.0 + da.arange(ncorr)[None, :]).astype(np.float32), ("row", "corr")),
    "TIME": ((row * 8.0).astype(np.float64), ("row",)),
    "FLAG_ROW": ((row % 7 == 0), ("row",)),
    "ANTENNA1": ((row % 61).astype(np.int32), ("row",)),
    "SCAN_NUMBER": ((row // 5000 + 1).astype(np.int32), ("row",)),
    "FIELD_ID": ((row // 20000).astype(np.int32), ("row",)),
}
data_vars = {k: (dims, arr.rechunk({0: chunk})) for k, (arr, dims) in v.items()}
coords = {"ROWID": ("row", row.astype(np.int64))} if mode == "update" else {}
ds = xr.Dataset(data_vars, coords=coords)
w0, c0 = time.perf_counter(), time.process_time()
dask.compute(xds_to_table(ds, path, "ALL", descriptor="ms"))
# Wall time is what a user feels; CPU time (user+sys, every dask worker
# thread) is what the write actually costs this process.  External load --
# a co-tenant container, memory pressure, page reclaim -- inflates the
# former and not the latter, which is why the scaling test checks both.
print(f"WRITE_SECONDS {time.perf_counter() - w0:.3f} "
      f"CPU_SECONDS {time.process_time() - c0:.3f}")
"""

_VERIFY = r"""
import sys
import numpy as np
import casacore
from casacore.tables import table
path, nrow, nchan, ncorr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
t = table(path, ack=False)
assert t.nrows() == nrow, (t.nrows(), nrow)
row = np.arange(nrow)
r3, chan, corr = row[:, None, None], np.arange(nchan)[None, :, None], np.arange(ncorr)[None, None, :]
want = {
    "DATA": (r3 + 1j * chan + 0 * corr).astype(np.complex64),
    "FLAG": ((r3 + chan + corr) % 3 == 0),
    "WEIGHT_SPECTRUM": (r3 * 0.5 + corr + 0 * chan).astype(np.float32),
    "SIGMA": (row[:, None] + np.arange(ncorr)[None, :]).astype(np.float32),
    "WEIGHT": (row[:, None] * 2.0 + np.arange(ncorr)[None, :]).astype(np.float32),
    "TIME": row * 8.0,
    "FLAG_ROW": row % 7 == 0,
    "ANTENNA1": row % 61,
    "SCAN_NUMBER": row // 5000 + 1,
    "FIELD_ID": row // 20000,
}
for name, expected in want.items():
    got = t.getcol(name)
    assert got.shape == expected.shape, (name, got.shape, expected.shape)
    assert np.array_equal(got, expected), f"{name} differs at rows {np.flatnonzero((got != expected).reshape(nrow, -1).any(axis=1))[:5]}"
print("VERIFIED", getattr(casacore, "__casacure_shim__", False))
"""


# Real casacore appends rows to (and rewrites cells of) the table casacure
# grew: its StandardStMan continues from the stored array-file length and
# index, its IncrementalStMan from the rewritten bucket index.
_EXTEND = r"""
import sys
import numpy as np
from casacore.tables import table
path, nrow, extra = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = table(path, readonly=False, ack=False)
t.addrows(extra)
rows = np.arange(nrow, nrow + extra)
t.putcol("SIGMA", (rows[:, None] + np.arange(4)[None, :]).astype(np.float32), nrow, extra)
t.putcol("TIME", rows * 8.0, nrow, extra)
t.putcol("SCAN_NUMBER", (rows // 5000 + 1).astype(np.int32), nrow, extra)
t.putcol("FIELD_ID", (rows // 20000).astype(np.int32), nrow, extra)
t.putcell("SIGMA", 3, np.full(4, -1.0, np.float32))
t.close()
print("EXTENDED")
"""

_CHECK_EXTENDED = r"""
import sys
import numpy as np
from casacore.tables import table
path, nrow, extra = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = table(path, ack=False)
n = nrow + extra
assert t.nrows() == n, (t.nrows(), n)
row = np.arange(n)
sigma = (row[:, None] + np.arange(4)[None, :]).astype(np.float32)
sigma[3] = -1.0
for name, want in {"SIGMA": sigma, "TIME": row * 8.0,
                   "SCAN_NUMBER": row // 5000 + 1, "FIELD_ID": row // 20000}.items():
    got = t.getcol(name)
    assert np.array_equal(got, want), name
print("CHECKED")
"""


def _write(tmp_path, nrow, mode):
    path = str(tmp_path / f"{mode}_{nrow}.ms")
    shutil.rmtree(path, ignore_errors=True)
    rc, out, rss = _run(_WRITE, [path, str(nrow), str(NCHAN), str(NCORR), str(CHUNK), mode],
                        _casacure_env())
    assert rc == 0, f"casacure write ({mode}, {nrow} rows) failed:\n{out}"
    wall = float(out.split("WRITE_SECONDS")[1].split()[0])
    cpu = float(out.split("CPU_SECONDS")[1].split()[0])
    return path, rss, wall, cpu


def _table_mib(nrow):
    # DATA (8 B) + WEIGHT_SPECTRUM (4 B) per visibility, plus the rest (small).
    return nrow * NCHAN * NCORR * 12 / 2**20


# The linear-cost bound: 4x the rows may cost at most this factor, so the
# quadratic flush this guards against (16x or worse) fails.  The floor keeps
# a sub-0.05 s, start-up-dominated sample from setting an unattainable bound.
TIME_FACTOR, MIN_SECONDS = 7, 0.05


def _write_cost(tmp_path, nrow, mode):
    """`(wall, cpu)` seconds for one fresh table of `nrow` rows."""
    _, _, wall, cpu = _write(tmp_path, nrow, mode)
    return wall, cpu


def _within_linear_bound(small, large):
    """Whether a `(wall, cpu)` pair for 4x the rows stayed within the bound
    on *either* signal (see the scaling test's docstring)."""
    (small_wall, small_cpu), (large_wall, large_cpu) = small, large
    return (large_wall < TIME_FACTOR * max(small_wall, MIN_SECONDS)
            or large_cpu < TIME_FACTOR * max(small_cpu, MIN_SECONDS))


def _best_linear_pair(tmp_path, mode, attempts=2):
    """A `(small, large)` pair of `(wall, cpu)` measurements, re-sampled
    while the pair still looks superlinear.

    A healthy run costs one sample per size, exactly as before.  A pair that
    fails the bound is re-measured with fresh tables (up to `attempts`
    samples) and the per-signal minimum kept: external load only ever *adds*
    to a run, so the minimum is the closest estimate of the write's own
    cost."""
    small = large = None
    for _ in range(attempts):
        sampled = (_write_cost(tmp_path, SMALL, mode), _write_cost(tmp_path, LARGE, mode))
        small = sampled[0] if small is None else tuple(map(min, small, sampled[0]))
        large = sampled[1] if large is None else tuple(map(min, large, sampled[1]))
        if _within_linear_bound(small, large):
            break
    return small, large


def test_scaling_check_ignores_a_loaded_wall_time():
    """The wall-time blow-up of a loaded host must not fail the scaling
    check when the writer's own CPU time stayed linear.  These are the
    numbers from the release-gate failure this guards against: 0.85 s ->
    38.79 s wall (46x) for code whose CPU cost scaled 4x."""
    loaded = ((0.85, 0.62), (38.79, 2.46))
    assert _within_linear_bound(*loaded)
    # A genuine quadratic flush raises both signals; that must still fail.
    assert not _within_linear_bound((0.85, 0.62), (38.79, 19.8))


def test_scaling_check_resamples_before_failing(monkeypatch, tmp_path):
    """A pair that fails on both signals is re-measured with fresh tables and
    the per-signal minimum kept, so one stalled (loaded, memory-pressured)
    sample does not fail the gate."""
    loaded = {"small": 0, "large": 0}

    def fake_cost(path, nrow, mode):
        size = "small" if nrow == SMALL else "large"
        loaded[size] += 1
        if size == "large" and loaded[size] == 1:
            return (30.0, 30.0)  # a large run stalled by load/memory pressure
        return (0.5, 0.6) if size == "small" else (2.0, 2.4)

    monkeypatch.setattr(sys.modules[__name__], "_write_cost", fake_cost)
    small, large = _best_linear_pair(tmp_path, "append")
    assert (loaded["small"], loaded["large"]) == (2, 2), "must re-sample both sizes"
    assert small == (0.5, 0.6) and large == (2.0, 2.4), "the cheapest sample must win"


@pytest.mark.parametrize("mode", ["append", "update"])
def test_new_table_write_memory_does_not_grow_with_the_table(tmp_path, mode):
    """Four times the rows at the same chunk: the peak may move by a fraction
    of the extra table, not by a multiple of it.  (Before casacure grew
    tables in place, each flush regrew the table from buffered cells and the
    peak rose by ~4x the table; now 64k -> 256k rows moves it ~16 MiB for
    ~280 MiB more table.)"""
    _, small, _, _ = _write(tmp_path, SMALL, mode)
    _, large, _, _ = _write(tmp_path, LARGE, mode)
    extra_table = _table_mib(LARGE) - _table_mib(SMALL)
    assert large - small < 0.25 * extra_table, (
        f"{mode}: peak {small:.0f} -> {large:.0f} MiB for {extra_table:.0f} MiB more table"
    )


@pytest.mark.parametrize("mode", ["append", "update"])
def test_new_table_write_time_is_linear_in_the_rows(tmp_path, mode):
    """Four times the rows at the same chunk may cost a few times as long,
    never tens of times.

    Wall time is what a user feels, but it is also what a loaded host — a
    co-tenant container, memory pressure, page reclaim — inflates
    arbitrarily: this test has been seen to fail at 0.9 s -> 38.8 s under
    load and pass at 0.9 s -> 4 s on an idle run of the same code.  The
    writer's own CPU time (user+sys, every dask worker thread) is immune to
    that, because other processes add no CPU to this one.  A flush that
    regrows the table from buffered cells re-encodes every buffered cell,
    so it inflates wall *and* CPU together: the run fails only when both
    signals say the cost grew superlinearly, and passing on the CPU signal
    alone warns that the host was loaded.
    """
    (small_wall, small_cpu), (large_wall, large_cpu) = _best_linear_pair(tmp_path, mode)
    wall_ok = large_wall < TIME_FACTOR * max(small_wall, MIN_SECONDS)
    cpu_ok = large_cpu < TIME_FACTOR * max(small_cpu, MIN_SECONDS)
    assert wall_ok or cpu_ok, (
        f"{mode}: {small_wall:.2f} s -> {large_wall:.2f} s wall and "
        f"{small_cpu:.2f} s -> {large_cpu:.2f} s CPU for 4x the rows (quadratic?)"
    )
    if not wall_ok:
        warnings.warn(
            f"{mode}: wall time {small_wall:.2f} s -> {large_wall:.2f} s looks "
            f"superlinear, but the writer's CPU time is linear "
            f"({small_cpu:.2f} s -> {large_cpu:.2f} s): treating as host load",
            RuntimeWarning,
            stacklevel=1,
        )


@pytest.mark.skipif(not CASACORE_AVAILABLE, reason="real python-casacore not installed")
@pytest.mark.parametrize("mode", ["append", "update"])
def test_real_casacore_reads_the_written_table(tmp_path, mode):
    """Every column, every row, read back by real python-casacore."""
    path, _, _, _ = _write(tmp_path, 16_000 + 777, mode)   # a partial last chunk
    rc, out, _ = _run(_VERIFY, [path, str(16_000 + 777), str(NCHAN), str(NCORR)],
                      _casacore_env())
    # The reader must be the real python-casacore, not the shim: the marker
    # is the only reliable signal (matching "casacure" against the module
    # path also matches this checkout's own directory name).
    assert rc == 0 and "VERIFIED False" in out, out
    # and casacure reads it back the same way
    rc, out, _ = _run(_VERIFY, [path, str(16_000 + 777), str(NCHAN), str(NCORR)],
                      _casacure_env())
    assert rc == 0 and "VERIFIED" in out, out
    shutil.rmtree(path, ignore_errors=True)
    assert not os.path.exists(path)


@pytest.mark.skipif(not CASACORE_AVAILABLE, reason="real python-casacore not installed")
@pytest.mark.parametrize("mode", ["append", "update"])
def test_real_casacore_extends_the_grown_table(tmp_path, mode):
    nrow, extra = 16_000 + 777, 1234
    path, _, _, _ = _write(tmp_path, nrow, mode)
    rc, out, _ = _run(_EXTEND, [path, str(nrow), str(extra)], _casacore_env())
    assert rc == 0 and "EXTENDED" in out, out
    for env in (_casacore_env(), _casacure_env()):
        rc, out, _ = _run(_CHECK_EXTENDED, [path, str(nrow), str(extra)], env)
        assert rc == 0 and "CHECKED" in out, out
