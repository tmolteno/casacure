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
assert "casacure" in casacore.__file__ or "shim" in casacore.__file__, casacore.__file__
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
t0 = time.perf_counter()
dask.compute(xds_to_table(ds, path, "ALL", descriptor="ms"))
print(f"WRITE_SECONDS {time.perf_counter() - t0:.3f}")
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
print("VERIFIED", casacore.__file__)
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
    seconds = float(out.split("WRITE_SECONDS")[1].split()[0])
    return path, rss, seconds


def _table_mib(nrow):
    # DATA (8 B) + WEIGHT_SPECTRUM (4 B) per visibility, plus the rest (small).
    return nrow * NCHAN * NCORR * 12 / 2**20


@pytest.mark.parametrize("mode", ["append", "update"])
def test_new_table_write_memory_does_not_grow_with_the_table(tmp_path, mode):
    """Four times the rows at the same chunk: the peak may move by a fraction
    of the extra table, not by a multiple of it.  (Before casacure grew
    tables in place, each flush regrew the table from buffered cells and the
    peak rose by ~4x the table; now 64k -> 256k rows moves it ~16 MiB for
    ~280 MiB more table.)"""
    _, small, _ = _write(tmp_path, SMALL, mode)
    _, large, _ = _write(tmp_path, LARGE, mode)
    extra_table = _table_mib(LARGE) - _table_mib(SMALL)
    assert large - small < 0.25 * extra_table, (
        f"{mode}: peak {small:.0f} -> {large:.0f} MiB for {extra_table:.0f} MiB more table"
    )


@pytest.mark.parametrize("mode", ["append", "update"])
def test_new_table_write_time_is_linear_in_the_rows(tmp_path, mode):
    _, _, small = _write(tmp_path, SMALL, mode)
    _, _, large = _write(tmp_path, LARGE, mode)
    assert large < 7 * max(small, 0.05), (
        f"{mode}: {small:.2f} s -> {large:.2f} s for 4x the rows (quadratic?)"
    )


@pytest.mark.skipif(not CASACORE_AVAILABLE, reason="real python-casacore not installed")
@pytest.mark.parametrize("mode", ["append", "update"])
def test_real_casacore_reads_the_written_table(tmp_path, mode):
    """Every column, every row, read back by real python-casacore."""
    path, _, _ = _write(tmp_path, 16_000 + 777, mode)   # a partial last chunk
    rc, out, _ = _run(_VERIFY, [path, str(16_000 + 777), str(NCHAN), str(NCORR)],
                      _casacore_env())
    assert rc == 0 and "VERIFIED" in out and "casacure" not in out.split("VERIFIED")[1], out
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
    path, _, _ = _write(tmp_path, nrow, mode)
    rc, out, _ = _run(_EXTEND, [path, str(nrow), str(extra)], _casacore_env())
    assert rc == 0 and "EXTENDED" in out, out
    for env in (_casacore_env(), _casacure_env()):
        rc, out, _ = _run(_CHECK_EXTENDED, [path, str(nrow), str(extra)], env)
        assert rc == 0 and "CHECKED" in out, out
