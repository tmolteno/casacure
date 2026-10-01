"""Multi-threaded use of the casacure backend.

The pyo3 bindings release the GIL around reads (`getcol` & co.), around
whole `putcol` batches, and around opens — and every writable handle of one
table directory shares a single cell store (the process-wide write
registry), so concurrent Python threads genuinely execute casacure code in
parallel.  That is the situation dask's default threaded scheduler creates.
These tests hammer that surface directly:

- many readers through one shared read handle (the GIL-released read path);
- parallel per-chunk column writers whose writes must MERGE through the
  shared backing — one handle's flush must never clobber another's column;
- a read handle overlapping a writer's putcol/flush cycles — every observed
  cell wholly its old or its new value, never a torn byte mix, never an
  exception;
- concurrent TaQL queries racing the scratch-directory registry.
"""

import glob
import os
import sys
import tempfile
import threading

import numpy as np
import pytest

from casacore.tables import maketabdesc, makescacoldesc, table, taql

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the casacure backend is exercised on POSIX"
)

NROW = 4096
CHUNK = 256


def make_scalar_table(path, nrow):
    """A one-double-column table `X = arange(nrow)`, closed and on disk."""
    desc = maketabdesc([makescacoldesc("X", 0.0)])
    t = table(str(path), tabledesc=desc, nrow=nrow, readonly=False)
    t.putcol("X", np.arange(nrow, dtype=np.float64))
    t.flush()
    t.close()
    return str(path)


def run_threads(n, target):
    """Run `target(i)` on `n` threads released together; return [(i, repr(e))]."""
    errors = []
    barrier = threading.Barrier(n)

    def wrap(i):
        try:
            barrier.wait()
            target(i)
        except BaseException as e:  # noqa: BLE001 - record, assert below
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=wrap, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_readers_share_one_read_handle(tmp_path):
    """Eight threads read through one readonly handle (dask's threaded
    scheduler holds exactly such a cached proxy): every read returns the
    snapshot's values, concurrently and repeatedly."""
    path = make_scalar_table(tmp_path / "readers.tab", NROW)
    want = np.arange(NROW, dtype=np.float64)
    t = table(path, readonly=True)

    def reader(i):
        for k in range(50):
            start = (i * 613 + k * 251) % (NROW - 128)
            np.testing.assert_array_equal(
                t.getcol("X", start, 128), want[start : start + 128]
            )
            row = (i * 977 + k * 31) % NROW
            assert t.getcell("X", row) == want[row]

    errors = run_threads(8, reader)
    t.close()
    assert not errors, errors


def test_parallel_column_writes_merge_into_one_table(tmp_path):
    """The write-registry contract: four threads each open their own
    writable handle of one table and stream a different column in chunks
    (dask-ms's parallel per-chunk putcol + flush pattern).  All handles
    share one cell store, so every flush must carry the OTHER threads'
    buffered writes too — without the shared backing, each handle's flush
    would rebuild the table from its own stale state and clobber the
    others' columns."""
    ncol = 4
    desc = maketabdesc([makescacoldesc(f"C{c}", 0.0) for c in range(ncol)])
    path = str(tmp_path / "merge.tab")
    t = table(path, tabledesc=desc, nrow=NROW, readonly=False)
    t.flush()
    t.close()

    def writer(c):
        w = table(path, readonly=False, lockoptions="nolock")
        for start in range(0, NROW, CHUNK):
            vals = np.arange(start, start + CHUNK, dtype=np.float64) + 1000.0 * c
            w.putcol(f"C{c}", vals, start)
            w.flush()
        w.close()

    errors = run_threads(ncol, writer)
    assert not errors, errors

    # A fresh readonly open reads the on-disk truth: every column complete.
    check = table(path, readonly=True)
    assert check.nrows() == NROW
    for c in range(ncol):
        np.testing.assert_array_equal(
            check.getcol(f"C{c}"), np.arange(NROW, dtype=np.float64) + 1000.0 * c
        )
    check.close()


def test_readers_survive_concurrent_writer_flushes(tmp_path):
    """A readonly handle (its own frozen snapshot) reads a full column in a
    tight loop while a writer thread rewrites and flushes it — both sides
    run with the GIL released, so the reads genuinely overlap the flushes.
    Every observed value must be wholly the original (`arange`) or wholly a
    rewritten one (`1e6 + iteration`): never a torn byte mix of the two,
    never a decode error."""
    nrow = 2048
    iterations = 300
    path = make_scalar_table(tmp_path / "rw.tab", nrow)
    want = np.arange(nrow, dtype=np.float64)
    reader = table(path, readonly=True)
    stop = threading.Event()
    failures = []

    def read_loop():
        while not stop.is_set():
            got = reader.getcol("X")
            ok = (got == want) | ((got >= 1e6) & (got < 1e6 + iterations))
            if not ok.all():
                failures.append(got[~ok][:8])
                return

    def write_loop():
        w = table(path, readonly=False, lockoptions="nolock")
        for it in range(iterations):
            w.putcol("X", np.full(nrow, 1e6 + it, dtype=np.float64))
            w.flush()
        w.close()

    reader_t = threading.Thread(target=read_loop)
    writer_t = threading.Thread(target=write_loop)
    reader_t.start()
    writer_t.start()
    writer_t.join()
    stop.set()
    reader_t.join()
    reader.close()
    assert not failures, [f.tolist() for f in failures]


def test_concurrent_taql_queries(tmp_path):
    """Concurrent taql() calls (each materialising a scratch result table
    under the temp dir, registered process-wide and removed when its last
    handle drops) must not interfere: every query returns its row count and
    the temp dir keeps no leftover result of THIS process."""
    path = make_scalar_table(tmp_path / "query.tab", 512)
    pattern = os.path.join(
        tempfile.gettempdir(), f"casacure-taql-{os.getpid()}-*"
    )
    before = set(glob.glob(pattern))

    def query(i):
        for _ in range(20):
            t = taql(f"select * from '{path}' where rownr() < 50")
            assert t.nrows() == 50
            t.close()

    errors = run_threads(6, query)
    assert not errors, errors
    assert set(glob.glob(pattern)) <= before, "a scratch result leaked"
