#! /usr/bin/env python3
"""dask-ms chunking benchmark on the casacure backend.

Verifies that dask-ms row chunking deliberately reduces the memory footprint
of a large-array read: builds an MS with a (nrow x nchan x ncorr) complex
DATA column (default ~1 GB), then reads it via `xds_from_table` at several
row-chunk sizes. Every chunk config runs in a fresh subprocess so `ru_maxrss`
reflects only that read; a synchronous dask scheduler ensures one chunk is in
flight at a time.

With casacure's memory-mapped data files, a read handle touches only the
pages of the rows it reads: peak RSS for a bounded read is the ~430 MiB
Python/dask stack baseline, and a full-column pass scales with the rows read.
Before the mapping change every open eagerly `fs::read` the whole data file,
so peak RSS was ~the full column for every chunk.

Usage (the casacure backend must be selected, and TMPDIR should be on disk so
casacure's taql temp tables do not fill a small tmpfs):
    DASK_MS_BACKEND=casacure \
    TMPDIR=/var/tmp/casacure-bench/tmp \
        python scripts/bench_daskms_chunking.py [ms.tab] [nrows] [nchan] [ncorr]
"""

import os
import resource
import subprocess
import sys
import time

import numpy as np

# Import dask-ms first: with DASK_MS_BACKEND=casacure it aliases
# sys.modules['casacore'] to casacure, so every later `casacore.tables`
# import (here and inside dask-ms) resolves to the casacure backend.
import daskms  # noqa: F401
import casacore.tables  # noqa: F401  (now the casacure backend)


def build_ms(path, nchan, ncorr, nrows, seed=7):
    from casacore.tables import makearrcoldesc, maketabdesc, makescacoldesc, table

    td = maketabdesc(
        [
            makescacoldesc("TIME", 0.0),
            makescacoldesc("ANTENNA1", 0),
            makearrcoldesc("DATA", 0j, 0, [nchan, ncorr], valuetype="complex"),
            makearrcoldesc("WEIGHT", 0.0, 0, [ncorr], valuetype="float"),
        ]
    )
    rng = np.random.default_rng(seed)
    t = table(path, td, nrows)
    data = rng.standard_normal((nrows, nchan, ncorr)).astype(np.complex64)
    data += 1j * rng.standard_normal((nrows, nchan, ncorr)).astype(np.complex64)
    t.putcol("DATA", data)
    t.putcol("TIME", rng.random(nrows))
    t.putcol("ANTENNA1", rng.integers(0, 2**31, nrows, dtype=np.int32))
    t.putcol("WEIGHT", rng.random((nrows, ncorr), dtype=np.float32))
    t.close()


READ_BODY = r"""
import resource, sys, time
import dask
dask.config.set(scheduler="sync")
import daskms  # activates the casacure alias first
from daskms import xds_from_table

path, chunk = sys.argv[1], int(sys.argv[2])
t0 = time.perf_counter()
ds = xds_from_table(path, chunks={"row": chunk})[0]
value = complex(ds.DATA.sum().compute())
rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"{rss_mib:.1f} {(time.perf_counter()-t0)*1e3:.0f}")
sys.stdout.flush()
"""


def read_sum(path, chunk):
    env = dict(os.environ, TMPDIR=os.environ.get("TMPDIR", "/var/tmp"))
    out = subprocess.run(
        [sys.executable, "-c", READ_BODY, path, str(chunk)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    rss, ms = out.stdout.split()
    return float(rss), float(ms)


def main(path, nrows, nchan, ncorr):
    total_mib = nrows * nchan * ncorr * 8 / 2**20
    print(f"MS rows={nrows} DATA cell={nchan*ncorr*8}B total={total_mib:.0f} MiB")
    print(f"{'chunk (rows)':>14} {'peak RSS (MiB)':>16} {'RSS/chunk':>11} {'read ms':>9}")
    for chunk in [nrows, max(nrows // 2, 1), max(nrows // 10, 1), max(nrows // 50, 1), max(nrows // 250, 1)]:
        rss, ms = read_sum(path, chunk)
        chunk_mib = chunk * nchan * ncorr * 8 / 2**20
        ratio = rss / chunk_mib if chunk_mib else 0.0
        tag = "all" if chunk == nrows else chunk
        print(f"{tag:>14} {rss:>16.1f} {ratio:>11.1f} {ms:>9.0f}")
        sys.stdout.flush()


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args else "/var/tmp/casacure-bench/ms.tab"
    nrows = int(args[1]) if len(args) > 1 else 250_000
    nchan = int(args[2]) if len(args) > 2 else 128
    ncorr = int(args[3]) if len(args) > 3 else 4
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        build_ms(path, nchan, ncorr, nrows)
    main(path, nrows, nchan, ncorr)
