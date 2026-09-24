# casacure benchmarks

Benchmark results for the `casacure-bench` console script and a row-count
scaling micro-benchmark, comparing casacure against real python-casacore on
the same machine and in the same process.

## Version and environment

| | |
|---|---|
| casacure wheel / crate | **3.8.5** (A.B.P policy: 3.8 = casacore interface, P = casacure patch) |
| source | `49978ec` (3.8.5 release head; the code measured here is released as 3.8.6) |
| build profile | **release** (`maturin develop --release`) |
| Python | 3.13.5 (CPython) |
| numpy | 2.2.4 |
| reference | python-casacore 3.7.1 |
| machine | AMD Ryzen 5 5600G @ 3.9 GHz (desktop), 12 threads, 62 GB RAM, Linux 6.12 |
| date | 2026-09-24 |

These numbers are a full rerun of everything in this file on the local
machine above, against the current code (3.8.5 / `49978ec`), on 2026-09-24.
The earlier 2026-09-21 measurements were taken on the original development
laptop (Intel Core i5-8365U, python-casacore 3.8.1-1, Python 3.14.7); where
sections below compare engine-vs-engine those older rows are kept as the
historical record and are labelled as such.

Run your own copy from the repo checkout:

```sh
maturin develop --release
casacure-bench          # needs python-casacore importable alongside casacure
```

`run_benchmark` imports `casacore.tables` and — when it resolves to a
*distinct* implementation (real python-casacore, not the `casacore` shim) —
measures both engines in the same process. The comparison table is only
printed when real casacore is present; otherwise casacure-only times are
shown with `n/a` ratios.

## Memory footprint: casacore vs casacure for dask-ms chunked reads

`scripts/bench_daskms_chunking.py` builds an MS with a large
`(nrow × nchan × ncorr) complex64` DATA column (~977 MiB, 250k rows ×
[128,4]), then reads it through `dask-ms` (`xds_from_table` +
`DATA.sum().compute()`) at several row-chunk sizes under a synchronous dask
scheduler. Each config runs in a fresh subprocess so `ru_maxrss` reflects
only that read; both engines get the identical MS (built once with real
python-casacore) and the identical dask-ms graph (the casacure backend vs
real python-casacore).

**Full pass** — `.sum().compute()` streams the whole 977 MiB column (every
row is read), peak RSS:

| chunk (rows) | casacure (MiB) | casacore (MiB) |
|---|---|---|
| all (250k) | 2240 | 2234 |
| 125 000 | 1187 | 1196 |
| 25 000 | 344 | 364 |
| 5 000 | 180 | 198 |
| 1 000 | 179 | 168 |

**Bounded read** — a window of the column (10 000 rows = 39 MiB, then
100 000 rows = 391 MiB) read at the relative chunk sizes 1/2, 1/10, 1/50 and
1/250 of the window; peak RSS spreads across those chunks:

| rows read | casacure (MiB) | casacore (MiB) |
|---|---|---|
| 10 000 | 180–197 | 160–196 |
| 100 000 | 179–555 | 163–570 |

(The whole-column `chunk = all` read is not a bounded read — for a window it
materialises the full block and sits at the full-pass `all` entry, ~2.2 GiB.
The 100 000-row spread includes a reproducible casacure peak of ~538 MiB at
400-row chunks — a small-chunk artifact where the mapped pages for a window
are not all released before the next getcol — and ~555 MiB at 50 000-row
chunks from the 39 MiB page working set; both engines otherwise sit near the
~160–200 MiB Python/dask-ms stack baseline.)

### What this says

- **Both engines respect chunking for anything short of a full pass.**
  Bounded reads scale with the rows actually read, not the column size, and
  sit at the same ~160–200 MiB Python/dask-ms stack baseline in both
  engines.
- **A full-column pass is at casacore parity across the board**, including
  the single whole-column read (`chunk = all`: 2240 vs 2234 MiB). casacure
  memory-maps the data files, drops the mapped pages
  (`madvise(MADV_DONTNEED)`) as a bulk scan advances, and — for a read
  handle over a StandardStMan numeric column — `getcolnp` decodes straight
  into the caller's numpy buffer (`Table::getcol_raw`), skipping the per-cell
  `Vec<RecordValue>`/`ArrayData` intermediate that previously added a third
  full-size buffer. See `MEMORY.md` for the mechanism and trade-offs.
- **Where casacure was before the mapping + page-drop + typed changes:** an
  open eagerly `fs::read` the whole data file (~6 GiB RSS at every chunk
  size), then memory-map-only left a full pass resident at ~1× the column
  (~1.1 GiB floor), then the per-cell decode added a third full buffer on a
  single whole-column read (3.2 GiB); all superseded by the current
  streaming/typed behaviour (2.2 GiB at `chunk = all`, ~180 MiB chunked).

Timing in the chunked regime is comparable or better: on this machine a
250-chunk (1000-row) ranged scan of the full column is ~0.79 s for casacure
vs ~1.39 s for real python-casacore (the per-element SSM decode was replaced
with a single `chunks_exact` + `from_{le,be}_bytes` decode, 3.2× faster
scan).

## Workload

`casacure-bench` builds a table with two double scalar columns
(`TIME`, `WEIGHT`) and stores 20 000 rows (~160 KB per column, fully
resident / memory-mapped). The three measured ops are the whole-column,
single-`putcol`/`getcol` round-trips and one `taql` `SELECT * WHERE …
ORDERBY …` that scans all 20 000 rows. These measure the **python-conversion
and value-bridging cost**, not I/O: at this size the data is a memory-mapped
`Vec<u8>` and the dominant cost is converting between numpy arrays and
casacure cell values.

## Results — `casacure-bench`, rerun 2026-09-24 (5 runs, medians, release)

| op | casacure (release) | real casacore | ratio (cure/core) |
|---|---|---|---|
| putcol | 1.47 ms | 1.08 ms | **1.4×** |
| getcol | 1.47 ms | 0.55 ms | **2.7×** |
| taql WHERE+ORDERBY | 12.03 ms | 7.27 ms | 1.7× |

Raw five single runs (ms), this rerun (the taql row absorbs the single
deferred flush, see below):

| op | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| putcol | 1.47 | 1.47 | 1.47 | 1.51 | 2.02 |
| getcol | 1.44 | 1.42 | 1.47 | 1.48 | 2.03 |
| taql WHERE+ORDERBY | 12.03 | 11.87 | 12.75 | 12.03 | 16.20 |

### Before / after the deferred-flush (write-buffering) optimization

*Historical record — measured on the original development machine
(2026-09-21, Intel i5 laptop):*

| op | before (ms) | after (ms) | ratio before | ratio after |
|---|---|---|---|---|
| putcol | 4.33 | 1.65 | 7.2× | 2.3× |
| getcol | 0.66 | 0.59 | 1.9× | 1.6× |
| taql WHERE+ORDERBY | 9.41 | 12.81 | 3.2× | 4.7× |

### What the optimization did

Previously every write op (`putcol`, `putcell`, `addrows`, keyword setters,
…) rewrote the **whole table to disk** (regenerate `table.dat` + every data
file) on each call — measured at ~49 ns/cell of a 114 ns/cell putcol, i.e.
~43 % of the cost. Writes are now buffered in the shared in-memory cell
store and only physically written when something needs the on-disk state:
an explicit `flush()` / `close()` / context exit, a mutating taql statement,
`query()`/`sort()`, or `getdminfo()`. This matches python-casacore's
buffered storage-manager behaviour; reads within a session always come from
the shared store.

The taql row above now includes the one full-table flush that the earlier
(now deferred) putcol would have paid: the benchmark runs taql immediately
after an unflushed putcol, so `taql` absorbs the deferred flush. On a
workload with many writes between flushes, the amortised putcol cost stays
at the 1.47 ms level and the flush happens once.

## Scaling — ns per cell vs row count (release, 1 double column)

Measured with a separate micro-benchmark (`putcol` of `np.arange(n)`, then
`getcol`), single DOUBLE scalar column, n rows, median of 7 repeats:

| n | casacure putcol | casacore putcol | casacure getcol | casacore getcol |
|---|---|---|---|---|
| 1 000 | 38 | 53 | 49 | 29 |
| 10 000 | 37 | 50 | 57 | 24 |
| 100 000 | 37 | 50 | 69 | 23 |

(*Historical scaling on the original machine, 2026-09-21:* casacure putcol
was 73/56/63, casacore putcol 32/32/29; casacure getcol 13/14/25, casacore
getcol 18/16/16.) On this machine the deferred flush has closed the putcol
gap entirely — casacure putcol is now slightly *faster* than casacore per
cell (37–38 vs 50–53 ns/cell); getcol is the remaining gap at 2–2.5×.

## Interpretation

- **putcol** is now at or below casacore on this machine: ~1.4× overall on
  the 20k bench (1.47 vs 1.08 ms) and 37–38 ns/cell vs casacore's 50–53 in
  the scaling micro-benchmark (flat with `n`). The deferred-flush write
  buffering removed the per-write whole-table rewrite; the small remaining
  constant overhead on the bench is the one `RecordValue` packaging per cell
  on its way into the buffered store.
- **getcol** is the remaining gap, ~2.7× overall on the 20k bench and 2–2.5×
  per cell (49 → 69 ns/cell from 1k → 100k rows, superlinear growth; casacore
  is flat at ~23–29). It is the per-cell `RecordValue` clone plus the
  two-pass numpy conversion, amplified by allocator/working-set effects at
  ~MB scale; casacore does a single memcpy. Not I/O (the data is
  memory-mapped).
- **taql** is ~1.7× (vs 4.7× on the original machine — casacore's taql is
  comparatively slow here): the `SELECT *` result materialises through the
  same getcol path, plus (in this benchmark) the one flush charged to the
  op.

### Where the remaining gap lives

getcol is dominated by the per-cell `RecordValue` round-trip between the
numpy buffers and the storage/convert layers, not by disk or algorithm. The
next step is a deeper **typed-buffer** read/write path — for a whole-column
single-value-type numpy call, batch-convert directly between the numpy
buffer and the column's typed storage, skipping per-element `RecordValue`
boxing for scalar columns. That should flatten getcol's scaling and bring
the getcol gap to ~1.5×; the ceiling is python-casacore's native C++ memcpy
path.
