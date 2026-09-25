# casacure benchmarks

Benchmark results for the `casacure-bench` console script and a row-count
scaling micro-benchmark, comparing casacure against real python-casacore on
the same machine and in the same process.

## Version and environment

| | |
|---|---|
| casacure wheel / crate | **3.8.8** (A.B.P policy: 3.8 = casacore interface, P = casacure patch) |
| build profile | **release** (`maturin develop --release`) |
| Python | 3.11.15 (CPython) |
| numpy | 2.4.6 |
| reference | python-casacore 3.8.1 |
| machine | Intel Core Ultra 7 258V (laptop), 8 cores, 30 GB RAM, Linux 7.1, load < 1 |
| date | 2026-09-26 |

The sections below were rerun on 2026-09-26 on this machine: the summary,
reads, writes and `casacure-bench`.  The end-to-end section ran on
schmalzburg.  The scaling micro-benchmark and the "before / after
deferred-flush" table are historical records from earlier machines, labelled
as such.

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

## Summary: casacure vs python-casacore (3.8.8)

Ratios are casacure / casacore, so **< 1 means casacure is faster or
lighter**.

| workload | time ratio | peak-RSS ratio |
|---|---|---|
| dask-ms chunked read, 977 MiB DATA, 25 000-row chunks | 1.03 | 1.04 |
| dask-ms chunked read, 1000-row chunks | **0.40** (526 vs 1316 ms) | 1.35 (179 vs 133 MiB) |
| dask-ms write of a new MS, 256k rows, 2000-row chunks | **0.60** (1.45 vs 2.4 s) | **0.80** (188 vs 234 MiB) |
| skarabina flag + 32x average + `--msout`, MeerKAT scan (schmalzburg) | **0.80** (17.4 vs 21.9 s) | **0.77** (7.4 vs 9.5 GB) |
| skarabina flag, `--write-changed-only`, same scan | **0.62** (7.6 vs 12.2 s) | **0.54** (2.4 vs 4.3 GB) |
| `casacure-bench` small ops (20k rows, fully cached) | 3–5x | — |

Where casacure is slower, it is per-call overhead on small, fully cached
tables (`casacure-bench`): the Python bridging and `RecordValue` packaging,
not I/O.  Where the work is I/O-shaped, as in dask-ms chunked scans and new
MS writes, casacure is at parity or ahead.

## dask-ms chunked reads: time and memory

`scripts/bench_daskms_chunking.py` builds an MS with a
`(250 000 × 128 × 4) complex64` DATA column (977 MiB), once, with real
python-casacore.  It then reads the MS through dask-ms (`xds_from_table` +
`DATA.sum().compute()`, synchronous scheduler) at several row chunks.  Each
read runs in a fresh subprocess, and its peak is that child's `/proc` VmHWM,
polled by the parent.  (The child's `ru_maxrss` inherits the parent's
build-time peak across fork+exec, which made every chunk report the same
number; the script now polls VmHWM.)

| chunk (rows) | casacure peak | casacore peak | casacure read | casacore read |
|---|---|---|---|---|
| all (250k) | 2209 MiB | 2198 MiB | 483 ms | 466 ms |
| 125 000 | 1170 MiB | 1160 MiB | 490 ms | 476 ms |
| 25 000 | 340 MiB | 327 MiB | 501 ms | 486 ms |
| 5 000 | 179 MiB | 162 MiB | 656 ms | 632–748 ms |
| 1 000 | 179 MiB | 133 MiB | **526 ms** | 1316–1518 ms |

Both engines bound memory by the chunk.  casacure sits ~12–46 MiB above
casacore: a larger import footprint and mapped-page slack.  At small chunks
casacure is 2.5x faster, because casacore's per-call cost dominates there.

## dask-ms writes of a new table (chunk-bounded since 3.8.8)

`tests/test_write_scaling.py` writes a new MS through dask-ms in 2000-row
chunks.  The columns are DATA (complex64 [32,4]), FLAG and WEIGHT_SPECTRUM
(tiled), SIGMA/WEIGHT (StandardStMan arrays), TIME/FLAG_ROW/ANTENNA1
(StandardStMan scalars) and SCAN_NUMBER/FIELD_ID (IncrementalStMan).  Peak
RSS and write time:

| rows (table) | casacure 3.8.7 | casacure 3.8.8 | python-casacore |
|---|---|---|---|
| 16 000 (23 MiB) | 242 MiB, 0.39 s | 137 MiB, 0.13 s | 188 MiB, 0.19 s |
| 64 000 (94 MiB) | 633 MiB, 4.1 s | 169 MiB, 0.42 s | 219 MiB, 0.67 s |
| 256 000 (375 MiB) | 1865 MiB, 57.6 s | 188 MiB, 1.45 s | 234 MiB, 2.4 s |
| 1 024 000 (1.5 GiB) | — | 224 MiB, 6.5 s | — |

The append pattern (`addrows` per chunk) and the update pattern (ROWID,
`addrows(nrow)` up front) measure the same.  3.8.7 regenerated the whole table
at every flush of a table larger than its files, so memory grew with the table
and time grew quadratically.  3.8.8 grows the files in place (see `MEMORY.md`).
Real python-casacore reads every written cell back, and extends the written
tables itself.

## End-to-end: skarabina on a MeerKAT scan

Host schmalzburg (12 cores, 62 GB, **load 7-11**, shared, so the timings are
load-dependent).  The input is a copy of scan 1 of a MeerKAT L-band MS:
143 716 rows x 2511 channels x 2 correlations, 11 GB.  The run is skarabina
`9b6b97b` with its stage-0 flag list (`save:imported`, `autos`,
`uv-above 2500`, `nan`, `clip 0 100`, `spectral-window`) and `--summary`.
The two backends ran in alternating order, twice each:

| workload | casacure 3.8.8 | python-casacore 3.8.1 |
|---|---|---|
| + 32x frequency average, `--msout` | 17.3 s, 7.40 GB / 17.5 s, 7.41 GB | 21.3 s, 10.1 GB / 22.5 s, 8.99 GB |
| + `--write-changed-only --msout` | 7.7 s, 2.51 GB / 7.6 s, 2.23 GB | 13.8 s, 4.16 GB / 10.6 s, 4.49 GB |

The averaged outputs of the two backends are identical in every readable
main-table column and in SPECTRAL_WINDOW.  (FLAG_CATEGORY cannot be read by
casacore in the input either.)  With casacure 3.8.7 the averaged run took
74.4 s at 21.7 GB, because the flag-version backup's whole table was
buffered.

## Workload

`casacure-bench` builds a table with two double scalar columns
(`TIME`, `WEIGHT`) and stores 20 000 rows (~160 KB per column, fully
resident / memory-mapped). The three measured ops are the whole-column,
single-`putcol`/`getcol` round-trips and one `taql` `SELECT * WHERE …
ORDERBY …` that scans all 20 000 rows. These measure the **python-conversion
and value-bridging cost**, not I/O: at this size the data is a memory-mapped
`Vec<u8>` and the dominant cost is converting between numpy arrays and
casacure cell values.

## Results — `casacure-bench`, 2026-09-26 (5 runs, medians, release)

Same workload as before (two double scalar columns, 20 000 rows).

| op | casacure | real casacore | ratio (cure/core) |
|---|---|---|---|
| putcol | 1.20 ms | 0.39 ms | 3.0× |
| getcol | 0.77 ms | 0.22 ms | 3.5× |
| taql WHERE+ORDERBY | 8.56 ms | 1.79 ms | 4.8× |

Raw runs (ms, casacure / casacore): putcol 1.20/0.41, 1.15/0.39, 1.20/0.43,
1.33/0.39, 1.17/0.39; getcol 0.77/0.22, 0.77/0.22, 0.81/0.22, 0.74/0.22,
0.76/0.22; taql 8.62/2.66, 8.56/1.79, 8.58/1.78, 8.53/2.04, 8.51/1.77.

*Historical (2026-09-24, Ryzen 5 5600G):* putcol 1.47 vs
1.08 ms (1.4×), getcol 1.47 vs 0.55 ms (2.7×), taql 12.03 vs 7.27 ms (1.7×).

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

## Interpretation (historical, 2026-09-24, casacure 3.8.5)

See the summary table at the top for the 3.8.8 ratios.

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
