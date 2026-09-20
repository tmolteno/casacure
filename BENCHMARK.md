# casacure benchmarks

Benchmark results for the `casacure-bench` console script and a row-count
scaling micro-benchmark, comparing casacure against real python-casacore on
the same machine and in the same process.

## Version and environment

| | |
|---|---|
| casacure wheel / crate | **3.8.2** (A.B.P policy: 3.8 = casacore interface, P = casacure patch) |
| source | `9cce6cc` (hot-path borrows) + deferred-flush write buffering + `casacure-bench` close-before-delete fix — the code measured here was released as 3.8.1.1 and renumbered 3.8.2 |
| build profile | **release** (`maturin develop --release`) |
| Python | 3.14.7 (CPython) |
| numpy | 2.4.6 |
| reference | python-casacore 3.8.1-1 (Debian, boost-python) |
| machine | Intel Core i5-8365U @ 1.60 GHz (laptop), 15 GB RAM, Linux 7.1.13 |
| date | 2026-09-21 |

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

## dask-ms chunked reads (memory footprint)

`scripts/bench_daskms_chunking.py` builds an MS with a large
`(nrow × nchan × ncorr) complex64` DATA column (~977 MiB, 250k rows ×
[128,4]), then reads it through `dask-ms` (`xds_from_table` +
`DATA.sum().compute()`) at several row-chunk sizes under a synchronous dask
scheduler, each config in a fresh subprocess so `ru_maxrss` reflects only
that read.

| chunk (rows) | peak RSS (MiB) | read ms |
|---|---|---|
| all (250k) | 4178 | 3270 |
| 125 000 | 2606 | 2134 |
| 25 000 | 1412 | 1471 |
| 5 000 | 1167 | 1451 |
| 1 000 | 1121 | 1725 |

Chunk size now **deliberately reduces memory**: a 1000-row chunk peaks at
~1.1 GiB vs ~4.2 GiB for a whole-column read (3.7×), and a *bounded* read
(2 000–10 000 rows of the 977 MiB column) peaks at the ~430 MiB
Python/dask-ms stack baseline — the actual column data adds ~0 because the
data files are memory-mapped and only the requested rows' pages are touched.

Before the fix, every `xds_from_table` eagerly `fs::read` the whole data
file per open, so peak RSS was ~the full column (~6 GiB observed) at every
chunk size — chunking did not reduce memory at all. Data files
(`table.f{seq}`, `table.f0i`, TSM tiles) are now `memmap2`-mapped.

A profiling pass (`perf record` during chunked ranged `getcolnp`) found the
per-element `aipsio::Reader` decode of SSM array cells at ~20 % of CPU;
replacing it with a direct `chunks_exact` + `from_{le,be}_bytes` pass made a
250-chunk scan 3.2× faster (2.4 s → 0.75 s) with no behavioural change.

## Workload

`casacure-bench` builds a table with two double scalar columns
(`TIME`, `WEIGHT`) and stores 20 000 rows (~160 KB per column, fully
resident / memory-mapped). The three measured ops are the whole-column,
single-`putcol`/`getcol` round-trips and one `taql` `SELECT * WHERE …
ORDERBY …` that scans all 20 000 rows. These measure the **python-conversion
and value-bridging cost**, not I/O: at this size the data is a memory-mapped
`Vec<u8>` and the dominant cost is converting between numpy arrays and
casacure cell values.

## Results — `casacure-bench` (5 runs, medians, release)

| op | casacure (release) | real casacore | ratio (cure/core) |
|---|---|---|---|
| putcol | 1.65 ms | 0.71 ms | **2.3×** |
| getcol | 0.59 ms | 0.36 ms | **1.6×** |
| taql WHERE+ORDERBY | 12.81 ms | 2.72 ms | 4.7× |

### Before / after the deferred-flush (write-buffering) optimization

| op | before (ms) | after (ms) | ratio before | ratio after |
|---|---|---|---|---|
| putcol | 4.33 | 1.65 | 7.2× | 2.3× |
| getcol | 0.66 | 0.59 | 1.9× | 1.6× |
| taql WHERE+ORDERBY | 9.41 | 12.81 | 3.2× | 4.7× |

Raw five single runs (ms) after the change:

| op | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| putcol | 1.54 | 1.54 | 1.68 | 1.68 | 1.65 |
| getcol | 0.29 | 0.61 | 0.44 | 0.59 | 0.59 |
| taql WHERE+ORDERBY | 12.81 | 13.02 | 12.72 | 13.45 | 12.17 |

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
at the 1.65 ms level and the flush happens once.

## Scaling — ns per cell vs row count (release, 1 double column)

Measured with a separate micro-benchmark (`putcol` of `np.arange(n)`, then
`getcol`), single DOUBLE scalar column, n rows, median of 7 repeats:

| n | casacure putcol | casacore putcol | casacure getcol | casacore getcol |
|---|---|---|---|---|
| 1 000 | 73 | 32 | 13 | 18 |
| 10 000 | 56 | 32 | 14 | 16 |
| 100 000 | 63 | 29 | 25 | 16 |

Before the optimization the same putcol cells were ~143–160 ns/cell; after
they are ~56–73 ns/cell (~2.4×, now within ~2× of casacore per cell).
Repeated `flush()` calls on a clean store measure ~0 ns/cell (no rewrite).

## Interpretation

- **putcol** is within **~2× of casacore per cell** (flat with `n`), down
  from ~4–5× before the flush deferral. The remaining constant-factor gap is
  the per-cell `RecordValue` packaging between the numpy view and the cell
  store; a deeper typed-buffer write path (store per-column buffers directly
  instead of one `RecordValue` per cell) would close most of it.
- **getcol** reaches parity with — and beats — casacore on small tables
  (13 vs 18 ns/cell at n=1000) and is ~1.6× overall on the 20k bench. Its
  mild superlinear growth (13 → 25 ns/cell from 1k → 100k rows) is the
  per-cell `RecordValue` clone plus the two-pass numpy conversion, amplified
  by allocator/working-set effects at ~MB scale; casacore does a single
  memcpy. Not I/O (the data is memory-mapped).
- **taql** is ~3–5×: the `SELECT *` result materialises through the same
  getcol path, plus (in this benchmark) the one flush charged to the op.

### Where the remaining gap lives

Both ops are dominated by the per-cell `RecordValue` round-trip between the
numpy buffers and the storage/convert layers, not by disk or algorithm. The
next step is a deeper **typed-buffer** read/write path — for a whole-column
single-value-type numpy call, batch-convert directly between the numpy
buffer and the column's typed storage, skipping per-element `RecordValue`
boxing for scalar columns. That should flatten getcol's scaling and bring
putcol to ~1.5×; the ceiling is python-casacore's native C++ memcpy path.
