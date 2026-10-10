# casacure benchmarks

Benchmark results for the `casacure-bench` console script and a row-count
scaling micro-benchmark, comparing casacure against real python-casacore on
the same machine and in the same process.

## Version and environment

| | |
|---|---|
| casacure wheel / crate | **3.8.32** (A.B.P policy: 3.8 = casacore interface, P = casacure patch), plus the unreleased tile-shape-defer fix described below |
| build profile | **release** (`maturin develop --release`) |
| Python | 3.13.5 (CPython) |
| numpy | 2.5.3 |
| reference | python-casacore 3.8.1 |
| dask-ms | the tmolteno fork, `origin/master` `a0d78e7` (`DASK_MS_BACKEND=casacure`, flush-once-per-column-write) |
| machine | schmalzburg: 12 cores, 62 GB RAM, Linux, **idle** (load < 1 before each section) |
| date | 2026-10-11 |

Every section below ran on 2026-10-11 on schmalzburg, both engines on the
same machine and (for `casacure-bench`) in the same process.  Only the
"before / after deferred-flush" table is a historical record from an earlier
machine, labelled as such.

This rerun needed one casacure fix to be possible: the flag-version tables
skarabina writes (`save:imported`) are variable-shape `TiledShapeStMan`
columns created at 0 rows with a `SPEC.DEFAULTTILESHAPE` shorter than the
cells that arrive later, and 3.8.32's dminfo-on-create validation (issue #20)
rejected the create.  casacore defers the hypercube until the first
`setShape`; the fix (unreleased, in `crates/casacure/src/tsm.rs`) defers the
requested tile the same way and keeps it in the column descriptor.  Without
it every skarabina run fails at its first stage-0 op.

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

## Summary: casacure vs python-casacore (3.8.32)

Ratios are casacure / casacore, so **< 1 means casacure is faster or
lighter**.

| workload | time ratio | peak-RSS ratio |
|---|---|---|
| dask-ms chunked read, 977 MiB DATA, 25 000-row chunks | 1.15 | 1.04 |
| dask-ms chunked read, 1000-row chunks | **0.55–0.70** | 1.37 (178 vs 130 MiB) |
| dask-ms write of a new MS, 256k rows, 2000-row chunks | 1.02 (3.89 vs 3.83 s) | **0.93** (250 vs 269 MiB) |
| skarabina flag + 32x average + `--msout`, MeerKAT scan | **0.92** (11.6 vs 12.6 s) | 0.99 (8.0 vs 8.2 GB) |
| skarabina flag, `--write-changed-only`, same scan | **0.96** | **0.68** (2.4 vs 3.6 GB) |
| `casacure-bench` small ops (20k rows, fully cached) | 1.3–1.7x (taql: 16x, see below) | — |

Where the work is I/O-shaped — dask-ms chunked scans, new-MS writes, the two
skarabina flagging pipelines — casacure is at parity or ahead.  On small,
fully cached tables the per-call bridging cost is 1.3x (putcol) and 1.7x
(getcol); the one large gap is a whole-column taql `SELECT *`, whose result
casacure materialises as a scratch table while casacore keeps it in memory
(see the `casacure-bench` section).

## dask-ms chunked reads: time and memory

`scripts/bench_daskms_chunking.py` builds an MS with a
`(250 000 × 128 × 4) complex64` DATA column (977 MiB), once, with real
python-casacore.  It then reads the MS through dask-ms (`xds_from_table` +
`DATA.sum().compute()`, synchronous scheduler) at several row chunks.  Each
read runs in a fresh subprocess, and its peak is that child's `/proc` VmHWM,
polled by the parent.  casacure medians are of 3 runs; casacore's of 2 (the
first casacore pass, on a cold page cache, is discarded).

| chunk (rows) | casacure peak | casacore peak | casacure read | casacore read |
|---|---|---|---|---|
| all (250k) | 2209 MiB | 2195 MiB | 693 ms | 602 ms |
| 125 000 | 1170 MiB | 1156 MiB | 677 ms | 602 ms |
| 25 000 | 339 MiB | 325 MiB | 709 ms | 617 ms |
| 5 000 | 178 MiB | 159 MiB | 873 ms | 787 ms |
| 1 000 | 178 MiB | 130 MiB | **625 ms** | 889–1133 ms |

Both engines bound memory by the chunk.  casacure sits ~13–48 MiB above
casacore: a larger import footprint and mapped-page slack.  At small chunks
casacore's per-call cost dominates and casacure is up to 1.8x faster; at
large chunks the two are within 15 %.

## dask-ms writes of a new table (chunk-bounded since 3.8.8)

`tests/test_write_scaling.py` writes a new MS through dask-ms in 2000-row
chunks (append mode).  The columns are DATA (complex64 [32,4]), FLAG and
WEIGHT_SPECTRUM (tiled), SIGMA/WEIGHT (StandardStMan arrays),
TIME/FLAG_ROW/ANTENNA1 (StandardStMan scalars) and SCAN_NUMBER/FIELD_ID
(IncrementalStMan).  Peak RSS (the child's VmHWM) and wall time; casacure is
the better of 2 samples (its CPU time is equal in both, the second sample's
extra wall is host tail), casacore the median of 2:

| rows (table) | casacure 3.8.32 | python-casacore 3.8.1 |
|---|---|---|
| 16 000 (23 MiB) | 166 MiB, 0.27 s | 190 MiB, 0.32 s |
| 64 000 (94 MiB) | 223 MiB, 0.95 s | 247 MiB, 1.03 s |
| 256 000 (375 MiB) | 250 MiB, 3.89 s | 269 MiB, 3.83 s |
| 1 024 000 (1.5 GiB) | 350 MiB, 19.1 s | 304 MiB, 16.2 s |

The append pattern (`addrows` per chunk) and the update pattern (ROWID,
`addrows(nrow)` up front) measure the same (the suite pins both).  Real
python-casacore reads every written cell back: it verified the
casacure-written tables (16k and 256k rows) exactly, every column and row.

*(Historical, laptop, casacure 3.8.8: 137/169/188/224 MiB and
0.13/0.42/1.45/6.5 s against python-casacore's 188/219/234 MiB and
0.19/0.67/2.4 s — casacure 0.60x casacore's time there.  On schmalzburg the
fork dask-ms's flush-once-per-column-write lands on both engines and
casacore's side gains, so the write race is now a dead heat.  3.8.7's
quadratic flush — 1865 MiB and 57.6 s at 256k rows — is the reason this
table exists; see `MEMORY.md`.)*

## End-to-end: skarabina on a MeerKAT scan

Host schmalzburg (12 cores, 62 GB, idle apart from the runs themselves).
The input is `.bench/scan1.ms`, a copy of scan 1 of a MeerKAT L-band MS:
143 716 rows x 2511 channels x 2 correlations, 11 GB.  The run is skarabina
`edf30ee` (v1.0.19-1) with its stage-0 flag list (`save:imported`, `autos`,
`uv-above 2500`, `nan`, `clip 0 100`, `spectral-window`) and `--summary`,
via `/usr/bin/time -v`:

```sh
DASK_MS_BACKEND=casacure /usr/bin/time -v skarabina --ms scan1.ms --summary \
  --clobber --time-average-factor 1 --frequency-average-factor 32 \
  --flag save:imported --flag autos --flag "uv-above 2500" --flag nan \
  --flag "clip 0 100" --flag "spectral-window ../bench/spectral-flags-L.yml" \
  --field-of-view 3.3deg --msout OUT
```

The two backends ran in alternating order (the casacore side with
`DASK_MS_BACKEND` set to a non-casacore value — skarabina selects casacure
by default when the variable is unset or empty):

| workload | casacure 3.8.32 | python-casacore 3.8.1 |
|---|---|---|
| + 32x frequency average, `--msout` | 11.5 s, 8.05 GB / 11.7 s, 8.17 GB | 12.7 s, 7.60 GB / 12.6 s, 8.15 GB / 11.1 s, 8.36 GB |
| + `--write-changed-only --msout` | 5.0 s, 2.43 GB / 5.1 s, 2.45 GB | 5.0 s, 3.54 GB / 5.5 s, 3.59 GB |

(each cell one run: wall time, peak RSS.  casacure's very first averaged run
of the session — 25.8 s, 5.0 GB — read the 11 GB input through a cold page
cache and is not counted; every run after it is warm.)

The averaged outputs of the two backends are identical in every readable
main-table column (all 23, DATA through UVW) and in SPECTRAL_WINDOW, ANTENNA,
FIELD and POLARIZATION, and the printed flagging reports match to the last
digit.  (FLAG_CATEGORY cannot be read by casacore in the input either.)

## Workload

`casacure-bench` builds a table with two double scalar columns
(`TIME`, `WEIGHT`) and stores 20 000 rows (~160 KB per column, fully
resident / memory-mapped). The three measured ops are the whole-column,
single-`putcol`/`getcol` round-trips and one `taql` `SELECT * WHERE …
ORDERBY …` that scans all 20 000 rows. These measure the **python-conversion
and value-bridging cost**, not I/O: at this size the data is a memory-mapped
`Vec<u8>` and the dominant cost is converting between numpy arrays and
casacure cell values.

## Results — `casacure-bench`, 2026-10-11 (5 runs, medians, release)

Same workload as before (two double scalar columns, 20 000 rows).

| op | casacure | real casacore | ratio (cure/core) |
|---|---|---|---|
| putcol | 1.32 ms | 1.01 ms | 1.3× |
| getcol | 0.79 ms | 0.47 ms | 1.7× |
| taql WHERE+ORDERBY | 39.37 ms | 2.46 ms | 16.0× |

Raw runs (ms, casacure / casacore): putcol 1.32/1.01, 1.32/1.01, 1.31/1.02,
1.31/1.02, 1.32/1.00; getcol 0.79/0.48, 0.80/0.47, 0.79/0.47, 0.79/0.48,
0.80/0.47; taql 39.20/4.98, 39.41/2.45, 39.30/2.45, 39.70/2.46, 39.37/2.46
(casacore's first taql is a cold-table outlier; the rest sit at 2.45–2.46).

The putcol/getcol gaps are the narrowest measured on any machine so far
(3.0×/3.5× on the 2026-09-26 laptop record below).  The taql number is new:
since 3.8.15 a TaQL result materialises as a real scratch table under the
temp dir (deleted when the handle closes — casacore keeps it in memory), and
on this op that materialisation dominates.  Measured separately on the same
table: the WHERE scan itself costs ~3 ms, a warm whole-result `SELECT *`
costs ~70 ms *before reading anything from it*, and writing the same rows
back through two `putcol` calls costs ~11 ms — the scratch write is going
row-by-row, not through the batch path.  On 3.8.8 (in-memory results) this
op measured 8.56 ms on a faster single core; the gap to close is batching
the scratch-table write, not the query engine.

*Historical (2026-09-26, Intel Core Ultra 7 258V laptop, casacure 3.8.8):*
putcol 1.20 vs 0.39 ms (3.0×), getcol 0.77 vs 0.22 ms (3.5×), taql 8.56 vs
1.79 ms (4.8×).

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

## Scaling — ns per cell vs row count (release, 1 double column)

Measured with a micro-benchmark (`putcol` of `np.arange(n)`, then `getcol`),
single DOUBLE scalar column, n rows, median of 7 repeats, schmalzburg
2026-10-11:

| n | casacure putcol | casacore putcol | casacure getcol | casacore getcol |
|---|---|---|---|---|
| 1 000 | 134 | 62 | 117 | 34 |
| 10 000 | 120 | 50 | 90 | 24 |
| 100 000 | 133 | 50 | 100 | 23 |

Unlike the 20k bench above, each repeat writes a *fresh* table, so putcol
pays the cell store's first-touch allocation (the 20k bench times a rewrite
of an already-stored column) — casacure ~2.5x and casacore ~1.2x above their
20k-bench per-cell costs for that reason.  The scaling is flat in `n` for
both engines on both ops.

(*Historical scaling on the 2026-09-24 laptop, casacure 3.8.5:* casacure
putcol 37/38/37, casacore putcol 50/50/50; casacure getcol 49/57/69,
casacore getcol 29/24/23 ns/cell at 1k/10k/100k.)

## Interpretation (2026-10-11, casacure 3.8.32, schmalzburg)

- **I/O-shaped work is at parity or ahead.**  Chunked dask-ms reads track
  casacore within 15 % on time at ≥ 25 000-row chunks and win up to 1.8x at
  1000-row chunks; new-MS writes are a dead heat in time and ~7 % lighter at
  256k rows; both skarabina pipelines finish 4–8 % faster, the
  `--write-changed-only` one in 32 % less memory.
- **The chunked-read footprint carries a fixed ~13–48 MiB premium** (import
  footprint + mapped-page slack): invisible at large chunks (1.04x), 1.37x
  at 1000-row chunks where the absolute numbers are small.
- **Small-table per-call costs keep narrowing**: putcol 1.3x, getcol 1.7x
  on the 20k bench.  The remaining cost is the per-cell `RecordValue`
  packaging between numpy and the cell store; casacore does a single
  memcpy.
- **taql `SELECT *` is the one wide gap (16x)**, and it is the scratch
  table: the query engine scans and filters at casacore-like speed (~3 ms
  for this WHERE), but writing the ~20k-row result into the scratch table
  goes row-by-row (~70 ms) where the batch write path costs ~11 ms.
  Batching that materialisation is the next step; casacore's in-memory
  result (1–2.8 ms) is the floor.

## Interpretation (historical, 2026-09-24, casacure 3.8.5)

- **putcol** was at or below casacore on that machine: ~1.4× overall on the
  20k bench (1.47 vs 1.08 ms) and 37–38 ns/cell vs casacore's 50–53 in the
  scaling micro-benchmark (flat with `n`). The deferred-flush write
  buffering removed the per-write whole-table rewrite; the small remaining
  constant overhead on the bench is the one `RecordValue` packaging per cell
  on its way into the buffered store.
- **getcol** was the remaining gap, ~2.7× overall on the 20k bench and
  2–2.5× per cell (49 → 69 ns/cell from 1k → 100k rows, superlinear growth;
  casacore is flat at ~23–29). It is the per-cell `RecordValue` clone plus
  the two-pass numpy conversion, amplified by allocator/working-set effects
  at ~MB scale; casacore does a single memcpy. Not I/O (the data is
  memory-mapped).
- **taql** was ~1.7× (vs 4.7× on the original machine — casacore's taql is
  comparatively slow there): the `SELECT *` result materialised through the
  same getcol path, plus (in this benchmark) the one flush charged to the
  op.

### Where the remaining gap lived (as of 3.8.5)

getcol was dominated by the per-cell `RecordValue` round-trip between the
numpy buffers and the storage/convert layers, not by disk or algorithm. The
typed-buffer read/write path landed since (getcolnp decodes SSM numeric
cells straight into the buffer), and the 2026-10-11 numbers above are the
result; the ceiling is python-casacore's native C++ memcpy path.
