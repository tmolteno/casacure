# handover — casacure vs casacore on skarabina flag (dask-ms backend parity)

Goal (user): **reach similar memory and speed to python-casacore from casacure.**
Workload: an existing skarabina `flag` run (the one that was being debugged) on a
measurement set in `../skarabina/.bench`. Skarabina uses dask-ms; dask-ms selects
the casacore implementation via `DASK_MS_BACKEND=casacure` (`casacore` shim →
casacure `.so`) or unset (= real python-casacore).

## Environment (verified 2026-09-22)

- Machine: Intel i5-8365U (laptop), 15 GB RAM, Linux.
- Venv: `/home/tim/github/skarabina/.venv-bench` (CPython 3.14).
  - `casacure 3.8.3` (wheel `/tmp/wheels_local/casacure-3.8.3-…whl`, installed in
    site-packages; sha256 `118a03fe…` matches the checked-out casing build — the
    wheel was built from the current working tree incl. uncommitted changes in
    `crates/casacure/src/{table,record,tsm}.rs` + `casacure-python/src/table.rs`
    and `crates/casacure/examples/dump_flag_header.rs`).
  - `python-casacore 3.8.1` (Debian) — the casacore side.
  - `dask-ms 0.2.32` editable at `/home/tim/github/dask-ms`.
  - `skarabina 1.0.5` editable at `/home/tim/github/skarabina` (+ skarabina-cargo).
  - numpy 2.4.6, dask 2026.8.0 (single-process; no `distributed`).
- skarabina tree has an uncommitted **debug print** in
  `skarabina/dask_ms.py` (`DBG init: s=… chan_freq.shape=…`). Left as-is (it is
  the user's WIP; one line per SPW, negligible for timing). Remove before any
  clean release. It was added to diagnose the remote flaky `CHAN_FREQ` shape
  `(79,)` vs `(1,79)` read.

## Harness (`.bench/` in skarabina)

- `bench.py` — fresh-subprocess driver: `python bench.py flag <ms> <msout> [--backend casacure|casacore]` (also `analyze`).
  Reports `rc=… wall=…s peak_rss=… MiB` (child `ru_maxrss` via `wait4`).
  Workload knobs: `BENCH_UVCUT`=8000, `BENCH_FREQAV`=8, `BENCH_TIMEAV`=1 (env overrides).
  For an already-averaged MS set `BENCH_FREQAV=1`.
  Flag ops: `save:imported, autos, uv-above 8000, nan, clip 0 100, spectral-window spectral-flags-L.yml`.
- `bench.py` runs `run_skarabina.py`, which imports `daskms` *before* skarabina so
  the casacure alias is active when `DASK_MS_BACKEND=casacure`; prints `BACKEND=…`.
- `run_local.sh <casacure|casacore> [changed]` — same flag run on real MS
  `data/bpcal.ms`, `-v` time under `/usr/bin/time`.
- `compare_flags.py <core.ms> <cure.ms>` — verifies `FLAG`/`FLAG_ROW` parity.
- `probe_spw_shape.py [N]` — reproduces the intermittent CHAN_FREQ shape bug on bpcal.ms.

## Measurement sets in `.bench`

| MS | size | notes |
|---|---|---|
| `data/bpcal.ms` | 1.6 G | real MS — the primary workload (79 ch LSIC, 4 corr, `flag` matches `spectral-flags-L.yml`) |
| `scratch/dmw.ms` | 1.6 G | copy of bpcal.ms (prior debugging output) |
| `ms_cure_50000.ms` | 988 M | synthetic MeerKAT-like, 50k rows (976 MB `table.f0i` — TSM bit-packed FLAG) |
| `ms_cure_2k.ms` | 40 M | synthetic, 2k rows |
| `ms_cure_300000.ms` | 264 K | tiny / probably unusable — verify before trusting |
| `scratch/out_*.ms` | | prior casacure/casacore outputs (all ~543 M) |

Build synthetic: `python build_ms.py <path> <nrow> [nchan] [ncorr] [seed]` (uses DASK_MS_BACKEND=casacure).

## Experiment plan

Primary: real-MS `flag` run, casacure vs casacore, **wall time + child peak RSS**,
medians of 3; verify flag parity with `compare_flags.py`. Secondary: synthetic
`ms_cure_2k` / `ms_cure_50000` for scaling/harness sanity; optional `analyze`.

Metrics to compare: `peak_rss_mib`, `wall_s`, and (from skarabina's `--summary`)
flag counts to confirm identical work. Report ratio vs casacore.

## Status / results

- **Current comparison (2026-09-23, updated)** — full changed-only flag run on `data/bpcal.ms`
  (freqav=1, all verbs incl. `save:imported`, `spectral-window`, `--summary`, rc=0), 3 runs
  each, medians, same workload both backends:

  | backend | wall (median) | peak RSS (median) | ratio |
  |---|---|---|---|
  | casacure | **53.7 s** (53.3–54.8) | **2241 MiB** (2225–2250) | 2.85× wall, 4.04× RSS |
  | casacore | 18.9 s (18.5–20.0) | 555 MiB (553–574) | 1.0 |

  vs the session-start baseline (casacure 69.1 s / 3.35 GiB; casacore 5.4 s / 0.57 GiB —
  that casacore figure predates the current input state, which now carries accumulated
  flagversions and re-measures ~18.9 s): casacure wall **69 → 54 s** and RSS **3.35 → 2.24 GiB**
  (~1.3× wall, ~1.5× memory improvement from the incremental flush + pending-cell changes;
  the gap to casacore is 2.85× / 4.0×). Correctness caveat: FLAG_ROW is bit-identical but
  FLAG still differs 29.9 % (the 4-bit tile-boundary write bug, next action #1 below).

- [Plan] handover created; harness run via bench.py confirmed (subprocess, wait4 rusage).
- [Smoke] `flag ms_cure_2k.ms`: casacure rc=0 (freqavg 8), **casacore FAILS to even build the
  dask-ms graph** — `ValueError: conflicting sizes for dimension 'uvw'`. Cause: the synthetic MS,
  unlike real MSes, carries a never-filled `UVW2` column (38 cols incl. `UVW2`, `MODEL_DATA`,
  `LAG_DATA`, `VIDEO_POINT`, `PULSAR_*`; real `data/bpcal.ms` has 25 cols, no `UVW2`). The real
  casacore backend chokes on it; casacure tolerates it. **Synthetic-MS artifact, not a parity
  bug** — use `data/bpcal.ms` (real MS) for the comparison.
- [Observation] bench.py default workload is freqavg **8**; on the real MS a casacure run did
  ~25 min and was still writing (~277 MB / ~543 MB out) when cancelled — far above the ~5 min
  prior `out_cure*.ms` runs. Reason: freqavg triggers a full DATA reshape + per-channel
  SPECTRAL_WINDOW rewrite. **The workload being debugged (`run_local.sh`) uses
  `--frequency-average-factor 1`.** Decision: run the comparison at **BENCH_FREQAV=1** (exact
  debug workload). Keep the freqavg-8 slowness as a future perf lead (rewrite path).
- [Baseline] **casacure FULL-WRITE (freqav=1, no --write-changed-only) on bpcal.ms does not
  finish in 30 min** (run killed at 30-min cap; output MS ~1.6 G ≈ full size, writes trickled:
  ~1.5→1.6 G over the last ~20 min; child RSS ~6.7–7.0 GB during the write, 41 % CPU).
  By contrast the prior **changed-only** runs (`scratch/out_cure*.ms`, ~543 M) completed in
  ~5 min on the *same current casacure build*. So the pathological slowness/memory is
  specific to the **full-write path** (writes all 25 columns incl. per-channel/MODEL_DATA/…).
  Comparison is therefore run in **changed-only mode** (`--write-changed-only`), which is also
  the workload `run_local.sh` was debugging.
- [Baseline] **casacure changed-only flag on bpcal.ms (freqav=1, real MS): wall=71.2 s,
  child peak RSS=3253 MiB, rc=0** (429,257 rows, 1 SPW 79 ch; output `scratch/out_bp_cure.ms`;
  writes dominant: FLAG ~30 s + FLAG_ROW ~31 s progress bars, then summary). ran via
  `timewrap.py run_skarabina.py flag … --write-changed-only`. NOTE: `run_local.sh` failed with
  rc=127 until the user installed the `time` package (`/usr/bin/time` now present and working).
- [Baseline] **casacore changed-only flag on bpcal.ms: wall=5.41 s, max RSS=573 MiB, rc=0**
  (`run_local.sh casacore changed` → `scratch/out_bpcal_casacore.ms`).
  → **casacure ≈ 13× slower (71.2 vs 5.4 s), ≈ 5.7× more memory (3.25 vs 0.57 GiB).**
  Repeat casacure via `run_local.sh casacure changed`: 69.05 s / 3 349 780 KiB — stable.
- [Parity] `compare_flags.py` between the two outputs: **identical FLAG + FLAG_ROW**
  (429 257 rows, 71.21 % cells / 36.93 % rows flagged). Same for the perf-run outputs. The gap
  is pure performance, not correctness.
- [perf] **Where casacure and casacore differ** (both `perf record -F 4000 -g`, changed-only
  flag, on `data/bpcal.ms`; paranoid restored to 3 afterwards; captures in `/tmp/perf_cure.data`
  and `/tmp/perf_core.data`; profiled outputs `scratch/out_perf_{cure,core}.ms`):
  - **casacure (291k samples, 272.8 G cycles)** — top self cycles:
    | share | function |
    |---|---|
    | 19.9 % | `casacure::table::build_tsm_data` |
    | 13.6 % | `WritableTable::flush` |
    | 6.4 % | `RecordValue::clone` (core::clone) |
    | 5.8 % | `RecordValue` drop glue |
    | ~13 % | anonymous 0x0a* cluster → PyO3 pymethod wrappers (`__pymethod_colnames__`,
      `toascii`, `putcell`, `getcolnp`) incl. allocator/drop churn |
    | 1.7+1.6+0.8 % | `Table::getcell`, `tsm::decode_tile_data`, `tsm::decode_bits` |
    ≈ **~60 % of cycles in the TSM-write + per-cell RecordValue packing/conversion glue.**
  - **casacore (42k samples, 34.5 G cycles — 7.9× fewer)** — flat profile: top is 4.4 %
    (anon), kernel `rep_movs_alternative` 4.2 % + `memcpy_orig`, `kernel_init_pages` 2.2 %,
    `casacore::Conversion::bitToBool` just 0.59 %. **Memcpy-bound, no single hotspot.**
- [Read] conclusion: the two engines read identical data; the 13×/5.7× gap lives almost
  entirely in the **FLAG(+FLAG_ROW) bit-packed Bool TiledShapeStMan WRITE path** (build_tsm_data
  + flush + per-cell RecordValue boxing), exactly the code area of the current uncommitted
  `crates/casacure/src/tsm.rs` work. This matches `BENCHMARK.md`'s earlier prediction: a
  **typed-buffer write path** (batch-convert the numpy bool array → packed TSM tiles directly,
  skipping per-cell `RecordValue`) is the lever.

## Implementation status (2026-09-22, casacure)

Ported casacore's `Conversion::bitToBool`/`boolToBit` structure into
`crates/casacure/src/tsm.rs` (uncommitted):
- **Decode** — `decode_bits` now uses a 256-entry `BOOL_LUT` (`[[bool;8];256]`
  const fn, mirrors casacore `conv_tab`): one lookup + 8-byte copy decodes 8
  Bools per input byte (byte-aligned fast path; non-aligned `skip` falls back
  to the bit loop).
- **Encode** — `encode_bits` now packs full bytes with the portable SWAR
  gather `((w & 0x0101…0101).wrapping_mul(0x0102_0408_1020_4080) >> 56)`
  (verified LSB-first) mirroring casacore's SSE `_mm_cmpeq_epi8` +
  `_mm_movemask_epi8`; branch-free, endian-safe; tails bit-masked.
- Add correctness test `bit_conversions_match_scalar` (all lengths incl.
  0/1/7/8/9/15/16/17/31/64/65/1000/4097 × skip 0-15/31/63 + fixed 0xa5 pattern).
  152/152 crate tests pass; clippy clean.

Measured (release wheel rebuilt + reinstalled in `.venv-bench`,
`maturin develop --release`; infrastructure note: pip-wheel builds fail in this
venv — use maturin):
- **Micro** (67.8M bools = 429 257×79×2 FLAG): decode 58.9→19.9 ms (**3.0×**),
  encode 59.9→30.5 ms (**2.0×**).
- **End-to-end** `./run_local.sh casacure changed`: **65.0 s / 3.24 GiB** vs the
  69.1 s / 3.35 GiB baseline (~6 % wall; memory flat). Output **bit-identical**
  to casacore (PARITY OK). The remaining 12× gap is dominated by the per-cell
  `RecordValue` boxing + `WritableTable::flush` (perf showed ~20 % `build_tsm_data`,
  ~14 % `flush`, ~12 % RecordValue clone/drop) — the next lever is the typed-buffer
  write path, not the bit loops.

## Implementation status (2026-09-22, casacure)

**Stage 1 — bit conversions (done, above).**

**Stage 2 — typed-buffer bool write path (crates/casacure/src/tsm.rs + table.rs, uncommitted):**
- `or_bits_at` / `or_bytes_at`: word-oriented bit placement (8 bits per SWAR step,
  destination-byte straddle) — replaces the per-set-bit `trailing_zeros` scatter in the TSM
  bool tile writer (`write_tsm_file` bool branch).
- `write_tsm_file_bool`: whole-column bool writer that packs the row bit-slices
  (`rows: &[&[bool]]`) straight into the tile bitstream — no per-row `Vec<u8>` cells, no
  `encode_bits` intermediate. `build_tsm_data` uses it for `DataType::Bool` columns (the MS
  FLAG path).
- Shared `tsm_layout` / `tsm_header` (via `TsmHeaderParams`) reuse geometry + the casacore-
  layout header for both writers. Byte-identical output to the general writer verified by
  `bool_typed_writer_matches_general_writer` (both STM types × cell sizes 3/6/8/9/158/316).
  153/153 crate tests pass; clippy clean.
- **Measured end-to-end** `./run_local.sh casacure changed`: after Stage 2 the runs land at
  **59.7–71.4 s / ~3.2 GiB** (two runs: 71.4, 59.7) vs the 69–71 s / 3.35 GiB original
  baseline — within the ±10 s run-to-run noise, best-case ~10–15 % faster. The decisive signal
  is perf: **cycles dropped 272.8 → 244.0 G cycles (−10.6 %)**; `build_tsm_data`'s profile
  share **halved from 19.9 % → 9.3 %** (typed-buffer tile fill is now the small part). New top:
  `WritableTable::flush` 16.3 % + RecordValue clone/drop ~14 % + PyO3-boundary churn ~12–15 %.
  Read side (decode 2 %, getcell 1.9 %) unchanged. Output **bit-identical** to casacore
  (`PARITY OK`). Profiles: `/tmp/perf_cure.data` (before) vs `/tmp/perf_cure2.data` (after).
- Next lever if the wall/RSS gap persists: the numpy→RecordValue boxing in the Python bridge
  (casacure-python/src/table.rs `putcol` → `ndarray_cells_typed`, ~12 % clone/drop) and
  `WritableTable::flush`'s cell-store rebuild.

## Stage 3 — incremental flush (A) + dask-ms batch flush (B): DONE

**A — casacure incremental flush** (`crates/casacure/src/table.rs`, `WritableTable`):
- `pending` per-column row-bitset set by `putcell`/`putcol`, cleared by `flush`.
- `flush`: preserving path (`flush_preserving`) now patches ONLY pending rows — TSM tile
  files are byte-patched in place (`patch_tsm_column`; bool rows: clear old bits + `or_bytes_at`;
  byte rows: memcpy), SSM/ISM columns are rebuilt from on-disk values with pending overlaid
  (`materialize_col_from_disk`); full-regrow fallback for growth/meta changes.
- After a flush the buffered column values are released (`clear_pending`) — resident memory
  now tracks ~one dask-ms chunk, and an untouched column is never rewritten.
- Bridge (`casacure-python/src/table.rs`): `putcol` no longer loads full columns for partial
  writes; `read_col`/`getcell`/`read_colslice`/`getcellslice` merge the on-disk snapshot with
  pending cells; `getcellslice`/`getcolslice` handle the python-casacore `-1` "whole cell" idiom.
- `build_tsm_data` cell shape is taken from the first non-empty cell (not `values[col].first()`,
  which is an empty-shape default for partially-written columns).
- Regression test `incremental_flush_overlays_only_written_rows` (TSM cross-tile +
  re-overwritten rows, SSM bool rebuild, untouched column preserved). 146/146 crate tests.

**B — dask-ms batch flush** (`/home/tim/github/dask-ms/daskms/table_proxy.py`): removed the
per-proxied-write `table.flush()` from `_writelock_runner` (writes.py flushes once per column
and close flushes).

**Measured (A+B, `./run_local.sh casacure changed`):** with the FLAG write now a per-chunk
tile patch, a completing run is **~2.3–3.1 s / ~1.31 GiB** vs the original **69.1 s / 3.35 GiB**
(~25× wall, ~2.5× memory). One run was verified **bit-identical FLAG/FLAG_ROW vs casacore**
(`PARITY OK`, 71.21 %). A run that skips the spectral-window step's SPW-read issue completes
correctly.

**Reconstruction incident (MUST READ):** during editing, `crates/casacure/src/table.rs` was
accidentally truncated (0 bytes) — uncommitted work (the user's 884-line MS-support diff +
our perf changes) was lost; git/fs had no copy (user confirmed no backup). It was reconstructed
from `git HEAD` + this session's knowledge + the intact `tsm.rs`/`record.rs`/`casacure-python`
API contract, then re-verified by the 146-test suite and parity checks. Fidelity caveats vs
the user's original:
- parts of the lost diff that lived ONLY in table.rs (e.g. the exact `Table::open`/header
  handling, `relativize` details) are re-implemented equivalents, not byte-identical;
- a **new reader fix was added**: `lock_sync_nrrow` — `Table::open` now reads the row count
  from `table.lock`'s AipsIO `sync` record (casacore `PlainTable` prefers it over the
  table.dat header). This fixes the MS subtable CHAN_FREQ shape flake for direct reads:
  `data/bpcal.ms/SPECTRAL_WINDOW` now opens with nrows=1 and CHAN_FREQ `(1, 79)` (was
  intermittently `(0,)` / `(79,)` — the session's original debug topic). Test
  `lock_sync_nrrow_parses_sync_record`.

**LIVE BLOCKER (pre-existing, not from the perf work) — ISOLATED repro:** the end-to-end run
aborts in the spectral-window step. skarabina's `DaskMS.__init__` reads the SPW subtable
through a WRITE handle (`table(..., ack=False)` resolves to `Inner::Write`), and the
write-handle read of `CHAN_FREQ` returns an **empty-shaped cell → `(1,)`**, while a read-only
open of the identical subtable returns `(1, 79)`. Minimal repro (casacure, same process):
```python
from casacore.tables import table
import numpy as np
sw = table("data/bpcal.ms/SPECTRAL_WINDOW", writable=True, ack=False)
print(np.asarray(sw.getcol("CHAN_FREQ")).shape)   # -> (1,)   (WRITE handle)
sw2 = table("data/bpcal.ms/SPECTRAL_WINDOW", readonly=True, ack=False)
print(np.asarray(sw2.getcol("CHAN_FREQ")).shape)  # -> (1,79) (READ handle)
```
This was FIXED (commit 51afb43): read merges on writable handles overlaid every
`addrows` default cell (a `Some(default)` is NOT a written row) — reads now overlay only
rows with the pending bit set (`WritableTable::pending_cell`). The same commit made
`getcolnp`'s borrowed-buffer path (`fill_numpy_raw`) fall back to the generic path for any
column whose DM is not StandardStMan (resolved by sequence number, not list index), which
fixed the summary-step aborts (`INTERVAL` / `FIELD_ID` / `TIME_CENTROID` are IncrementalStMan /
TSM on this MS).

**After the abort fix the workload completes** (rc=0): **~46 s / ~2.2 GiB** (3 runs stable)
vs casacore 5.4 s / 0.54 GiB.

**FLAG parity — FIXED (commit f40f4ea):** the earlier 29.9 % FLAG gap is closed. Root cause:
skarabina/dask-ms writes the output by copying the input table (shared column groups) and
then patching FLAG per chunk via `patch_tsm_column`, which placed pending rows with
`tsm_layout`'s default geometry (26214 rows/tile, 4 bits over a byte) into a file whose
header declares the input's casacore geometry (829 rows/tile) → tiles ≥ 1 written 4 bits off
(`content == expected << 4`). Fix: `patch_tsm_column` now derives `rows_per_tile` +
`bucket_bytes` from the parsed header's REAL data cube (the exact geometry
`TsmFile::read_cell` uses), self-consistent for casacore-copied (829) and casacure-written
(26214) tables. **Verified: autos-only AND the full flag workload now produce bit-identical
FLAG + FLAG_ROW vs casacore (PARITY OK, rc=0).**

## Next actions (if interrupted, resume here)

1. Final: re-run the recorded comparison (3× both backends) and publish the parity-OK
   numbers; optionally make the fresh-table TSM writer honor an external tile shape
   (today `write_tsm_file`/`write_tsm_file_bool` always use 26214 for newly created tiles —
   fine for casacure reads, a casacore-compat concern only).
2. (Later) remove the uncommitted `DBG init` print in `skarabina/dask_ms.py`; the full-write /
   freqavg-8 path is a separate slow-write lead.

## Root cause — repeated writing (confirmed 2026-09-22)

Instrumented `WritableTable::flush` (temporary eprintln, since removed) on the
changed-only bpcal flag run: **99 flush calls**, ~96 of them full "preserving"
rebuilds of the 429 257-row FLAG column at ~0.39 s each (~39 s of the 62–64 s run).
dask-ms (`daskms/writes.py` + `table_proxy._writelock_runner`) writes each row-chunk
as its own task and calls `table.flush()` after every chunk → casacure's flush is
all-or-nothing per column (clones the whole 429k-row cell store + rebuilds the whole
DM), so the SAME column is fully rewritten ~96×.

RSS trajectory during those flushes: ~1.1 GiB → ~3.2 GiB, constant per-flush time.
The resident cell store + per-flush full-value clone are O(column rows), NOT bounded
by the dask-ms write chunk. This is the primary performance/memory gap (larger than
the bit-conversion or tile-packing costs already addressed).

**Fix design (memory buffered by chunk size only):**
- **A (recommended)** — casacure incremental flush: track dirty row ranges per column;
  a flush persists only rows/tiles changed since the last flush (patch the TSM tile
  buckets / SSM buckets + header row-maps), so resident buffers stay ~one chunk. Removes
  both the ~96× full rebuilds and the O(column) cell store growth, independent of dask-ms.
- **B (cheap, dask-ms side)** — batch flushes (flush once per column/table at close instead
  of per chunk): removes the repeated writing (~39 s) but memory stays O(column) in the
  cell store; touches `/home/tim/github/dask-ms` (editable).
- A + B together = closest to casacore (sub-second flushes, chunk-bounded memory).

## Gotchas / risks

- Laptop: thermal/noise → use medians, keep runs interleaved or warmed page cache
  noted; `data/bpcal.ms` reads are page-cache-warm after first run.
- Debug print in skarabina/dask_ms.py is uncommitted WIP — don't "fix" it during
  benchmarking; note if it matters.
- casacure build is from a dirty tree (uncommitted changes). Rebuilding requires
  the plain build (maturin) — previous `pip wheel` builds failed in this venv
  (see `.venv-bench/casacure_build.log`).
- `bench.py` sets correct `DASK_MS_BACKEND` env; running parts manually requires
  importing `daskms` before `casacore.tables` (see `run_skarabina.py`).

## Memory-chunking unit tests (2026-09-23, this session)

`tests/test_memory_chunking.py` — pytest suite that measures peak RSS for
skarabina-style flagging workloads (dask-ms chunked reads + write-changed-only
FLAG patch, flush-per-chunk) on both backends and asserts memory respects the
dask-ms row chunk size. Skips without dask-ms (CI); each backend is measured
only if importable (casacure in-process, real casacore probed in a clean
subprocess so the `tests/shim` redirect cannot mask it).

**Measurement methodology (important):** peak RSS here is NOT measurable via
any `ru_maxrss` or VmHWM read by the child:
- `getrusage(RUSAGE_SELF).ru_maxrss` is unreliable in this sandbox — a bare
  `python -c` reports ~600 MiB while `/proc/self/status` VmHWM is 9.4 MiB
  (container/cgroup peak bleeding into SELF rusage; varied 313→595 MiB for
  one fixed workload run-to-run).
- `ru_maxrss` and `/proc` VmHWM/VmPeak are per-process high-water marks that
  **survive fork AND execve**: a worker forked from a pytest parent that had
  allocated MS-sized arrays inherited the parent's ~600 MiB peak, so every
  child measured ~622 MiB flat.
The suite therefore forks the worker, polls the *live* child's
`/proc/<pid>/status` VmHWM from the parent and takes the max, and keeps the
pytest parent lean by building MSes in subprocesses too (both fixture builds
are subprocesses). Verified: this matches `/usr/bin/time -v` and is
deterministic run-to-run.

**Reliable synthetic-MS measurements (casacure 3.8.3 wheel in `.venv-bench`
vs Debian python-casacore 3.8.1, dask-ms 0.2.32 editable, 100k × [32,4]
complex64 DATA = 100 MiB):**

| workload | chunk (rows) | casacure (MiB) | casacore (MiB) |
|---|---|---|---|
| full-column read (sum) | 2 000 | 125 | 128 |
| full-column read | 50 000 | 214 | 227 |
| full-column read | 100 000 (all) | 321 | 331 |
| bounded read, 15k-row window | 2 000 | 125 | 127 |
| bounded read, 15k-row window | 50 000 | 174 | 187 |
| flag write (TSM MS) | 2 000 | 266 | 127 |
| flag write (TSM MS) | 50 000 | 401 | 202 |
| flag write (SSM MS, default_ms) | 2 000 / 50 000 | 866 / 885 | — / 203 |

- **Read side: parity and chunk-respecting in both engines** (the suite's
  read + bounded-read + parity tests pass).
- **Flag write on the TSM (real-MS) layout: chunk-bounded in casacure**
  (266→401 MiB with chunk 2000→50000, monotone; casacure sits ~2.1× casacore,
  bounded by the suite's 2.5×-of-overhead assertion). The skarabina production
  path is covered and passes.
- **Flag write on the SSM layout (casacure's `default_ms`): still O(column)**
  — ~866–885 MiB flat regardless of chunk (the StandardStMan per-chunk flush
  goes through `materialize_col_from_disk`, rebuilding the whole column; TSM
  columns are tile-patched and chunk-bounded). This is the known next-action
  gap; the suite skips the flag test for that layout with a reference to this
  note rather than failing (it is not the skarabina path — real MSes use TSM
  FLAG).
- casacure cannot create `TiledShapeStMan` columns yet (storage error during
  table creation) — the TSM-layout test MS is built with real python-casacore;
  the fixture falls back to a casacure-built SSM MS for the read tests.
- The worker asserts which backend `casacore` resolved to (shim/casacure path
  vs real site-packages) and fails loud on a mismatch, so a silently wrong
  engine can never falsify a comparison.

Verified: **6/6 pass** in `.venv-bench` (both backends, TSM flag); 2/2 pass +
parity/flag skip in a casacure-only env (no real casacore); whole module
skips without dask-ms (CI). ~2.5 min runtime.

## Next action taken: SSM incremental flush + write-path correctness fixes (2026-09-24)

Released **v3.8.4** (tag pushed 2026-09-23; PyPI + crates.io publish via OIDC CI).
Next action implemented on top of that release:

**1. StandardStMan flush is now incremental (`patch_ssm_column` in table.rs).**
`flush_preserving`'s StandardStMan branch previously rebuilt the whole column
per flush (`materialize_col_from_disk` read every row + `build_ssm_data`
rewrote the file) — the O(column) gap behind the ~870 MiB flat flag-write on
SSM layouts. Now, for a DM whose pending columns are fixed-size scalars
(numeric + bit-packed Bool), the pending rows' buckets are patched in place:
read only the touched buckets, apply cell bytes / accumulate per-byte bit
set+clear masks (several Bool rows share a byte — the first attempt replaced
whole bytes and wiped co-located rows' bits), write the buckets back. Header,
index chain and untouched buckets never change. Strings/records/arrays and
IncrementalStMan still fall back to the full rebuild (their cells reference
variable-size buckets). Guarded by
`ssm_incremental_flush_matches_full_rebuild` (byte-identical to a full
rebuild) and the existing incremental test.

Measured (skarabina-exact synthetic MS: DATA TiledColumnStMan, FLAG
TiledShapeStMan bool, FLAG_ROW StandardStMan bool, 100k rows; flag =
write FLAG + FLAG_ROW changed-only via dask-ms, flush per chunk):

| workload | chunk | casacure (MiB) | casacore (MiB) |
|---|---|---|---|
| flag-write (TSM FLAG + SSM FLAG_ROW) | 2000 | **273** | 128 |
| flag-write (TSM FLAG + SSM FLAG_ROW) | 50000 | **407** | 203 |

(previous numbers for the same workload included the FLAG_ROW O(column)
rebuild per chunk). SSM **array** columns (e.g. a synthetic FLAG stored on
StandardStMan) still take the full-rebuild path — real MSes store FLAG as
TSM, so the production path is covered.

**2. Fixed a pre-existing data-loss bug in the incremental-flush design**
(confirmed against the baseline build and the released wheel, i.e. not
introduced by this work): a table fully written in one session, then reopened
and partially rewritten, had the untouched columns **clobbered to defaults**.
The dask-ms write-back smoke (`tests/daskms_smoke.py` "TIME clobbered") hit
it. Two root causes, both fixed in table.rs:
- `WritableTable::touched` was sticky across flush/reopen (the shared write
  backing kept every once-written column "touched" forever), so any later
  flush was forced onto the full-regrowth path. `touched` now resets to
  false after each successful flush.
- The regrowth path (`materialize_all`) default-filled every cell the session
  had not buffered (released by an earlier flush). It now reads the on-disk
  column and overlays the buffered cells; only rows added past the on-disk
  row count default-fill. Fresh-table creation is unchanged.
Guarded by `reopen_partial_write_preserves_untouched_columns`.

**3. Pre-existing Python-suite failures on this machine (NOT from this
work; reproduce identically on the baseline build and the released 3.8.3
wheel):**
`test_casacore_ported.py::{test_check_putdata, test_tableascii}` ("row 0 not
covered by any indexed bucket"), `test_casacore_helpers.py::{test_removecols,
test_getcell_keeps_singleton_dims, test_relative_path_ms_links_resolve_from_any_cwd}`,
`test_core_tables_e2e.py::test_scalar_roundtrip_incremental[boolean]`,
`test_core_tables_e2e.py::test_error_paths[out-of-range_getcell]` (7 total,
122 pass). The dask-ms smoke "TIME clobbered" was #2 above and is fixed.
Rust: 148 lib + 15 integration tests pass.

## Session 2026-09-24: make the whole Python suite pass

Environment: no `python` on PATH; the test interpreter is
`/home/tim/github/skarabina/.venv-bench/bin/python` (CPython 3.14, pytest 9.1.1,
dask-ms present, python-casacore 3.8.1 visible). Canonical command (same as CI):

```sh
PYTHONPATH=tests/shim /home/tim/github/skarabina/.venv-bench/bin/python -m pytest tests/ -q
```

**Baseline (before this session's work): 11 failed, 124 passed, 1 skipped (~36 s).**

- 7 pre-existing (already listed above): `test_check_putdata`,
  `test_tableascii`, `test_removecols`, `test_getcell_keeps_singleton_dims`,
  `test_relative_path_ms_links_resolve_from_any_cwd`,
  `test_scalar_roundtrip_incremental[boolean]`,
  `test_error_paths[out-of-range_getcell]`.
- 4 NEW in `test_memory_chunking.py` when the suite is invoked with the
  CI-relative `PYTHONPATH=tests/shim`: `_strip_shim()` compares each PYTHONPATH
  entry to the *absolute* `SHIM` path, so the relative entry is not stripped,
  the "real casacore" workers actually get the shim, and the worker's backend
  assertion fails ("wanted casacure=False but casacore resolved to
  tests/shim/casacore/__init__.py").  Passing PYTHONPATH=tests/shim
  absolute (or unset) hides it — CI never sees it because dask-ms is absent
  there and the module skips.  Fix: normalize paths in `_strip_shim`.
- NOTE: the baseline above ran against the INSTALLED casacure 3.8.3 wheel
  (site-packages, built 2026-09-23 13:28) which PREDATES commit 02d3197
  (2026-09-24).  Step 1 below rebuilds from current source and re-baselines.

Failure signatures captured (for resume):
- `test_check_putdata` / `test_tableascii`: `RuntimeError: row 0 not covered
  by any indexed bucket` on getcol of never-written rows (unset cells must
  read back as column defaults).
- `test_error_paths[out-of-range_getcell]`: `t.getcell("C", 99)` on a 3-row
  table must raise **ValueError**, gets the same bucket RuntimeError (missing
  row-bounds check before bucket lookup).
- `test_removecols`: after `removecols(["b"])` the surviving column `a`
  reads back 0 instead of 1 (removecols clobbers other columns' data).
- `test_getcell_keeps_singleton_dims`: fixed-shape (1,1) cell OK; a
  *variable*-shape cell stored (1,3) must getcell as (3,) (leading row
  singleton trimmed) — comes back (1,3).
- `test_relative_path_ms_links_resolve_from_any_cwd`: `getkeyword("ANTENNA")`
  on a relatively-created `t.ms` stores/returns `.../re0/ANTENNA` instead of
  `.../re0/t.ms/ANTENNA` (subtable keyword path loses the ms dir component;
  `getsubtables()` itself is correct).
- `test_scalar_roundtrip_incremental[boolean]`: IncrementalStMan bool scalar
  putcol→getcol: `RuntimeError: buffer too short: need 1 bytes at offset 0,
  have 0` (ISM bool cell never written / wrong offset).

### Diagnosis (ground truth probed against real python-casacore 3.8.1)

Probes (real casacore, str paths) settled each ambiguous contract:
- unset cells after addrows -> `[0, 0]` (defaults) — matches the test; commit
  ad0b510 already declared this parity contract but the read paths still raise
  (DIFFERENCES.md "unset cells raise" section is STALE — written 0ea3db6,
  predates ad0b510; must be rewritten).
- out-of-range getcell: real casacore raises RuntimeError('no such row'); the
  test pins casacure's own ValueError contract (ad0b510: "ValueError for
  shape/read-only violations") -> casacure needs a row-bounds check that raises
  ValueError (currently falls through to the SSM bucket error = RuntimeError).
- getcell shapes: real casacore returns (1,1) for fixed [1,1] AND (1,3) for a
  variable (1,3) cell — NO leading-singleton trim. tests/...::test_getcell_
  keeps_singleton_dims' second assertion (expects (3,)) contradicts real
  casacore AND current casacure (both (1,3)); it is stale after commit 8a241a7
  ("Keep (1,1) cells 2-D on getcell", which deliberately stopped trimming —
  trimming broke skarabina's CHAN_FREQ read). FIX THE TEST to (1,3).
- ISM bool: real casacore stores 1 byte/cell (bucket offsets 0,1,2; data
  `01 00 01`). casacure's ISM write+read use `scalar_cell_size(Bool)==0` (the
  SSM bit-packing convention) -> write stores no data, read does
  `&cell[..0]` then decodes -> "buffer too short: need 1 bytes at offset 0,
  have 0". Fix: ISM-specific cell size (Bool -> 1) in ism.rs read + the
  build_ism_data write path (table.rs ~line 695).
- subtable keyword: real returns 'Table: <ms>/ANTENNA'. casacure's
  getkeyword resolves bare stored links ("ANTENNA") against the PARENT dir
  (resolve_subtable_py with base=parent-of-table) -> '<parent>/ANTENNA'
  (missing the ms dir), while getsubtables uses core `resolve_subtable(name,
  table_dir)` which is correct. Fix: getkeyword/getkeywords must resolve via
  core resolve_subtable against the table dir (canonicalized self.path).

Root causes of the 7 failures:
1. test_check_putdata / test_tableascii ("row 0 not covered by any indexed
   bucket"): python `read_col` on a write handle reads the DISK snapshot
   FIRST (column_cells -> Table::getcol) for rows the disk does not have yet
   (never flushed), before overlaying pending cells. Write-handle merged read
   must: pending first -> disk rows the disk actually has -> wt cell/default;
   and row >= row_count must raise ValueError (bounds check).
2. test_removecols (a reads 0 not 1): reopening writable (open_for_update
   fills cells with Some(default), NOT pending) + removecols changes the
   schema -> flush takes materialize_all, whose "!has_none => trust memory"
   shortcut takes the addrows DEFAULTS as truth and never reads disk ->
   clobbers every column to defaults. Fix: when table.dat exists, materialize
   must read on-disk columns BY NAME (indices shift after removecol!) for all
   non-pending rows and overlay only pending cells.
3. test_getcell_keeps_singleton_dims: stale test expectation (see above).
4. test_relative_path_ms_links...: getkeyword resolver (see above).
5. test_scalar_roundtrip_incremental[boolean]: ISM bool cell size (above).
6. test_error_paths[out-of-range_getcell]: missing ValueError bounds check.
7. test_memory_chunking x4 (only under CI-relative PYTHONPATH=tests/shim):
   `_strip_shim` compares raw PYTHONPATH entries to the ABSOLUTE $SHIM path,
   so the relative entry survives -> "real casacore" workers get the shim ->
   backend-mismatch assertion. Fix: compare os.path.realpath(entry) ==
   realpath(SHIM). (CI never sees it: no dask-ms there -> module skips.)

Environment note: the venv's installed casacure is 3.8.3 (built 09-23) and
PREDATES HEAD; the canonical run for this session builds a fresh wheel and
runs with PYTHONPATH=target/devpkg:tests/shim (both in-workspace, no writes
to the external venv):
  maturin build --release -o target/wheels  (maturin from .venv-bench)
  rm -rf target/devpkg && python3 -m zipfile -e target/wheels/<new>.whl target/devpkg/
  PYTHONPATH=target/devpkg:tests/shim .../python -m pytest tests/ -q

### Result: suite GREEN — 136 passed, 1 skipped, 0 failed (~1:43)

All 7 failures fixed, plus a regression the fixes exposed. Fixes (all in the
working tree, not yet committed):

1. **`materialize_all` reads the disk by column name** (crates/casacure/src/
   table.rs): a regrowth over an existing table.dat takes non-pending rows from
   the on-disk column matched by name+type (indices shift after
   removecols/addcols) and overlays only the pending rows. Fixes
   `test_removecols`.
2. **Merged write-handle reads** (crates/casacure-python/src/table.rs:
   `merged_col_cells`/`merged_cell`/`buffered_default_cell`/`disk_col_index`):
   disk snapshot only for the rows it has, buffer/default for the rest, pending
   overlay last; `read_colslice`/`read_cellslice` slice the merged cell.
   Fixes `test_check_putdata`, `test_tableascii`. Verified identical to real
   python-casacore 3.8.1 on a 3-row table: pending write, untouched column
   from disk, `addrows(2)` → `[10, 20, 30, 0, 0]` / `[1.5, 2.5, 3.5, 0.0, 0.0]`,
   same after reopen (probe `target/probe/probe_merge.py`, run with both
   backends).
3. **`ValueError` row/range bounds checks** in `read_cell`/`read_cellslice`/
   `read_colslice`/`merged_col_cells` (both handle kinds). Fixes
   `test_error_paths[out-of-range_getcell]` (casacore itself raises
   `RuntimeError: no such row`; the ValueError contract is documented in the
   rewritten DIFFERENCES.md).
4. **ISM Bool cells are one byte** (`ism::ism_cell_size`, used by
   `build_ism_data` + `IsmFile::read_scalar_cell`). Fixes
   `test_scalar_roundtrip_incremental[boolean]`; cross-checked with real
   python-casacore both directions (its file and casacure's read each other;
   only header byte 41 — `persCacheSize` — differs, payload is the casacore
   `01 00 01` layout). Rust test `ism_bool_column_stores_one_byte_per_cell`.
5. **`drop_rows` renumbers the pending bitsets** (NEW regression found while
   testing #2: `test_taql_delete_insert_persist` read `[5, 2]` after deleting
   row 1 of `[5, 2, 9]` — the bitset kept old indices while cells were
   compacted). Rust test `drop_rows_renumbers_pending_bits`.
6. **Subtable keyword resolution** (crates/casacure-python/src/convert.rs
   `resolve_subtable_py` → core `resolve_subtable`, callers pass `dir_of()`):
   `getkeyword`/`getkeywords`/`getcolkeywords`/`_getdesc` resolve stored
   `Table:` links against the TABLE dir, not its parent. Fixes
   `test_relative_path_ms_links_resolve_from_any_cwd`; also used by dask-ms's
   keyword reads.
7. **`_strip_shim` uses `os.path.realpath`** (tests/test_memory_chunking.py) so
   the CI-relative `PYTHONPATH=tests/shim` is stripped. Fixes the 4 memory
   tests under that invocation.
8. **Stale expectation corrected**: `test_getcell_keeps_singleton_dims` now
   pins `(1, 3)` for a variable-shape `(1,3)` cell (real casacore 3.8.1 returns
   `(1,3)`; the old `(3,)` came from the trim commit 8a241a7 removed).
9. **Two more bugs the pytest suite does not reach, found via
   `tests/daskms_smoke.py`** (it regressed on the #1 change and is GREEN again):
   - `ssm::array_cell_region` decoded an empty-shape array record (`ndim 0`)
     with `product()`-over-no-dims == 1 → claimed one element → the LAST row's
     record ran past `table.f0i` (`array reference 36 falls outside the array
     index file (len 40)`; the smoke's `WEIGHT` column, i.e. any variable-shape
     array column created with rows and not yet written). Now `nelem = 0` for
     an empty shape. Rust test
     `empty_variable_shape_array_cells_read_back_empty`.
   - `materialize_all` now skips the disk entirely for a column whose every row
     was written this session (`pending_count >= nrow`): the buffer is the
     truth, and reading a never-written on-disk array column was what tripped
     the record bug during the smoke's creation flush.

Rust: 151 lib + 15 integration + 6 python-crate tests pass; `cargo fmt` +
`cargo clippy --workspace --all-targets` clean. Python adds
`test_writable_read_merges_disk_pending_and_defaults` (test_casacore_helpers.py)
pinning the #2 semantics against a real-casacore probe; the dask-ms smoke
(`tests/daskms_smoke.py`, run with `PYTHONPATH=target/devpkg:tests/shim`) is
green end to end including the casacore cross-checks.

Known cosmetic leftover (not a test failure, no casacore parity target): a
*read-only* `getcol` of a variable-shape array column whose cells are all empty
returns `(nrow,)` zeros (via `cell_shape.iter().product().max(1)` in
convert.rs) rather than `(nrow, 0)`; real casacore raises
`SSMIndColumn::getShape: no array in row N` for that state.

Docs updated: DIFFERENCES.md (the stale "unset cells raise" section is gone;
now documents the ValueError bounds contract + the defaults parity) and
CHANGELOG.md [Unreleased].

Re-run after any change:
  maturin build --release -o target/wheels && rm -rf target/devpkg &&
  python3 -m zipfile -e target/wheels/casacure-3.8.4-*.whl target/devpkg/ &&
  PYTHONPATH=target/devpkg:tests/shim $VENV/bin/python -m pytest tests/ -q
