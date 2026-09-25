# casacure memory footprint

How casacure uses memory for large tables, how that compares with real
casacore under dask-ms-style chunked reads, and what casacure does to keep a
long scan resident at ~the working set instead of the whole file.

## TL;DR

- Data files (`table.f{seq}`, `table.f0i`, TSM tiles) are **memory-mapped**
  (`memmap2`), so opening a table costs ~0 memory and a read touches only the
  pages of the rows it reads.
- After a **bulk** ranged read (`getcol` ≥ 512 rows) casacure drops the
  mapped pages (`madvise(MADV_DONTNEED)`) — the data was already copied into
  the caller's buffer — so a long chunked scan (dask-ms `xds_from_table`) is
  resident at ~the current chunk, not the whole column. This is casacure's
  analogue of casacore's bounded LRU storage-manager cache.
- Small / random reads (< 512 rows) do **not** drop pages, so the page cache
  is retained for repeated random access.
- A **writable** open allocates nothing either: `WritableTable` holds the row
  count and grows a column's cell store only when that column is written, so
  opening the output MS for update (the first thing dask-ms's
  write-changed-only path does) is at casacore parity — see "The write path"
  below.

## The measured numbers

`scripts/bench_daskms_chunking.py` builds an MS with a `(250000 × 128 × 4)`
`complex64` DATA column (~977 MiB) and reads it via dask-ms
(`xds_from_table` + `DATA.sum().compute()`, synchronous scheduler, one
config per fresh subprocess so `ru_maxrss` is exact).

**Full pass** (every row read), peak RSS:

| chunk (rows) | casacure now | casacure typed-buffer | casacure before typed | casacore |
|---|---|---|---|---|
| all (250k) | 2209 MiB | 2209 MiB | 3197 MiB | 2202 MiB |
| 125 000 | 1163 MiB | 1163 MiB | 2643 MiB | 1163 MiB |
| 25 000 | 327 MiB | 327 MiB | 1411 MiB | 331 MiB |
| 5 000 | 164 MiB | 164 MiB | 1166 MiB | 165 MiB |
| 1 000 | 164 MiB | 164 MiB | 1121 MiB | 136 MiB |

casacure is now at **casacore parity across the board**, including the
single whole-column read (`chunk = all`, where the pre-typed path held a
third full-size buffer: the per-cell `Vec<RecordValue>` decode). Chunked
full-column passes sit at the ~Python/dask-ms stack baseline in both
engines.

**Bounded read** (a 10k-row window = 39 MiB, and 100k = 391 MiB, read at
1000/10000-row chunks), peak RSS:

| rows read | casacure | casacore |
|---|---|---|
| 10 000 | 362 MiB | 366 MiB |
| 100 000 | 429–529 MiB | 366 MiB |

Both engines stay at the ~350–430 MiB stack baseline; neither grows with the
column size.

## The write path (dask-ms write-changed-only)

Read parity is not the whole contract: dask-ms opens the **output** MS
writable before its first chunk and then patches FLAG chunk by chunk
(skarabina `--write-changed-only`). Opening a table for update must therefore
cost ~nothing.

Peak RSS of a writable open with **no writes at all**:

| MS opened writable | casacore | casacure before | casacure now |
|---|---|---|---|
| synthetic 100k rows × 5 cols (fixed [32,4] DATA) | 80.4 MiB | 162.0 MiB | **80.4 MiB** |
| `bpcal.ms` (429 257 rows × 25 cols) | 80.3 MiB | **800.8 MiB** | **80.3 MiB** |

`WritableTable` now keeps the row count and allocates a column's cell store
only when that column is written; an unwritten row reads back as the column
default and a flush writes that default out. Before, `addrows` cloned one
buffered default cell per row per column — O(rows × columns) memory, 800 MiB
on `bpcal.ms` before a single value was written — and a flushed column's
per-row slots were blanked rather than released.

End-to-end, the skarabina changed-only flag workload on `bpcal.ms` (VmHWM,
this machine, same input state):

| backend | wall | peak RSS | vs casacore |
|---|---|---|---|
| casacore 3.8.1 | 23.8 s | 565 MiB | 1.0 |
| casacure before | 51.6 s | 2265 MiB | 4.01× |
| casacure now | 49.4 s | 864 MiB | **1.53×** |

And the memory suite's synthetic TSM MS (100k × [32,4], DATA+FLAG, dask-ms
per-chunk flush):

| flag write | casacore | casacure before | casacure now |
|---|---|---|---|
| 2000-row chunks | 127.5 MiB | 266.7 MiB (2.09×) | **125.7 MiB (0.98×)** |
| 50 000-row chunks | 202.7 MiB | 401.8 MiB (1.98×) | 253.3 MiB (1.25×) |

**Remaining gap: the large-chunk working set.** A whole 100 MiB column read in
one chunk peaks 417 MiB in casacure vs 331 MiB in casacore (1.26×), and the
50 000-row flag write likewise 1.25×; both are inside the suite's
`1.5 × casacore + 15 MiB` bound. Wall time is untouched by this work
(casacure is still ~2.1× casacore on the flag workload — the separate speed
lead).

## The write path (dask-ms writing a NEW table)

dask-ms writes a new output table (an averaged `--msout`, a flag-version
backup) chunk by chunk. It either appends each chunk's rows (`addrows` +
`putcol`) or adds all rows up front (`addrows(nrow)`, rows with a ROWID), and
it flushes after every column. Each such flush sees a table with more rows
than its files. casacure used to regenerate the whole table from buffered
cells at every one, so the peak grew with the table and time was quadratic
in it.

casacure now **grows the files in place** (`crates/casacure/src/grow.rs`):
- TSM tile files are zero-extended.
- StandardStMan appends default buckets and rewrites its index.
- IncrementalStMan re-encodes its last bucket and appends new ones.

The written chunk is then patched in:
- TSM cells are patched byte-wise.
- StandardStMan scalar cells are patched in their buckets, and array records
  in the array file.
- IncrementalStMan re-encodes only the buckets that hold the chunk.

Resident memory is one chunk of cells; I/O and time are linear in the rows.

`tests/test_write_scaling.py` measures DATA/FLAG/WEIGHT_SPECTRUM + SSM
scalars/arrays + ISM columns in 2000-row chunks (peak RSS, write time):

| rows (table) | casacure before | casacure now | casacore |
|---|---|---|---|
| 16 000 (23 MiB) | 242 MiB, 0.39 s | 137 MiB, 0.13 s | 188 MiB, 0.19 s |
| 64 000 (94 MiB) | 633 MiB, 4.1 s | 169 MiB, 0.42 s | 219 MiB, 0.67 s |
| 256 000 (375 MiB) | 1865 MiB, 57.6 s | 188 MiB, 1.45 s | 234 MiB, 2.4 s |
| 1 024 000 (1.5 GiB) | — | 224 MiB, 6.5 s | — |

The append and update patterns measure the same. A layout that cannot be
grown in place keeps the whole-table rewrite. Such layouts are:
- a casacore-written file with several SSM column groups;
- string or record cells in a grown StandardStMan;
- a casacore-tiled hypercube.

## History

| stage | memory behaviour |
|---|---|
| eager `fs::read` per open | every open copied the whole data file into RAM; a 1000-row chunk of a 1 GiB column cost ~6 GiB RSS |
| `memmap2` mapping | open is O(1); chunked reads touch only their pages; a *full* pass left the file resident (~1× column, 1.1 GiB floor) |
| + `MADV_DONTNEED` on bulk reads | full pass streams at ~the working set (164 MiB), no whole-file floor |
| **+ typed-buffer `getcolnp`** | SSM numeric cells decode straight from the map into the numpy buffer (no per-cell `Vec<RecordValue>`), so a single whole-column read holds ~1 full buffer + the read window (2.2 GiB, casacore parity) |
| **+ lazy cell store** | a writable open allocates nothing (the row count is authoritative; a column's cells are allocated on first write), so the dask-ms changed-only write path reaches parity: 801 → 80 MiB to open `bpcal.ms` writable, and 4.0× → 1.5× peak RSS on the flag workload |
| **+ growth in place** | a flush on a table with more rows than its files appends default rows to each storage manager instead of regenerating the table, so writing a NEW table through dask-ms is chunk-bounded: 1865 → 188 MiB and 57.6 → 1.45 s for 256k rows, below casacore |

## How it works

`crate::datafile::Buffer` holds a data file as `Owned(Vec<u8>)` (small /
in-memory / tests) or `Mapped(memmap2::Mmap)`.

- `Buffer::from_file` maps the file read-only; unix builds also hint
  `madvise(MADV_SEQUENTIAL)` (best-effort; affects read-ahead, does not by
  itself bound residency).
- `Table::getcol` (the ranged column read behind `getcol`/`getcolnp`/\
  `getvarcol`) calls `Table::drop_data_file_pages()` when `nrow ≥ 512`,
  which `MADV_DONTNEED`s every data file's mapping. The cell values were
  already copied into `Vec<RecordValue>` / the numpy buffer, so the pages are
  clean and re-fault from disk on the next read — no data loss, just page
  cache eviction.
- **Typed-buffer `getcolnp`:** for a read handle over a StandardStMan
  numeric column, `getcolnp` no longer materialises a `Vec<RecordValue>`.
  `Table::getcol_raw` walks the mapped cells (`array_cell_region` /
  `scalar_cell_raw`, borrowed slices, no copy) and the pyo3 layer fills the
  caller's numpy buffer cell-by-cell (`fill_numpy_raw`), also dropping mapped
  pages as a long scan advances. Scalar/array bool (bit-packed) is handled;
  strings, records, ISM and TSM, and variable-shape arrays keep the generic
  path.
- A read handle stays an open snapshot (documented): a concurrent flush
  rewrites the file, so a stale handle reads its own captured state.
- **Lazy, sparse write store:** `WritableTable` keeps the table's row count
  and, per column, a map from row to buffered cell that holds ONLY the rows
  written (or loaded) — never one slot per table row. `addrows` only bumps
  the row count; `putcell` buffers the cell flagged as pending, and `flush`
  patches only the pending rows and then releases the column's buffer. (A
  per-table-row slot vector, re-allocated after every flush, made each
  one-row dask-ms chunk flush O(table rows); `tests/flush_cost.rs` pins the
  allocation per one-row flush as independent of the table size.) A read of a row with no buffered cell (never
  written, or released by an earlier flush) answers with the column default or
  the on-disk value, so nothing has to be pre-filled. That is what makes
  `open_for_update` — dask-ms's writable open of the output MS — O(1) instead
  of O(rows × columns).

## Trade-offs / caveats

- **One-pass scans are ideal.** Pages dropped after a bulk read are re-read
  from disk if touched again, so workloads that repeatedly bulk-scan the same
  huge column pay page faults on each pass (the same trade-off a bounded
  LRU cache makes when it evicts).
- **Random single-row access keeps the cache** (below the 512-row
  threshold), so coordinate-style lookups do not thrash.
- **Dropping is process-wide** to the file's page cache: another handle or
  process mapping the same table may re-fault the evicted pages (correct,
  just I/O). Write handles read from the in-memory store, so they are
  unaffected.
- **`chunk = all` (a single whole-column `getcolnp`) is at casacore parity**
  (2.2 GiB): the typed path holds ~1 full buffer plus a read window. The
  generic (non-raw) `getcol` still materialises per-cell values (~3.2 GiB
  for this column); dask-ms uses `getcolnp`, which is the typed path.
- **Large chunks hold more than casacore** (1.14–1.28× on the synthetic MS;
  see "Remaining gap" above). Small chunks — the skarabina setting — are at
  parity or below.
- The knob is a compile-time policy, not configurable at runtime.

## Reproduce

```sh
# casacure backend, and TMPDIR on disk so taql temp tables don't fill a small tmpfs
DASK_MS_BACKEND=casacure TMPDIR=/var/tmp python scripts/bench_daskms_chunking.py
# same script without DASK_MS_BACKEND runs the identical graph on real casacore
```

The same numbers, plus the casacure-vs-casacore `casacure-bench` op table,
are in `BENCHMARK.md`.
