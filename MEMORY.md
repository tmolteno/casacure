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

## The measured numbers

`scripts/bench_daskms_chunking.py` builds an MS with a `(250000 × 128 × 4)`
`complex64` DATA column (~977 MiB) and reads it via dask-ms
(`xds_from_table` + `DATA.sum().compute()`, synchronous scheduler, one
config per fresh subprocess so `ru_maxrss` is exact).

**Full pass** (every row read), peak RSS:

| chunk (rows) | casacure now | casacure before drop | casacore |
|---|---|---|---|
| all (250k) | 3197 MiB | 4185 MiB | 2202 MiB |
| 125 000 | 1658 MiB | 2643 MiB | 1163 MiB |
| 25 000 | 425 MiB | 1411 MiB | 331 MiB |
| 5 000 | 180 MiB | 1166 MiB | 165 MiB |
| 1 000 | 164 MiB | 1121 MiB | 136 MiB |

Chunked full-column passes now sit at **casacore parity** (164 vs 136 MiB at
a 1000-row chunk ≈ the Python/dask-ms stack baseline). The remaining gap at
`chunk = all` is the single whole-column read materialising the full result
(per-cell `RecordValue` decode + the numpy buffer) inside one call — the
same call shape costs casacore 2.2 GiB; the "deeper typed-buffer getcolnp"
would close it.

**Bounded read** (a 10k-row window = 39 MiB, and 100k = 391 MiB, read at
1000/10000-row chunks), peak RSS:

| rows read | casacure | casacore |
|---|---|---|
| 10 000 | 362 MiB | 366 MiB |
| 100 000 | 429–529 MiB | 366 MiB |

Both engines stay at the ~350–430 MiB stack baseline; neither grows with the
column size.

## History

| stage | memory behaviour |
|---|---|
| eager `fs::read` per open | every open copied the whole data file into RAM; a 1000-row chunk of a 1 GiB column cost ~6 GiB RSS |
| `memmap2` mapping | open is O(1); chunked reads touch only their pages; a *full* pass left the file resident (~1× column, 1.1 GiB floor) |
| **+ `MADV_DONTNEED` on bulk reads** | full pass streams at ~the working set (164 MiB), no whole-file floor |

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
- A read handle stays an open snapshot (documented): a concurrent flush
  rewrites the file, so a stale handle reads its own captured state.

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
- **`chunk = all` still materialises a whole-column `getcol`** inside one
  call (~3.2 GiB here vs casacore's 2.2 GiB); chunking avoids this, which is
  the dask-ms contract.
- The knob is a compile-time policy, not configurable at runtime.

## Reproduce

```sh
# casacure backend, and TMPDIR on disk so taql temp tables don't fill a small tmpfs
DASK_MS_BACKEND=casacure TMPDIR=/var/tmp python scripts/bench_daskms_chunking.py
# same script without DASK_MS_BACKEND runs the identical graph on real casacore
```

The same numbers, plus the casacure-vs-casacore `casacure-bench` op table,
are in `BENCHMARK.md`.
