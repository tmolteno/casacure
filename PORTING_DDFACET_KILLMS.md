# Porting DDFacet and killMS to casacure

What `../DDFacet` and `../killMS` actually use casacore for, what casacure
already covers, what has to be added, and what is better replaced by
astropy / pyephem — plus the Rust-crate integration that lets killMS's own
native extension read the MS directly.

## TL;DR

- **Tables**: the entire `pyrap.tables` surface both codebases use is already
  in casacure. One module-level helper is missing:
  `casacore.tables.addImagingColumns` (used by `killMS/Data/ClassMS.py`).
- **Quanta**: used thinly (`quantity`, `get_value`, `to_unix_time`) — all
  present. One risk item: ISO8601 **date-string** quantities for the
  `TimeRange` selection.
- **Measures**: seven calls (`epoch`, `direction`, `position`, `do_frame`,
  `posangle`, `measure`, `get_value`) in `ClassFITSBeam` / `GiveDate` need a
  new `casacure.measures` subset. Everything else that looks like measures
  use is already astropy or pyephem.
- **Images**: `pyrap.images` is a separate subsystem and **out of scope**
  here.
- **Rust crate**: killMS already runs its solver as an in-process thread pool
  around its own `killms-core`/`native` crates. That crate can depend on
  `casacure` directly and do the MS chunk reads/writes in Rust, in parallel,
  with no Python marshalling per call — the multiprocessing payoff.

## 1. Audit: what the two codebases call

### 1.1 Rust extensions in the consumers (the crate integration point)

| Repo | Crates | What they do | How Python calls them |
|---|---|---|---|
| killMS | `rust/killms-core` (pure Rust) + `rust/native` (cdylib `killMS._native`) | `dot`, gridder points, `assembleJacobian`/`jhjGram`, `extractAntennaData`, `flagZeroKernel`, `giveScalarTaskData`, predict/model kernels | `from killMS._native import ...`; wrapped in an **in-process thread pool** (`ClassWirtingerSolver._giveInProcessPool`, killMS#14 — the old forked `WorkerAntennaPool` is retired) |
| DDFacet | `rust/ddfacet-gridder` (cdylib + rlib) | BDA gridding backend, batched per-facet fan-out on `rayon` threads | `ClassDDEGridMachine` / `ClassFacetMachine` via pyo3 |

Both already use `rayon` inside the native extension.

killMS's I/O and solver run in **threads in one process**, so a single Rust
`Table` handle can be shared across the pool — casacure's `Table` is
`Send + Sync` (`crates/casacure/src/table.rs:986-1010`). This is the direct
crate-dependency opportunity.

DDFacet's I/O runs on `AsyncProcessPool` (`DDFacet/Other/AsyncProcessPool.py`)
— **forked** worker processes (`multiprocessing.Process`, `_start_worker`,
lines 624-633) that each open their own table handle. Its workers use the
Python `casacure` module; the forked children inherit the parent's page cache
for the mapped data files, which is already a win. Its gridder crate could
also grow a `casacure` dependency later if profiling justifies it.

### 1.2 Import surface

| Module | DDFacet | killMS | Notes |
|---|---|---|---|
| `pyrap.tables` / `casacore.tables` | `DDF.py`, `Restore.py`, `Data/ClassMS.py`, `Data/ClassJones.py`, `Data/ClassVisServer.py`, `Data/ClassEveryBeam.py`, `Imager/ClassWeighting.py`, `ToolsDir/ModEstimateMemory.py`, `ToolsDir/ModRotate.py`, tests | `Data/ClassMS.py`, `Data/ClassVisServer.py`, `Data/ClassWeighting.py`, `Data/ClassBeam.py`, `Weights/W_*.py`, `kMS.py`, `BLCal.py`, `ClipCal.py`, `AQWeight.py`, `SmoothSols.py`, `InterpSols.py`, `Simul/DoSimul.py`, `Simul/MakeClusterCat.py`, `Predict/ClassImageSM2.py`, `Predict/PredictGaussPoints_NumExpr.py` | load-bearing |
| `pyrap.quanta` | `Data/ClassMS.py`, `Data/PointingProvider.py`, `Data/ClassFITSBeam.py`, `ToolsDir/ModRotate.py` | `Data/ClassMS.py`, `Simul/MakeClusterCat.py` | thin |
| `pyrap.measures` | `Data/ClassMS.py`, `Data/ClassFITSBeam.py`, `Imager/ClassMontblancMachine.py` | `Data/ClassMS.py`, `Simul/MakeClusterCat.py` | thin but real |
| `pyrap.images` | `Restore.py`, `Imager/ClassCasaImage.py`, `Imager/ClassDeconvMachine.py`, `Imager/ClassImageNoiseMachine.py`, `Imager/ClassFacetMachineTessel.py`, `Imager/MultiSliceDeconv/*`, `Imager/SSD3/*`, `Imager/MSMF/*`, `ToolsDir/ModMosaic.py`, `ToolsDir/ModFitPSF.py`, `ToolsDir/casapy2bbs.py`, `fits2png.py` | `Simul/MakeModelImage.py`, `Predict/ClassImageSM2.py` | **out of scope** |
| `astropy` | `astropy.time.Time`, `astropy.io.fits`, `astropy.io.ascii`, `astropy.coordinates.SkyCoord` | `astropy.io.fits` | already used |
| `pyephem` | `ephem.Date` in `ClassMS.GiveDate` | `ephem.Date` in `ClassMS.GiveDate`, `Simul/MakeClusterCat.GiveDate` | trivial |

### 1.3 Tables API census (instance methods on `table(...)` objects)

Total call sites across both repos (receivers `t`, `tab`, `table_all`, `to`,
`tf`, `ta`, `ta_ddid`, `ta_spectral`, `tp`, `tspw`, `tField`, `t0`, `t1`):

| Method | DDFacet | killMS | casacure |
|---|---|---|---|
| `getcol(name[, startrow, nrow])` | 72 | 146 | ✅ |
| `putcol(name, value[, startrow, nrow])` | 18 | 43 | ✅ |
| `close()` | 94 | 72 | ✅ |
| `colnames()` | — | 2 | ✅ |
| `nrows()` | 7 | 1 | ✅ |
| `getkeyword(name)` | 9 | 4 | ✅ |
| `putkeyword(name, value)` | — | 2 (`PutLOFARKeys`) | ✅ |
| `getcoldesc(colname)` | 6 | 5 | ✅ |
| `addcols(desc)` | 6 | 5 | ✅ |
| `removecols(names)` | 1 (AddCol, commented) | — | ✅ |
| `getcolslicenp(col, npbuf, blc, trc, inc, startrow, nrow)` | 6 | — | ✅ |
| `getcolslice(col, blc, trc, inc, startrow, nrow)` | 2 | — | ✅ |
| `query(taql_where)` | 2 | 1 | ✅ |
| `sort("TIME")` | 1 (`GiveMainTable`) | — | ✅ |
| `flush()` | 6 | 2 | ✅ |

### 1.4 Module-level helpers

| API | Where | casacure |
|---|---|---|
| `pyrap.tables.addImagingColumns(msname, ack=False)` | `killMS/Data/ClassMS.py:1213` (`PutCasaCols`); `DDFacet/Data/ClassMS.py:1542` (commented) | ❌ **missing** |
| `maketabdesc` / `makescacoldesc` / `makearrcoldesc` / `makecoldesc` | implicit via `addcols` descriptors | ✅ |
| `tableexists` / `tabledelete` / `tablecopy` | not called | ✅ (unneeded) |
| `default_ms` / `default_ms_subtable` | not called by these two | ✅ (unneeded) |

### 1.5 Quanta

| Call | Where | casacure |
|---|---|---|
| `qa.quantity(value, unit)` / `qa.quantity(str)` | ClassMS (both), PointingProvider, ClassFITSBeam, ModRotate, MakeClusterCat | ✅ |
| `q.get_value([unit])` | ClassFITSBeam, GiveDate, PointingProvider | ✅ |
| `q.to_unix_time()` | ClassMS (TimeRange sel), PointingProvider | ✅ |
| `q.get_unit()` / `q.get()` / `q.canonical()` | incidental | ✅ |
| `qa.quantity("2000/01/01/00:00:00")` date strings | `TimeRange` option (ISO8601) | ⚠️ **to verify** |

### 1.6 Measures

| Call | Where | casacure |
|---|---|---|
| `pm.measures()` | ClassMS (both), ClassFITSBeam, MakeClusterCat | ❌ **now added** (`casacure::measures`) |
| `me.epoch('utc', q)` | ClassMS (both), MakeClusterCat | ❌ → added |
| `me.direction('J2000', q, q)` / `('AZEL','0deg','90deg')` / `('AZELGEO',...)` / `('itrf',...)` | ClassFITSBeam | ❌ → added |
| `me.position('itrf', q, q, q)` | ClassFITSBeam | ❌ → added |
| `me.do_frame(m)` | ClassFITSBeam | ❌ → added |
| `me.posangle(dir1, dir2)` → quantity | ClassFITSBeam | ❌ → added |
| `me.measure(dir, 'AZELGEO')` → dict | ClassFITSBeam | ❌ → added |
| `me.get_value(measure_dict)` → list of quantities | ClassFITSBeam | ❌ → added |

`Imager/ClassMontblancMachine.py` also uses `me.uvw` / `me.baseline` /
`me.doptofreq` — but only under the optional montblanc extra; out of scope.

## 2. Classification

### 2.1 Already in casacure (tables) — no new work
The whole `pyrap.tables` surface is implemented and tested
(`crates/casacure-python/src/table.rs`, `helpers.rs`). What is **not yet
tested against these exact call patterns**: the `getcolslicenp` +
`cs_tlc/cs_brc/cs_inc` tuple-slice convention, the `t.query(...).sort("TIME")`
chain, and `addcols` with an IMAGING_WEIGHT-shaped `ColDesc` (option 4 /
shape [nchan]).

### 2.2 Replaceable by astropy / pyephem (drop the casacore call)

1. `qa.quantity(x).to_unix_time()` in `Data/ClassMS.py` TimeRange selection →
   `astropy.time.Time(x, format='unix').unix` (both packages already import
   `astropy.time.Time`).
2. `ephem.Date(JD).datetime()` in `GiveDate` (both `ClassMS` +
   `Simul/MakeClusterCat`) → `astropy.time.Time(JD, format='jd').to_datetime()`;
   drops `pyephem`.
3. `qa.quantity(str(x)+'s').to_unix_time()` in `PointingProvider` →
   `astropy.time.Time(float(x), format='unix')`.
4. `astropy.coordinates.SkyCoord` is already used in `ClassFITSBeam` for the
   pointing-centre parse; the `dm.direction('J2000', ...)` wrapper around it
   is only there to feed `dm.posangle`/`dm.measure`, which the new
   `casacure.measures` covers.

After these, the only casacore things left are `pyrap.tables` (= casacure) and
the seven measures calls.

### 2.3 New casacure functionality (added this session)

#### A. `casacure.tables.addImagingColumns(msname, ack=True)` (+ `removeImagingColumns`)

Reproduce python-casacore's `casacore/tables/msutil.py::addImagingColumns`
exactly (ground truth: `casacure/.venv/lib/python3.14/site-packages/
casacore/tables/msutil.py`):

- open `table(msname, readonly=False, ack=False)`;
- add `MODEL_DATA`, `CORRECTED_DATA` (clone of `DATA`'s `getcoldesc`, new
  comment, tiled when `DATA` is tiled, `TiledShapeStMan` otherwise) and
  `IMAGING_WEIGHT` (`makearrcoldesc`, `ndim=1`, `shape=[nchan]`,
  `valuetype='float'`, `TiledShapeStMan`);
- set `MODEL_DATA`'s `CHANNEL_SELECTION` column keyword (int32 `[0,nch]` per
  SPW, read from `SPECTRAL_WINDOW.NUM_CHAN`);
- `flush()`.
- `removeImagingColumns(msname)` removes those three columns and flushes.

Implemented in `crates/casacure/src/msutil.rs` (Rust) — works on both the
`required` MS schema (which has `DATA` only in the `complete` variant) and
real MSes that already carry `DATA`.

#### B. `casacure.measures` — the seven-call subset

New pure-Rust `casacure.measures` (`crates/casacure/src/measures.rs`)
mirroring `casacore.measures.measures`:

- `measures()` constructor;
- `direction(refer, v0, v1)` accepting quantities **or** strings;
- `position(refer, v0, v1, v2)` (ITRF Cartesian metres or WGS84 lon/lat/h);
- `epoch(refer, quantity)` → `{'type','refer','m0':{'value','unit'}}` dict;
- `do_frame(measure)` → `True`;
- `posangle(dir1, dir2)` → radians (great-circle PA at `dir1`);
- `measure(dir, target_refer)` → converted direction dict;
- `get_value(measure)` → **list of quantities**.

Semantics pinned to real python-casacore (probed live on this machine):

- `epoch('utc', q)` treats `q` as **days since MJD 0**: so
  `epoch('utc', quantity(1000,'s'))` gives `m0.value = 0.011574…` with
  `m0.unit = 'd'`.
- Return values are plain `dict`s with `type`/`refer`/`m0`(/`m1`,`m2`) keys.
- `get_value` on a direction returns `[quantity(m0), quantity(m1)]`.
- `posangle` converts `m1` to `m0`'s refer first, then measures the
  great-circle PA at `m0` from its increasing-declination direction —
  exact for same-frame directions, and the mixed-frame path (J2000 source vs
  AZELGEO zenith) reproduces casacore's value.
- `measure(d, 'AZELGEO')` uses the frame's `position` + `epoch`, converts
  J2000 → apparent place (first-order precession), computes hour angle from
  IAU 1982 GMST, and returns azimuth as `az_east + π` (casacore's
  convention). Verified to ~10–20 arcsec against real casacore over a
  multi-epoch sweep.

**Accuracy note.** The full casacore measures stack applies nutation and IERS
data; the analytic model here omits both (casacore itself degrades to
"less precision" without IERS files). The residual in `AZELGEO` is
~10–20 arcsec in azimuth/altitude and <1 arcsec in the same-frame `posangle`
path — far finer than DDFacet's parallactic-angle sampling granularity
(`pa_inc` is degrees). `GiveDate` does not use any direction conversion at
all (only `epoch('utc', s)` → MJD days) and is exact.

#### C. `casacure` crate: Rust-facing MS I/O (for killMS's native extension)

killMS's solver already runs an in-process thread pool around
`killms-core`/`native`. The `casacure` crate (`Table` is `Send + Sync`,
`crates/casacure/src/table.rs:986-1010`) can be consumed directly from that
crate for the chunked MS reads/writes, in parallel across rayon threads,
with no Python marshalling per call. This is the multiprocessing payoff:
`getcol`'s Python↔Rust bridging cost (3–5x slower per call than casacore on
cached tables, `BENCHMARK.md`) disappears when the read happens inside the
native extension.

`killms-core` can add `casacure` as a dependency and expose
`read_vis_chunk` / `write_imaging_weight` through `killMS/rust/native`,
A/B-able behind `KILLMS_NATIVE_IO=1` exactly like `KILLMS_NATIVE_JACOBIAN`.
DDFacet's `AsyncProcessPool` is fork-based, so its workers keep using the
Python `casacure` module (the forked children inherit the parent's page
cache for the mapped data files, which is already a win).

## 3. Test suite

`tests/test_ddfacet_killms_usage.py` replays the exact call patterns against
a mini-MS built with `default_ms` + a `TiledColumnStMan` `DATA` column (the
MS convention: DATA shape is `(nchan, ncorr)`). Each test cites its source
file:line. **All 24 tests pass** (`PYTHONPATH=tests/shim python -m pytest
tests/test_ddfacet_killms_usage.py`). Sections:

- A `GiveMainTable` (`query` + `sort("TIME")`) — DDFacet `Data/ClassMS.py:230-234`
- B chunked `getcol`/`getcolslicenp`/`getcolslice` with `cs_tlc/cs_brc/cs_inc`
- C column management (`getcoldesc`→`addcols`, IMAGING_WEIGHT-shaped desc)
- D `addImagingColumns` / `removeImagingColumns`
- E subtable linkage (`getkeyword` + `::SUBTABLE` and `/SUBTABLE` opens)
- F weight/flag write path (`putcol` partial windows)
- G `query("FIELD_ID==%d")` and chained `sort`
- H quanta call forms (incl. the ISO8601 date-string risk item)
- I measures call forms (pinned to the values probed from real casacore)
- J cross-implementation interop (opt-in, real python-casacore)

## 4. Out of scope

- `pyrap.images` (CASA image subsystem) — a separate port.
- `Imager/ClassMontblancMachine.py` (montblanc extra; `me.uvw`/`me.baseline`).
- `measures` features beyond the seven used (e.g. `doppler`, `frequency`,
  `radialvelocity`, `separation`, `uvw`, `baseline`).
