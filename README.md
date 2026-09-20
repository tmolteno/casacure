# casacure and python-casacure

[![Crates.io](https://img.shields.io/crates/v/casacure?style=flat-square&logo=rust&logoColor=white)](https://crates.io/crates/casacure)
[![Crates.io downloads](https://img.shields.io/crates/d/casacure?style=flat-square&logo=rust&logoColor=white)](https://crates.io/crates/casacure)
[![PyPI](https://img.shields.io/pypi/v/casacure?style=flat-square&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/casacure/)
[![PyPI downloads](https://img.shields.io/pypi/dm/casacure?style=flat-square&logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/casacure/)

This repository contains two user-facing packages that share one engine — a
pure-Rust implementation of the CASA table system that reads and writes real
casacore Measurement Sets and tables without needing the C++ casacore library:

| Package | Kind | Install | Entry point |
|---|---|---|---|
| **casacure** | Rust crate | `cargo add casacure` (crates.io) | `use casacure;` |
| **python-casacure** | Python package | `pip install casacure` (PyPI) | `import casacure.tables` |

"python-casacure" is the name of the Python package. Its published PyPI name
stays **`casacure`** (module `casacure.tables`) — the Python package is
**not** renamed. The Python package is built from the internal
`crates/casacure-python` crate, which is not published to crates.io and is
not intended for direct use.

Both packages expose the same capabilities — the `StandardStMan`,
`IncrementalStMan` and `TiledColumnStMan` on-disk formats, table metadata,
keywords and subtable linkage, and a TaQL subset — without building or linking
the C++ casacore library. Install is fast (a wheel or the crate), there is no
multi-hour C++ dependency build, and it works everywhere a Rust `cdylib` can
be built, including ARM (`aarch64`).

The Python binding's **`casacure.tables`** is an interface-compatible
replacement for `casacore.tables`, so existing tooling — most notably
**dask-ms** — can run on casacure unchanged (CPython 3.9–3.14; wheels for
Linux, macOS and Windows).

| | |
|---|---|
| On-disk formats | StandardStMan, IncrementalStMan, TiledColumnStMan — read and write |
| Table surface | columns (scalar / fixed / variable-shape arrays, strings, records), keywords, dminfo, subtable linkage (`::SUBTABLE`) |
| TaQL | `SELECT` (WHERE / ORDERBY / GROUPBY / UNIQUE) and `CREATE TABLE` |
| Measurement Sets | full MS + 17 standard subtable schemas, `default_ms`, writable in place |
| Interop | tables written by casacure open in real casacore and vice-versa |

Status is tracked in [ARE_WE_CURED.md](ARE_WE_CURED.md); the outline is in
[CASACORE_TO_CASA_RS.md](CASACORE_TO_CASA_RS.md).

## Why choose casacure?

- **Drop-in replacement for `casacore.tables`.** `casacure.tables` mirrors the
  python-casacore interface, so existing packages — most notably **dask-ms** —
  run unchanged. The full dask-ms 0.2.32 test suite passes against it
  (219/219 on Python 3.13 and 3.14), reading, writing and updating real
  Measurement Sets.
- **No C++ casacore library.** The table engine is pure Rust. There is no
  casacore build, no multi-hour C++ dependency compile, and no Fortran/C
  system library (`wcs`, `measures`, `images`, …) to install. `pip install
  casacure` is the entire setup; building from source needs only a Rust
  toolchain plus maturin — still no C++ compiler.
- **Real on-disk interop.** casacure is byte-compatible with the casacore
  table formats it implements (StandardStMan, IncrementalStMan,
  TiledColumnStMan): tables it writes open in real casacore and vice-versa,
  verified against casacore 3.8.1 — including full Measurement Sets with
  their subtable tree and multi-manager tables.
- **Endian-correct anywhere.** The engine reads and writes both big- and
  little-endian tables, so it behaves identically on any host endianness.
- **Self-contained wheels.** Prebuilt wheels for Linux, macOS and Windows on
  CPython 3.9–3.14 — including `aarch64`, the machines where casacore is
  most painful to build (see "Platforms and architectures").
- **One engine, two packages.** The same Rust core ships as the `casacure`
  crate on crates.io (Rust) and as the `casacure` Python package on PyPI.
- **Memory-safe by construction.** A Rust implementation removes the segfault
  and memory-management warts of the C++ python-casacore bindings: for
  example, `putcol` accepts numpy object arrays directly — behaviours that
  crash the C++ build are absent by construction.
- **TaQL included.** A real TaQL subset ships in the same package: `SELECT`
  with WHERE / ORDERBY / GROUPBY / HAVING / UNIQUE / subqueries, a
  casacore-style function library (numeric/string/date-time/stats families,
  `g*` aggregates, `LIKE` / `IN`), and the write statements (`UPDATE`,
  `DELETE`, `INSERT INTO`, `ALTER TABLE`, `DROPTABLE`, `SHOW`/`CALC`) — all
  reachable through `taql()` / `table.taql()`, so no separate query layer is
  needed.

## Using casacure in place of casacore

casacure provides a python-casacore-compatible surface. The whole dask-ms 0.2.32
test suite passes against it (219/219 on both Python 3.13 and 3.14), including
reading, writing and updating real Measurement Sets.

### Direct API

`casacure.tables` mirrors `casacore.tables`:

```python
import numpy as np
import casacure.tables as ct   # drop-in for `import casacore.tables`

t = ct.table("test.tab",
             {"TIME": {"valueType": "double", "option": 0, "comment": "",
                       "keywords": {}},
              "DATA": {"valueType": "dcomplex", "option": 0, "comment": "",
                       "ndim": 2, "shape": [8, 4], "keywords": {}}},
             nrow=3)
t.putcol("TIME", np.array([0.0, 1.5, 3.0]))
t.putcol("DATA", np.random.rand(3, 8, 4).astype(np.complex128))
t.flush()

r = ct.table("test.tab", ack=False)          # read-only view
print(r.getcol("DATA").shape)                # (3, 8, 4)

q = ct.taql("SELECT * FROM $1 WHERE TIME > 1.0", tables=[r])
print(q.getcol("TIME"))

ms = ct.default_ms("observation.ms")         # full MS + 17 subtables
```
Also available: `taql` `CREATE TABLE`, `getcell`/`putcell`, `getvarcol`/
`putvarcol`, `getcolslice`/`putcolslice`, `getkeywords`/`putkeywords`,
`addcols`, `addrows`, `required_ms_desc`/`complete_ms_desc`,
`default_ms_subtable`, `tablefromascii`.

### Units and quantities

`casacure.quanta` mirrors `casacore.quanta` — the unit system and `Quantity`
(a value with a unit), with SI conversion, arithmetic, the time/angle value
classes, and the physical-constant / unit / prefix tables:

```python
import casacure.quanta as qa        # drop-in for `import casacore.quanta`

q = qa.quantity(1.5, "Jy")          # Quantity
q.get_value("mJy")                  # 1500.0
q.canonical()                       # '1.5e-26 kg.s-2'
qa.quantity(3, "km") + qa.quantity(500, "m")   # 3.5 km
qa.quantity(45, "deg").formatted()  # '+045.00.00'
qa.quantity(51544, "d").to_unix_time()  # 946684800.0 (2000-01-01T00:00Z)
```

The full `casacore.measures` engine (direction/epoch/uvw transforms, backed
by the casacore ephemeris data) is not implemented.

### Driving dask-ms on casacure

dask-ms imports `casacore.tables`; point it at casacure in one of two ways:

**1. A drop-in `casacore` shim** — a two-file package that re-exports
casacure, placed on `PYTHONPATH` ahead of any real python-casacore (this is
exactly what the project's own `tests/daskms_smoke.py` runs):

```
casacore/__init__.py          # (empty)
casacore/tables.py            # from casacure.tables import *
```

**2. Backend selection** (the cleaner upstream path) — an env-gated
`DASK_MS_BACKEND=casacure` that aliases `casacore.tables` → `casacure.tables`
in-process. It is a ~10-line `sys.modules` swap in dask-ms's `__init__`,
validated against the full dask-ms suite; it is the prototype for the upstream
store-dispatch change.

Either way, the dask-ms workflow is unchanged:

```python
import dask
import dask.array as da
import numpy as np
import xarray as xr
from daskms import xds_from_ms, xds_to_table
import casacure.tables as ct

n = 4
data = (np.random.rand(n, 8, 4) + 1j * np.random.rand(n, 8, 4)).astype(np.complex64)
ds = xr.Dataset({
    "TIME":     (("row",), da.from_array(np.array([0.0, 1.0, 2.0, 3.0]), chunks=n)),
    "ANTENNA1": (("row",), da.from_array(np.array([0, 1, 0, 1], dtype=np.int32), chunks=n)),
    "DATA":     (("row", "chan", "corr"), da.from_array(data, chunks=(n, 8, 4))),
}, coords={"row": np.arange(n)})

# Create a Measurement Set from datasets (partitioned, chunked writes).
dask.compute(xds_to_table(ds, "example.ms", descriptor="ms"))

# Read it back as lazily-chunked dask arrays.
xds = xds_from_ms("example.ms", columns=["TIME", "ANTENNA1", "DATA"])[0]

# Modify in xarray and write back in place.
xds2 = xds.assign(DATA=(xds.DATA.dims, xds.DATA.data * 2))
dask.compute(xds_to_table(xds2, "example.ms", ["DATA"]))

# The result is a real MS: open it with casacure (or real casacore).
t = ct.table("example.ms", ack=False)
print(t.getcol("DATA")[0, 0, :2])   # doubled visibilities
```

Because casacure is implemented in Rust, none of this needs a C++ toolchain:
the entire dependency stack for reading and writing Measurement Sets is the
published wheel.

## Install

```sh
pip install casacure          # prebuilt wheels once published
# or build from source:
pip install .                # maturin builds the cdylib for your interpreter
```

The Rust core is published to crates.io as the `casacure` crate; add
`casacure = "3.8"` to `Cargo.toml` (the crate tracks the casacore interface:
3.8 == casacore 3.8.x) if you want the table engine in Rust directly.

**Versions are `<casacore-major>.<casacore-minor>.<casacure-patch>`
(e.g. `3.8.2`):** the first two digits are the casacore *interface* level the
package implements (3.8 == casacore 3.8.x), and the last digit is casacure's
own patch counter — incremented on every fix/feature release and reset to 1
when the mirrored interface moves (3.8 → 3.9 …). The crates.io `casacure`
crate and the PyPI wheel share the same three-part version, so a release
publishes both together. `casacure.__version__` reports the same number.

`pip install casacure` on Python 3.9–3.14 installs a self-contained package —
no `casacore` C++ library, no `wcs/measures` harness — which is the point: MS
support on machines (including `aarch64`) where casacore is painful to build.

## Dependencies

**Rust core (`casacure` crate).** A single small runtime dependency:
`thiserror` (error-type plumbing). `serde` / `serde_json` are dev-only
dependencies, used by the casacore-comparison fixture tests
(`crates/casacure/tests/compat_fixtures.rs`). The core is pure Rust and
compiles for any Rust target; there is no unsafe system linkage.

**Python bindings (`casacure` wheel).** Built from `crates/casacure-python`,
which links the two standard Rust↔Python bridge crates — `pyo3` (CPython
binding, 0.27) and `numpy` (0.27) — against the core crate. The *installed*
Python package has exactly one runtime dependency — `numpy` — declared in
`pyproject.toml` (`dependencies = ["numpy"]`): `getcol`/`putcol`/`getvarcol`
return and accept the same numpy-typed values as python-casacore (which also
hard-depends on numpy), so `pip install casacure` pulls it automatically and
the `casacure-test` / `casacure-bench` console scripts run out of the box.
Everything else — including the C++ casacore library — is absent.

**Explicitly not needed — the point of the project:**

- the C++ casacore library or its build system (no `wcs`, `measures`,
  `images` harnesses);
- any system C / Fortran library or header — the only native code is Rust;
- a C++ toolchain to install or to build wheels.

Building from source adds only the Rust toolchain plus `maturin` (the
pyproject `[build-system]`):
`pip install .` compiles the `cdylib` for your interpreter without invoking a
C++ compiler.

## Platforms and architectures

The pure-Rust core is portable to any Rust target and handles both byte
orders, so it runs on any architecture where a Rust toolchain exists. On top
of that, the `Publish Python package` workflow (maturin) builds **prebuilt
wheels** for the following matrix:

| Platform | Architectures | CPython versions |
|---|---|---|
| Linux (manylinux) | `x86_64`, `aarch64` | 3.9 – 3.14 |
| macOS | `aarch64` (Apple Silicon) | 3.10 – 3.14 |
| Windows (MSVC) | `x86_64` | 3.10 – 3.14 |

Notes:

- Linux wheels are built from the manylinux image, whose `--find-interpreter`
  pass covers every supported CPython (3.9–3.14) in a single build per
  target; the source distribution is produced from the `x86_64` Linux job.
- Intel-macOS (`x86_64`) wheels are deliberately not built: GitHub no longer
  hosts Intel-mac runners. Intel Mac users install the source distribution,
  which still needs only Rust + maturin — no C++.
- The Rust `casacure` crate itself is endian- and target-agnostic: cargo
  builds it for any supported target (the CI gate runs on Linux, but the
  engine's portability is what makes the cross-compiled wheel matrix above
  possible).

## Development

* [TODO.md](TODO.md) — the live task list; [CHANGELOG.md](CHANGELOG.md) records
  completed steps.
* [ARE_WE_CURED.md](ARE_WE_CURED.md) — progress / test counts.
* [CASACORE_TO_CASA_RS.md](CASACORE_TO_CASA_RS.md) — the implementation
  outline (the upstream `casacore/` submodule is the reference C++ source and
  is never modified).
* `tests/daskms_smoke.py` — end-to-end dask-ms MS create/read/write smoke;
  `tests/test_types_compat.py` — type-system comparison vs real casacore.

Rust work is linted with clippy and rustfmt as we go — CI runs these as gates
(`.github/workflows/ci.yml`), so run them locally before committing:

```sh
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
```

`clippy` is run with `-D warnings` (warnings are errors); `cargo fmt` keeps
the formatting canonical. Everything merged must pass both.
