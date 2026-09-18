# casacure

A pure-Rust implementation of the CASA table system (casacore), pip-installable
on any machine as a Python package.

casacure reads and writes real casacore Measurement Sets and tables — the
`StandardStMan`, `IncrementalStMan` and `TiledColumnStMan` on-disk formats,
table metadata, keywords and subtable linkage, and a TaQL subset — without
needing to build or link the C++ casacore library. It is fast to install
(source or a prebuilt wheel), drops the multi-hour C++ dependency hell, and
works everywhere a Rust `cdylib` can be built, including ARM (`aarch64`).

The Python bindings expose **`casacure.tables`**, an interface-compatible
replacement for `casacore.tables`, so existing tooling — most notably
**dask-ms** — can run on casacure unchanged.

- **Pure Rust core**: `crates/casacure` (the `casacure` crate on crates.io).
- **Python bindings**: `crates/casacure-python` → `pip install casacure`
  (CPython 3.9–3.14; wheels for Linux, macOS and Windows). Bring your own
  Measurement Sets — casacure never modifies the upstream C++ source.

| | |
|---|---|
| On-disk formats | StandardStMan, IncrementalStMan, TiledColumnStMan — read and write |
| Table surface | columns (scalar / fixed / variable-shape arrays, strings, records), keywords, dminfo, subtable linkage (`::SUBTABLE`) |
| TaQL | `SELECT` (WHERE / ORDERBY / GROUPBY / UNIQUE) and `CREATE TABLE` |
| Measurement Sets | full MS + 17 standard subtable schemas, `default_ms`, writable in place |
| Interop | tables written by casacure open in real casacore and vice-versa |

Status is tracked in [ARE_WE_CURED.md](ARE_WE_CURED.md); the outline is in
[CASACORE_TO_CASA_RS.md](CASACORE_TO_CASA_RS.md).

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
`casacure = "0.1"` to `Cargo.toml` if you want the table engine in Rust
directly.

`pip install casacure` on Python 3.9–3.14 installs a self-contained package —
no `casacore` C++ library, no `wcs/measures` harness — which is the point: MS
support on machines (including `aarch64`) where casacore is painful to build.

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
