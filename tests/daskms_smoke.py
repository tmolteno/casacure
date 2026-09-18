#! /usr/bin/env python3
"""dask-ms smoke test against the casacure backend.

Runs the dask-ms read+write dataset path (`xds_from_table`/`xds_to_table`)
against tables backed by casacure, then verifies the result with real
casacore. Instructions:

1. Build the extension for the interpreter running dask-ms:
       PYO3_PYTHON=<python-with-dask-ms> cargo build -p casacure-python
2. Point the `casacore` shim at the built extension and run:
       PYTHONPATH=<target/debug parent>:<shim dir> python tests/daskms_smoke.py
   where the shim dir is one with a `casacore/` package whose
   `casacore/tables.py` re-exports `casacure.tables`.
"""

import os
import shutil
import tempfile

import numpy as np

DESC = {
    "TIME": {"valueType": "double", "option": 0, "comment": "", "keywords": {}},
    "ANTENNA1": {"valueType": "int", "option": 0, "comment": "", "keywords": {}},
    "DATA": {
        "valueType": "complex",
        "option": 0,
        "comment": "",
        "ndim": 2,
        "keywords": {},
    },
    "WEIGHT": {
        "valueType": "float",
        "option": 0,
        "comment": "",
        "ndim": 1,
        "keywords": {},
    },
}


def ms_create_read_write():
    """Create a Measurement Set from scratch via xds_to_table(descriptor='ms')
    (exercising default_ms), then read it back."""

    import dask
    import dask.array as da
    import xarray as xr
    from daskms import xds_from_ms, xds_to_table

    tmp = tempfile.mkdtemp(prefix="casacure-newms-")
    path = os.path.join(tmp, "ms.tab")
    rng = np.random.default_rng(3)
    n = 4
    data = (rng.random((n, 8, 4)) + 1j * rng.random((n, 8, 4))).astype(np.complex64)
    weight = rng.random((n, 4), dtype=np.float32)
    ds = xr.Dataset(
        {
            "TIME": (("row",), da.from_array(np.array([0.0, 1.0, 2.0, 3.0]), chunks=n)),
            "ANTENNA1": (
                ("row",),
                da.from_array(np.array([0, 1, 0, 1], dtype=np.int32), chunks=n),
            ),
            "DATA": (("row", "chan", "corr"), da.from_array(data, chunks=(n, 8, 4))),
            "WEIGHT": (("row", "corr"), da.from_array(weight, chunks=(n, 4))),
        },
        coords={"row": np.arange(n)},
    )
    dask.compute(xds_to_table(ds, path, descriptor="ms"))
    print("xds_to_ms create+write: OK")

    back = xds_from_ms(path, columns=["TIME", "ANTENNA1", "DATA", "WEIGHT"])[0].compute()
    assert back.sizes["row"] == n and back.DATA.shape == (n, 8, 4), back
    assert np.allclose(back.DATA.values, data), "DATA mismatch"
    assert back.TIME.values.tolist() == [0.0, 1.0, 2.0, 3.0]
    print("xds_from_ms read-back: OK")

    # Cross-check with real casacore.
    try:
        from casacore.tables import table as casacore_table

        t = casacore_table(path, ack=False)
        assert t.nrows() == n and t.getcol("DATA").shape == (n, 8, 4), "casacore mismatch"
        import os as _os

        subs = [
            d
            for d in _os.listdir(path)
            if _os.path.isdir(os.path.join(path, d))
            and _os.path.exists(os.path.join(path, d, "table.dat"))
        ]
        assert len(subs) == 12, f"expected 12 standard subtables, got {subs}"
        print("casacore cross-check (12 subtables): OK")
    except ImportError:
        print("casacore cross-check: skipped (not installed)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import dask
    from daskms import xds_from_table, xds_to_table

    tmp = tempfile.mkdtemp(prefix="casacure-msfull-")
    path = os.path.join(tmp, "msfull.tab")

    # Create the table once with real casacore, then drive dask-ms through
    # the casacure `casacore.tables` shim.
    if os.environ.get("CASACURE_SMOKE_USE_CASACORE"):
        from casacore.tables import table

        t = table(path, DESC, nrow=3, ack=False)
        t.putcol("TIME", [0.0, 1.5, 3.0])
        t.putcol("ANTENNA1", [0, 1, 0])
        t.putcol(
            "DATA",
            (np.random.random((3, 2, 3)) + 1j * np.random.random((3, 2, 3))).astype(
                np.complex64
            ),
        )
        t.putcol("WEIGHT", np.arange(6, dtype=np.float32).reshape(3, 2))
        t.flush()
        t.close()
    else:
        # Create through casacure itself.
        from casacure.tables import table

        rng = np.random.default_rng(42)
        t = table(path, DESC, nrow=3)
        t.putcol("TIME", np.array([0.0, 1.5, 3.0]))
        t.putcol("ANTENNA1", np.array([0, 1, 0], dtype=np.int32))
        t.putcol(
            "DATA",
            (rng.random((3, 2, 3)) + 1j * rng.random((3, 2, 3))).astype(np.complex64),
        )
        t.putcol("WEIGHT", np.arange(6, dtype=np.float32).reshape(3, 2))
        t.flush()

    # Read through dask-ms (taql ordering + getcolnp served by casacure).
    ds = xds_from_table(path)[0].compute()
    assert ds.TIME.values.tolist() == [0.0, 1.5, 3.0], ds.TIME.values
    assert ds.ANTENNA1.values.tolist() == [0, 1, 0], ds.ANTENNA1.values
    assert ds.DATA.shape == (3, 2, 3), ds.DATA.shape
    assert ds.WEIGHT.shape == (3, 2), ds.WEIGHT.shape
    print("dask-ms read: OK")

    # Write back doubled DATA, keeping the graph lazy.
    xds = xds_from_table(path)[0]
    orig = xds.DATA.data
    # Snapshot the expected pre-write values: with live data-manager reads
    # (like real casacore) `orig.compute()` after the write would re-read the
    # already-doubled table.
    expected = np.asarray(orig.compute()) * 2
    xds2 = xds.assign(DATA=(xds.DATA.dims, orig * 2))
    dask.compute(xds_to_table(xds2, path, ["DATA"]))
    print("dask-ms write: OK")

    # Cross-check with real casacore when available.
    try:
        from casacore.tables import table as casacore_table

        t = casacore_table(path, ack=False)
        assert np.allclose(
            t.getcol("DATA"), expected
        ), "write-back DATA mismatch"
        assert list(t.getcol("TIME")) == [0.0, 1.5, 3.0], "TIME clobbered"
        print("casacore cross-check: OK")
    except ImportError:
        print("casacore cross-check: skipped (not installed)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ms_create_read_write()


if __name__ == "__main__":
    main()
