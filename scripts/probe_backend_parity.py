#! /usr/bin/env python3
"""Print the read/write semantics this project pins to python-casacore.

Run it twice — once with the real python-casacore and once with the casacure
shim — and diff the output to see every behavioural difference at a glance:

    # casacure (the shim re-exports casacure.tables)
    PYTHONPATH=target/devpkg:tests/shim python scripts/probe_backend_parity.py

    # real python-casacore (no shim on PYTHONPATH)
    python scripts/probe_backend_parity.py

The report is stable line-per-fact text; the only *documented* divergence is
the exception type of an out-of-range `getcell` (casacore `RuntimeError`,
casacure `ValueError`) — see DIFFERENCES.md.
"""

import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np


def _report(label, fn):
    try:
        print(f"{label}: {fn()}")
    except Exception as exc:  # noqa: BLE001 - the failure IS the observation
        print(f"{label}: {type(exc).__name__}: {exc}")


def scalar_defaults(d):
    from casacore.tables import makescacoldesc, maketabdesc, table

    t = table(str(d / "scalars.tab"), maketabdesc([makescacoldesc("coli", 0)]), ack=False)
    t.addrows(2)
    out = np.asarray(t.getcol("coli")).tolist()
    t.putcol("coli", (1, 2))
    out2 = np.asarray(t.getcol("coli")).tolist()
    t.close()
    return f"unset rows {out}, after putcol {out2}"


def merged_write_handle_read(d):
    from casacore.tables import makescacoldesc, maketabdesc, table

    p = str(d / "merged.tab")
    td = maketabdesc([makescacoldesc("a", 1), makescacoldesc("b", 0.0)])
    t = table(p, td, 3, ack=False)
    t.putcol("a", [1, 2, 3])
    t.putcol("b", [1.5, 2.5, 3.5])
    t.close()

    t = table(p, readonly=False, ack=False)
    t.putcol("a", [10, 20, 30])  # pending
    pending = np.asarray(t.getcol("a")).tolist()
    untouched = np.asarray(t.getcol("b")).tolist()  # from disk
    t.addrows(2)  # rows with no on-disk value -> column defaults
    grown_a = np.asarray(t.getcol("a")).tolist()
    grown_b = np.asarray(t.getcol("b")).tolist()
    t.close()
    t = table(p, ack=False)
    reopened = (np.asarray(t.getcol("a")).tolist(), np.asarray(t.getcol("b")).tolist())
    t.close()
    return f"pending {pending}, untouched {untouched}, +addrows {grown_a}/{grown_b}, reopened {reopened}"


def cell_shapes(d):
    from casacore.tables import makearrcoldesc, maketabdesc, table

    out = []
    cases = (("fixed [1,1]", [1, 1], [[[7]]]), ("variable (1,3)", None, [[[7, 8, 9]]]))
    for name, shape, value in cases:
        p = str(d / f"{name.split()[0]}.tab")
        desc = (
            makearrcoldesc("arr", 1, 0, shape)
            if shape
            else makearrcoldesc("arr", 1, ndim=2)
        )
        t = table(p, maketabdesc([desc]), ack=False)
        t.addrows(1)
        t.putcol("arr", np.array(value, dtype=np.int64))
        t.close()
        t = table(p, ack=False)
        out.append(f"{name} getcell {np.shape(t.getcell('arr', 0))}")
        t.close()
    return "; ".join(out)


def out_of_range_getcell(d):
    from casacore.tables import makescacoldesc, maketabdesc, table

    p = str(d / "range.tab")
    t = table(p, maketabdesc([makescacoldesc("C", 0.0)]), 3, ack=False)
    t.close()
    t = table(p, readonly=False, ack=False)
    try:
        t.getcell("C", 99)
        result = "no error"
    except Exception as exc:  # noqa: BLE001
        result = f"{type(exc).__name__}"
    t.close()
    return result


def ism_bool_roundtrip(d):
    from casacore.tables import makescacoldesc, maketabdesc, table

    cd = makescacoldesc("C", True, valuetype="boolean")
    cd["desc"]["dataManagerType"] = "IncrementalStMan"
    p = d / "ism_bool.tab"
    t = table(str(p), maketabdesc([cd]), 3, ack=False)
    t.putcol("C", np.array([True, False, True]))
    t.flush()
    t.close()
    t = table(str(p), ack=False)
    got = np.asarray(t.getcol("C")).tolist()
    t.close()
    f0 = (p / "table.f0").read_bytes()
    packed = "01 00 01" if b"\x01\x00\x01" in f0 else "not-found"
    return f"values {got}, cell bytes {packed}"


def subtable_keyword_path(cwd):
    from casacore.tables import default_ms, table

    ms = cwd / "t.ms"
    default_ms("t.ms")
    t = table("t.ms", readonly=False, ack=False)
    keyword = t.getkeyword("ANTENNA")
    sub_in_getsubtables = str(ms / "ANTENNA") in t.getsubtables()
    t.close()
    return f"getkeyword {keyword!r}, getsubtables has it: {sub_in_getsubtables}"


def main():
    import casacore

    print(f"casacore resolves to: {casacore.__file__}")
    d = pathlib.Path(tempfile.mkdtemp(prefix="casacure-parity-"))
    cwd = pathlib.Path(tempfile.mkdtemp(prefix="casacure-parity-cwd-"))
    try:
        _report("unset scalar cells", lambda: scalar_defaults(d))
        _report("writable handle read", lambda: merged_write_handle_read(d))
        _report("getcell shapes", lambda: cell_shapes(d))
        _report("out-of-range getcell", lambda: out_of_range_getcell(d))
        _report("ISM bool round trip", lambda: ism_bool_roundtrip(d))
        old = os.getcwd()
        os.chdir(cwd)
        try:
            _report("relative MS subtable keyword", lambda: subtable_keyword_path(cwd))
        finally:
            os.chdir(old)
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(cwd, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
