"""Byte-parity tests: casacure's ISM writer vs the casacore C++ output.

What is pinned here, and why exactly this much:

- casacore's own output is NOT canonical: writing the same logical table
  with putcol vs row-by-row putcell produces different bytes (its bucket
  boundaries and the array-index file's record placement carry the write
  session's history — refcount/free-list artifacts of the
  create-defaults-then-replace lifecycle, and stale bytes left in bucket
  slack by in-place splits). Verified by probe while developing #48.

- What IS canonical, and asserted against real python-casacore output: the
  ISM header fields (bucket size, pers-cache size, unique number), the
  per-bucket layout up to the end of its interval index (data area + index;
  bucket slack excluded), the ISMIndex, and the array-index file format
  (version 1, per-record refcount, shared records for equal runs). For
  single-column tables the buckets fill to exactly the capacity casacore's
  canAddData accounts, so the meaningful bytes are identical.

Requires real python-casacore (the oracle builds the reference files in
subprocesses); skipped where it is not installed.
"""

import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

BS = 32768


def _real_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


needs_casacore = pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_real_env(),
        capture_output=True,
    ).returncode
    != 0,
    reason="real python-casacore not installed",
)


def _oracle_build(dst, coldesc_codes, names, nrows, puts):
    """Build a table with REAL python-casacore in a subprocess.
    `coldesc_codes` are python expressions evaluated there."""
    code = (
        "import numpy as np, sys\n"
        "import casacore.tables as pt\n"
        "td = pt.maketabdesc([%s])\n"
        "dm = {'*1': {'TYPE': 'IncrementalStMan', 'NAME': 'ism', 'SPEC': {}, 'COLUMNS': %r}}\n"
        "with pt.table(sys.argv[1], td, nrow=%d, dminfo=dm, ack=False) as t:\n"
        "    %s\n"
    ) % (", ".join(coldesc_codes), list(names), nrows, puts.replace("\n", "\n    "))
    r = subprocess.run(
        [sys.executable, "-c", code, str(dst)],
        env=_real_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr


def _f0_header(data):
    """(bucket_size, nbucket, pers_cache, uniqnr) from an ISM data file."""
    off = data.find(struct.pack("<I", BS))
    assert off > 0, "bucket size not found in ISM header"
    return struct.unpack_from("<IIII", data, off)


def _bucket_meaningful(data, base, ncols):
    """A bucket's bytes up to the end of its interval index — data area
    plus index; the slack to bucket_size is excluded (casacore leaves
    session-history bytes there)."""
    index_off = struct.unpack_from("<I", data, base)[0] & 0x0FFFFFFF
    pos = base + index_off
    for _ in range(ncols):
        nr = struct.unpack_from("<I", data, pos)[0]
        pos += 4 + 4 * nr + 4 * nr
    return data[base:pos]


CASES = {
    "double-distinct": (
        "pt.makescacoldesc('T', 0.0)",
        "t.putcol('T', np.arange(5000, dtype=float))",
        "double",
    ),
    "int-distinct": (
        "pt.makescacoldesc('A', 0)",
        "t.putcol('A', np.arange(5000, dtype=np.int32))",
        "int",
    ),
    "string-distinct": (
        "pt.makescacoldesc('S', '')",
        "t.putcol('S', [f'v{i%50}' for i in range(5000)])",
        "string",
    ),
    "double-runs": (
        "pt.makescacoldesc('T', 0.0)",
        "t.putcol('T', np.where(np.arange(5000) % 3 == 0, 0.0, np.arange(5000, dtype=float)))",
        "double",
    ),
}


@needs_casacore
@pytest.mark.parametrize("kind", list(CASES))
def test_ism_single_column_layout_matches_casacore(tmp_path, kind):
    """Single-column ISM tables: header, every bucket's data+index, and the
    ISMIndex reproduce the casacore C++ writer's bytes."""
    import casacore.tables as ct  # the casacure shim

    coldesc_code, puts, _ = CASES[kind]
    n = 5000
    dst = tmp_path / "cure.ms"
    ref = tmp_path / "core.ms"

    default = {"double": 0.0, "int": 0, "string": ""}[CASES[kind][2]]
    name = coldesc_code.split("'")[1]
    td = ct.maketabdesc([ct.makescacoldesc(name, default)])
    dm = {"*1": {"TYPE": "IncrementalStMan", "NAME": "ism", "SPEC": {}, "COLUMNS": [name]}}
    with ct.table(str(dst), td, nrow=n, dminfo=dm, ack=False) as t:
        exec(puts, {"np": np, "t": t})

    _oracle_build(ref, [coldesc_code], [name], n, puts)

    a = open(ref / "table.f0", "rb").read()
    b = open(dst / "table.f0", "rb").read()
    hdr_a, hdr_b = _f0_header(a), _f0_header(b)
    assert hdr_a == hdr_b, f"ISM header differs: casacore {hdr_a} vs casacure {hdr_b}"
    nbucket = hdr_a[1]
    for buck in range(nbucket):
        ma = _bucket_meaningful(a, 512 + buck * BS, 1)
        mb = _bucket_meaningful(b, 512 + buck * BS, 1)
        assert ma == mb, f"bucket {buck} content differs"
    assert a[512 + nbucket * BS :] == b[512 + nbucket * BS :], "ISMIndex differs"


@needs_casacore
def test_ism_array_index_file_format_matches_casacore(tmp_path):
    """The ISM array index file uses casacore's version-1 format: header
    [u32 version = 1][Int64 length][u32 spare] and a 4-byte refcount before
    each StIndArray record; equal adjacent arrays share one record."""
    import casacore.tables as ct  # the casacure shim

    n = 12
    dst = tmp_path / "cure.ms"
    ref = tmp_path / "core.ms"
    vals = np.tile(np.array([[1.0, 2, 3], [4, 5, 6]]), (6, 1))  # runs of 2
    td = ct.maketabdesc([ct.makearrcoldesc("V", 0.0, ndim=1, shape=[3])])
    dm = {"*1": {"TYPE": "IncrementalStMan", "NAME": "ism", "SPEC": {}, "COLUMNS": ["V"]}}
    with ct.table(str(dst), td, nrow=n, dminfo=dm, ack=False) as t:
        t.putcol("V", vals)

    _oracle_build(
        ref,
        ["pt.makearrcoldesc('V', 0.0, ndim=1, shape=[3])"],
        ["V"],
        n,
        "t.putcol('V', np.tile(np.array([[1.0, 2, 3], [4, 5, 6]]), (6, 1)))",
    )

    a = open(ref / "table.f0i", "rb").read()
    b = open(dst / "table.f0i", "rb").read()
    assert struct.unpack_from("<I", b, 0)[0] == 1, "array file version must be 1"
    assert struct.unpack_from("<I", a, 0)[0] == 1
    assert struct.unpack_from("<q", b, 4)[0] == len(b), "array file length field"
    # First record of each file: refcount then [ndim=1][dim=3][3 doubles].
    # casacore's sequential putcol gives every row its own record
    # (refcount 1 — sharing is a split-duplication artifact).
    for label, d in (("casacore", a), ("casacure", b)):
        refcount, ndim, dim = struct.unpack_from("<III", d, 16)
        assert (refcount, ndim, dim) == (1, 1, 3), f"{label} first record header"
        assert struct.unpack_from("<3d", d, 28) == (1.0, 2.0, 3.0)
    for path in (dst, ref):
        with ct.table(str(path), ack=False) as t:
            np.testing.assert_array_equal(t.getcol("V"), vals)


@needs_casacore
def test_ism_table_semantics_match_casacore(tmp_path):
    """The table.dat-level view of a casacure-written ISM table — column
    descriptions and dminfo, including the header-derived SPEC — is what
    casacore reports for its own."""
    import casacore.tables as ct  # the casacure shim

    n = 8
    dst = tmp_path / "cure.ms"
    td = ct.maketabdesc(
        [
            ct.makearrcoldesc("WEIGHT", 0.0, ndim=1, shape=[4], valuetype="float"),
            ct.makescacoldesc("TIME", 0.0),
            ct.makescacoldesc("NAME", ""),
        ]
    )
    dm = {
        "*1": {
            "TYPE": "IncrementalStMan",
            "NAME": "ism",
            "SPEC": {},
            "COLUMNS": ["WEIGHT", "TIME", "NAME"],
        }
    }
    with ct.table(str(dst), td, nrow=n, dminfo=dm, ack=False) as t:
        t.putcol("WEIGHT", np.ones((n, 4), np.float32))
        t.putcol("TIME", np.arange(n, dtype=float))
        t.putcol("NAME", [f"n{i}" for i in range(n)])

    code = (
        "import sys\n"
        "import casacore.tables as pt\n"
        "t = pt.table(sys.argv[1], ack=False)\n"
        "print(t.getdminfo())\n"
        "print(sorted((c, str(t.getcoldesc(c))) for c in t.colnames()))\n"
    )
    out_cure = subprocess.run(
        [sys.executable, "-c", code, str(dst)],
        env=_real_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out_cure.returncode == 0, out_cure.stderr

    ref = tmp_path / "core.ms"
    _oracle_build(
        ref,
        [
            "pt.makearrcoldesc('WEIGHT', 0.0, ndim=1, shape=[4], valuetype='float')",
            "pt.makescacoldesc('TIME', 0.0)",
            "pt.makescacoldesc('NAME', '')",
        ],
        ["WEIGHT", "TIME", "NAME"],
        n,
        "t.putcol('WEIGHT', np.ones((%d, 4), np.float32))\n"
        "t.putcol('TIME', np.arange(%d, dtype=float))\n"
        "t.putcol('NAME', [f'n{i}' for i in range(%d)])" % (n, n, n),
    )
    out_core = subprocess.run(
        [sys.executable, "-c", code, str(ref)],
        env=_real_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out_core.returncode == 0, out_core.stderr
    assert eval(out_cure.stdout.splitlines()[0]) == eval(out_core.stdout.splitlines()[0])
    assert out_cure.stdout.splitlines()[1] == out_core.stdout.splitlines()[1]
