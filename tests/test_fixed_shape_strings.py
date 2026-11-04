"""Fixed-shape string array columns interoperate with casacore both ways.

Casacore stores a fixed-shape string array cell in the SSM string buckets
as length-prefixed strings only — no shape header (`SSMIndStringColumn`
delegates fixed-shape cells to `SSMDirColumn`, whose string path calls
`SSMStringHandler::put/get(..., handleShape=False)`; the shape comes from
the column descriptor).  Casacure up to 3.8.36 wrote and expected the
`[ndim][dims][filled flag]` header there, so

* casacore-written fixed-shape string columns read as "bad string
  reference" (#24), and
* casacore could not read fixed-shape string columns casacure wrote
  ("ArrayIterator: cannot iterate by scalars", because the descriptor also
  needs the `FixedShape` option bit that `BaseColumnDesc` derives from a
  shape or the `Direct` option) (#28).

Casacure now writes the headerless layout, derives the reader's shape from
the descriptor, and still accepts the 3.8.36 header layout when it matches
the descriptor exactly (files written by the old release keep reading).
Undefined cells read as the column shape filled with "" in both
libraries.

Real python-casacore runs in subprocesses without ``tests/shim`` on the
path (the shim stands in for it); tests needing it are skipped when it is
not installed.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import casacure.tables as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return env


def _real_casacore_available():
    r = subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_real_casacore_env(),
        capture_output=True,
    )
    return r.returncode == 0


needs_casacore = pytest.mark.skipif(
    not _real_casacore_available(), reason="real python-casacore not installed"
)


def _real(code, *args):
    """Run `code` under real python-casacore; return its stdout."""
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


CASACORE_WRITE = r"""
import sys
import numpy as np
import casacore.tables as pt
path, options, shape_json = sys.argv[1], int(sys.argv[2]), sys.argv[3]
shape = eval(shape_json)
d = pt.maketabdesc([pt.makearrcoldesc("S", "", shape=shape, options=options,
                                      valuetype="string")])
t = pt.table(path, d, nrow=3, ack=False)
for r in range(3):
    t.putcell("S", r, np.array(
        [[f"p{r}" + "x" * i for i in range(int(np.prod(shape)))]]
    ).reshape(shape))
t.close()
"""

CASACORE_CHECK = r"""
import sys
import numpy as np
import casacore.tables as pt
path, shape_json, written = sys.argv[1], sys.argv[2], int(sys.argv[3])
shape = eval(shape_json)
t = pt.table(path, ack=False)

def cell_array(cell):
    v = cell["array"] if isinstance(cell, dict) else cell
    return np.asarray(v).reshape(shape)

for r in range(written):
    want = np.array(
        [[f"p{r}" + "x" * i for i in range(int(np.prod(shape)))]]
    ).reshape(shape)
    got = cell_array(t.getcell("S", r))
    assert got.shape == tuple(shape), (path, r, got.shape)
    assert got.tolist() == want.tolist(), (path, r, got, want)
# An unwritten row reads as the column shape filled with the default "".
if written < 3:
    got = cell_array(t.getcell("S", 2))
    assert got.tolist() == [[""] * shape[-1]] * int(np.prod(shape[:-1])), got
desc = t.getcoldesc("S")
assert desc["shape"].tolist() == shape, desc
assert desc["option"] & 4, f"FixedShape bit lost: {desc}"
print("CASACORE-CHECKED")
"""

CASACURE_CHECK = r"""
import sys
import numpy as np
import casacure.tables as ct
path, shape_json, written = sys.argv[1], sys.argv[2], int(sys.argv[3])
shape = eval(shape_json)
t = ct.table(path, ack=False)
nelem = int(np.prod(shape))

def flat(cell):
    return list(cell["array"]) if isinstance(cell, dict) else \
        np.asarray(cell).astype(str).reshape(-1).tolist()

def cellshape(cell):
    return list(cell["shape"]) if isinstance(cell, dict) else \
        np.asarray(cell).shape

for r in range(written):
    cell = t.getcell("S", r)
    assert list(cellshape(cell)) == shape, (path, r, cell)
    assert flat(cell) == [f"p{r}" + "x" * i
                          for i in range(nelem)], (path, r, cell)
col = t.getcol("S")
assert list(col["shape"]) == [3, *shape], col
assert list(col["array"]) == [f"p{r}" + "x" * i
                              for r in range(3)
                              for i in range(nelem)], col
# An unwritten row reads as the column shape filled with the default ""
# (SSMStringHandler::get with a zero-length reference and a pre-sized
# result), matching casacore.
if written < 3:
    cell = t.getcell("S", 2)
    assert flat(cell) == [""] * nelem, cell
print("CASACURE-CHECKED")
"""


def _casacure_write(path, options, shape, nrow=3):
    t = ct.table(
        path,
        ct.maketabdesc(
            [ct.makearrcoldesc("S", "", shape=shape, options=options,
                               valuetype="string")]
        ),
        nrow=nrow,
        ack=False,
    )
    # putcol with a properly shaped numpy array (per-cell putcell with
    # numpy/nested string values is the separate #29 binding gap).
    values = np.array(
        [[f"p{r}" + "x" * i for i in range(int(np.prod(shape)))]
         for r in range(nrow)]
    ).reshape([nrow, *shape])
    t.putcol("S", values)
    t.close()


@pytest.mark.parametrize("options", [0, 1, 4, 5])
@pytest.mark.parametrize("shape", [[2], [4], [2, 3]])
@needs_casacore
def test_casacore_written_fixed_shape_strings_read(tmp_path, options, shape):
    """#24: a fixed-shape string column written by casacore reads."""
    path = str(tmp_path / f"c{options}_{'x'.join(map(str, shape))}.tab")
    _real(CASACORE_WRITE, path, str(options), repr(shape))
    _real(CASACORE_CHECK, path, repr(shape), "3")
    _real(CASACURE_CHECK, path, repr(shape), "3")


@pytest.mark.parametrize("options", [0, 1, 4, 5])
@pytest.mark.parametrize("shape", [[2], [4], [2, 3]])
@needs_casacore
def test_casacure_written_fixed_shape_strings_read_by_casacore(tmp_path,
                                                               options,
                                                               shape):
    """#28: casacore reads a fixed-shape string column casacure wrote."""
    path = str(tmp_path / f"u{options}_{'x'.join(map(str, shape))}.tab")
    _casacure_write(path, options, shape)
    _real(CASACORE_CHECK, path, repr(shape), "3")
    _real(CASACURE_CHECK, path, repr(shape), "3")
    # The descriptor keeps casacore's invariant: a shape or the Direct
    # option implies the FixedShape bit (BaseColumnDesc; #28's second half
    # — without it casacore sees a variable-shape column and its getcol
    # iterates by scalars).
    desc = _real(
        "import sys; import casacore.tables as pt; "
        "d = pt.table(sys.argv[1], ack=False).getcoldesc('S'); "
        "assert d['option'] & 4, d; assert d['shape'].tolist() == "
        + repr(shape),
        path,
    )


@needs_casacore
def test_unwritten_fixed_shape_string_row_reads_as_defaults(tmp_path):
    """A never-written cell reads as the column shape filled with "" in
    both libraries (casacore: filled flag 0 -> emptyString per element)."""
    path = str(tmp_path / "unwritten.tab")
    _casacure_write(path, 5, [2], nrow=3)
    # Only rows 0..1 were written? _casacure_write writes all 3; make a
    # sparse variant: rewrite with an extra row appended and never written.
    t = ct.table(path, ack=False, readonly=False)
    t.addrows(1)
    t.flush()
    t.close()
    cell = ct.table(path, ack=False).getcell("S", 3)
    got = cell["array"] if isinstance(cell, dict) else np.asarray(cell)
    assert list(np.asarray(got).reshape(-1)) == ["", ""]
    _real(CASACORE_CHECK, path, "[2]", "3")  # rows 0..2 unchanged
    _real(
        "import sys; import numpy as np; import casacore.tables as pt; "
        "t = pt.table(sys.argv[1], ack=False); "
        "got = t.getcell('S', 3); "
        "got = np.asarray(got['array'] if isinstance(got, dict) else got); "
        "assert got.reshape(-1).tolist() == ['', ''], got",
        path,
    )


def test_casacure_roundtrip_without_casacore(tmp_path):
    """The pure-casacure round trip keeps working (no casacore involved)."""
    path = str(tmp_path / "roundtrip.tab")
    _casacure_write(path, 5, [3])
    t = ct.table(path, ack=False)
    col = t.getcol("S")
    assert list(col["shape"]) == [3, 3]
    assert list(col["array"]) == [f"p{r}{'x' * i}" for r in range(3)
                                  for i in range(3)]
    t.close()


def test_variable_shape_strings_unchanged(tmp_path):
    """Regression: variable-shape string columns keep the header layout
    (ndim, dims, filled flag), both directions."""
    path = str(tmp_path / "variable.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makearrcoldesc("V", "", ndim=1,
                                          valuetype="string")]),
        nrow=2,
        ack=False,
    )
    t.putcell("V", 0, ["a", "bb", "ccc"])
    t.putcell("V", 1, ["z"])
    t.close()
    t = ct.table(path, ack=False)
    cell0 = t.getcell("V", 0)
    cell1 = t.getcell("V", 1)
    got0 = cell0["array"] if isinstance(cell0, dict) else list(cell0)
    got1 = cell1["array"] if isinstance(cell1, dict) else list(cell1)
    assert got0 == ["a", "bb", "ccc"], cell0
    assert got1 == ["z"], cell1
    t.close()
    if _real_casacore_available():
        out = _real(
            "import sys; import casacore.tables as pt; "
            "t = pt.table(sys.argv[1], ack=False); "
            "assert list(t.getcell('V', 0)) == ['a', 'bb', 'ccc'], "
            "t.getcell('V', 0); "
            "assert list(t.getcell('V', 1)) == ['z'], t.getcell('V', 1); "
            "print('OK')",
            path,
        )
        assert "OK" in out


@needs_casacore
def test_descriptor_option_normalization_matches_casacore(tmp_path):
    """Casacore never writes option=1-with-shape: BaseColumnDesc forces
    FixedShape when Direct is set or a shape is given. Casacure's stored
    descriptor now does the same (read back here through casacore)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = str(os.path.join(td, "opts.tab"))
        _casacure_write(path, 1, [2])  # Direct without FixedShape as given
        out = _real(
            "import sys; import casacore.tables as pt; "
            "d = pt.table(sys.argv[1], ack=False).getcoldesc('S'); "
            "assert d['option'] == 5, d; print('OK')",
            path,
        )
        assert "OK" in out
