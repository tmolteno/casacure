"""casacore locking protocol: the Python surface (lockoptions, lock/unlock/
haslock/lockoptions/ismultiused), cross-process exclusion between two
casacure processes, and cross-implementation interop with real
python-casacore.

Interop children run WITHOUT the tests/shim on PYTHONPATH, so `casacore`
there is the real python-casacore the shim replaces; they are skipped when
it is not installed.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from casacore.tables import table, maketabdesc, makescacoldesc

PYTHON = sys.executable


def _child_code(body):
    return textwrap.dedent(body)


def _spawn(code, extra_env=None, timeout=30):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run([PYTHON, "-c", code], capture_output=True, text=True,
                          env=env, timeout=timeout)


@pytest.fixture
def ms_path(tmp_path):
    d = str(tmp_path / "l.ms")
    desc = maketabdesc([makescacoldesc("DATA", 0.0)])
    t = table(d, desc, nrow=4, ack=False)
    t.putcol("DATA", np.arange(4, dtype="f8"))
    t.close()
    return d


def test_lockoptions_validation(ms_path):
    t = table(ms_path, ack=False)
    assert t.lockoptions()["option"] == "default"
    assert t.lockoptions()["interval"] == 5
    t.close()
    with pytest.raises(RuntimeError, match="unknown lock option"):
        table(ms_path, lockoptions="bogus", ack=False)


def test_lockoptions_dict_form(ms_path):
    t = table(ms_path, ack=False,
              lockoptions={"option": "user", "interval": 2.4, "maxwait": 3})
    lo = t.lockoptions()
    assert lo["option"] == "user"
    assert lo["interval"] == 3  # sub-second intervals round up
    assert lo["maxwait"] == 3
    t.close()


def test_lock_unlock_haslock(ms_path):
    t = table(ms_path, readonly=False, lockoptions="user", ack=False)
    assert not t.haslock(write=True)
    t.lock(write=True, nattempts=1)
    assert t.haslock(write=True)
    assert t.haslock(write=False)  # a write lock covers reads
    t.unlock()
    assert not t.haslock(write=True)
    t.close()


def test_ismultiused_without_other_process(ms_path):
    t = table(ms_path, ack=False)
    assert not t.ismultiused()
    t.close()


def test_nolock_option_never_locks(ms_path):
    """`nolock` takes no byte-0 lock, but like casacore's no-locking mode
    every lock *request* succeeds (`haslock` reflects the request)."""
    t = table(ms_path, readonly=False, lockoptions="nolock", ack=False)
    assert not t.haslock(write=True)
    t.lock(write=True, nattempts=1)
    assert t.haslock(write=True)
    t.unlock()
    assert not t.haslock(write=True)
    t.close()


CROSS = _child_code("""
    import sys, time
    from casacore.tables import table   # casacure via tests/shim
    mode, path, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    t = table(path, readonly=False, lockoptions=mode, ack=False)
    t.lock(write=True, nattempts=0)
    print("HELD", flush=True)
    time.sleep(seconds)
    t.unlock()
    t.close()
    print("DONE", flush=True)
""")


def test_cross_process_writer_exclusion(ms_path, tmp_path):
    """A casacure writer holding the lock excludes another casacure
    process's writable `permanent` open, and releases it on unlock."""
    child = subprocess.Popen([PYTHON, "-c", CROSS, "permanentwait", ms_path, "3"],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "HELD"
        with pytest.raises(RuntimeError, match="Permanent lock on table"):
            table(ms_path, readonly=False, lockoptions="permanent", ack=False)
        # A reader that takes no byte-0 lock still opens and sees the table.
        r = table(ms_path, lockoptions="autonoread", ack=False)
        assert r.ismultiused()
        r.close()
    finally:
        out = child.communicate(timeout=15)[0]
    assert "DONE" in out
    # Released: a writable open succeeds again and the table is not in use.
    t = table(ms_path, readonly=False, lockoptions="permanent", ack=False)
    assert not t.ismultiused()
    t.close()


def test_cross_process_sees_grown_table(ms_path, tmp_path):
    """Rows added by a locked writer are visible to a reader opened
    afterwards (the lock file's sync record carries the row count)."""
    writer = _child_code("""
        import sys
        import numpy as np
        from casacore.tables import table
        path = sys.argv[1]
        t = table(path, readonly=False, lockoptions="permanentwait", ack=False)
        t.addrows(2)
        t.putcol("DATA", np.full(2, 9.0), 4, 2)
        t.flush()
        t.unlock()
        t.close()
        print("WROTE")
    """)
    env = dict(os.environ)
    out = subprocess.run([PYTHON, "-c", writer, ms_path], capture_output=True,
                         text=True, env=env, timeout=30)
    assert out.returncode == 0, out.stderr
    r = table(ms_path, ack=False)
    assert r.nrows() == 6
    got = r.getcol("DATA")
    assert np.array_equal(got[4:], np.full(2, 9.0))
    assert np.array_equal(got[:4], np.arange(4, dtype="f8"))
    r.close()


# --- interop with real python-casacore (children run without the shim) ---

def _real_casacore_available():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    r = subprocess.run([PYTHON, "-c", "import casacore.tables"], env=env,
                       capture_output=True)
    return r.returncode == 0


def _run_real_casacore(code, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return subprocess.run([PYTHON, "-c", _child_code(code), *args],
                          capture_output=True, text=True, env=env, timeout=60)


REAL_HOLDS_WRITE = """
    import sys, time
    from casacore.tables import table
    path, seconds = sys.argv[1], float(sys.argv[2])
    t = table(path, readonly=False, lockoptions="permanentwait", ack=False)
    t.lock(write=True)
    print("HELD", flush=True)
    time.sleep(seconds)
    t.unlock()
    t.close()
    print("DONE", flush=True)
"""


@pytest.mark.skipif(not _real_casacore_available(),
                    reason="real python-casacore not installed")
def test_casacore_writer_excludes_casacure(ms_path):
    """A real casacore write lock blocks a casacure writable open."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.Popen([PYTHON, "-c", _child_code(REAL_HOLDS_WRITE),
                              ms_path, "3"],
                             stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "HELD"
        with pytest.raises(RuntimeError, match="Permanent lock on table"):
            table(ms_path, readonly=False, lockoptions="permanent", ack=False)
        r = table(ms_path, lockoptions="autonoread", ack=False)
        assert r.ismultiused(), "casacore's open must show as in-use"
        r.close()
    finally:
        out = child.communicate(timeout=15)[0]
    assert "DONE" in out


@pytest.mark.skipif(not _real_casacore_available(),
                    reason="real python-casacore not installed")
def test_casacure_writer_excludes_casacore(ms_path):
    """A casacure write lock (user mode, held) blocks a real casacore
    writable open, and is gone after unlock."""
    t = table(ms_path, readonly=False, lockoptions="user", ack=False)
    t.lock(write=True, nattempts=0)
    blocked = _run_real_casacore("""
        from casacore.tables import table
        import sys
        t = table(sys.argv[1], readonly=False, lockoptions="permanent", ack=False)
        print("OPENED")
    """, ms_path)
    assert blocked.returncode != 0
    assert "Permanent lock" in blocked.stderr
    t.unlock()
    t.close()
    ok = _run_real_casacore("""
        from casacore.tables import table
        import sys
        t = table(sys.argv[1], readonly=False, lockoptions="permanent", ack=False)
        print("OPENED")
    """, ms_path)
    assert ok.returncode == 0, ok.stderr


@pytest.mark.skipif(not _real_casacore_available(),
                    reason="real python-casacore not installed")
def test_casacore_reads_casacure_grown_table(ms_path):
    """casacure grows a table under the lock; real casacore re-opens it and
    sees the new row count (sync record) and the written values."""
    writer = _child_code("""
        import sys
        import numpy as np
        from casacore.tables import table
        path = sys.argv[1]
        t = table(path, readonly=False, lockoptions="permanentwait", ack=False)
        t.addrows(2)
        t.putcol("DATA", np.full(2, 7.5), 4, 2)
        t.flush()
        t.unlock()
        t.close()
        print("WROTE")
    """)
    env = dict(os.environ)
    out = subprocess.run([PYTHON, "-c", writer, ms_path], capture_output=True,
                         text=True, env=env, timeout=30)
    assert out.returncode == 0, out.stderr
    check = _run_real_casacore("""
        import sys
        import numpy as np
        from casacore.tables import table
        t = table(sys.argv[1], ack=False)
        assert t.nrows() == 6, t.nrows()
        got = t.getcol("DATA")
        assert np.array_equal(got[4:], np.full(2, 7.5)), got[4:]
        assert np.array_equal(got[:4], np.arange(4, dtype="f8"))
        print("CASACORE-VERIFIED")
    """, ms_path)
    assert check.returncode == 0, check.stderr
    assert "CASACORE-VERIFIED" in check.stdout


@pytest.mark.skipif(not _real_casacore_available(),
                    reason="real python-casacore not installed")
def test_casacure_resyncs_after_casacore_grows(ms_path):
    """A casacure reader opened before a casacore write resyncs on
    `lock()`: rows and values written by casacore become visible."""
    reader = table(ms_path, lockoptions="user", ack=False)
    assert reader.nrows() == 4
    grow = _run_real_casacore("""
        import sys
        import numpy as np
        from casacore.tables import table
        t = table(sys.argv[1], readonly=False, lockoptions="permanentwait", ack=False)
        t.addrows(2)
        t.putcol("DATA", np.full(2, 3.25), 4, 2)
        t.flush()
        t.close()
        print("GREW")
    """, ms_path)
    assert grow.returncode == 0, grow.stderr
    reader.lock(write=False, nattempts=0)  # fresh acquire -> resync
    assert reader.nrows() == 6
    got = reader.getcol("DATA")
    assert np.array_equal(got[4:], np.full(2, 3.25))
    reader.unlock()
    reader.close()
