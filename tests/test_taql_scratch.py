"""TaQL query results must not outlive their handles in the temp directory.

casacure materialises a TaQL query result as a real table under the temp dir
(`casacure-taql-<pid>-<nanos>`), where casacore keeps it as a reference table in
memory (or a scratch table deleted on close).  Until 3.8.15 nothing removed those
directories: schmalzburg's /tmp (a RAM tmpfs) held 848 of them, 710 MB, up to 85
per process -- dask-ms caches table proxies for the life of a process, so its
results were never dropped before exit.

casacure-only: python-casacore has no temp directories to check, and
`_cleanup_scratch` is casacure's.
"""
import gc
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import pytest

from casacore.tables import table, taql, maketabdesc, makescacoldesc

pytestmark = pytest.mark.skipif(
    not hasattr(sys.modules["casacore.tables"], "_cleanup_scratch"),
    reason="casacure-specific: python-casacore keeps TaQL results in memory")


def _mktable(path):
    t = table(str(path), maketabdesc(makescacoldesc("a", 1)), ack=False)
    t.addrows(5)
    t.putcol("a", [0, 1, 2, 3, 4])
    t.close()
    return str(path)


def _is_scratch(name):
    return os.path.basename(name).startswith("casacure-taql-")


def test_a_dropped_result_removes_its_directory(tmp_path):
    t = table(_mktable(tmp_path / "t.tab"), ack=False)
    r = taql("select a from $1 where a > 1 order by a", tables=[t])
    name = r.name()
    assert _is_scratch(name) and os.path.isdir(name)
    assert list(r.getcol("a")) == [2, 3, 4]
    r.close()
    del r
    gc.collect()
    assert not os.path.exists(name), "the TaQL result's temp directory outlived its handle"
    t.close()


def test_the_method_form_result_is_scratch_too(tmp_path):
    t = table(_mktable(tmp_path / "t.tab"), ack=False)
    r = t.taql("select a from $1 where a > 2")
    name = r.name()
    assert os.path.isdir(name)
    del r
    gc.collect()
    assert not os.path.exists(name)
    t.close()


def test_a_result_still_referenced_at_exit_is_removed(tmp_path):
    path = _mktable(tmp_path / "t.tab")
    code = textwrap.dedent(f"""
        import sys
        from casacore.tables import table, taql
        t = table({path!r}, ack=False)
        KEEP = taql("select a from $1", tables=[t])   # a module global: alive at exit
        print(KEEP.name())
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ), check=True)
    name = out.stdout.strip().splitlines()[-1]
    assert _is_scratch(name), out.stdout
    assert not os.path.exists(name), "the atexit hook did not remove a live result"


def test_the_stale_sweep_removes_only_dead_and_old(tmp_path):
    if not os.path.exists("/proc/self"):
        pytest.skip("the stale sweep uses /proc")
    tmp = tempfile.gettempdir()
    dead = 2 ** 22 + 12345                     # above pid_max's default: never alive
    while os.path.exists("/proc/%d" % dead):
        dead += 1
    old_dead = os.path.join(tmp, "casacure-taql-%d-111" % dead)
    fresh_dead = os.path.join(tmp, "casacure-taql-%d-222" % dead)
    old_live = os.path.join(tmp, "casacure-taql-1-333")        # pid 1 is always alive
    for d in (old_dead, fresh_dead, old_live):
        os.makedirs(d, exist_ok=True)
    two_days = time.time() - 2 * 86400
    os.utime(old_dead, (two_days, two_days))
    os.utime(old_live, (two_days, two_days))
    try:
        # the sweep runs once per process, before its first TaQL result: use a fresh one
        path = _mktable(tmp_path / "t.tab")
        code = textwrap.dedent(f"""
            from casacore.tables import table, taql
            t = table({path!r}, ack=False)
            taql("select a from $1", tables=[t]).close()
        """)
        subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))
        assert not os.path.exists(old_dead), "a day-old directory of a dead pid was kept"
        assert os.path.exists(fresh_dead), "a fresh directory was swept (pid namespaces!)"
        assert os.path.exists(old_live), "a live pid's directory was swept"
    finally:
        for d in (old_dead, fresh_dead, old_live):
            if os.path.isdir(d):
                os.rmdir(d)
