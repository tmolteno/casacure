"""IncrementalStMan non-scalar columns: reads must raise, not return garbage.

The `ism_arrays.tab` fixture is written by python-casacore (see
`tests/make_fixtures.py`): one IncrementalStMan group binding a fixed-shape
float array (WEIGHT), a variable-shape float array (SIGMA), a scalar double
(TIME) and a string (NAME). casacore stores an ISM array cell as the whole
array per interval entry, not one scalar per row; before the issue-#47
stop-gap casacure decoded each row as a single float at the wrong offset and
returned plausible-looking garbage with no error. These tests pin the loud
failure and the unaffected scalar column.

Run as part of the project suite (the `casacore` import must resolve to the
casacure shim):
    PYTHONPATH=tests/shim python -m pytest tests/test_ism_unsupported.py -q
"""

import numpy as np
import pytest
from pathlib import Path

from casacore.tables import table

CASACORE_FIXTURES = ("ism_arrays.tab", "manifest.json")

FIXTURE = Path(__file__).parent / "fixtures" / "ism_arrays.tab"


@pytest.mark.skipif(not FIXTURE.exists(), reason="ism_arrays.tab fixture not generated")
def test_ism_array_columns_raise_instead_of_garbage():
    t = table(str(FIXTURE), ack=False)
    for col in ("WEIGHT", "SIGMA"):
        with pytest.raises(RuntimeError, match="not supported yet"):
            t.getcol(col)
        with pytest.raises(RuntimeError, match="not supported yet"):
            t.getcell(col, 0)
    with pytest.raises(RuntimeError, match="not supported yet"):
        t.getcol("NAME")
    # The scalar column in the same ISM group still reads.
    assert np.array_equal(t.getcol("TIME"), np.arange(10, dtype=float))
    t.close()


@pytest.mark.skipif(not FIXTURE.exists(), reason="ism_arrays.tab fixture not generated")
def test_ism_scalar_column_in_mixed_group_reopens(tmp_path):
    # A range read (getcell after a getcol) exercises the same flush-gate
    # path through which the raises above travel.
    t = table(str(FIXTURE), ack=False)
    assert t.getcell("TIME", 5) == 5.0
    assert t.nrows() == 10
    t.close()
