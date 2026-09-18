"""Comparison tests: casacure vs the system python-casacore.

Run with: .venv/bin/python -m pytest tests/

These tests create real CASA tables with python-casacore (TaQL DDL) and check
that casacure's view of the CASA type system matches what casacore actually
writes into column descriptors and returns from getcol.
"""

import numpy as np
import pytest

import casacure
from casacore.tables import table, taql

# TaQL DDL type code -> (casacore valueType, dtype of getcol data, sample value)
#
# Ground truth from python-casacore 3.8.1 on this machine:
#  - TaQL cannot create USHORT columns (casacore limitation), so USMALLINT is
#    covered only in the pure mapping tests below.
#  - getcol PROMOTES uchar columns to uint16 (python-casacore quirk).
#  - getcolnp FAILS for uchar/short/uint columns ("Unknown data type") — the
#    zero-copy path only supports bool/int/float/double/complex/dcomplex.
#  - 1-D string columns come back as plain Python lists.
COLUMN_CASES = [
    ("B", "boolean", np.dtype(np.bool_), True),
    ("U1", "uchar", np.dtype(np.uint16), 7),  # quirk: promoted to uint16
    ("I2", "short", np.dtype(np.int16), -300),
    ("I4", "int", np.dtype(np.int32), -70000),
    ("U4", "uint", np.dtype(np.uint32), 4000000000),
    ("R4", "float", np.dtype(np.float32), 1.5),
    ("R8", "double", np.dtype(np.float64), 1.5e300),
    ("C4", "complex", np.dtype(np.complex64), 1.5 + 2.5j),
    ("C8", "dcomplex", np.dtype(np.complex128), 1.5e300 + 2.5e300j),
    ("S", "string", None, "hello"),
]


@pytest.fixture()
def typed_table(tmp_path):
    """A casacore-created table with one column per CASA type."""
    colnames = [f"COL_{code}" for code, *_ in COLUMN_CASES]
    decl = ", ".join(f"{name} {code}" for name, (code, *_) in zip(colnames, COLUMN_CASES))
    path = str(tmp_path / "typed.tab")
    taql(f"CREATE TABLE {path} [{decl}] LIMIT 1")
    with table(path, readonly=False, ack=False) as t:
        for name, (_, _, _, value) in zip(colnames, COLUMN_CASES):
            t.putcol(name, [value])
        yield t, dict(zip(colnames, COLUMN_CASES))


def test_valuetypes_match_casacure(typed_table):
    """casacure must parse every valueType string casacore emits, and the
    mapped numpy dtype must match the dtype of the data casacore returns."""
    t, cases = typed_table
    for name, (_, value_type, getcol_dtype, _) in cases.items():
        desc = t.getcoldesc(name)
        assert desc["valueType"] == value_type
        numpy_name = casacure.numpy_dtype(desc["valueType"])
        data = t.getcol(name)
        if value_type == "string":
            # 1-D string columns come back as plain Python lists
            # (CASACORE_TO_CASA_RS.md §3), so there is no dtype to compare.
            assert numpy_name == "object"
            assert isinstance(data, list)
        elif value_type == "uchar":
            # python-casacore promotes uchar columns to uint16 on read;
            # casacure's mapping describes the *storage* type (uint8).
            assert numpy_name == "uint8"
            assert data.dtype == getcol_dtype == np.dtype(np.uint16)
        else:
            assert np.dtype(numpy_name) == data.dtype == getcol_dtype


def test_roundtrip_values(typed_table):
    """Values written by casacore read back with the dtype casacure predicts."""
    t, cases = typed_table
    for name, (_, value_type, _, value) in cases.items():
        data = t.getcol(name)
        if value_type == "string":
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0] == value
        else:
            assert data.shape == (1,)
            assert data[0] == np.array(value).astype(data.dtype)


def test_reverse_mapping():
    """casacure.casa_type is the inverse of casacure.numpy_dtype, returning
    casacure's canonical CASA names."""
    canonical = {
        "boolean": "BOOL",
        "uchar": "BYTE",
        "short": "SHORT",
        "int": "INT",
        "uint": "UINT",
        "float": "FLOAT",
        "double": "DOUBLE",
        "complex": "COMPLEX",
        "dcomplex": "DCOMPLEX",
        "string": "STRING",
    }
    for _, value_type, _, _ in COLUMN_CASES:
        numpy_name = casacure.numpy_dtype(value_type)
        assert casacure.casa_type(numpy_name) == canonical[value_type]


def test_aliases_accepted():
    """Every alias from dask-ms's _TABLE_TO_PY must parse."""
    for alias in ["BOOL", "BOOLEAN", "BYTE", "UCHAR", "SHORT", "SMALLINT",
                  "USHORT", "USMALLINT", "INT", "INTEGER", "UINT", "UINTEGER",
                  "FLOAT", "DOUBLE", "FCOMPLEX", "COMPLEX", "DCOMPLEX", "STRING"]:
        casacure.numpy_dtype(alias)  # must not raise


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        casacure.numpy_dtype("LARGEINT")
