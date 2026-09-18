#!/usr/bin/env python3
"""Generate casacore fixture tables + manifest.json for the Rust core
comparison tests (`crates/casacure/tests/compat_fixtures.rs`).

Run with the system python-casacore available, e.g.:

    .venv/bin/python tests/make_fixtures.py

Fixtures land in `tests/fixtures/` (gitignored). The Rust tests read only the
manifest for now; the table directories become the ground truth for byte-level
read interop once §1 (on-disk format) is implemented.
"""

import json
import shutil
import sys
from pathlib import Path

import casacore.tables as ct
from casacore.tables import table, taql

FIXTURES = Path(__file__).parent / "fixtures"

# TaQL DDL type code -> sample value. Keep in sync with the casacore quirks
# documented in tests/test_types_compat.py (no USHORT in TaQL DDL).
COLUMN_CASES = [
    ("B", True),
    ("U1", 7),
    ("I2", -300),
    ("I4", -70000),
    ("U4", 4000000000),
    ("R4", 1.5),
    ("R8", 1.5e300),
    ("C4", 1.5 + 2.5j),
    ("C8", 1.5e300 + 2.5e300j),
    ("S", "hello"),
]


def make_typed_table(fixtures: Path) -> dict:
    path = fixtures / "typed.tab"
    colnames = [f"COL_{code}" for code, _ in COLUMN_CASES]
    decl = ", ".join(f"{name} {code}" for name, (code, _) in zip(colnames, COLUMN_CASES))
    taql(f"CREATE TABLE {path} [{decl}] LIMIT 1")
    columns = {}
    with table(str(path), readonly=False, ack=False) as t:
        for name, (_, value) in zip(colnames, COLUMN_CASES):
            t.putcol(name, [value])
        nrows = t.nrows()
        for name in colnames:
            data = t.getcol(name)
            columns[name] = {
                "value_type": t.getcoldesc(name)["valueType"],
                "getcol_dtype": getattr(data, "dtype", None).str if hasattr(data, "dtype") else "list",
            }
    return {
        "path": path.name,
        "nrows": nrows,
        # casacore writes data files in host byte order; table.dat itself is
        # always big-endian canonical AipsIO.
        "big_endian": sys.byteorder == "big",
        "columns": columns,
    }


def make_array_table(fixtures: Path) -> dict:
    """A table with a fixed-shape (2x3 complex) array column plus a scalar
    int column, across several rows — byte ground truth for array columns."""
    import numpy as np

    path = fixtures / "array.tab"
    taql(f"CREATE TABLE {path} [ARR C4 [NDIM=2, SHAPE=[2,3]], IDX I4] LIMIT 2")
    with table(str(path), readonly=False, ack=False) as t:
        for row in range(2):
            base = 10 * row
            cells = np.array(
                [
                    [base + 1j, base + 2j, base + 3j],
                    [base + 4j, base + 5j, base + 6j],
                ],
                dtype=np.complex64,
            )
            t.putcell("ARR", row, cells)
            t.putcell("IDX", row, row)
        nrows = t.nrows()
        first = t.getcell("ARR", 0)
        return {
            "path": path.name,
            "nrows": nrows,
            "big_endian": sys.byteorder == "big",
            "array_shape": list(first.shape),
            "array_dtype": first.dtype.str,
            "columns": {
                "ARR": {
                    "value_type": t.getcoldesc("ARR")["valueType"],
                    "getcol_dtype": first.dtype.str,
                },
                "IDX": {
                    "value_type": t.getcoldesc("IDX")["valueType"],
                    "getcol_dtype": t.getcol("IDX").dtype.str,
                },
            },
        }


def make_long_string_table(fixtures: Path) -> dict:
    """A table whose variable string column holds strings longer than 8
    chars — the SSMStringHandler 'string bucket' case (byte ground truth)."""
    path = fixtures / "longstr.tab"
    taql(f"CREATE TABLE {path} [TXT S, IDX I4] LIMIT 2")
    values = [
        "hello world this is a longer string than eight chars",
        "another quite long string that certainly exceeds eight characters",
    ]
    with table(str(path), readonly=False, ack=False) as t:
        for row in range(len(values)):
            t.putcell("TXT", row, values[row])
            t.putcell("IDX", row, row)
        nrows = t.nrows()
        return {
            "path": path.name,
            "nrows": nrows,
            "big_endian": sys.byteorder == "big",
            "values": {
                "TXT": [str(t.getcell("TXT", row)) for row in range(nrows)],
            },
            "columns": {
                "TXT": {
                    "value_type": t.getcoldesc("TXT")["valueType"],
                    "getcol_dtype": "list",
                },
                "IDX": {
                    "value_type": t.getcoldesc("IDX")["valueType"],
                    "getcol_dtype": t.getcol("IDX").dtype.str,
                },
            },
        }


def make_ism_table(fixtures: Path) -> dict:
    """A table with IncrementalStMan (Direct, option 1) index-style columns
    (TIME double, ANT1 int) plus a StandardStMan column, with repeated
    values so the ISM interval compression is exercised."""
    path = fixtures / "ism.tab"
    scd1 = ct.makescacoldesc("TIME", 0.0, "IncrementalStMan", "IncrementalStMan", 1)
    scd2 = ct.makescacoldesc("ANT1", 0, "IncrementalStMan", "IncrementalStMan", 1)
    scd3 = ct.makescacoldesc("VAL", 0.0)
    td = ct.maketabdesc([scd1, scd2, scd3])
    nrow = 6
    time_vals = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0]
    ant1_vals = [0, 0, 1, 1, 1, 2]
    with ct.table(str(path), td, nrow=nrow, ack=False) as t:
        for r in range(nrow):
            t.putcell("TIME", r, time_vals[r])
            t.putcell("ANT1", r, ant1_vals[r])
            t.putcell("VAL", r, float(r))
        return {
            "path": path.name,
            "nrows": nrow,
            "big_endian": sys.byteorder == "big",
            "values": {
                "TIME": [float(t.getcell("TIME", r)) for r in range(nrow)],
                "ANT1": [int(x) for x in t.getcol("ANT1")],
                "VAL": [float(x) for x in t.getcol("VAL")],
            },
            "columns": {
                "TIME": {
                    "value_type": t.getcoldesc("TIME")["valueType"],
                    "getcol_dtype": t.getcol("TIME").dtype.str,
                },
                "ANT1": {
                    "value_type": t.getcoldesc("ANT1")["valueType"],
                    "getcol_dtype": t.getcol("ANT1").dtype.str,
                },
                "VAL": {
                    "value_type": t.getcoldesc("VAL")["valueType"],
                    "getcol_dtype": t.getcol("VAL").dtype.str,
                },
            },
        }


def make_tsm_table(fixtures: Path) -> dict:
    """A table whose DATA column (fixed-shape 2x3 dcomplex) is stored with
    TiledColumnStMan — the MS visibility-data storage pattern."""
    import numpy as np

    path = fixtures / "tsm.tab"
    acd = ct.makearrcoldesc(
        "DATA", 0.0 + 0.0j, 2, [2, 3], "TiledColumnStMan", "TiledData_GROUP", 4
    )
    scd = ct.makescacoldesc("IDX", 0)
    td = ct.maketabdesc([acd, scd])
    nrow = 3
    with ct.table(str(path), td, nrow=nrow, ack=False) as t:
        for r in range(nrow):
            cells = np.array(
                [
                    [r + 1j, r + 2j, r + 3j],
                    [r + 4j, r + 5j, r + 6j],
                ],
                dtype=np.complex128,
            )
            t.putcell("DATA", r, cells)
            t.putcell("IDX", r, r)
        first = t.getcell("DATA", 0)
        return {
            "path": path.name,
            "nrows": nrow,
            "big_endian": sys.byteorder == "big",
            "array_shape": list(first.shape),
            "array_dtype": first.dtype.str,
            "columns": {
                "DATA": {
                    "value_type": t.getcoldesc("DATA")["valueType"],
                    "getcol_dtype": first.dtype.str,
                },
                "IDX": {
                    "value_type": t.getcoldesc("IDX")["valueType"],
                    "getcol_dtype": t.getcol("IDX").dtype.str,
                },
            },
        }


def make_keyword_table(fixtures: Path) -> dict:
    """A table with table and column keywords (nested records) — metadata
    round-trip ground truth."""
    path = fixtures / "kw.tab"
    scd = ct.makescacoldesc("A", 0)
    td = ct.maketabdesc([scd])
    nrow = 1
    with ct.table(str(path), td, nrow=nrow, ack=False) as t:
        t.putkeyword("VER", "1.0")
        t.putkeyword("MAXROWS", 1000)
        t.putkeyword("NEST", {"HH": {"II": 5}, "S": "x"})
        t.putcolkeyword("A", "UNITS", "Jy")
        t.putcolkeyword("A", "MULTI", 3)
        return {
            "path": path.name,
            "nrows": nrow,
            "big_endian": sys.byteorder == "big",
            "keywords": {k: v for k, v in t.getkeywords().items()},
            "colkeywords": dict(t.getcolkeywords("A")),
            "columns": {
                "A": {
                    "value_type": t.getcoldesc("A")["valueType"],
                    "getcol_dtype": t.getcol("A").dtype.str,
                }
            },
        }


def main() -> None:
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    import casacore

    manifest = {
        "casacore_version": casacore.__version__,
        "tables": {
            "typed": make_typed_table(FIXTURES),
            "array": make_array_table(FIXTURES),
            "longstr": make_long_string_table(FIXTURES),
            "ism": make_ism_table(FIXTURES),
            "tsm": make_tsm_table(FIXTURES),
            "kw": make_keyword_table(FIXTURES),
        },
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {FIXTURES / 'manifest.json'} (casacore {manifest['casacore_version']})")


if __name__ == "__main__":
    sys.exit(main())
