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
        for name in colnames:
            data = t.getcol(name)
            columns[name] = {
                "value_type": t.getcoldesc(name)["valueType"],
                "getcol_dtype": getattr(data, "dtype", None).str if hasattr(data, "dtype") else "list",
            }
    return {"path": path.name, "columns": columns}


def main() -> None:
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    import casacore

    manifest = {
        "casacore_version": casacore.__version__,
        "tables": {"typed": make_typed_table(FIXTURES)},
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {FIXTURES / 'manifest.json'} (casacore {manifest['casacore_version']})")


if __name__ == "__main__":
    sys.exit(main())
