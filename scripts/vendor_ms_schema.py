#! /usr/bin/env python3
"""Regenerate crates/casacure/src/ms_schema.rs from casacore.

Requires python-casacore. Dumps the canonical MS / subtable descriptors via
``required_ms_desc()`` / ``complete_ms_desc()`` and writes them as embedded
JSON string data for the casacure crate.

Usage::

    .venv/bin/python scripts/vendor_ms_schema.py
"""

import json
import sys
from pathlib import Path

from casacore.tables import complete_ms_desc, required_ms_desc

# The main MS plus every subtable name dask-ms's ``table_schemas.SUBTABLES``
# lists (dask-ms accepts all of these; ``default_ms`` creates the 12 standard
# ones).
NAMES = [
    None,
    "ANTENNA",
    "DATA_DESCRIPTION",
    "DOPPLER",
    "FEED",
    "FIELD",
    "FLAG_CMD",
    "FREQ_OFFSET",
    "HISTORY",
    "OBSERVATION",
    "POINTING",
    "POLARIZATION",
    "PROCESSOR",
    "SOURCE",
    "SPECTRAL_WINDOW",
    "STATE",
    "SYSCAL",
    "WEATHER",
]


def norm(o):
    """JSON-normalise a dict, including numpy arrays -> lists."""
    if isinstance(o, dict):
        return {k: norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [norm(v) for v in o]
    if hasattr(o, "tolist"):
        return norm(o.tolist())
    return o


def main() -> None:
    entries = {}
    for name in NAMES:
        table = "MS" if name is None else name
        entries["required:" + table] = norm(
            required_ms_desc(name) if name else required_ms_desc()
        )
        entries["complete:" + table] = norm(
            complete_ms_desc(name) if name else complete_ms_desc()
        )

    out = Path(__file__).parent.parent / "crates/casacure/src/ms_schema.rs"
    lines = [
        "//! Vendored canonical Measurement Set descriptors (generated from",
        "//! casacore `required_ms_desc()`/`complete_ms_desc()`; regenerate with",
        "//! `scripts/vendor_ms_schema.py`). Each entry is the dict (in the",
        "//! python-casacore table-desc format) for the main MS (`\"MS\"`) or one",
        "//! of its 17 subtables.",
        "",
        "/// (table, complete) -> canonical descriptor dict as JSON.",
        "pub static SCHEMAS: &[(&str, bool, &str)] = &[",
    ]
    for key in sorted(entries):
        table = key.split(":", 1)[1]
        complete = key.startswith("complete:")
        compact = json.dumps(entries[key], separators=(",", ":"))
        lines.append(f'    ("{table}", {str(complete).lower()}, r##"{compact}"##),')
    lines += ["];", ""]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(entries)} entries)")


if __name__ == "__main__":
    sys.exit(main())
