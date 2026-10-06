"""Shared pytest configuration for the casacure test-suite.

`tests/fixtures/` holds casacore-generated ground truth — a CASA image cube,
the FITS cube it came from, CASA tables and a `manifest.json` describing
them.  It is produced by `tests/make_fixtures.py`, which needs real
python-casacore, and is gitignored, so a bare checkout (or the free-threaded
CI job, which installs only numpy/pytest/maturin) has no fixtures at all.

A fixture-dependent module declares what it needs at module level:

    CASACORE_FIXTURES = ("image.image", "image.fits")

and every test in it is skipped with one clear reason when any of those
paths is missing, instead of failing with the misleading
"no such image (not a CASA table directory or a FITS file)".  Modules
without the declaration are untouched, so this is opt-in and costs a
`getattr` per collected item.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config, items):
    for item in items:
        module = getattr(item, "module", None)
        needed = getattr(module, "CASACORE_FIXTURES", ())
        if not needed:
            continue
        missing = [name for name in needed if not (FIXTURES_DIR / name).exists()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason="casacore fixtures not generated: "
                    + ", ".join(missing)
                    + " (run `python tests/make_fixtures.py` with python-casacore)"
                )
            )
