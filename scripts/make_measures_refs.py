#!/usr/bin/env python3
"""Regenerate the pinned astropy reference table used by the Rust tests.

`crates/casacure/src/astro.rs` carries a small table of J2000 ->
(azimuth, altitude) values produced by astropy's `AltAz` frame.  Those are
the values `casacure.measures` is held to (the full contract lives in
`MEASURES_ACCURACY.md`); this script prints the Rust source for the table
so it can be refreshed when astropy's own IERS data changes.

Usage:
    python scripts/make_measures_refs.py

The three ITRF site vectors are MeerKAT, the VLA and ALMA.  Astropy's
`AltAz` is used with its defaults, i.e. no refraction (pressure 0) and
IERS polar motion applied — exactly what `casacure::astro` implements.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

SITES = [
    ("MEERKAT", (5109360.133, 2006852.586, -3238948.127)),
    ("VLA", (-1601192.0, -5043315.0, 3554630.0)),
    ("ALMA", (2225142.2, -5440307.3, -2642727.9)),
]

# (mjd, ra_deg, dec_deg, site_index) -- 1926 to 2127 plus the issue #15
# field (J2000 0.5, -0.5 rad from MeerKAT at MJD 57844.5) and the EOP
# series boundaries.
CASES = [
    (51544.5, 10.0, -20.0, 0),
    (57844.5, 28.64788975654116, -28.64788975654116, 0),
    (60000.5, 280.0, 60.0, 0),
    (24262.5, 100.0, 20.0, 0),
    (95000.5, 200.0, -60.0, 0),
    (57844.5, 145.0, -30.0, 1),
    (51544.5, 355.0, 75.0, 2),
    (98000.5, 60.0, 5.0, 2),
    (33543.5, 350.0, -45.0, 0),
    (76675.0, 30.0, 10.0, 1),
    (41684.0, 210.0, 0.0, 2),
    (61145.0, 75.0, -10.0, 0),
]


def main():
    out = [
        "/// ITRF (x, y, z) metres for the three reference sites.",
        "const SITES: [[f64; 3]; 3] = [",
    ]
    for name, xyz in SITES:
        out.append(f"    // {name}")
        out.append(f"    [{xyz[0]!r}, {xyz[1]!r}, {xyz[2]!r}],")
    out += [
        "];",
        "",
        "/// (mjd, ra_deg, dec_deg, site_index, astropy_az_deg, astropy_alt_deg),",
        f"/// generated from astropy {__import__('astropy').__version__} "
        "(`AltAz`, no refraction, IERS polar",
        "/// motion) — see `scripts/make_measures_refs.py`.",
        f"const ASTROPY_REFS: [(f64, f64, f64, usize, f64, f64); {len(CASES)}] = [",
    ]
    for mjd, ra, dec, si in CASES:
        xyz = SITES[si][1]
        loc = EarthLocation.from_geocentric(*xyz, unit=u.m)
        t = Time(mjd, format="mjd", scale="utc")
        aa = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(
            AltAz(obstime=t, location=loc)
        )
        out.append(
            f"    ({float(mjd)!r}, {float(ra)!r}, {float(dec)!r}, {si}, "
            f"{float(aa.az.deg)!r}, {float(aa.alt.deg)!r}),"
        )
    out.append("];")
    print("\n".join(out))


if __name__ == "__main__":
    main()
