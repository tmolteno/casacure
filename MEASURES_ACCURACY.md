# MEASURES_ACCURACY.md

How accurate `casacure.measures` is, what it is measured against, and where
it deliberately differs from casacore.  The executable form of this
document is `tests/test_measures.py` (Python, astropy reference) plus the
`astro` module tests in `crates/casacure/src/astro.rs` (pinned astropy
values).

## The contract

| | |
|---|---|
| **Reference** | astropy's `AltAz` frame (independent implementation of the same IAU algorithms, and the reference this project prefers over casacore) |
| **Epoch range** | 1926-01-01 … 2126-01-01 (a century either side of the release) |
| **Accuracy** | **< 1 arcsec** from astropy, worst measured ≈ **0.02 arcsec** |
| **Sites** | MeerKAT, VLA, ALMA (WGS84) |
| **Directions** | the whole sky, compared as great-circle separations |
| **Documented exceptions** | one IERS-prediction window (see below) and casacore's own `AZEL` divergence |

`casacure.measures` is *better* than 1 arcsec in practice; the 1 arcsec
figure is the asserted bound, so that coarse changes to the model or to the
bundled data cannot pass unnoticed.

## The model

The conversion is the IAU [SOFA](http://iausofa.org/) chain, evaluated
through the pure-Rust [`sofars`](https://crates.io/crates/sofars) port:

1. `pnm06a` — IAU 2006/2000A bias-precession-nutation matrix, with the CIO
   locator `s06` (`x`, `y` from `bpn2xy`);
2. `era00` — Earth rotation angle, on UT1 (`UTC + DUT1`);
3. `apco` — the astrometry context: the observer's topocentric position and
   velocity in the GCRS (WGS84 geodetic site from the ITRS/ITRF frame via
   `gc2gd`), the Sun's direction and distance, and refraction constants;
4. `atciq` — ICRS → CIRS: light deflection by the Sun, annual aberration
   with the observer's *rotating-frame* velocity (so diurnal aberration is
   included here);
5. `atioq` — CIRS → observed azimuth/altitude, with polar motion applied,
   and the redundant diurnal-aberration step disabled (`diurab = 0`), which
   is exactly what SOFA's `atco13` does.  The inverse path uses `atoiq` and
   `aticq` with the same context, so `measure(..., 'AZEL')` and
   `measure(..., 'J2000')` invert each other to ~10⁻¹³ rad.

Deliberate simplifications, both matching astropy's defaults:

* **no atmospheric refraction** (pressure 0) — casacore only applies it when
  a pressure is supplied, and DDFacet's call sites do not;
* **directions are at infinite distance** (no parallax, no proper motion) —
  astropy `SkyCoord(ra, dec)` without a distance behaves the same way.

### Bundled data (no `casadata` needed)

casacore reads IERS tables and leap seconds from an external `casadata`
installation; that missing data is exactly the failure mode of
[ratt-ru/QuartiCal#330](https://github.com/ratt-ru/QuartiCal/issues/330).
casacure compiles its data in:

* **UT1−UTC and polar motion** — the IERS `finals2000A` daily series bundled
  by the [`celestial-eop-data`](https://crates.io/crates/celestial-eop-data)
  crate (MJD 41684 … 61547, i.e. 1973-01-02 … 2027-05-22), linearly
  interpolated.  Outside that
  span casacure uses the same constants astropy falls back to:
  `UT1−UTC = +0.807841 s` before the series, `−0.147817 s` after it, and a
  polar motion of `(0.035″, 0.29″)` (astropy's 50-year-mean `_DEFAULT_PM`).
  Note that UTC itself only exists from 1960 and no `UT1−UTC` observation
  reaches back to 1926; the pre-1973 constants keep casacure and astropy in
  agreement where neither has data.
* **The Earth's heliocentric position and velocity** — a Keplerian model
  built from the JPL approximate elements, instead of SOFA's `epv00`.
  `epv00` is only defined for 1900–2100, and the `sofars` port turns its
  out-of-range status into `None`, which makes `sofars`'s `apco13` wrapper
  *panic* outside that window — a non-starter for the ±century contract.
  Driving the lower-level `apco` with our own ephemeris removes the limit
  and keeps the library panic-free from MJD 0 to 150 000 (tested).  The
  Keplerian model matches `epv00` to ~4×10⁻⁵ AU in position and ~70 mas in
  velocity direction where both exist, which contributes well under 10⁻⁴
  arcsec to the aberration.

## Measured agreement

### casacure vs astropy

| Case set | Worst separation |
|---|---|
| 12 pinned Rust references, 1926 … 2127, 3 sites (`astro.rs::matches_astropy_altaz`) | **0.019″** (asserted < 0.1″) |
| Python grid, 1926 … 2126 every 10 years × 3 sites × 6 fields = 378 cases (`test_azel_matches_astropy_over_a_century`) | **0.026″** (asserted < 1″ per case) |
| `gmst_deg` vs astropy `Time.sidereal_time("mean")` | **4×10⁻⁵″** |
| `measures` forward/inverse round trip | **~10⁻¹³ rad** |
| Issue #15 field (MeerKAT, J2000 (0.5, −0.5) rad, MJD 57844.5) | **0.005″** in AZEL, 0.09″ in PA |

### casacure vs casacore

casacore's own astrometry is not a fixed target: it needs `casadata` and
degrades without it.  Measured against `casacore.measures` 3.8.1 with its
IERS data present:

| Epochs | casacure vs casacore `AZELGEO` | casacore `AZELGEO` vs astropy |
|---|---|---|
| 1973 – 2126 | ≤ 2.6″ | ≤ 2.6″ |
| 1926, 1960 | ≤ 12.3″ | ≤ 12.3″ (casacore logs "outside the range of the IERS … less precision") |

So casacure reproduces casacore's astropy-compatible `AZELGEO` reference to
about the accuracy casacore itself has.

## Documented differences

### 1. casacore's `AZEL` uses the geocentric latitude

casacore's two horizontal references are **not** the same transform: its
`AZEL` builds the local horizon from the **geocentric** latitude of the
site, while `AZELGEO` — and astropy, and casacure for both references —
use the **geodetic (WGS84)** latitude.  For MeerKAT that is a 0.169°
difference in the horizon pole, which is ~600 arcsec of sky at a field
3° from the zenith, and it is the entire explanation of issue #15's
parallactic-angle gap:

| engine (MeerKAT, J2000 (0.5,−0.5), MJD 57844.5) | AZEL az | AZEL alt | PA toward the zenith |
|---|---|---|---|
| casacure 3.8.24 (before this work) | −49.546770 | +86.933124 | 131.675708° |
| **casacure 3.8.25** | **−47.701503** | **+86.855931** | **133.551306°** |
| astropy `AltAz` | 312.298521 | +86.855931 | 133.551330° |
| casacore `AZEL` | −50.057201 | +86.966811 | 131.192182° |
| casacore `AZELGEO` | −47.702093 | +86.855856 | 133.550707° |

casacure follows astropy (and casacore's `AZELGEO`): the geodetic vertical
is the physically correct horizon normal.  The divergence is proved in
`tests/test_measures.py::test_casacore_azel_uses_the_geocentric_latitude`,
which reproduces casacore's `AZEL` by handing casacure a WGS84 site whose
latitude *is* the geocentric latitude.  It is also recorded in
`DIFFERENCES.md`.

### 2. The IERS prediction window (2026-06 … 2027-10)

Two libraries can only agree about Earth orientation where their data do.
Inside 1973 … 2026-06 (MJD 41684 … 61197) the bundled `finals2000A` series and astropy's
own IERS table agree to **0.02 ms** of UT1−UTC (2×10⁻⁴ arcsec) and 0.04″ of
polar motion.  Beyond MJD 61197 (**2026-06-06**) both libraries are extrapolating, from
`finals2000A` files of different vintages: the gap grows past 0.5″ on
2026-07-15, reaches at most ~3.4″ in the bundled tail (which ends
2027-05-22), and closes again to under 1″ once astropy's own plateau starts
at MJD 61680 (2027-10-02).  This is the only
window in 1926–2126 where the <1″ contract does not hold; the test asserts
the documented bound instead
(`test_eop_prediction_window_is_the_only_documented_gap`).

Refresh the bundled data with `cargo update -p celestial-eop-data` — the
crate publishes revised IERS files weekly.

## Reproducing / regenerating

```bash
pip install -e '.[dev]'          # astropy + skyfield + pytest
python -m pytest tests/test_measures.py -q   # astropy contract + casacore parity
cargo test -p casacure --lib astro           # pinned astropy values in Rust
python scripts/make_measures_refs.py         # regenerate the pinned table
```

The pinned astropy table in `crates/casacure/src/astro.rs` is generated by
`scripts/make_measures_refs.py`; regenerate it if astropy's bundled IERS
data changes enough to move the values (they are used with a 0.1″ tolerance
while the model holds ~0.02″).

## Credits

The algorithms are the IAU
[Standards of Fundamental Astronomy](http://iausofa.org/) (SOFA) routines,
which the library calls through the pure-Rust
[`sofars`](https://crates.io/crates/sofars) port (MIT).  In line with the
SOFA terms of use, results obtained through casacure's `measures` module
depend on algorithms from the SOFA ANSI C source code and should be
acknowledged as such.  The IERS Earth-orientation data come from the
[`celestial-eop-data`](https://crates.io/crates/celestial-eop-data) crate
(MIT OR Apache-2.0), which bundles the IERS `finals2000A` series.  The
accuracy contract is measured against [astropy](https://www.astropy.org/)
(BSD-3-Clause), and the port followed
[skyfield](https://rhodesmill.org/skyfield/)'s (MIT) implementation notes.
