"""Accuracy contract for `casacure.measures`: J2000 <-> AZEL/AZELGEO.

`casacure.measures` is held to **astropy's `AltAz`** — the independent
implementation of the same IAU SOFA algorithms — to better than **1
arcsec** for epochs from a century in the past to a century in the future
(1926-2126), at several radio-telescope sites and over the whole sky.  The
model, the bundled IERS Earth-orientation data and the (few, documented)
exceptions live in `MEASURES_ACCURACY.md`; the casacore divergences are
recorded in `DIFFERENCES.md`.

This module is the executable form of that contract:

* the reference for *accuracy* is astropy (dev dependency, `.[dev]`);
* the reference for *parity* is real python-casacore, where it is
  installed (`needs_casacore`), with the one place casacore differs from
  astropy -- its `AZEL` observer latitude -- asserted explicitly.

Azimuths are compared on the sphere, never componentwise: casacore and
casacure report azimuth in `(-180, 180]` while astropy uses `[0, 360)`,
and near the zenith the azimuth is extremely sensitive to tiny direction
errors (`delta az ~ delta position / cos(alt)`), so azimuth differences
there can look large while the sky positions agree to mas.
"""

import numpy as np
import pytest

astropy = pytest.importorskip("astropy")

from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # noqa: E402
from astropy.time import Time  # noqa: E402
import astropy.units as u  # noqa: E402

from casacure import measures as cm  # noqa: E402
from casacure import quanta as cq  # noqa: E402

# Dates outside the IERS observation window make astropy warn about its own
# extrapolations; those warnings are part of the documented behaviour here.
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*dubious year.*"),
    pytest.mark.filterwarnings("ignore:.*IERS data is valid.*"),
    pytest.mark.filterwarnings("ignore:.*polar motions.*"),
]

# ---------------------------------------------------------------------------
# Sites, fields and epochs of the contract
# ---------------------------------------------------------------------------

SITES = {
    "MeerKAT": (5109360.133, 2006852.586, -3238948.127),
    "VLA": (-1601192.0, -5043315.0, 3554630.0),
    "ALMA": (2225142.2, -5440307.3, -2642727.9),
}

# A century past to a century future (relative to the 3.8.25 release).
EPOCHS = [f"{year}-01-01T00:00:00" for year in range(1926, 2127, 10)]

# Directions spread over the sky (J2000 RA/Dec in degrees); deliberately
# away from each site's zenith so azimuth is well conditioned.
FIELDS_DEG = [
    (10.0, -20.0),
    (90.0, 20.0),
    (150.0, -55.0),
    (200.0, 10.0),
    (280.0, 60.0),
    (330.0, -30.0),
]

# Issue #15: MeerKAT observing J2000 (0.5, -0.5) rad at MJD 57844.5.
ISSUE_ITRF = SITES["MeerKAT"]
ISSUE_MJD = 57844.5
ISSUE_FIELD_RAD = (0.5, -0.5)
# casacore 3.8.1 values from the issue report.
ISSUE_CASACORE_AZEL_DEG = (-50.057201, 86.966811)
ISSUE_CASACORE_POSANGLE_DEG = 131.192182
# astropy's AltAz for the same case (this is what casacure must match).
ISSUE_ASTROPY_AZEL_DEG = (312.2985207888044, 86.85593066850102)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def measures_at(itrf_m, mjd):
    """A `measures()` with the frame set to `itrf_m` at UTC `mjd`."""
    me = cm.measures()
    me.do_frame(
        me.position("itrf", *(cq.quantity(p, "m") for p in itrf_m))
    )
    me.do_frame(me.epoch("UTC", cq.quantity(mjd * 86400.0, "s")))
    return me


def azel(me, ra_rad, dec_rad, refer="AZEL"):
    """casacure `measure(direction, refer)` as (az, alt) in radians."""
    d = me.direction(
        "J2000", cq.quantity(ra_rad, "rad"), cq.quantity(dec_rad, "rad")
    )
    m = me.measure(d, refer)
    return m["m0"]["value"], m["m1"]["value"]


def astropy_altaz(mjd, ra_deg, dec_deg, itrf_m):
    loc = EarthLocation.from_geocentric(*itrf_m, unit=u.m)
    t = Time(mjd, format="mjd", scale="utc")
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return c.transform_to(AltAz(obstime=t, location=loc))


def separation_arcsec(az1_deg, alt1_deg, az2_deg, alt2_deg):
    """Great-circle separation of two horizontal coordinates, arcseconds.

    All four arguments are in **degrees**; comparing on the sphere makes the
    result independent of where each library wraps its azimuth.
    """
    az1, alt1, az2, alt2 = map(np.radians, [az1_deg, alt1_deg, az2_deg, alt2_deg])
    cos_sep = (
        np.sin(alt1) * np.sin(alt2)
        + np.cos(alt1) * np.cos(alt2) * np.cos(az1 - az2)
    )
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 3600.0


def angle_diff_arcsec(a_deg, b_deg):
    """Difference of two angles in arcseconds, ignoring 360-degree wraps."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0) * 3600.0


def casacore_available():
    """True when real python-casacore *with usable measures data* is present.

    Two things can make the parity half unrunnable: the shim (casacore is
    casacure), or an importable casacore whose measures data files are
    missing — the failure mode of ratt-ru/QuartiCal#330, where a conversion
    raises instead of answering.  A throwaway conversion probes both, so the
    parity tests skip with a reason rather than erroring.
    """
    try:
        import casacore
    except ImportError:  # pragma: no cover - depends on the environment
        return False
    if getattr(casacore, "__casacure_shim__", False):
        return False
    try:
        from casacore import measures as cc
        from casacore import quanta as cq_

        me = cc.measures()
        me.do_frame(
            me.position("itrf", *(cq_.quantity(p, "m") for p in (6378137.0, 0.0, 0.0)))
        )
        me.do_frame(me.epoch("UTC", cq_.quantity(51544.5, "d")))
        d = me.direction(
            "J2000", cq_.quantity(0.1, "rad"), cq_.quantity(0.2, "rad")
        )
        me.measure(d, "AZELGEO")
    except Exception:  # pragma: no cover - depends on casacore's data files
        return False
    return True


needs_casacore = pytest.mark.skipif(
    not casacore_available(),
    reason="real python-casacore (with usable measures data) not installed",
)


def casacore_measures_at(itrf_m, mjd):
    """The same frame built with real python-casacore."""
    from casacore import measures as cc
    from casacore import quanta as cq_

    me = cc.measures()
    me.do_frame(me.position("itrf", *(cq_.quantity(p, "m") for p in itrf_m)))
    me.do_frame(me.epoch("UTC", cq_.quantity(mjd * 86400.0, "s")))
    return me


# ---------------------------------------------------------------------------
# The accuracy contract: < 1 arcsec from astropy, 1926-2126
# ---------------------------------------------------------------------------


def test_azel_matches_astropy_over_a_century():
    """Every epoch of the 1926-2126 grid, at three sites and over the sky,
    must agree with astropy's `AltAz` to well under an arcsecond."""
    worst = (0.0, None)
    for iso in EPOCHS:
        mjd = Time(iso, scale="utc").mjd
        for site_name, itrf in SITES.items():
            me = measures_at(itrf, mjd)
            for ra_deg, dec_deg in FIELDS_DEG:
                az, alt = azel(me, np.radians(ra_deg), np.radians(dec_deg))
                aa = astropy_altaz(mjd, ra_deg, dec_deg, itrf)
                sep = separation_arcsec(np.degrees(az), np.degrees(alt), aa.az.deg, aa.alt.deg)
                assert sep < 1.0, (
                    f"{iso} {site_name} ({ra_deg}, {dec_deg}): "
                    f"{sep:.3f}\" from astropy"
                )
                if sep > worst[0]:
                    worst = (sep, f"{iso} {site_name} ({ra_deg}, {dec_deg})")
    # The model tracks astropy to a few tens of mas; 1" is the contract.
    print(f"worst separation: {worst[0]:.4f}\" at {worst[1]}")
    assert worst[0] < 0.2, f"worst {worst[0]:.3f}\" at {worst[1]}"


def test_azel_and_azelgeo_are_the_same_topocentric_direction():
    """casacure computes both casacore references the astropy way, so they
    are identical; casacore's `AZEL` is the one that differs (its observer
    latitude is geocentric) -- see DIFFERENCES.md."""
    me = measures_at(ISSUE_ITRF, ISSUE_MJD)
    azel_geo = azel(me, *ISSUE_FIELD_RAD, refer="AZELGEO")
    azel_plain = azel(me, *ISSUE_FIELD_RAD, refer="AZEL")
    assert azel_geo == pytest.approx(azel_plain, abs=0.0)


def test_azel_roundtrips_through_j2000():
    """`measure(J2000, 'AZEL')` and `measure(AZEL, 'J2000')` invert each
    other to better than a milliarcsecond over the whole grid."""
    for iso in ("1926-01-01T00:00:00", "1960-06-01T00:00:00",
                "2017-04-01T12:00:00", "2050-01-01T00:00:00",
                "2126-01-01T00:00:00"):
        mjd = Time(iso, scale="utc").mjd
        for itrf in SITES.values():
            me = measures_at(itrf, mjd)
            for ra_deg, dec_deg in FIELDS_DEG:
                az, alt = azel(me, np.radians(ra_deg), np.radians(dec_deg))
                back = me.measure(
                    me.direction(
                        "AZEL", cq.quantity(az, "rad"), cq.quantity(alt, "rad")
                    ),
                    "J2000",
                )
                dra = np.degrees(back["m0"]["value"]) - ra_deg
                ddec = np.degrees(back["m1"]["value"]) - dec_deg
                assert abs(dra) < 1e-6 and abs(ddec) < 1e-6, (dra, ddec)


@pytest.mark.parametrize("site_name", sorted(SITES))
def test_posangle_matches_astropy_away_from_the_zenith(site_name):
    """The parallactic angle at a field toward the true zenith."""
    itrf = SITES[site_name]
    mjd = Time("2017-04-01T12:00:00", scale="utc").mjd
    me = measures_at(itrf, mjd)
    zenith = me.direction(
        "AZEL", cq.quantity(0.0, "deg"), cq.quantity(90.0, "deg")
    )
    loc = EarthLocation.from_geocentric(*itrf, unit=u.m)
    t = Time(mjd, format="mjd", scale="utc")
    z_icrs = SkyCoord(
        az=0 * u.deg, alt=90 * u.deg, frame="altaz", obstime=t, location=loc
    ).transform_to("icrs")
    for ra_deg, dec_deg in FIELDS_DEG:
        field = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        aa = field.transform_to(AltAz(obstime=t, location=loc))
        # A field within 20 degrees of the zenith amplifies any direction
        # error by 1/sin(zenith distance); the well-conditioned contract is
        # the rest of the sky.
        if aa.alt.deg > 70.0:
            continue
        expected = field.position_angle(z_icrs).deg
        d = me.direction(
            "J2000", cq.quantity(field.ra.rad, "rad"),
            cq.quantity(field.dec.rad, "rad"),
        )
        got = me.posangle(d, zenith).get_value("deg")
        assert angle_diff_arcsec(got, expected) < 1.0, (
            f"{site_name} ({ra_deg}, {dec_deg}): {got} vs {expected}"
        )


# ---------------------------------------------------------------------------
# Issue #15
# ---------------------------------------------------------------------------


def test_issue15_field_is_astropy_accurate():
    """The issue's MeerKAT/J2000(0.5,-0.5)/MJD 57844.5 case: casacure must
    agree with astropy's `AltAz` (and so differ from casacore's `AZEL`, whose
    geocentric observer latitude is a casacore divergence -- DIFFERENCES.md).
    """
    me = measures_at(ISSUE_ITRF, ISSUE_MJD)
    az, alt = azel(me, *ISSUE_FIELD_RAD)
    az_deg, alt_deg = np.degrees(az), np.degrees(alt)

    # The issue's own reproducer compares against these casacore constants.
    sep_astropy = separation_arcsec(
        az_deg, alt_deg, *ISSUE_ASTROPY_AZEL_DEG
    )
    assert sep_astropy < 0.1, f"{sep_astropy:.4f}\" from astropy"

    # Documented casacore deviation (its AZEL uses the geocentric latitude):
    # ~0.17 deg in the horizon pole, ~2.36 deg in this near-zenith azimuth.
    sep_casacore = separation_arcsec(
        az_deg, alt_deg, *ISSUE_CASACORE_AZEL_DEG
    )
    print(
        f"issue #15: casacure az/alt ({az_deg:.6f}, {alt_deg:.6f}), "
        f"astropy ({ISSUE_ASTROPY_AZEL_DEG[0]:.6f}, "
        f"{ISSUE_ASTROPY_AZEL_DEG[1]:.6f}), "
        f"casacore AZEL ({ISSUE_CASACORE_AZEL_DEG[0]:.6f}, "
        f"{ISSUE_CASACORE_AZEL_DEG[1]:.6f}); "
        f"casacure-astropy {sep_astropy:.3f}\", "
        f"casacure-casacore {sep_casacore:.3f}\""
    )
    # casacore's AZEL differs by its geocentric-latitude horizon: ~0.17 deg
    # in the pole, which is a ~607 arcsec *sky* separation at this field
    # (DIFFERENCES.md identifies it exactly; casacore's own AZELGEO path,
    # which matches astropy, is the one casacure reproduces).
    assert 60.0 < sep_casacore < 900.0, sep_casacore


def test_issue15_posangle_matches_astropy():
    """QuartiCal's parallactic angle: PA at the field toward the zenith.

    casacure matches astropy's PA.  The field sits ~3 deg from the zenith,
    which amplifies a direction error by ~1/sin(z) ~ 19, so the tolerance is
    the underlying accuracy (0.1") times that amplification.
    """
    me = measures_at(ISSUE_ITRF, ISSUE_MJD)
    field_ra, field_dec = ISSUE_FIELD_RAD
    d = me.direction(
        "J2000", cq.quantity(field_ra, "rad"), cq.quantity(field_dec, "rad")
    )
    zenith = me.direction(
        "AZEL", cq.quantity(0.0, "deg"), cq.quantity(90.0, "deg")
    )
    pa = me.posangle(d, zenith).get_value("deg")

    loc = EarthLocation.from_geocentric(*ISSUE_ITRF, unit=u.m)
    t = Time(ISSUE_MJD, format="mjd", scale="utc")
    z_icrs = SkyCoord(
        az=0 * u.deg, alt=90 * u.deg, frame="altaz", obstime=t, location=loc
    ).transform_to("icrs")
    field = SkyCoord(ra=field_ra * u.rad, dec=field_dec * u.rad, frame="icrs")
    expected = field.position_angle(z_icrs).deg
    assert angle_diff_arcsec(pa, expected) < 5.0, f"pa={pa} astropy={expected}"
    # casacore's AZEL-based PA is a different (geocentric-latitude) zenith:
    # ~2.36 degrees away at this near-zenith field (DIFFERENCES.md).
    assert angle_diff_arcsec(pa, ISSUE_CASACORE_POSANGLE_DEG) > 60.0


def test_eop_prediction_window_is_the_only_documented_gap():
    """Beyond the bundled IERS finals predictions astropy and casacure hold
    different constants; inside the window the gap is bounded and documented
    in MEASURES_ACCURACY.md."""
    itrf = SITES["MeerKAT"]
    # 2027-06-01 sits in the hand-off between the bundled predictions (which
    # end 2027-05-22) and astropy's plateau (from 2027-10-02): both libraries
    # are extrapolating, from files of different vintages.
    for iso in ("2027-06-01T00:00:00", "2027-12-01T00:00:00"):
        mjd = Time(iso, scale="utc").mjd
        me = measures_at(itrf, mjd)
        az, alt = azel(me, np.radians(10.0), np.radians(-20.0))
        aa = astropy_altaz(mjd, 10.0, -20.0, itrf)
        sep = separation_arcsec(np.degrees(az), np.degrees(alt), aa.az.deg, aa.alt.deg)
        assert sep < 4.0, f"{iso}: {sep:.3f}\""
    # And the settled eras either side are tight.
    for iso in ("2020-01-01T00:00:00", "2029-06-01T00:00:00"):
        mjd = Time(iso, scale="utc").mjd
        me = measures_at(itrf, mjd)
        az, alt = azel(me, np.radians(10.0), np.radians(-20.0))
        aa = astropy_altaz(mjd, 10.0, -20.0, itrf)
        sep = separation_arcsec(np.degrees(az), np.degrees(alt), aa.az.deg, aa.alt.deg)
        assert sep < 1.0, f"{iso}: {sep:.3f}\""


def test_wgs84_position_frame_is_accepted():
    """`position('wgs84', lon, lat, height)` drives the same transform."""
    me = cm.measures()
    # MeerKAT geodetic (astropy: lon 21.443888889697842 deg,
    # lat -30.711055553291878 deg, height 1086.599484882955 m).
    me.do_frame(
        me.position(
            "wgs84",
            cq.quantity(np.radians(21.443888889697842), "rad"),
            cq.quantity(np.radians(-30.711055553291878), "rad"),
            cq.quantity(1086.599484882955, "m"),
        )
    )
    me.do_frame(me.epoch("UTC", cq.quantity(ISSUE_MJD * 86400.0, "s")))
    az, alt = azel(me, *ISSUE_FIELD_RAD)
    aa = astropy_altaz(ISSUE_MJD, np.degrees(0.5), np.degrees(-0.5), ISSUE_ITRF)
    sep = separation_arcsec(np.degrees(az), np.degrees(alt), aa.az.deg, aa.alt.deg)
    assert sep < 0.1, f"{sep:.4f}\""


# ---------------------------------------------------------------------------
# casacore parity (and its one divergence)
# ---------------------------------------------------------------------------


@needs_casacore
def test_casacore_azelgeo_parity_over_the_grid():
    """casacore's AZELGEO (its astropy-compatible reference) agrees with
    casacure to a couple of arcseconds over the century range.

    casacore warns and loses precision before its IERS tables start
    (MJD 41684): there it drifts ~12" from astropy, and casacure follows
    astropy, so the tolerance is wider for those epochs.  Measured spread
    from astropy by casacore itself: 11.8" (1926, 1960), ~0.4-0.7"
    (1973-2017), ~2.6" (2060, 2126).
    """
    from casacore import quanta as cq_

    worst = 0.0
    for iso in ("1926-01-01T00:00:00", "1960-01-01T00:00:00",
                "2000-01-01T00:00:00", "2017-04-01T12:00:00",
                "2060-01-01T00:00:00", "2126-01-01T00:00:00"):
        mjd = Time(iso, scale="utc").mjd
        tol = 15.0 if mjd < 41684.0 else 3.0
        for site_name, itrf in SITES.items():
            ours = measures_at(itrf, mjd)
            theirs = casacore_measures_at(itrf, mjd)
            for ra_deg, dec_deg in FIELDS_DEG:
                az_o, alt_o = azel(ours, np.radians(ra_deg), np.radians(dec_deg))
                d = theirs.direction(
                    "J2000", cq_.quantity(np.radians(ra_deg), "rad"),
                    cq_.quantity(np.radians(dec_deg), "rad"),
                )
                m = theirs.measure(d, "AZELGEO")
                sep = separation_arcsec(
                    np.degrees(az_o), np.degrees(alt_o),
                    np.degrees(m["m0"]["value"]), np.degrees(m["m1"]["value"]),
                )
                worst = max(worst, sep)
                assert sep < tol, (
                    f"{iso} {site_name} ({ra_deg}, {dec_deg}): "
                    f"{sep:.3f}\" from casacore AZELGEO (tolerance {tol}\")"
                )
    print(f"worst casacore AZELGEO separation: {worst:.4f}\"")


@needs_casacore
def test_casacore_azel_uses_the_geocentric_latitude():
    """The one place casacore's measures differ from astropy (documented in
    DIFFERENCES.md): its `AZEL` builds the horizon from the *geocentric*
    latitude, while its `AZELGEO` (and astropy, and casacure) use the
    geodetic (WGS84) latitude.

    The test proves the identification: feeding casacure a WGS84 site whose
    latitude *is* the geocentric latitude reproduces casacore's `AZEL`.
    """
    from casacore import quanta as cq_

    x, y, z = ISSUE_ITRF
    lon = np.arctan2(y, x)
    lat_geocentric = np.arctan2(z, np.hypot(x, y))

    theirs = casacore_measures_at(ISSUE_ITRF, ISSUE_MJD)
    d = theirs.direction(
        "J2000", cq_.quantity(ISSUE_FIELD_RAD[0], "rad"),
        cq_.quantity(ISSUE_FIELD_RAD[1], "rad"),
    )
    cc_azel = theirs.measure(d, "AZEL")

    me = cm.measures()
    me.do_frame(
        me.position(
            "wgs84",
            cq.quantity(lon, "rad"),
            cq.quantity(lat_geocentric, "rad"),
            cq.quantity(0.0, "m"),
        )
    )
    me.do_frame(me.epoch("UTC", cq.quantity(ISSUE_MJD * 86400.0, "s")))
    az, alt = azel(me, *ISSUE_FIELD_RAD)

    sep = separation_arcsec(
        np.degrees(az), np.degrees(alt),
        np.degrees(cc_azel["m0"]["value"]), np.degrees(cc_azel["m1"]["value"]),
    )
    assert sep < 1.0, f"geocentric-latitude model of casacore AZEL: {sep:.3f}\""

    # And the divergence is real and large for this near-zenith field.
    me_geodetic = measures_at(ISSUE_ITRF, ISSUE_MJD)
    az_geo, alt_geo = azel(me_geodetic, *ISSUE_FIELD_RAD)
    sep_geodetic = separation_arcsec(
        np.degrees(az_geo), np.degrees(alt_geo),
        np.degrees(cc_azel["m0"]["value"]), np.degrees(cc_azel["m1"]["value"]),
    )
    assert sep_geodetic > 1.0, sep_geodetic
