//! Topocentric astrometry: ICRS/J2000 directions <-> observed azimuth and
//! altitude — the engine behind `casacure::measures`' `AZEL`/`AZELGEO`
//! conversions.
//!
//! The transforms are the IAU [SOFA](http://iausofa.org/) routines
//! (IAU 2006/2000A precession-nutation, Earth rotation angle, annual +
//! diurnal aberration, solar light deflection, polar motion) evaluated
//! through the pure-Rust [`sofars`] port — the same algorithms ERFA/pyerfa
//! and therefore **astropy** use. `tests/test_measures.py` holds
//! `casacure.measures` to astropy's `AltAz` at **< 1 arcsec over 1926–2126**
//! (see `MEASURES_ACCURACY.md`).
//!
//! Two inputs the SOFA wrappers take from the caller are produced here
//! instead, so that casacure needs no external data files at runtime (the
//! pain that is ratt-ru/QuartiCal#330 for casacore's `casadata`):
//!
//! * **UT1−UTC and polar motion** — the IERS `finals2000A` daily series
//!   bundled by the `celestial-eop-data` crate (MJD 41684 … 61547,
//!   1973-01-02 … 2027-05-22),
//!   with the same constants astropy falls back to outside that span.
//! * **The Earth's heliocentric position and velocity** — a Keplerian
//!   model from the JPL approximate elements. SOFA's `epv00` is only
//!   defined for 1900–2100 and the `sofars` port turns its out-of-range
//!   status into `None` (its `apco13` wrapper would panic outside that
//!   window, which casacure's ±century contract needs). Driving the lower
//!   level `apco` with our own ephemeris removes the limit; the Keplerian
//!   model is verified against `epv00` where `epv00` is valid (see the
//!   `earth_pv_matches_epv00` test).
//!
//! No atmospheric refraction is applied (pressure 0) and directions are
//! treated as at infinite distance (no parallax), both matching astropy's
//! default `AltAz`/`SkyCoord` inputs.

use celestial_eop_data::finals_data;
use sofars::astro::{apco, atciq, aticq, atioq, atoiq, refco, IauAstrom};
use sofars::consts::DAS2R;
use sofars::coords::gc2gd;
use sofars::erst::{era00, gmst06};
use sofars::pnp::{bpn2xy, pnm06a, s06, sp00};
use sofars::ts::{taitt, utctai, utcut1};

use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// MJD = JD − 2_400_000.5.
const JD_MJD: f64 = 2_400_000.5;
/// Mean obliquity of the ecliptic at J2000.0 (degrees), used to take the
/// Keplerian orbit from the ecliptic to the equatorial frame.
const EPS_J2000_DEG: f64 = 23.439_291_111;
/// Gaussian gravitational parameter of the Sun (AU^3/day^2).
const GM_SUN: f64 = 2.959_122_082_855_911e-4;
/// Days per Julian century.
const JULIAN_CENTURY: f64 = 36_525.0;

/// First MJD of the bundled IERS `finals2000A` series (1973-01-02).
const FINALS_START_MJD: f64 = 41_684.0;
/// Last MJD of the bundled series (2027-05-22).
const FINALS_END_MJD: f64 = 61_547.0;
/// UT1−UTC before the series: the constant astropy falls back to before
/// its own IERS data begins (verified against astropy 8.0.1).
const DUT1_PRE_FINALS: f64 = 0.807_841;
/// UT1−UTC after the series: astropy's post-data plateau (astropy holds
/// this value from MJD 61680 on; nothing is knowable that far ahead).
const DUT1_POST_FINALS: f64 = -0.147_817;
/// Polar motion fallback, astropy's 50-year-mean default (`_DEFAULT_PM`),
/// in arcseconds.
const XP_DEFAULT_ASEC: f64 = 0.035;
const YP_DEFAULT_ASEC: f64 = 0.29;

/// Errors from the astrometry engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstroError {
    /// A SOFA time-scale conversion (leap-second) call failed.
    Time(i32),
    /// A coordinate or frame conversion call failed.
    Status(i32),
}

impl std::fmt::Display for AstroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstroError::Time(s) => write!(f, "time scale conversion failed (status {s})"),
            AstroError::Status(s) => write!(f, "coordinate conversion failed (status {s})"),
        }
    }
}

impl std::error::Error for AstroError {}

/// An observer site on the WGS84 ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Site {
    /// East-positive longitude (radians).
    pub elong: f64,
    /// Geodetic latitude (radians).
    pub phi: f64,
    /// Height above the ellipsoid (metres).
    pub hm: f64,
}

impl Site {
    /// WGS84 site from ITRF Cartesian coordinates (metres), the form
    /// `measures.position("itrf", ...)` stores.
    pub fn from_itrf(xyz: [f64; 3]) -> Result<Site, AstroError> {
        // n = 1: WGS84 reference ellipsoid.
        let (elong, phi, hm) = gc2gd(1, xyz).map_err(AstroError::Status)?;
        Ok(Site { elong, phi, hm })
    }

    /// WGS84 site from geodetic (longitude, latitude, height), the form
    /// `measures.position("wgs84", ...)` stores.
    pub fn wgs84(elong: f64, phi: f64, hm: f64) -> Site {
        Site { elong, phi, hm }
    }
}

/// Earth-orientation parameters at `mjd`: `(UT1-UTC seconds, xp, yp)` with
/// the pole offsets in radians.
///
/// Inside the bundled IERS `finals2000A` span this is the same data
/// astropy reads, so the two libraries agree to sub-milliarcsecond;
/// outside it, the constants astropy itself falls back to are used (see
/// `MEASURES_ACCURACY.md` for the one window where the bundled prediction
/// tail and astropy's newer file disagree).
pub fn eop(mjd: f64) -> (f64, f64, f64) {
    if mjd < FINALS_START_MJD || mjd > FINALS_END_MJD {
        let dut1 = if mjd < FINALS_START_MJD {
            DUT1_PRE_FINALS
        } else {
            DUT1_POST_FINALS
        };
        return (dut1, XP_DEFAULT_ASEC * DAS2R, YP_DEFAULT_ASEC * DAS2R);
    }
    let table = finals_data();
    // First row strictly after `mjd` (the series is daily and sorted).
    let hi = table.partition_point(|e| e.mjd <= mjd);
    let interp = |a: f64, b: f64, w: f64| a + (b - a) * w;
    let row = |i: usize| {
        let e = &table[i];
        (e.ut1_utc, e.x_p * DAS2R, e.y_p * DAS2R)
    };
    if hi == 0 {
        return row(0);
    }
    if hi >= table.len() {
        return row(table.len() - 1);
    }
    let (a, b) = (&table[hi - 1], &table[hi]);
    let w = ((mjd - a.mjd) / (b.mjd - a.mjd)).clamp(0.0, 1.0);
    let (dut1, xp, yp) = row(hi - 1);
    let (dut1_b, xp_b, yp_b) = row(hi);
    (
        interp(dut1, dut1_b, w),
        interp(xp, xp_b, w),
        interp(yp, yp_b, w),
    )
}

/// Two parts `(d1, d2)` with `d1 + d2` the UTC Julian date of `mjd`, split
/// so the integer day count sits in `d1` exactly: SOFA routines subtract
/// their reference epoch from `d1` first, which keeps the arithmetic at
/// sub-microsecond precision (a single JD double would lose ~40 us).
fn two_part(mjd: f64) -> (f64, f64) {
    let day = mjd.floor();
    (JD_MJD + day, mjd - day)
}

/// Heliocentric Earth state from the JPL approximate Keplerian elements
/// (Standish), in the equatorial J2000 frame: `([position, velocity],
/// position)` in AU and AU/day.
///
/// `sofars::astro::apco` takes the observer's barycentric position and
/// velocity and its heliocentric position (for the Sun's direction in the
/// light-deflection term). The Sun's own motion around the Solar System
/// barycentre (~12 m/s) is neglected in the barycentric velocity, which
/// limits the annual-aberration error to ~8 mas; the Keplerian elements
/// themselves differ from a perturbed ephemeris by a few tens of
/// milliarcseconds of aberration (see `earth_pv_matches_epv00`).
fn earth_pv(mjd: f64) -> ([[f64; 3]; 2], [f64; 3]) {
    let t = (mjd - 51_544.5) / JULIAN_CENTURY;
    // Orbital elements (epoch J2000, ecliptic of J2000; node = 0 in this
    // element set, so the longitude of perihelion is the argument of
    // perihelion).
    let a = 1.000_002_61 + 0.000_005_62 * t;
    let e = 0.016_711_23 - 0.000_043_92 * t;
    let l = 100.464_571_66 + 35_999.372_449_81 * t;
    let peri = 102.937_681_93 + 0.323_273_64 * t;
    let inc = (-0.012_946_68 * t).to_radians();

    let argp = peri.to_radians();
    let m = (l - peri).rem_euclid(360.0).to_radians();
    // Kepler's equation by Newton's method (e ~ 0.017: four steps is
    // convergence to machine precision).
    let mut ea = m;
    for _ in 0..4 {
        ea = (ea - (ea - e * ea.sin() - m) / (1.0 - e * ea.cos())).rem_euclid(TAU);
    }
    let (sea, cea) = ea.sin_cos();
    let root = (1.0 - e * e).sqrt();
    // Position and velocity in the orbital plane (AU, AU/day).
    let xp = a * (cea - e);
    let yp = a * root * sea;
    let n = (GM_SUN / (a * a * a)).sqrt();
    let denom = 1.0 - e * cea;
    let vx = -a * sea * n / denom;
    let vy = a * root * cea * n / denom;

    // Perifocal -> ecliptic (node = 0): rotate by the argument of perihelion
    // about z, then by the inclination about x.
    let peri_vec = rot3(argp, [xp, yp, 0.0]);
    let peri_vel = rot3(argp, [vx, vy, 0.0]);
    let ecl_pos = rot1(inc, peri_vec);
    let ecl_vel = rot1(inc, peri_vel);

    // Ecliptic -> equatorial: rotate by the mean obliquity about x.
    let eps = EPS_J2000_DEG.to_radians();
    let pos = rot1(eps, ecl_pos);
    let vel = rot1(eps, ecl_vel);
    ([pos, vel], pos)
}

/// Right-handed rotation about z.
fn rot3(a: f64, v: [f64; 3]) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
}

/// Right-handed rotation about x.
fn rot1(a: f64, v: [f64; 3]) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

/// The SOFA `apco13` astrometry context (ICRS -> CIRS), with `epv00`
/// replaced by [`earth_pv`] and refraction set to zero.
fn apco_astrom(mjd: f64, site: &Site) -> Result<IauAstrom, AstroError> {
    let (utc1, utc2) = two_part(mjd);
    let (dut1, xp, yp) = eop(mjd);
    let (tai1, tai2) = utctai(utc1, utc2).map_err(AstroError::Time)?;
    let (tt1, tt2) = taitt(tai1, tai2).map_err(AstroError::Time)?;
    let (ut11, ut12) = utcut1(utc1, utc2, dut1).map_err(AstroError::Time)?;
    let rbpn = pnm06a(tt1, tt2);
    let (x, y) = bpn2xy(&rbpn);
    let s = s06(tt1, tt2, x, y);
    let theta = era00(ut11, ut12);
    let sp = sp00(tt1, tt2);
    let (refa, refb) = refco(0.0, 10.0, 0.5, 0.55);
    let (ebpv, ehp) = earth_pv(mjd);
    let mut astrom = IauAstrom::default();
    apco(
        tt1,
        tt2,
        &ebpv,
        &ehp,
        x,
        y,
        s,
        theta,
        site.elong,
        site.phi,
        site.hm,
        xp,
        yp,
        sp,
        refa,
        refb,
        &mut astrom,
    );
    Ok(astrom)
}

/// Observed (azimuth, altitude) of the ICRS/J2000 direction `(ra, dec)`,
/// radians, for an observer at `site` at UTC `mjd`.
///
/// Azimuth is measured **east from north** and wrapped to `(-pi, pi]`,
/// which is casacore's `AZEL` convention and the sign astropy reports
/// shifted by 360 degrees; altitude is above the true horizon.
///
/// One `apco` context drives both steps, exactly as SOFA's `atco13` does:
/// `apco` folds the observer's topocentric (rotating-frame) velocity into
/// the aberration of the ICRS -> CIRS step and disables the redundant
/// diurnal-aberration step of `atioq` (`diurab = 0`). Building a separate
/// `apio13` context for the second step would apply diurnal aberration a
/// second time — a 0.1-0.3 arcsec error depending on geometry.
pub fn j2000_to_azel(mjd: f64, ra: f64, dec: f64, site: &Site) -> Result<(f64, f64), AstroError> {
    let mut acrs = apco_astrom(mjd, site)?;
    let (ri, di) = atciq(ra, dec, 0.0, 0.0, 0.0, 0.0, &mut acrs);
    let (aob, zob, _, _, _) = atioq(ri, di, &acrs);
    Ok((wrap_pi(aob), FRAC_PI_2 - zob))
}

/// The inverse of [`j2000_to_azel`]: the ICRS/J2000 direction `(ra, dec)`
/// of the observed point `(az, alt)` (radians; azimuth east from north,
/// any wrap).
pub fn azel_to_j2000(mjd: f64, az: f64, alt: f64, site: &Site) -> Result<(f64, f64), AstroError> {
    // Same single `apco` context as the forward transform, so this is the
    // exact inverse of `j2000_to_azel`.
    let mut acrs = apco_astrom(mjd, site)?;
    let (ri, di) = atoiq("a", az.rem_euclid(TAU), FRAC_PI_2 - alt, &acrs);
    Ok(aticq(ri, di, &mut acrs))
}

/// Wrap an angle to `(-pi, pi]`.
fn wrap_pi(a: f64) -> f64 {
    let a = a.rem_euclid(TAU);
    if a > PI {
        a - TAU
    } else {
        a
    }
}

/// Greenwich Mean Sidereal Time in degrees at UTC `mjd`: the IAU 2006
/// relation (Earth rotation angle from UT1 plus the precession-in-RA
/// polynomial), the model astropy's `Time.sidereal_time("mean")` returns
/// (to ~0.01 arcsec — both use `sofars`/ERFA `gmst06`).
pub fn gmst_deg(mjd: f64) -> f64 {
    let (dut1, _, _) = eop(mjd);
    let (utc1, utc2) = two_part(mjd);
    let (ut11, ut12) = match utcut1(utc1, utc2, dut1) {
        Ok(v) => v,
        // Outside the leap-second table: fall back to UTC itself; the
        // difference is < 0.9 s of UT1 and only matters here at the
        // milliarcsecond level.
        Err(_) => (utc1, utc2),
    };
    let (tai1, tai2) = match utctai(utc1, utc2) {
        Ok(v) => v,
        Err(_) => (utc1, utc2),
    };
    let (tt1, tt2) = match taitt(tai1, tai2) {
        Ok(v) => v,
        Err(_) => (utc1, utc2),
    };
    gmst06(ut11, ut12, tt1, tt2).to_degrees().rem_euclid(360.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arcseconds per radian.
    const ARCSEC_PER_RAD: f64 = 206_264.806_247_096_36;

    /// ITRF (x, y, z) metres for the three reference sites.
    const SITES: [[f64; 3]; 3] = [
        // MEERKAT
        [5109360.133, 2006852.586, -3238948.127],
        // VLA
        [-1601192.0, -5043315.0, 3554630.0],
        // ALMA
        [2225142.2, -5440307.3, -2642727.9],
    ];

    /// (mjd, ra_deg, dec_deg, site_index, astropy_az_deg, astropy_alt_deg),
    /// generated from astropy 8.0.1 (`AltAz`, no refraction, IERS polar
    /// motion) — see `scripts/make_measures_refs.py`.
    const ASTROPY_REFS: [(f64, f64, f64, usize, f64, f64); 12] = [
        (
            51544.5,
            10.0,
            -20.0,
            0,
            97.51856735929418,
            28.43234807837067,
        ),
        (
            57844.5,
            28.64788975654116,
            -28.64788975654116,
            0,
            312.2985207888044,
            86.85593066850102,
        ),
        (
            60000.5,
            280.0,
            60.0,
            0,
            328.8665444711708,
            -19.987162929251024,
        ),
        (
            24262.5,
            100.0,
            20.0,
            0,
            48.141150700844285,
            21.79211107733157,
        ),
        (
            95000.5,
            200.0,
            -60.0,
            0,
            212.86003377963897,
            25.362493261670668,
        ),
        (
            57844.5,
            145.0,
            -30.0,
            1,
            255.89072467628947,
            -37.50298221324216,
        ),
        (
            51544.5,
            355.0,
            75.0,
            2,
            11.250365496626177,
            -35.78872831315423,
        ),
        (
            98000.5,
            60.0,
            5.0,
            2,
            109.55404382234043,
            -46.93998069117311,
        ),
        (
            33543.5,
            350.0,
            -45.0,
            0,
            159.14950381173216,
            -9.29419721754662,
        ),
        (
            76675.0,
            30.0,
            10.0,
            1,
            71.02041703903299,
            -8.912366023934842,
        ),
        (
            41684.0,
            210.0,
            0.0,
            2,
            170.13544276328756,
            -65.45546022625422,
        ),
        (
            61145.0,
            75.0,
            -10.0,
            0,
            220.8588864175319,
            -39.77589014317221,
        ),
    ];

    /// Great-circle separation in arcseconds between two (az, alt) radian
    /// pairs — the only meaningful way to compare horizontal coordinates,
    /// whose azimuth blows up near the zenith.
    fn sep_arcsec(a: (f64, f64), b: (f64, f64)) -> f64 {
        let (s1, c1) = a.1.sin_cos();
        let (s2, c2) = b.1.sin_cos();
        let cos_sep = (s1 * s2 + c1 * c2 * (a.0 - b.0).cos()).clamp(-1.0, 1.0);
        cos_sep.acos() * ARCSEC_PER_RAD
    }

    #[test]
    fn matches_astropy_altaz() {
        let mut worst = 0.0f64;
        for (mjd, ra, dec, si, az, alt) in ASTROPY_REFS {
            let site = Site::from_itrf(SITES[si]).expect("site");
            let ours = j2000_to_azel(mjd, ra.to_radians(), dec.to_radians(), &site).expect("azel");
            let s = sep_arcsec(ours, (az.to_radians(), alt.to_radians()));
            println!("mjd={mjd} site={si} field=({ra},{dec}): {s:.4}\"");
            assert!(
                s < 0.1,
                "mjd={mjd} field=({ra},{dec}) site={si}: {s:.4}\" from astropy"
            );
            worst = worst.max(s);
        }
        println!("worst separation from astropy: {worst:.4} arcsec");
    }

    #[test]
    fn roundtrip_j2000_azel_j2000() {
        for (mjd, ra, dec, si, _, _) in ASTROPY_REFS {
            let site = Site::from_itrf(SITES[si]).expect("site");
            let (az, alt) =
                j2000_to_azel(mjd, ra.to_radians(), dec.to_radians(), &site).expect("fwd");
            let (ra2, dec2) = azel_to_j2000(mjd, az, alt, &site).expect("inv");
            let dra = (ra2.to_degrees() - ra + 540.0).rem_euclid(360.0) - 180.0;
            let ddec = dec2.to_degrees() - dec;
            assert!(
                dra.abs() < 1e-9 && ddec.abs() < 1e-9,
                "mjd={mjd}: dRA={dra} dDec={ddec}"
            );
        }
    }

    #[test]
    fn site_from_itrf_is_wgs84() {
        // astropy: EarthLocation.from_geocentric(MeerKAT) ->
        // lon 21.443888889697842 deg, lat -30.711055553291878 deg,
        // height 1086.599484882955 m.
        let site = Site::from_itrf(SITES[0]).expect("site");
        assert!((site.elong.to_degrees() - 21.443888890).abs() < 1e-7);
        assert!((site.phi.to_degrees() - -30.711055553).abs() < 1e-7);
        assert!((site.hm - 1086.599484882955).abs() < 1e-3);
    }

    #[test]
    fn eop_outside_finals_uses_astropy_constants() {
        // Before MJD 41684 (1973-01-02) astropy has no IERS data and holds
        // UT1-UTC at +0.807841 s with its 50-year-mean pole; so do we.
        let (d, xp, yp) = eop(FINALS_START_MJD - 1.0);
        assert!((d - 0.807841).abs() < 1e-12, "dut1={d}");
        assert!((xp.to_degrees() * 3600.0 - 0.035).abs() < 1e-12);
        assert!((yp.to_degrees() * 3600.0 - 0.29).abs() < 1e-12);
        // Far past and far future both take the fallbacks.
        for mjd in [0.0, 24262.5, FINALS_END_MJD + 1.0, 120_000.0] {
            let (d, xp, yp) = eop(mjd);
            let expect = if mjd < FINALS_START_MJD {
                0.807841
            } else {
                -0.147817
            };
            assert!((d - expect).abs() < 1e-12, "mjd={mjd} dut1={d}");
            assert!((xp.to_degrees() * 3600.0 - 0.035).abs() < 1e-12);
            assert!((yp.to_degrees() * 3600.0 - 0.29).abs() < 1e-12);
        }
    }

    #[test]
    fn eop_interpolates_the_finals_series() {
        // First and last bundled finals rows (from the raw file):
        let (d0, _, _) = eop(FINALS_START_MJD);
        assert!((d0 - 0.808418).abs() < 1e-6, "d0={d0}");
        let (d1, _, _) = eop(FINALS_END_MJD);
        assert!((d1 - 0.011404).abs() < 1e-6, "d1={d1}");
        // Linear interpolation: the midpoint is the mean of the endpoints
        // at daily spacing.
        let (a, _, _) = eop(50_000.0);
        let (b, _, _) = eop(50_001.0);
        let (m, _, _) = eop(50_000.5);
        assert!((m - 0.5 * (a + b)).abs() < 1e-12, "m={m}");
        // Inside the series UT1-UTC stays physical (|DUT1| < 0.9 s).
        for mjd in [42_000.0, 50_000.0, 57_844.5, 61_000.0] {
            let (d, _, _) = eop(mjd);
            assert!(d.abs() < 0.9, "mjd={mjd} dut1={d}");
        }
    }

    #[test]
    fn earth_pv_matches_epv00_in_its_valid_window() {
        // SOFA's own low-precision Earth ephemeris is valid for 1900-2100
        // (and the sofars port refuses to answer outside it). Our Keplerian
        // model must track it closely: it feeds the annual aberration.
        for mjd in [51_544.5, 57_844.5, 61_145.0] {
            let (ebpv, ehp) = earth_pv(mjd);
            let (pvh, _pvb) = sofars::eph::epv00(JD_MJD, mjd).expect("epv00 in window");
            let d3 = |a: [f64; 3], b: [f64; 3]| {
                (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
            };
            let dp = d3(ehp, pvh[0]);
            let dv = d3(ebpv[1], pvh[1]);
            // The velocity *direction* is what annual aberration cares about.
            let dot = (0..3).map(|i| ebpv[1][i] * pvh[1][i]).sum::<f64>();
            let n1 = (0..3).map(|i| ebpv[1][i].powi(2)).sum::<f64>().sqrt();
            let n2 = (0..3).map(|i| pvh[1][i].powi(2)).sum::<f64>().sqrt();
            let ang = (dot / (n1 * n2)).clamp(-1.0, 1.0).acos() * ARCSEC_PER_RAD;
            println!("mjd={mjd}: dpos={dp:.6} AU dvel={dv:.6} AU/day vel angle={ang:.4} mas");
            assert!(dp < 5e-4, "mjd={mjd} dpos={dp}");
            assert!(ang < 500.0, "mjd={mjd} vel angle={ang} mas");
        }
    }

    #[test]
    fn no_panics_over_the_supported_and_unsupported_epoch_range() {
        // sofars' `apco13` panics outside 1900-2100 (its epv00 unwrap);
        // we drive `apco` directly, so every date from before the MJD
        // epoch to well past 2126 must work. Sweep a coarse grid over the
        // whole range and a slightly finer one over the century contract
        // (the full 1926-2126 grid lives in tests/test_measures.py).
        let site = Site::from_itrf(SITES[0]).expect("site");
        for mjd in (0..150_000).step_by(1499) {
            let (az, alt) = j2000_to_azel(mjd as f64, 1.0, -0.5, &site).expect("fwd");
            assert!(az.is_finite() && alt.is_finite(), "mjd={mjd}");
            let (ra, dec) = azel_to_j2000(mjd as f64, az, alt, &site).expect("inv");
            assert!(ra.is_finite() && dec.is_finite(), "mjd={mjd}");
        }
        for mjd in (24_000..99_000).step_by(997) {
            let _ = j2000_to_azel(mjd as f64, 2.5, 0.25, &site).expect("contract range");
        }
    }
}
