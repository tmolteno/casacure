//! Measures: a port of the casacore `Measures` subsystem subset that
//! DDFacet and killMS actually use (`casacore.measures.measures`).
//!
//! The full casacore measures system is a large stack (IERS data, frame
//! conversions across a dozen reference types, observatory catalogs). The
//! two imaging codes only exercise seven calls, all against
//! `direction`/`position`/`epoch` measures with `J2000`, `ITRF`, `AZEL`,
//! `AZELGEO` and `UTC`/`GMST` references:
//!
//! | call | where |
//! |---|---|
//! | `measures()` | `ClassMS` (both), `ClassFITSBeam`, `Simul/MakeClusterCat` |
//! | `epoch(refer, quantity)` | `ClassMS.GiveDate`, `ClassFITSBeam` |
//! | `direction(refer, v0, v1)` | `ClassFITSBeam` |
//! | `position(refer, v0, v1, v2)` | `ClassFITSBeam` |
//! | `do_frame(measure)` | `ClassFITSBeam` |
//! | `posangle(dir1, dir2)` | `ClassFITSBeam.getBeamSampleTimes`/`evaluateBeam` |
//! | `measure(dir, refer)` | `ClassFITSBeam.evaluateBeam` (J2000 -> AZELGEO) |
//! | `get_value(measure)` | `ClassFITSBeam` (list of quantities) |
//!
//! Semantics pinned to python-casacore 3.8.1 on this machine:
//!
//! * `epoch('utc', q)` interprets `q` as **days since MJD 0** (so
//!   `epoch('utc', quantity(1000,'s'))` gives `m0 = 0.011574…` in `'d'`).
//! * `measure(e, refer)` between `UTC`/`TAI`/`UT1`/`UT2`/`TT` applies the
//!   real offsets: a leap-second table for TAI−UTC, the bundled IERS
//!   series for UT1−UTC, +32.184 s for TT (returned as `TDT`).  `TDB`
//!   raises rather than relabelling.  ITRF positions are
//!   (lon, geocentric lat, radius); `ITRF`↔`WGS84` converts through the
//!   WGS84 ellipsoid.
//! * `measure(d, 'AZEL')`/`measure(d, 'AZELGEO')` convert via the frame's
//!   `position` and `epoch`; the azimuth is measured **east from north**
//!   and wrapped to `(-π, π]` (the range python-casacore reports), and the
//!   source place is the apparent place (J2000 precessed, nutated and
//!   aberrated to date). casacore's two references differ only in the
//!   observer latitude *it* applies (its `AZEL` uses the geocentric
//!   latitude — see `DIFFERENCES.md`); casacure computes both the astropy
//!   way, i.e. the geodetic latitude.
//! * `posangle(m0, m1)` is the great-circle position angle **at `m0`**
//!   toward `m1`, measured from the direction of increasing declination of
//!   `m0`'s refer frame (`m1` is converted to `m0`'s refer first).
//! * `get_value(m)` returns a `list` of quantities, one per component.
//!
//! The frame conversion itself is the SOFA algorithm stack driven through
//! [`crate::astro`]: IAU 2006/2000A precession-nutation, the Earth
//! rotation angle on UT1 (IERS Earth-orientation data compiled into the
//! library — no `casadata` installation needed), annual and diurnal
//! aberration, solar light deflection and polar motion, with refraction
//! switched off. That is held to **&lt; 1 arcsec of astropy's `AltAz`
//! over 1926–2126** by `tests/test_measures.py` (contract and caveats in
//! `MEASURES_ACCURACY.md`), and to about an arcsecond of real casacore's
//! `AZELGEO` path. The same-frame `posangle` path is exact by
//! construction (a great-circle formula).

use crate::quanta::{mjd_ymd, Quantity};

/// Errors from the measures subset.
#[derive(Debug)]
pub enum MeasuresError {
    /// A quantity could not be interpreted (bad unit, non-conforming).
    Quantity(String),
    /// A required frame (`position` or `epoch`) is missing for the
    /// conversion — casacore's "Cannot convert due to missing frame
    /// information".
    MissingFrame(String),
    /// An unsupported reference code for a measure type.
    BadReference(String),
    /// Any other conversion failure.
    Convert(String),
}

impl std::fmt::Display for MeasuresError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasuresError::Quantity(e) => write!(f, "quantity error: {e}"),
            MeasuresError::MissingFrame(e) => write!(f, "missing frame information: {e}"),
            MeasuresError::BadReference(r) => write!(f, "bad reference type: {r}"),
            MeasuresError::Convert(e) => write!(f, "conversion error: {e}"),
        }
    }
}

impl std::error::Error for MeasuresError {}

fn qerr(e: impl std::fmt::Display) -> MeasuresError {
    MeasuresError::Quantity(e.to_string())
}

/// The reference codes a measure type accepts (uppercase, casacore style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasureType {
    Direction,
    Position,
    Epoch,
}

/// A raw measure value as casacore hands it around: a `type`, a `refer`
/// code, and one or more components. Components are kept in radians (for
/// direction/position angles) or days (for epoch), matching the unit
/// casacore's `m0`/`m1`/`m2` fields use.
#[derive(Debug, Clone)]
pub struct Measure {
    pub mtype: MeasureType,
    pub refer: String,
    /// m0, m1, m2 in the measure's canonical unit (rad for
    /// direction/position angles, m for position height, d for epoch).
    pub values: [f64; 3],
}

impl Measure {
    /// The `{'type':..., 'refer':..., 'm0':{'value':..,'unit':..}, ...}`
    /// dict shape python-casacore returns.
    pub fn to_dict(&self) -> String {
        let (t, u0, u1, u2) = match self.mtype {
            MeasureType::Direction => ("direction", "rad", "rad", ""),
            MeasureType::Position => ("position", "rad", "rad", "m"),
            MeasureType::Epoch => ("epoch", "d", "", ""),
        };
        let mut s = format!(
            "{{\"type\":\"{t}\",\"refer\":\"{}\",\"m0\":{{\"value\":{},\"unit\":\"{u0}\"}}",
            self.refer,
            fmt_f(self.values[0])
        );
        if !u1.is_empty() {
            s.push_str(&format!(
                ",\"m1\":{{\"value\":{},\"unit\":\"{u1}\"}}",
                fmt_f(self.values[1])
            ));
        }
        if !u2.is_empty() {
            s.push_str(&format!(
                ",\"m2\":{{\"value\":{},\"unit\":\"{u2}\"}}",
                fmt_f(self.values[2])
            ));
        }
        s.push('}');
        s
    }
}

fn fmt_f(v: f64) -> String {
    // Plain shortest round-trip form; casacore's dict values are f64.
    format!("{v:?}")
}

/// Greenwich Mean Sidereal Time in degrees at UTC `mjd`: IAU 2006,
/// evaluated on UT1 with the bundled IERS data — the same model astropy's
/// `Time.sidereal_time("mean")` returns. Re-exported from
/// [`crate::astro`], where the astrometry engine lives.
pub use crate::astro::gmst_deg;

/// The measures server: holds the current frame (position + epoch), as
/// casacore's `measures` object does.
pub struct Measures {
    frame: Vec<Measure>,
}

impl Default for Measures {
    fn default() -> Self {
        Self::new()
    }
}

impl Measures {
    pub fn new() -> Measures {
        Measures { frame: Vec::new() }
    }

    /// `do_frame(m)` — record a measure as the current frame component.
    /// Returns `true` (casacore returns whether the frame accepts it).
    pub fn do_frame(&mut self, m: Measure) -> bool {
        self.frame.retain(|f| f.mtype != m.mtype);
        self.frame.push(m);
        true
    }

    fn frame_of(&self, t: MeasureType) -> Option<&Measure> {
        self.frame.iter().find(|f| f.mtype == t)
    }

    /// `epoch(refer, quantity)` — a time measure. `refer` is case-insensitive
    /// (`'utc'`, `'UTC'`, `'GMST1'`, …); the quantity is interpreted as
    /// **days since MJD 0** in the `UTC`/`TAI`/`UT1` family (casacore's
    /// `MEpoch` constructor).
    pub fn epoch(&self, refer: &str, q: &Quantity) -> Result<Measure, MeasuresError> {
        let r = refer.to_ascii_uppercase();
        let days = q.value_in("d").map_err(qerr)?;
        // The family casacore stores relative to MJD 0.
        match r.as_str() {
            "UTC" | "TAI" | "UT1" | "UT2" | "GMST" | "GMST1" | "GAST" | "LAST" | "LMST" | "TDT"
            | "TCG" | "TDB" | "TCD" => Ok(Measure {
                mtype: MeasureType::Epoch,
                refer: r,
                values: [days, 0.0, 0.0],
            }),
            other => Err(MeasuresError::BadReference(other.to_string())),
        }
    }

    /// `direction(refer, v0, v1)` — a sky direction. `v0`/`v1` are
    /// quantities (or raw radians when given as f64). Refer is
    /// case-insensitive; `J2000`, `JMEAN`, `APP`/`APPARENT`, `B1950`, `AZEL`
    /// and `AZELGEO` are accepted (the rest raise).
    pub fn direction(
        &self,
        refer: &str,
        v0: &Quantity,
        v1: &Quantity,
    ) -> Result<Measure, MeasuresError> {
        let r = normalize_dir_refer(refer)?;
        let lon = v0.value_in("rad").map_err(qerr)?;
        let lat = v1.value_in("rad").map_err(qerr)?;
        Ok(Measure {
            mtype: MeasureType::Direction,
            refer: r,
            values: [lon, lat, 0.0],
        })
    }

    /// `position(refer, v0, v1, v2)` — an observer position. `ITRF` takes
    /// Cartesian metres (x, y, z) and stores casacore's (lon, geocentric
    /// lat, radius) form; `WGS84` takes (lon, lat, height) as
    /// (rad, rad, m).
    pub fn position(
        &self,
        refer: &str,
        v0: &Quantity,
        v1: &Quantity,
        v2: &Quantity,
    ) -> Result<Measure, MeasuresError> {
        let r = refer.to_ascii_uppercase();
        match r.as_str() {
            "ITRF" => {
                let x = v0.value_in("m").map_err(qerr)?;
                let y = v1.value_in("m").map_err(qerr)?;
                let z = v2.value_in("m").map_err(qerr)?;
                Ok(Measure {
                    mtype: MeasureType::Position,
                    refer: "ITRF".into(),
                    values: geocentric_itrf([x, y, z]),
                })
            }
            "WGS84" => {
                let lon = v0.value_in("rad").map_err(qerr)?;
                let lat = v1.value_in("rad").map_err(qerr)?;
                let h = v2.value_in("m").map_err(qerr)?;
                Ok(Measure {
                    mtype: MeasureType::Position,
                    refer: "WGS84".into(),
                    values: [lon, lat, h],
                })
            }
            other => Err(MeasuresError::BadReference(other.to_string())),
        }
    }

    /// `measure(m, refer)` — convert `m` to `refer` using the frame.
    pub fn measure(&self, m: &Measure, refer: &str) -> Result<Measure, MeasuresError> {
        match m.mtype {
            MeasureType::Direction => self.convert_direction(m, refer),
            MeasureType::Epoch => self.convert_epoch(m, refer),
            MeasureType::Position => self.convert_position(m, refer),
        }
    }

    fn convert_position(&self, m: &Measure, refer: &str) -> Result<Measure, MeasuresError> {
        let r = refer.to_ascii_uppercase();
        let out = match (m.refer.as_str(), r.as_str()) {
            ("ITRF", "ITRF") | ("WGS84", "WGS84") => return Ok(m.clone()),
            // ITRF measures are (lon, geocentric lat, radius); WGS84 are
            // (lon, geodetic lat, height).  Both frames share the WGS84
            // ellipsoid, so the conversion goes through Cartesian metres.
            ("ITRF", "WGS84") => {
                let site = crate::astro::Site::from_itrf(itrf_cartesian(m.values))
                    .map_err(|e| MeasuresError::Convert(e.to_string()))?;
                [site.elong, site.phi, site.hm]
            }
            ("WGS84", "ITRF") => {
                let xyz = sofars::coords::gd2gc(1, m.values[0], m.values[1], m.values[2])
                    .map_err(|e| MeasuresError::Convert(format!("gd2gc status {e}")))?;
                geocentric_itrf(xyz)
            }
            _ => return Err(MeasuresError::BadReference(r)),
        };
        Ok(Measure {
            mtype: MeasureType::Position,
            refer: r,
            values: out,
        })
    }

    fn convert_epoch(&self, m: &Measure, refer: &str) -> Result<Measure, MeasuresError> {
        let r = refer.to_ascii_uppercase();
        if m.refer == r {
            return Ok(m.clone());
        }
        if matches!(r.as_str(), "GMST" | "GMST1" | "GAST" | "LAST" | "LMST")
            || matches!(
                m.refer.as_str(),
                "GMST" | "GMST1" | "GAST" | "LAST" | "LMST"
            )
        {
            return Err(MeasuresError::Convert(
                "sidereal-time epoch conversion needs a position frame".into(),
            ));
        }
        // Convert through a UTC pivot: TAI-UTC from the leap-second table,
        // UT1-UTC from the bundled IERS series (the same data GMST uses).
        let utc = epoch_to_utc(&m.refer, m.values[0])?;
        let (days, out_refer) = utc_to_epoch(&r, utc)?;
        Ok(Measure {
            mtype: MeasureType::Epoch,
            refer: out_refer,
            values: [days, 0.0, 0.0],
        })
    }

    fn convert_direction(&self, m: &Measure, refer: &str) -> Result<Measure, MeasuresError> {
        let r = normalize_dir_refer(refer)?;
        if r == m.refer {
            return Ok(m.clone());
        }
        // Only the J2000<->AZEL/AZELGEO family is implemented (what
        // ClassFITSBeam uses). Everything needs the frame.
        let need_azel = matches!(r.as_str(), "AZEL" | "AZELGEO");
        let have_azel = matches!(m.refer.as_str(), "AZEL" | "AZELGEO");
        if !need_azel && !have_azel && !(m.refer == "J2000" && r == "J2000") {
            // e.g. J2000 -> B1950: not in the supported subset.
            return Err(MeasuresError::Convert(format!(
                "direction {} -> {} is not in the supported subset",
                m.refer, r
            )));
        }
        if need_azel {
            self.j2000_to_azel(m, &r)
        } else if have_azel {
            self.azel_to_j2000(m, &r)
        } else {
            Err(MeasuresError::Convert(format!(
                "unsupported direction {} -> {}",
                m.refer, r
            )))
        }
    }

    /// J2000/ICRS -> AZEL / AZELGEO. Needs the frame's position (ITRF or
    /// WGS84) and epoch; the SOFA-stack conversion in [`crate::astro`] does
    /// the work (precession/nutation/aberration to the apparent place, then
    /// the topocentric rotation — astropy-accurate).
    ///
    /// Only J2000/ICRS inputs are accepted: the astrometry context converts
    /// *from* the ICRS. An `APP`/`JMEAN`/`B1950` input would be silently
    /// misinterpreted, so it is refused instead.
    fn j2000_to_azel(&self, m: &Measure, target: &str) -> Result<Measure, MeasuresError> {
        if !matches!(m.refer.as_str(), "J2000" | "ICRS") {
            return Err(MeasuresError::Convert(format!(
                "{} -> {}: only J2000/ICRS direction inputs are supported",
                m.refer, target
            )));
        }
        let pos = self.frame_of(MeasureType::Position).ok_or_else(|| {
            MeasuresError::MissingFrame("position frame required for AZEL".into())
        })?;
        let ep = self
            .frame_of(MeasureType::Epoch)
            .ok_or_else(|| MeasuresError::MissingFrame("epoch frame required for AZEL".into()))?;
        let site = site_of(pos)?;
        let mjd = ep.values[0]; // days since MJD 0 = MJD itself
        let (az, alt) = crate::astro::j2000_to_azel(mjd, m.values[0], m.values[1], &site)
            .map_err(|e| MeasuresError::Convert(e.to_string()))?;
        Ok(Measure {
            mtype: MeasureType::Direction,
            refer: target.to_string(),
            values: [az, alt, 0.0],
        })
    }

    /// AZEL / AZELGEO -> J2000/ICRS (the inverse of [`j2000_to_azel`]; used
    /// by `posangle` when `m0` is J2000 and `m1` is the AZEL/AZELGEO
    /// zenith). Only a J2000/ICRS target is supported, for the same reason.
    fn azel_to_j2000(&self, m: &Measure, target: &str) -> Result<Measure, MeasuresError> {
        if !matches!(target, "J2000" | "ICRS") {
            return Err(MeasuresError::Convert(format!(
                "AZEL -> {target}: only J2000/ICRS direction outputs are supported"
            )));
        }
        let pos = self.frame_of(MeasureType::Position).ok_or_else(|| {
            MeasuresError::MissingFrame("position frame required for AZEL".into())
        })?;
        let ep = self
            .frame_of(MeasureType::Epoch)
            .ok_or_else(|| MeasuresError::MissingFrame("epoch frame required for AZEL".into()))?;
        let site = site_of(pos)?;
        let mjd = ep.values[0];
        let (ra, dec) = crate::astro::azel_to_j2000(mjd, m.values[0], m.values[1], &site)
            .map_err(|e| MeasuresError::Convert(e.to_string()))?;
        Ok(Measure {
            mtype: MeasureType::Direction,
            refer: target.to_string(),
            values: [ra, dec, 0.0],
        })
    }

    /// `posangle(m0, m1)` — the position angle at `m0` toward `m1`, in
    /// radians (a quantity in degrees is returned through
    /// [`Measures::posangle_deg`]). `m1` is converted to `m0`'s refer first
    /// (casacore's rule), then the great-circle PA from `m0`'s increasing-
    /// declination direction to `m1` is measured.
    pub fn posangle(&self, m0: &Measure, m1: &Measure) -> Result<f64, MeasuresError> {
        if m0.mtype != MeasureType::Direction || m1.mtype != MeasureType::Direction {
            return Err(MeasuresError::Convert(
                "posangle takes direction measures".into(),
            ));
        }
        let m1c = if m1.refer == m0.refer {
            m1.clone()
        } else {
            self.convert_direction(m1, &m0.refer)?
        };
        Ok(pa_at(
            m0.values[0],
            m0.values[1],
            m1c.values[0],
            m1c.values[1],
        ))
    }

    /// `get_value(m)` — the measure's components as quantities, one per
    /// field, in the measure's canonical unit (rad for direction/position
    /// angles, m for position height, d for epoch).
    pub fn get_value(&self, m: &Measure) -> Vec<Quantity> {
        let (u0, u1, u2) = match m.mtype {
            MeasureType::Direction => ("rad", Some("rad"), None),
            MeasureType::Position => ("rad", Some("rad"), Some("m")),
            MeasureType::Epoch => ("d", None, None),
        };
        let mut out = vec![Quantity::new(m.values[0], u0).expect("unit parses")];
        if let Some(u) = u1 {
            out.push(Quantity::new(m.values[1], u).expect("unit parses"));
        }
        if let Some(u) = u2 {
            out.push(Quantity::new(m.values[2], u).expect("unit parses"));
        }
        out
    }
}

fn normalize_dir_refer(refer: &str) -> Result<String, MeasuresError> {
    let r = refer.to_ascii_uppercase();
    match r.as_str() {
        "J2000" | "JMEAN" | "JTRUE" | "APP" | "APPARENT" | "B1950" | "BMEAN" | "BTRUE" | "AZEL"
        | "AZELGEO" | "GALACTIC" | "SUPERGALACTIC" | "ECLIPTIC" | "ICRS" => Ok(r),
        other => Err(MeasuresError::BadReference(other.to_string())),
    }
}

/// The observer's site for a `position` measure: ITRF measures are
/// (lon, geocentric lat, radius) — Cartesian metres are reconstructed for
/// the geodetic conversion; WGS84 is already (lon, lat, height).
fn site_of(pos: &Measure) -> Result<crate::astro::Site, MeasuresError> {
    match pos.refer.as_str() {
        "ITRF" => crate::astro::Site::from_itrf(itrf_cartesian(pos.values))
            .map_err(|e| MeasuresError::Convert(e.to_string())),
        "WGS84" => Ok(crate::astro::Site::wgs84(
            pos.values[0],
            pos.values[1],
            pos.values[2],
        )),
        other => Err(MeasuresError::BadReference(other.to_string())),
    }
}

/// ITRF measure values (lon, geocentric lat, radius) back to Cartesian
/// metres.
fn itrf_cartesian(v: [f64; 3]) -> [f64; 3] {
    let (lon, lat, r) = (v[0], v[1], v[2]);
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    [r * cos_lat * cos_lon, r * cos_lat * sin_lon, r * sin_lat]
}

/// Cartesian metres as casacore's ITRF position: (lon, geocentric lat,
/// radius).
fn geocentric_itrf([x, y, z]: [f64; 3]) -> [f64; 3] {
    let r = (x * x + y * y + z * z).sqrt();
    [y.atan2(x), z.atan2(x.hypot(y)), r]
}

/// TAI-UTC (leap seconds) at UTC MJD `mjd`, from the published leap-second
/// table (piecewise constant).  Dates before 1972-01-01 (MJD 41317, the
/// first table entry) clamp to its 10 s value — the bundled IERS series
/// does not reach back that far either.
fn tai_minus_utc(mjd: f64) -> f64 {
    // (UTC MJD of the leap, TAI-UTC from that instant).
    const LEAPS: &[(f64, f64)] = &[
        (41317.0, 10.0), // 1972-01-01
        (41499.0, 11.0), // 1972-07-01
        (41683.0, 12.0), // 1973-01-01
        (42048.0, 13.0), // 1974-01-01
        (42413.0, 14.0), // 1975-01-01
        (42778.0, 15.0), // 1976-01-01
        (43144.0, 16.0), // 1977-01-01
        (43509.0, 17.0), // 1978-01-01
        (43874.0, 18.0), // 1979-01-01
        (44239.0, 19.0), // 1980-01-01
        (44426.0, 20.0), // 1981-07-01
        (44791.0, 21.0), // 1982-07-01
        (45156.0, 22.0), // 1983-07-01
        (45887.0, 23.0), // 1985-07-01
        (46247.0, 24.0), // 1988-01-01
        (46977.0, 25.0), // 1990-01-01
        (47342.0, 26.0), // 1991-01-01
        (47519.0, 27.0), // 1992-07-01
        (47884.0, 28.0), // 1993-07-01
        (48249.0, 29.0), // 1994-07-01
        (48803.0, 30.0), // 1996-01-01
        (48990.0, 31.0), // 1997-07-01
        (49522.0, 32.0), // 1999-01-01
        (52091.0, 33.0), // 2006-01-01
        (53187.0, 34.0), // 2009-01-01
        (54470.0, 35.0), // 2012-07-01
        (55566.0, 36.0), // 2015-07-01
        (56109.0, 37.0), // 2017-01-01
    ];
    let i = LEAPS.partition_point(|&(at, _)| at <= mjd);
    if i == 0 {
        LEAPS[0].1
    } else {
        LEAPS[i - 1].1
    }
}

/// `days` on the `from` scale as UTC days (MJD-0 scale, like casacore).
/// UT2 converts as UT1: python-casacore 3.8.1 returns a bit-identical
/// value for the two scales (probed over 45000-59444).
fn epoch_to_utc(from: &str, days: f64) -> Result<f64, MeasuresError> {
    let secs = match from {
        "UTC" => 0.0,
        "TAI" => tai_minus_utc(days),
        "UT1" | "UT2" => crate::astro::eop(days).0,
        "TDT" | "TT" => tai_minus_utc(days) + 32.184,
        other => return Err(MeasuresError::BadReference(other.to_string())),
    };
    Ok(days - secs / 86400.0)
}

/// UTC `days` on the `to` scale.  `TT` comes back as casacore's `TDT`;
/// `TDB` keeps raising (its periodic terms are not modelled) rather than
/// relabelling the input.
fn utc_to_epoch(to: &str, days: f64) -> Result<(f64, String), MeasuresError> {
    let secs = match to {
        "UTC" => 0.0,
        "TAI" => tai_minus_utc(days),
        "UT1" | "UT2" => crate::astro::eop(days).0,
        "TDT" | "TT" => tai_minus_utc(days) + 32.184,
        other => return Err(MeasuresError::BadReference(other.to_string())),
    };
    let refer = if to == "TT" { "TDT" } else { to };
    Ok((days + secs / 86400.0, refer.to_string()))
}

/// WGS84 geodetic (lon, lat) from ITRF Cartesian (x, y, z) in metres.
/// Bowring's iteration on the WGS84 ellipsoid.
pub fn itrf_to_geodetic(x: f64, y: f64, z: f64) -> (f64, f64) {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    let e2 = F * (2.0 - F);
    let lon = y.atan2(x);
    let p = x.hypot(y);
    let mut lat = z.atan2(p * (1.0 - e2));
    for _ in 0..12 {
        let n = A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
        lat = (z + e2 * n * lat.sin()).atan2(p);
    }
    (lon, lat)
}

/// The great-circle position angle at `(ra1, dec1)` toward `(ra2, dec2)`,
/// measured from the direction of increasing declination at `(ra1, dec1)`
/// (casacore's `posangle` for same-frame directions).
fn pa_at(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let d_ra = ra2 - ra1;
    d_ra.sin()
        .atan2(dec1.cos() * dec2.tan() - dec1.sin() * d_ra.cos())
}

/// Calendar date (year, month, day) of an MJD — re-exported for tests that
/// cross-check `epoch` against a known date.
pub fn mjd_to_ymd(mjd: f64) -> (i64, i64, i64) {
    mjd_ymd(mjd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: f64, u: &str) -> Quantity {
        Quantity::new(v, u).unwrap()
    }

    /// The session position from the DDFacet/killMS probes: ITRF metres near
    /// (6100000, 100000, 100000) -> geodetic (0.939 deg, 0.946 deg).
    const POS: (f64, f64, f64) = (6_100_000.0, 100_000.0, 100_000.0);
    const J2000: (f64, f64) = (0.3, -0.5);
    /// MJD 57844.5 = 2017-04-01 12:00 UTC.
    const MJD: f64 = 57844.5;

    #[test]
    fn epoch_days_since_mjd0() {
        let m = Measures::new();
        // epoch('utc', quantity(1000, 's')) -> m0 = 1000/86400 days.
        let e = m.epoch("utc", &q(1000.0, "s")).unwrap();
        assert_eq!(e.refer, "UTC");
        assert!((e.values[0] - 1000.0 / 86400.0).abs() < 1e-12);
        // A quantity in days passes through unchanged.
        let e2 = m.epoch("UTC", &q(MJD, "d")).unwrap();
        assert!((e2.values[0] - MJD).abs() < 1e-12);
    }

    #[test]
    fn get_value_returns_quantities() {
        let m = Measures::new();
        let d = m
            .direction("J2000", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        let vals = m.get_value(&d);
        assert_eq!(vals.len(), 2);
        assert!((vals[0].value - J2000.0).abs() < 1e-12);
        assert_eq!(vals[0].unit.display, "rad");
    }

    #[test]
    fn posangle_same_frame_exact() {
        // casacore's posangle for two J2000 directions is the analytic
        // great-circle PA (verified live: posangle((0.3,-0.5),(0,0)) =
        // -32.831026912 deg).
        let m = Measures::new();
        let d1 = m
            .direction("J2000", &q(0.3, "rad"), &q(-0.5, "rad"))
            .unwrap();
        let d2 = m
            .direction("J2000", &q(0.0, "rad"), &q(0.0, "rad"))
            .unwrap();
        let pa = m.posangle(&d1, &d2).unwrap().to_degrees();
        assert!((pa - (-32.831026912)).abs() < 1e-6, "pa={pa}");
    }

    #[test]
    fn posangle_reversal_is_pi_complement() {
        // posangle(m0,m1) and posangle(m1,m0) are the great-circle PA at
        // each point; they sum to pi only when the two points share the same
        // reference (increasing-declination) direction. For arbitrary points
        // the sum is the difference of the two reference directions — the
        // analytic value is exact, so pin it.
        let m = Measures::new();
        let d1 = m
            .direction("J2000", &q(0.3, "rad"), &q(-0.5, "rad"))
            .unwrap();
        let d2 = m
            .direction("J2000", &q(0.0, "rad"), &q(0.0, "rad"))
            .unwrap();
        let a = m.posangle(&d1, &d2).unwrap().to_degrees();
        let b = m.posangle(&d2, &d1).unwrap().to_degrees();
        // Verified against real casacore: -32.831 and 151.589 deg.
        assert!((a - (-32.831026912)).abs() < 1e-6, "a={a}");
        assert!((b - 151.589000582).abs() < 1e-6, "b={b}");
    }

    #[test]
    fn j2000_to_azelgeo_matches_casacore() {
        // casacore (probed live at this session position + MJD):
        //   measure(J2000(0.3,-0.5), 'AZELGEO') -> az=2.945114291, alt=1.044796770.
        // casacure computes the same quantity through the SOFA stack and
        // lands 0.6 arcsec from that casacore value (az 2.945111616,
        // alt 1.044795251). casacure is held to <1" of astropy over
        // 1926-2126 by tests/test_measures.py; casacore's own AZELGEO is
        // within ~1" of astropy too (see MEASURES_ACCURACY.md).
        // Tolerance: 1e-5 rad = 2 arcsec.
        let mut m = Measures::new();
        assert!(m.do_frame(
            m.position("ITRF", &q(POS.0, "m"), &q(POS.1, "m"), &q(POS.2, "m"))
                .unwrap()
        ));
        assert!(m.do_frame(m.epoch("UTC", &q(MJD, "d")).unwrap()));
        let d = m
            .direction("J2000", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        let azel = m.measure(&d, "AZELGEO").unwrap();
        assert_eq!(azel.refer, "AZELGEO");
        let az = azel.values[0];
        let alt = azel.values[1];
        assert!((az - 2.945114291).abs() < 1e-5, "az={az}");
        assert!((alt - 1.044796770).abs() < 1e-5, "alt={alt}");
    }

    #[test]
    fn j2000_to_azelgeo_missing_frame_errors() {
        let m = Measures::new();
        let d = m
            .direction("J2000", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        assert!(matches!(
            m.measure(&d, "AZELGEO"),
            Err(MeasuresError::MissingFrame(_))
        ));
    }

    #[test]
    fn azel_conversions_only_accept_j2000() {
        // The engine converts from the ICRS: an APP/JMEAN/B1950 input (or
        // output) would be silently misinterpreted, so it is refused.
        let mut m = Measures::new();
        m.do_frame(
            m.position("ITRF", &q(POS.0, "m"), &q(POS.1, "m"), &q(POS.2, "m"))
                .unwrap(),
        );
        m.do_frame(m.epoch("UTC", &q(MJD, "d")).unwrap());
        let app = m
            .direction("APP", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        assert!(matches!(
            m.measure(&app, "AZEL"),
            Err(MeasuresError::Convert(_))
        ));
        let j2000 = m
            .direction("J2000", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        let azel = m.measure(&j2000, "AZEL").unwrap();
        assert!(matches!(
            m.measure(&azel, "APP"),
            Err(MeasuresError::Convert(_))
        ));
        // ICRS is the same frame as J2000 here and is accepted.
        let icrs = m
            .direction("ICRS", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        assert!(m.measure(&icrs, "AZEL").is_ok());
    }

    #[test]
    fn azelgeo_to_j2000_roundtrip() {
        let mut m = Measures::new();
        m.do_frame(
            m.position("ITRF", &q(POS.0, "m"), &q(POS.1, "m"), &q(POS.2, "m"))
                .unwrap(),
        );
        m.do_frame(m.epoch("UTC", &q(MJD, "d")).unwrap());
        let d = m
            .direction("J2000", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        let azel = m.measure(&d, "AZELGEO").unwrap();
        let back = m.measure(&azel, "J2000").unwrap();
        assert_eq!(back.refer, "J2000");
        // The SOFA forward/inverse pair round-trips to ~3e-13 rad.
        assert!(
            (back.values[0] - J2000.0).abs() < 1e-9 && (back.values[1] - J2000.1).abs() < 1e-9,
            "roundtrip {} {}",
            back.values[0],
            back.values[1]
        );
    }

    #[test]
    fn posangle_with_mixed_frames() {
        // casacore: posangle(J2000 src, AZELGEO zenith) = the great-circle
        // PA at src toward zenith-as-J2000 (verified live = -12.875850547 deg
        // at this position/epoch). casacure gives -12.876001978 deg, 0.55
        // arcsec away (casacure is held to <1" of astropy; see
        // MEASURES_ACCURACY.md). The mixed-frame path converts m1 to m0's
        // refer first.
        let mut m = Measures::new();
        m.do_frame(
            m.position("ITRF", &q(POS.0, "m"), &q(POS.1, "m"), &q(POS.2, "m"))
                .unwrap(),
        );
        m.do_frame(m.epoch("UTC", &q(MJD, "d")).unwrap());
        let src = m
            .direction("J2000", &q(J2000.0, "rad"), &q(J2000.1, "rad"))
            .unwrap();
        let zen = m
            .direction("AZELGEO", &q(0.0, "deg"), &q(90.0, "deg"))
            .unwrap();
        let pa = m.posangle(&src, &zen).unwrap().to_degrees();
        // The PA formula is exact; the residual is the zenith->J2000
        // conversion difference between the two engines (0.55 arcsec here).
        assert!((pa - (-12.875850547)).abs() < 5e-4, "pa={pa}");
    }

    #[test]
    fn gmst_anchors() {
        // IAU 2006 GMST on UT1 (the model astropy's
        // `Time.sidereal_time("mean")` returns) at MJD 51544.5:
        // astropy 280.462105800 deg; casacure 280.462105812 deg.
        let g = gmst_deg(51544.5);
        assert!((g - 280.4621058).abs() < 1e-6, "gmst={g}");
        // Pre-IERS-data date: astropy holds UT1-UTC at +0.807841 s there,
        // and so do we (see astro::eop).
        let g1950 = gmst_deg(33543.5);
        assert!((g1950 - 177.825888838).abs() < 1e-6, "gmst={g1950}");
    }

    #[test]
    fn itrf_to_geodetic_anchor() {
        // (6100000, 100000, 100000) -> lon 0.939190946 deg, lat 0.945681226
        // deg (Bowring on WGS84; verified independently).
        let (lon, lat) = itrf_to_geodetic(POS.0, POS.1, POS.2);
        assert!((lon.to_degrees() - 0.939190946).abs() < 1e-6, "lon={lon}");
        assert!((lat.to_degrees() - 0.945681226).abs() < 1e-6, "lat={lat}");
    }

    /// Seconds from a converted epoch measure back to the input MJD.
    fn seconds_from(m: &Measures, e: &Measure, refer: &str) -> f64 {
        (m.measure(e, refer).unwrap().values[0] - e.values[0]) * 86400.0
    }

    #[test]
    fn epoch_offsets_match_casacore() {
        // python-casacore 3.8.1 at MJD 59444.5 (issue #43): UTC->TAI +37 s,
        // UTC->UT1 -0.1301 s (bundled IERS series), UTC->TT +69.184 s
        // (returned as TDT); TDB keeps raising.
        let m = Measures::new();
        let e = m.epoch("UTC", &q(59444.5, "d")).unwrap();
        assert!((seconds_from(&m, &e, "TAI") - 37.0).abs() < 1e-4);
        assert!((seconds_from(&m, &e, "UT1") - (-0.1301)).abs() < 1e-3);
        let tt = m.measure(&e, "TT").unwrap();
        assert_eq!(tt.refer, "TDT");
        assert!(((tt.values[0] - 59444.5) * 86400.0 - 69.184).abs() < 1e-6);
        assert!(m.measure(&e, "TDB").is_err());
        // And back: TAI->UTC -37 s.
        let tai = m.epoch("TAI", &q(59444.5, "d")).unwrap();
        assert!((seconds_from(&m, &tai, "UTC") - (-37.0)).abs() < 1e-4);
        // UT2 converts as UT1 (casacore returns a bit-identical value).
        assert!((seconds_from(&m, &e, "UT2") - seconds_from(&m, &e, "UT1")).abs() < 1e-12);
    }

    #[test]
    fn leap_second_table_boundaries() {
        assert_eq!(tai_minus_utc(59444.5), 37.0); // 2021
        assert_eq!(tai_minus_utc(56108.999), 36.0); // just before 2017-01-01
        assert_eq!(tai_minus_utc(56109.0), 37.0);
        assert_eq!(tai_minus_utc(41316.999), 10.0); // before the first leap
        assert_eq!(tai_minus_utc(50000.0), 32.0); // 1995-ish
    }

    #[test]
    fn itrf_position_is_lon_lat_radius() {
        // python-casacore 3.8.1 (issue #43): ITRF Cartesian metres come
        // back as (lon, geocentric lat, radius), units rad/rad/m.
        let m = Measures::new();
        let p = m
            .position(
                "ITRF",
                &q(5109360.133, "m"),
                &q(2006852.586, "m"),
                &q(-3238948.127, "m"),
            )
            .unwrap();
        assert!((p.values[0] - 0.374266).abs() < 1e-6, "lon={}", p.values[0]);
        assert!(
            (p.values[1] - (-0.533065)).abs() < 1e-6,
            "lat={}",
            p.values[1]
        );
        assert!((p.values[2] - 6373680.51).abs() < 0.01, "r={}", p.values[2]);
        // ITRF->WGS84: geodetic (lon, lat, height); casacore gives
        // (0.374266, -0.536009, 1086.599).
        let w = m.measure(&p, "WGS84").unwrap();
        assert!((w.values[0] - 0.374266).abs() < 1e-6);
        assert!(
            (w.values[1] - (-0.536009)).abs() < 1e-6,
            "lat={}",
            w.values[1]
        );
        assert!((w.values[2] - 1086.599).abs() < 1e-3, "h={}", w.values[2]);
        // And back again.
        let back = m.measure(&w, "ITRF").unwrap();
        for (a, b) in back.values.iter().zip(p.values.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
