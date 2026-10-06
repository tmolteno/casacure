//! Image coordinate systems (issue #14): the subset of casacore's
//! `CoordinateSystem` that DDFacet exercises — a direction coordinate
//! (zenithal SIN projection), plus linear per-axis coordinates (Stokes,
//! spectral, generic), converted to/from pixel coordinates and to/from the
//! nested-record layout casacore stores in a CASA image's `coords` keyword
//! (and that `coordinates().dict()` exposes).
//!
//! Ground truth is `scripts/probe_pyrap_images.py`: pixel->world through
//! the direction coordinate matches casacore's `image.toworld` to 3e-16 rad
//! and the inverse to 1e-9 pixel on the probe image; the record layout is
//! the on-disk `coords` keyword of a casacore-written `.image`.

use crate::record::{ArrayData, ArrayValue, RecordValue, TableRecord};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoordError {
    #[error("unsupported direction projection {0:?} (SIN is implemented)")]
    UnsupportedProjection(String),
    #[error("world coordinate is outside the projection hemisphere (r = {value:.6})")]
    OffProjection { value: f64 },
    #[error("coordinate record field {field}: expected {expected}")]
    BadField {
        field: String,
        expected: &'static str,
    },
    #[error("pixel/world vector length {got}, image has {want} axes")]
    AxisCount { got: usize, want: usize },
}

/// One axis family of the coordinate system, in casacore CS order.
#[derive(Debug, Clone)]
pub enum Coordinate {
    /// A two-axis direction coordinate (`direction0`): long (RA), lat
    /// (Dec).  All angles radians; reference pixel 0-based (casacore).
    Direction {
        name: String,
        crval: [f64; 2],
        crpix: [f64; 2],
        cdelt: [f64; 2],
        /// The 2x2 pixel->intermediate rotation (identity in practice).
        pc: [[f64; 2]; 2],
        system: String,
        projection: String,
        /// The per-axis units ("rad" from records; arcmin "'" in casacore's
        /// default image template).
        units: [String; 2],
        /// The CS pixel axes this coordinate occupies (long, lat).
        pixel_axes: [usize; 2],
    },
    /// A linear axis (Stokes, spectral, or generic linear): world =
    /// crval + (pixel - crpix) * cdelt * pc per axis.
    Linear {
        name: String,
        crval: Vec<f64>,
        crpix: Vec<f64>,
        cdelt: Vec<f64>,
        pc: Vec<f64>,
        pixel_axes: Vec<usize>,
        /// Stokes letters when this is a StokesCoordinate.
        stokes: Vec<String>,
    },
}

/// The image coordinate system: coordinates in casacore CS order, over an
/// image of `nimaxes` pixel axes (CASA order: axis 0 is the FITS NAXIS1 /
/// fastest-varying axis; the pyrap/numpy order is the reverse).
#[derive(Debug, Clone)]
pub struct CoordinateSystem {
    pub coords: Vec<Coordinate>,
    pub nimaxes: usize,
    /// The raw `coords` record when the source was a CASA image (kept for
    /// `dict()` fidelity).
    record: Option<TableRecord>,
}

fn f64s(value: &RecordValue) -> Option<Vec<f64>> {
    match value {
        RecordValue::Double(v) => Some(vec![*v]),
        RecordValue::Float(v) => Some(vec![f64::from(*v)]),
        RecordValue::Int(v) => Some(vec![f64::from(*v)]),
        RecordValue::Int64(v) => Some(vec![*v as f64]),
        RecordValue::Array(ArrayValue {
            data: ArrayData::Double(v),
            ..
        }) => Some(v.clone()),
        RecordValue::Array(ArrayValue {
            data: ArrayData::Float(v),
            ..
        }) => Some(v.iter().map(|f| f64::from(*f)).collect()),
        RecordValue::Array(ArrayValue {
            data: ArrayData::Int(v),
            ..
        }) => Some(v.iter().map(|i| f64::from(*i)).collect()),
        _ => None,
    }
}

fn field_f64s(rec: &TableRecord, field: &str) -> Option<Vec<f64>> {
    rec.get(field).and_then(f64s)
}

fn field_str(rec: &TableRecord, field: &str) -> Option<String> {
    match rec.get(field) {
        Some(RecordValue::String(s)) => Some(s.trim_end_matches('\0').to_string()),
        _ => None,
    }
}

const COORD_PREFIXES: [&str; 4] = ["direction", "stokes", "spectral", "linear"];

impl CoordinateSystem {
    /// An empty system over `nimaxes` axes (an image with no coordinates
    /// yet — every axis reads as a bare pixel index).
    pub fn empty(nimaxes: usize) -> CoordinateSystem {
        CoordinateSystem {
            coords: Vec::new(),
            nimaxes,
            record: None,
        }
    }

    /// The default system casacore builds for
    /// `image(imagename=, shape=)` with no coordsys: a J2000/SIN direction
    /// on the two spatial axes (unit arcmin, crpix at the centre), Stokes
    /// I, and an LSRK spectral axis — the template DDFacet's
    /// ClassCasaimage.createScratch creates, mutates and re-creates with.
    pub fn default_for(shape: &[usize]) -> CoordinateSystem {
        let ndim = shape.len();
        // casa pixel axis p = numpy axis ndim-1-p; direction on the last
        // two numpy axes = casa axes 0 (long/x) and 1 (lat/y).
        let (nx, ny) = (shape[ndim - 1], shape[ndim - 2]);
        let direction = Coordinate::Direction {
            name: "direction0".into(),
            crval: [0.0, 0.0],
            crpix: [nx as f64 / 2.0, ny as f64 / 2.0],
            cdelt: [-1.0, 1.0],
            pc: [[1.0, 0.0], [0.0, 1.0]],
            system: "J2000".into(),
            projection: "SIN".into(),
            units: ["'".into(), "'".into()],
            pixel_axes: [0, 1],
        };
        let stokes = Coordinate::Linear {
            name: "stokes1".into(),
            crval: vec![1.0],
            crpix: vec![0.0],
            cdelt: vec![1.0],
            pc: vec![1.0],
            pixel_axes: vec![ndim - 2],
            stokes: vec!["I".into()],
        };
        let spectral = Coordinate::Linear {
            name: "spectral2".into(),
            crval: vec![1.415e9],
            crpix: vec![1.0],
            cdelt: vec![1000.0],
            pc: vec![1.0],
            // casa axis ndim-1 (the channel axis; numpy axis 0).
            pixel_axes: vec![ndim - 1],
            stokes: Vec::new(),
        };
        CoordinateSystem {
            coords: vec![direction, stokes, spectral],
            nimaxes: ndim,
            record: None,
        }
    }

    /// Parse a CASA image's `coords` keyword record.  `nimaxes` is the
    /// image's axis count (the raster cell's dimensions).
    pub fn from_record(rec: &TableRecord, nimaxes: usize) -> Result<CoordinateSystem, CoordError> {
        // Coordinate fields, ordered by their trailing index: direction0,
        // stokes1, spectral2, linear3, ...
        let mut named: Vec<(usize, &str, &TableRecord)> = Vec::new();
        for i in 0..rec.desc.fields.len() {
            let name = rec.desc.fields[i].name.clone();
            for prefix in COORD_PREFIXES {
                if let Some(idx) = name
                    .strip_prefix(prefix)
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    if let Some(RecordValue::Record(sub)) = rec.get(&name) {
                        named.push((idx, prefix, sub));
                    }
                }
            }
        }
        named.sort_by_key(|(i, _, _)| *i);
        let mut coords = Vec::with_capacity(named.len());
        for (idx, prefix, sub) in named {
            let field = format!("{prefix}{idx}");
            // This coordinate's pixel axes in CS order (parent pixelmapK).
            let pixel_axes = field_f64s(rec, &format!("pixelmap{idx}"))
                .map(|v| v.iter().map(|f| *f as usize).collect::<Vec<usize>>())
                .unwrap_or_default();
            match prefix {
                "direction" => {
                    let crval = field_f64s(sub, "crval").ok_or(CoordError::BadField {
                        field: format!("{field}.crval"),
                        expected: "a numeric array",
                    })?;
                    let crpix = field_f64s(sub, "crpix").unwrap_or_else(|| vec![0.0; 2]);
                    let cdelt = field_f64s(sub, "cdelt").ok_or(CoordError::BadField {
                        field: format!("{field}.cdelt"),
                        expected: "a numeric array",
                    })?;
                    let projection = field_str(sub, "projection").unwrap_or_else(|| "SIN".into());
                    if projection != "SIN" {
                        return Err(CoordError::UnsupportedProjection(projection));
                    }
                    let pc = match field_f64s(sub, "pc") {
                        Some(v) if v.len() == 4 => [[v[0], v[1]], [v[2], v[3]]],
                        _ => [[1.0, 0.0], [0.0, 1.0]],
                    };
                    if crval.len() < 2 || cdelt.len() < 2 {
                        return Err(CoordError::BadField {
                            field: format!("{field}.crval/cdelt"),
                            expected: "two values (long, lat)",
                        });
                    }
                    let units = match sub.get("units") {
                        Some(RecordValue::Array(a)) => match &a.data {
                            ArrayData::String(sv) => [
                                sv.first().cloned().unwrap_or_else(|| "rad".into()),
                                sv.get(1).cloned().unwrap_or_else(|| "rad".into()),
                            ],
                            _ => ["rad".into(), "rad".into()],
                        },
                        _ => ["rad".into(), "rad".into()],
                    };
                    coords.push(Coordinate::Direction {
                        name: field,
                        crval: [crval[0], crval[1]],
                        crpix: [
                            crpix.first().copied().unwrap_or(0.0),
                            crpix.get(1).copied().unwrap_or(0.0),
                        ],
                        cdelt: [cdelt[0], cdelt[1]],
                        pc,
                        system: field_str(sub, "system").unwrap_or_default(),
                        projection,
                        units,
                        pixel_axes: [
                            pixel_axes.first().copied().unwrap_or(0),
                            pixel_axes.get(1).copied().unwrap_or(1),
                        ],
                    });
                }
                other => {
                    // Stokes / spectral / linear: all linear here.  The
                    // linear parameters are top-level for Stokes/Linear,
                    // under the `wcs` sub-record for Spectral.
                    let (crval, crpix, cdelt, pc) = if let (Some(crval), Some(cdelt)) =
                        (field_f64s(sub, "crval"), field_f64s(sub, "cdelt"))
                    {
                        let crpix =
                            field_f64s(sub, "crpix").unwrap_or_else(|| vec![0.0; crval.len()]);
                        let pc = field_f64s(sub, "pc").unwrap_or_else(|| vec![1.0; crval.len()]);
                        (crval, crpix, cdelt, pc)
                    } else if let Some(RecordValue::Record(wcs)) = sub.get("wcs") {
                        (
                            field_f64s(wcs, "crval").unwrap_or_default(),
                            field_f64s(wcs, "crpix").unwrap_or_default(),
                            field_f64s(wcs, "cdelt").unwrap_or_default(),
                            field_f64s(wcs, "pc").unwrap_or_else(|| vec![1.0]),
                        )
                    } else {
                        return Err(CoordError::BadField {
                            field: format!("{field}.crval"),
                            expected: "a numeric array or a wcs sub-record",
                        });
                    };
                    let stokes = match sub.get("stokes") {
                        Some(RecordValue::Array(a)) => match &a.data {
                            ArrayData::String(s) => s.clone(),
                            _ => Vec::new(),
                        },
                        _ => Vec::new(),
                    };
                    let n = crval.len();
                    coords.push(Coordinate::Linear {
                        name: field,
                        crval: crval.clone(),
                        crpix: if crpix.len() == n {
                            crpix
                        } else {
                            vec![0.0; n]
                        },
                        cdelt: if cdelt.len() == n {
                            cdelt
                        } else {
                            vec![1.0; n]
                        },
                        pc: if pc.len() == n { pc } else { vec![1.0; n] },
                        pixel_axes: if pixel_axes.len() == n {
                            pixel_axes
                        } else {
                            (0..n).collect()
                        },
                        stokes,
                    });
                    let _ = other;
                }
            }
        }
        Ok(CoordinateSystem {
            coords,
            nimaxes,
            record: Some(rec.clone()),
        })
    }

    /// Build from a FITS primary-image header (astropy-style WCS cards).
    /// Direction axes (RA/Dec, Gal lon/lat) collapse into one direction
    /// coordinate placed first, as casacore's FITSCoordinateUtil does; the
    /// remaining axes become linear coordinates in FITS order.  Angles are
    /// converted to radians and CRPIX to 0-based, matching the record path.
    pub fn from_fits(hdr: &super::fits::FitsHeader, nimaxes: usize) -> CoordinateSystem {
        let ct = |n: usize| hdr.axis_str("CTYPE", n).unwrap_or("").to_string();
        let get = |prefix: &str, n: usize| hdr.axis_f64(prefix, n);
        // casacore pixel axis p = FITS axis p+1 (NAXIS1 is the fastest /
        // pixel axis 0); the pyrap/numpy order is the reverse.
        let fits_axis = |p: usize| p + 1;
        let is_direction = |c: &str| {
            c.starts_with("RA")
                || c.starts_with("DEC")
                || c.starts_with("GLON")
                || c.starts_with("GLAT")
        };
        let mut coords = Vec::new();
        let dir_axes: Vec<usize> = (0..nimaxes)
            .filter(|p| is_direction(&ct(fits_axis(*p))))
            .collect();
        if dir_axes.len() == 2 {
            let (lp, tp) = (dir_axes[0], dir_axes[1]); // long, lat pixel axes
            let (ln, tn) = (fits_axis(lp), fits_axis(tp)); // their FITS axes
            let rad = |v: f64| v.to_radians();
            // PCi_j cards (world axis i, pixel axis j, 1-based FITS axes).
            let pij = |i: usize, j: usize| {
                hdr.f64_of(&format!("PC{i}_{j}"))
                    .unwrap_or(if i == j { 1.0 } else { 0.0 })
            };
            coords.push(Coordinate::Direction {
                name: "direction0".into(),
                crval: [
                    rad(get("CRVAL", ln).unwrap_or(0.0)),
                    rad(get("CRVAL", tn).unwrap_or(0.0)),
                ],
                crpix: [
                    get("CRPIX", ln).unwrap_or(1.0) - 1.0,
                    get("CRPIX", tn).unwrap_or(1.0) - 1.0,
                ],
                cdelt: [
                    rad(get("CDELT", ln).unwrap_or(1.0)),
                    rad(get("CDELT", tn).unwrap_or(1.0)),
                ],
                pc: [[pij(ln, ln), pij(ln, tn)], [pij(tn, ln), pij(tn, tn)]],
                system: "ICRS".into(),
                projection: ct(ln).split("---").nth(1).unwrap_or("SIN").to_string(),
                units: ["rad".into(), "rad".into()],
                pixel_axes: [lp, tp],
            });
        }
        let mut lin_idx = 0usize;
        for p in 0..nimaxes {
            if dir_axes.contains(&p) {
                continue;
            }
            let n = fits_axis(p);
            let ctype = ct(n);
            if ctype.is_empty() {
                continue;
            }
            let name = if ctype.contains("STOKES") {
                "stokes1".to_string()
            } else if ["FREQ", "VRAD", "VELO", "WAVE", "AWAV"]
                .iter()
                .any(|k| ctype.contains(k))
            {
                "spectral2".to_string()
            } else {
                lin_idx += 1;
                format!("linear{lin_idx}")
            };
            coords.push(Coordinate::Linear {
                name,
                crval: vec![get("CRVAL", n).unwrap_or(0.0)],
                crpix: vec![get("CRPIX", n).unwrap_or(1.0) - 1.0],
                cdelt: vec![get("CDELT", n).unwrap_or(1.0)],
                pc: vec![1.0],
                pixel_axes: vec![p],
                stokes: Vec::new(),
            });
        }
        CoordinateSystem {
            coords,
            nimaxes,
            record: None,
        }
    }

    /// Pixel -> world, both in CASA axis order (axis 0 = fastest/FITS
    /// NAXIS1).  Every axis is covered exactly once by the coordinates.
    pub fn to_world(&self, pixel: &[f64]) -> Result<Vec<f64>, CoordError> {
        if pixel.len() != self.nimaxes {
            return Err(CoordError::AxisCount {
                got: pixel.len(),
                want: self.nimaxes,
            });
        }
        let mut world = vec![0.0; self.nimaxes];
        for c in &self.coords {
            match c {
                Coordinate::Direction {
                    crval,
                    crpix,
                    cdelt,
                    pc,
                    pixel_axes,
                    ..
                } => {
                    let dx = (pixel[pixel_axes[0]] - crpix[0]) * cdelt[0];
                    let dy = (pixel[pixel_axes[1]] - crpix[1]) * cdelt[1];
                    let ix = pc[0][0] * dx + pc[0][1] * dy;
                    let iy = pc[1][0] * dx + pc[1][1] * dy;
                    let (ra, dec) = sin_to_world(crval, ix, iy)?;
                    world[pixel_axes[0]] = ra;
                    world[pixel_axes[1]] = dec;
                }
                Coordinate::Linear {
                    crval,
                    crpix,
                    cdelt,
                    pc,
                    pixel_axes,
                    ..
                } => {
                    for (k, &pa) in pixel_axes.iter().enumerate() {
                        world[pa] = crval[k] + (pixel[pa] - crpix[k]) * cdelt[k] * pc[k];
                    }
                }
            }
        }
        Ok(world)
    }

    /// World -> pixel (CASA axis order); the inverse of [`Self::to_world`].
    pub fn to_pixel(&self, world: &[f64]) -> Result<Vec<f64>, CoordError> {
        if world.len() != self.nimaxes {
            return Err(CoordError::AxisCount {
                got: world.len(),
                want: self.nimaxes,
            });
        }
        let mut pixel = vec![0.0; self.nimaxes];
        for c in &self.coords {
            match c {
                Coordinate::Direction {
                    crval,
                    crpix,
                    cdelt,
                    pc,
                    pixel_axes,
                    ..
                } => {
                    let (ix, iy) =
                        sin_to_intermediate(crval, world[pixel_axes[0]], world[pixel_axes[1]])?;
                    // Invert the pc rotation, then the per-axis scale.
                    let det = pc[0][0] * pc[1][1] - pc[0][1] * pc[1][0];
                    if det.abs() < f64::MIN_POSITIVE {
                        return Err(CoordError::BadField {
                            field: "direction.pc".into(),
                            expected: "an invertible 2x2 matrix",
                        });
                    }
                    let dx = (pc[1][1] * ix - pc[0][1] * iy) / det;
                    let dy = (pc[0][0] * iy - pc[1][0] * ix) / det;
                    pixel[pixel_axes[0]] = dx / cdelt[0] + crpix[0];
                    pixel[pixel_axes[1]] = dy / cdelt[1] + crpix[1];
                }
                Coordinate::Linear {
                    crval,
                    crpix,
                    cdelt,
                    pc,
                    pixel_axes,
                    ..
                } => {
                    for (k, &pa) in pixel_axes.iter().enumerate() {
                        pixel[pa] = (world[pa] - crval[k]) / (cdelt[k] * pc[k]) + crpix[k];
                    }
                }
            }
        }
        Ok(pixel)
    }

    /// The direction coordinate's cdelt (radians, long, lat) — the field
    /// DDFacet reads as `dict()["direction0"]["cdelt"]`.
    pub fn direction_cdelt(&self) -> Option<[f64; 2]> {
        self.coords.iter().find_map(|c| match c {
            Coordinate::Direction { cdelt, .. } => Some(*cdelt),
            _ => None,
        })
    }

    /// The `dict()` pyrap exposes: the raw coords record when the image is
    /// a CASA table (plus the runtime `_image_axes`/`_axes_sizes` entries
    /// casacore adds to every coordinate), or a synthesised record for
    /// FITS-sourced systems.
    pub fn to_record(&self) -> TableRecord {
        let mut rec = self.raw_record();
        // casacore annotates each coordinate with the image axes it spans
        // (numpy/pyrap order, ascending) and their sizes.
        for c in &self.coords {
            let (name, paxes) = match c {
                Coordinate::Direction {
                    name, pixel_axes, ..
                } => (name.clone(), pixel_axes.to_vec()),
                Coordinate::Linear {
                    name, pixel_axes, ..
                } => (name.clone(), pixel_axes.clone()),
            };
            let image_axes: Vec<i64> = paxes
                .iter()
                .map(|&p| (self.nimaxes - 1 - p) as i64)
                .collect();
            let mut sorted = image_axes.clone();
            sorted.sort_unstable();
            if let Some(pos) = rec.desc.fields.iter().position(|f| f.name == name) {
                if let Some(RecordValue::Record(sub)) = rec.values.get_mut(pos) {
                    sub.set(
                        "_image_axes",
                        RecordValue::Array(ArrayValue {
                            shape: vec![sorted.len() as u32],
                            data: ArrayData::Int64(sorted.clone()),
                        }),
                    );
                }
            }
        }
        rec
    }

    /// The coords record as it is stored on disk (no runtime
    /// `_image_axes` annotations): the record the image was opened with,
    /// or a synthesised one for FITS-sourced systems.
    pub fn raw_record(&self) -> TableRecord {
        match &self.record {
            Some(rec) => rec.clone(),
            None => self.completed_record(),
        }
    }

    /// A synthesised record with everything casacore's coordinate restore
    /// paths read: the spectral wcs block and conversion frame measures,
    /// the per-coordinate world/pixel axis maps and replacements, and the
    /// ObsInfo records.  Shared by the default template and the FITS
    /// synthesis — a persisted coordinate system must open in casacore.
    fn completed_record(&self) -> TableRecord {
        let mut rec = self.synthesise_record();
        // The per-coordinate world/pixel axis mappings and replacements
        // CoordinateSystem::restore reads back after the coordinates, in
        // coordinate order (each coordinate's casa pixel axes).
        for (idx, c) in self.coords.iter().enumerate() {
            let paxes: Vec<i64> = match c {
                Coordinate::Direction { pixel_axes, .. } => {
                    pixel_axes.iter().map(|&p| p as i64).collect()
                }
                Coordinate::Linear { pixel_axes, .. } => {
                    pixel_axes.iter().map(|&p| p as i64).collect()
                }
            };
            let worldreplace = match c {
                Coordinate::Direction { crval, .. } => crval.to_vec(),
                Coordinate::Linear { crval, .. } => crval.clone(),
            };
            let n = paxes.len();
            rec.set(
                &format!("worldmap{idx}"),
                RecordValue::Array(ArrayValue {
                    shape: vec![n as u32],
                    data: ArrayData::Int64(paxes.clone()),
                }),
            );
            rec.set(
                &format!("worldreplace{idx}"),
                RecordValue::Array(ArrayValue {
                    shape: vec![n as u32],
                    data: ArrayData::Double(worldreplace),
                }),
            );
            rec.set(
                &format!("pixelmap{idx}"),
                RecordValue::Array(ArrayValue {
                    shape: vec![n as u32],
                    data: ArrayData::Int64(paxes),
                }),
            );
            rec.set(
                &format!("pixelreplace{idx}"),
                RecordValue::Array(ArrayValue {
                    shape: vec![n as u32],
                    data: ArrayData::Double(vec![0.0; n]),
                }),
            );
        }
        // ObsInfo.
        rec.set("telescope", RecordValue::String("UNKNOWN".into()));
        rec.set("observer", RecordValue::String("UNKNOWN".into()));
        let mut obsdate = TableRecord::default();
        obsdate.set("type", RecordValue::String("epoch".into()));
        obsdate.set("refer", RecordValue::String("UTC".into()));
        let mut om0 = TableRecord::default();
        om0.set("value", RecordValue::Double(0.0));
        om0.set("unit", RecordValue::String("d".into()));
        obsdate.set("m0", RecordValue::Record(om0));
        rec.set("obsdate", RecordValue::Record(obsdate));
        let mut pointingcenter = TableRecord::default();
        pointingcenter.set(
            "value",
            RecordValue::Array(ArrayValue {
                shape: vec![2],
                data: ArrayData::Double(vec![0.0, 0.0]),
            }),
        );
        pointingcenter.set("initial", RecordValue::Bool(true));
        rec.set("pointingcenter", RecordValue::Record(pointingcenter));
        let mut telescopeposition = TableRecord::default();
        telescopeposition.set("type", RecordValue::String("position".into()));
        telescopeposition.set("refer", RecordValue::String("ITRF".into()));
        for (name, value, unit) in [("m0", 0.0, "rad"), ("m1", 0.0, "rad"), ("m2", 0.0, "m")] {
            let mut m = TableRecord::default();
            m.set("value", RecordValue::Double(value));
            m.set("unit", RecordValue::String(unit.into()));
            telescopeposition.set(name, RecordValue::Record(m));
        }
        rec.set("telescopeposition", RecordValue::Record(telescopeposition));
        // StokesCoordinate::restore reads the axes name and pc matrix; a
        // FITS-sourced system has no letters (derive them from the crval
        // Stokes codes, 1=I 2=Q 3=U 4=V).
        let stokes_pos = rec
            .desc
            .fields
            .iter()
            .position(|f| f.name.starts_with("stokes"));
        if let Some(pos) = stokes_pos {
            if let Some(RecordValue::Record(stokes)) = rec.values.get_mut(pos) {
                if !stokes.desc.fields.iter().any(|f| f.name == "axes") {
                    let crval = stokes
                        .get("crval")
                        .and_then(f64s)
                        .unwrap_or_else(|| vec![1.0]);
                    let n = crval.len().max(1);
                    let letters: Vec<String> = crval
                        .iter()
                        .map(|&v| {
                            ["I", "Q", "U", "V"]
                                .get(v.round() as usize)
                                .unwrap_or(&"I")
                                .to_string()
                        })
                        .collect();
                    let letters = if letters.is_empty() {
                        vec!["I".to_string()]
                    } else {
                        letters
                    };
                    let _ = n;
                    stokes.set(
                        "axes",
                        RecordValue::Array(ArrayValue {
                            shape: vec![1],
                            data: ArrayData::String(vec!["Stokes".into()]),
                        }),
                    );
                    stokes.set(
                        "pc",
                        RecordValue::Array(ArrayValue {
                            shape: vec![1, 1],
                            data: ArrayData::Double(vec![1.0]),
                        }),
                    );
                    stokes.set(
                        "stokes",
                        RecordValue::Array(ArrayValue {
                            shape: vec![letters.len() as u32],
                            data: ArrayData::String(letters),
                        }),
                    );
                }
            }
        }
        // The spectral coordinate's linear wcs block and conversion frame
        // measures SpectralCoordinate::restore reads.
        let spectral_pos = rec
            .desc
            .fields
            .iter()
            .position(|f| f.name.starts_with("spectral"));
        if let Some(pos) = spectral_pos {
            if let Some(RecordValue::Record(spectral)) = rec.values.get_mut(pos) {
                if !spectral.desc.fields.iter().any(|f| f.name == "wcs") {
                    let crval = spectral
                        .get("crval")
                        .and_then(f64s)
                        .and_then(|v| v.first().copied())
                        .unwrap_or(1.415e9);
                    let crpix = spectral
                        .get("crpix")
                        .and_then(f64s)
                        .and_then(|v| v.first().copied())
                        .unwrap_or(0.0);
                    let cdelt = spectral
                        .get("cdelt")
                        .and_then(f64s)
                        .and_then(|v| v.first().copied())
                        .unwrap_or(1000.0);
                    let mut wcs = TableRecord::default();
                    wcs.set("crval", RecordValue::Double(crval));
                    wcs.set("crpix", RecordValue::Double(crpix));
                    wcs.set("cdelt", RecordValue::Double(cdelt));
                    wcs.set("pc", RecordValue::Double(1.0));
                    wcs.set(
                        "ctype",
                        RecordValue::String("FREQ\u{0}\u{0}\u{0}\u{0}\u{0}".into()),
                    );
                    spectral.set("wcs", RecordValue::Record(wcs));
                }
                // The scalar/enum fields SpectralCoordinate::restore
                // reads (set-if-absent: records parsed from casacore
                // already carry them).
                let set_absent = |sp: &mut TableRecord, name: &str, v: RecordValue| {
                    if !sp.desc.fields.iter().any(|f| f.name == name) {
                        sp.set(name, v);
                    }
                };
                set_absent(spectral, "version", RecordValue::Int(2));
                set_absent(spectral, "system", RecordValue::String("LSRK".into()));
                set_absent(spectral, "restfreq", RecordValue::Double(0.0));
                set_absent(
                    spectral,
                    "restfreqs",
                    RecordValue::Array(ArrayValue {
                        shape: vec![1],
                        data: ArrayData::Double(vec![0.0]),
                    }),
                );
                set_absent(spectral, "velType", RecordValue::Int(0));
                set_absent(spectral, "nativeType", RecordValue::Int(0));
                set_absent(spectral, "velUnit", RecordValue::String("km/s".into()));
                set_absent(spectral, "waveUnit", RecordValue::String("mm".into()));
                set_absent(spectral, "formatUnit", RecordValue::String(String::new()));
                set_absent(spectral, "unit", RecordValue::String("Hz".into()));
                set_absent(spectral, "name", RecordValue::String("Frequency".into()));
                if !spectral.desc.fields.iter().any(|f| f.name == "conversion") {
                    let mut conversion = TableRecord::default();
                    let mut direction = TableRecord::default();
                    direction.set("type", RecordValue::String("direction".into()));
                    direction.set("refer", RecordValue::String("J2000".into()));
                    let mut m1 = TableRecord::default();
                    m1.set("value", RecordValue::Double(std::f64::consts::FRAC_PI_2));
                    m1.set("unit", RecordValue::String("rad".into()));
                    direction.set("m1", RecordValue::Record(m1));
                    let mut m0 = TableRecord::default();
                    m0.set("value", RecordValue::Double(0.0));
                    m0.set("unit", RecordValue::String("rad".into()));
                    direction.set("m0", RecordValue::Record(m0));
                    conversion.set("direction", RecordValue::Record(direction));
                    let mut position = TableRecord::default();
                    position.set("type", RecordValue::String("position".into()));
                    position.set("refer", RecordValue::String("ITRF".into()));
                    for (name, value, unit) in
                        [("m2", 0.0, "m"), ("m1", 0.0, "rad"), ("m0", 0.0, "rad")]
                    {
                        let mut m = TableRecord::default();
                        m.set("value", RecordValue::Double(value));
                        m.set("unit", RecordValue::String(unit.into()));
                        position.set(name, RecordValue::Record(m));
                    }
                    conversion.set("position", RecordValue::Record(position));
                    let mut epoch = TableRecord::default();
                    epoch.set("type", RecordValue::String("epoch".into()));
                    epoch.set("refer", RecordValue::String("LAST".into()));
                    let mut em0 = TableRecord::default();
                    em0.set("value", RecordValue::Double(0.0));
                    em0.set("unit", RecordValue::String("d".into()));
                    epoch.set("m0", RecordValue::Record(em0));
                    conversion.set("epoch", RecordValue::Record(epoch));
                    conversion.set("system", RecordValue::String("LSRK".into()));
                    spectral.set("conversion", RecordValue::Record(conversion));
                }
            }
        }
        rec
    }

    fn synthesise_record(&self) -> TableRecord {
        let mut rec = TableRecord::default();
        for (idx, c) in self.coords.iter().enumerate() {
            match c {
                Coordinate::Direction {
                    crval,
                    crpix,
                    cdelt,
                    pc,
                    system,
                    projection,
                    units,
                    pixel_axes,
                    ..
                } => {
                    let mut d = TableRecord::default();
                    d.set("system", RecordValue::String(system.clone()));
                    d.set("projection", RecordValue::String(projection.clone()));
                    d.set(
                        "units",
                        RecordValue::Array(ArrayValue {
                            shape: vec![2],
                            data: ArrayData::String(units.to_vec()),
                        }),
                    );
                    d.set(
                        "projection_parameters",
                        RecordValue::Array(ArrayValue {
                            shape: vec![2],
                            data: ArrayData::Double(vec![0.0, 0.0]),
                        }),
                    );
                    // casacore's DirectionCoordinate::restore requires the
                    // axis names and the pole fields.
                    d.set(
                        "axes",
                        RecordValue::Array(ArrayValue {
                            shape: vec![2],
                            data: ArrayData::String(vec![
                                "Right Ascension".into(),
                                "Declination".into(),
                            ]),
                        }),
                    );
                    d.set("conversionSystem", RecordValue::String(system.clone()));
                    d.set("longpole", RecordValue::Double(180.0));
                    d.set("latpole", RecordValue::Double(0.0));
                    d.set("crval", arr2(*crval));
                    d.set("crpix", arr2(*crpix));
                    d.set("cdelt", arr2(*cdelt));
                    d.set(
                        "pc",
                        RecordValue::Array(ArrayValue {
                            shape: vec![2, 2],
                            data: ArrayData::Double(vec![pc[0][0], pc[0][1], pc[1][0], pc[1][1]]),
                        }),
                    );
                    rec.set("direction0", RecordValue::Record(d));
                    rec.set(
                        &format!("pixelmap{idx}"),
                        RecordValue::Array(ArrayValue {
                            shape: vec![2],
                            data: ArrayData::Int(vec![pixel_axes[0] as i32, pixel_axes[1] as i32]),
                        }),
                    );
                }
                Coordinate::Linear {
                    crval,
                    crpix,
                    cdelt,
                    pc,
                    pixel_axes,
                    name,
                    stokes,
                    ..
                } => {
                    let mut s = TableRecord::default();
                    if !stokes.is_empty() {
                        // StokesCoordinate::restore reads the axis name and
                        // the pc matrix.
                        s.set(
                            "axes",
                            RecordValue::Array(ArrayValue {
                                shape: vec![1],
                                data: ArrayData::String(vec!["Stokes".into()]),
                            }),
                        );
                        s.set(
                            "pc",
                            RecordValue::Array(ArrayValue {
                                shape: vec![1, 1],
                                data: ArrayData::Double(vec![pc.first().copied().unwrap_or(1.0)]),
                            }),
                        );
                    }
                    for (k, v) in [("crval", crval), ("crpix", crpix), ("cdelt", cdelt)] {
                        s.set(
                            k,
                            RecordValue::Array(ArrayValue {
                                shape: vec![v.len() as u32],
                                data: ArrayData::Double(v.clone()),
                            }),
                        );
                    }
                    if !stokes.is_empty() {
                        s.set(
                            "stokes",
                            RecordValue::Array(ArrayValue {
                                shape: vec![stokes.len() as u32],
                                data: ArrayData::String(stokes.clone()),
                            }),
                        );
                    }
                    rec.set(name, RecordValue::Record(s));
                    rec.set(
                        &format!("pixelmap{idx}"),
                        RecordValue::Array(ArrayValue {
                            shape: vec![pixel_axes.len() as u32],
                            data: ArrayData::Int(pixel_axes.iter().map(|&p| p as i32).collect()),
                        }),
                    );
                }
            }
        }
        rec
    }

    /// get_increment: per coordinate in CS order, its per-axis increment.
    pub fn increments(&self) -> Vec<Vec<f64>> {
        self.coords
            .iter()
            .map(|c| match c {
                Coordinate::Direction { cdelt, .. } => vec![cdelt[0], cdelt[1]],
                Coordinate::Linear { cdelt, .. } => cdelt.clone(),
            })
            .collect()
    }

    pub fn set_increments(&mut self, inc: &[Vec<f64>]) {
        for (c, v) in self.coords.iter_mut().zip(inc) {
            match c {
                Coordinate::Direction { cdelt, .. } => {
                    if v.len() == 2 {
                        *cdelt = [v[0], v[1]];
                    }
                }
                Coordinate::Linear { cdelt, .. } => {
                    if v.len() == cdelt.len() {
                        *cdelt = v.clone();
                    }
                }
            }
        }
    }

    pub fn reference_values(&self) -> Vec<Vec<f64>> {
        self.coords
            .iter()
            .map(|c| match c {
                Coordinate::Direction { crval, .. } => vec![crval[0], crval[1]],
                Coordinate::Linear { crval, .. } => crval.clone(),
            })
            .collect()
    }

    pub fn set_reference_values(&mut self, vals: &[Vec<f64>]) {
        for (c, v) in self.coords.iter_mut().zip(vals) {
            match c {
                Coordinate::Direction { crval, .. } => {
                    if v.len() == 2 {
                        *crval = [v[0], v[1]];
                    }
                }
                Coordinate::Linear { crval, .. } => {
                    if v.len() == crval.len() {
                        *crval = v.clone();
                    }
                }
            }
        }
    }

    pub fn reference_pixels(&self) -> Vec<Vec<f64>> {
        self.coords
            .iter()
            .map(|c| match c {
                Coordinate::Direction { crpix, .. } => vec![crpix[0], crpix[1]],
                Coordinate::Linear { crpix, .. } => crpix.clone(),
            })
            .collect()
    }

    pub fn set_reference_pixels(&mut self, vals: &[Vec<f64>]) {
        for (c, v) in self.coords.iter_mut().zip(vals) {
            match c {
                Coordinate::Direction { crpix, .. } => {
                    if v.len() == 2 {
                        *crpix = [v[0], v[1]];
                    }
                }
                Coordinate::Linear { crpix, .. } => {
                    if v.len() == crpix.len() {
                        *crpix = v.clone();
                    }
                }
            }
        }
    }
}

fn arr2(v: [f64; 2]) -> RecordValue {
    RecordValue::Array(ArrayValue {
        shape: vec![2],
        data: ArrayData::Double(v.to_vec()),
    })
}

/// Zenithal SIN (orthographic) deprojection: intermediate plane (ix, iy)
/// radians -> (ra, dec).  Verified against casacore to 3e-16 rad
/// (scripts/probe_pyrap_images.py).
fn sin_to_world(crval: &[f64; 2], ix: f64, iy: f64) -> Result<(f64, f64), CoordError> {
    let r2 = ix * ix + iy * iy;
    if r2 >= 1.0 {
        return Err(CoordError::OffProjection { value: r2.sqrt() });
    }
    let w = (1.0 - r2).sqrt(); // cos(theta), theta = angle from tangent point
    let (a0, d0) = (crval[0], crval[1]);
    let sin_d = d0.sin() * w + d0.cos() * iy;
    let cd_ca = d0.cos() * w - d0.sin() * iy;
    let dec = sin_d.clamp(-1.0, 1.0).asin();
    let ra = a0 + ix.atan2(cd_ca);
    Ok((ra, dec))
}

/// Projection: (ra, dec) -> intermediate plane (ix, iy); the inverse of
/// `sin_to_world`.
fn sin_to_intermediate(crval: &[f64; 2], ra: f64, dec: f64) -> Result<(f64, f64), CoordError> {
    let (a0, d0) = (crval[0], crval[1]);
    let (sd, cd) = (dec.sin(), dec.cos());
    let da = ra - a0;
    let w = sd * d0.sin() + cd * da.cos() * d0.cos();
    if w <= 0.0 {
        return Err(CoordError::OffProjection { value: dec });
    }
    let iy = sd * d0.cos() - cd * da.cos() * d0.sin();
    let ix = cd * da.sin();
    Ok((ix, iy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::fits::CardValue;

    /// Ground truth: the probe image's `coords` record values (recorded in
    /// tests/fixtures/manifest.json by scripts/probe_pyrap_images.py) and
    /// casacore's `image.toworld` answers for two pixels.
    fn probe_system() -> CoordinateSystem {
        let json = r#"{
            "telescope": "UNKNOWN",
            "observer": "UNKNOWN",
            "obsdate": {"type": "epoch", "refer": "LAST", "m0": {"value": 0.0, "unit": "d"}},
            "pointingcenter": {"value": [0.0, 0.0], "initial": true},
            "direction0": {
                "system": "ICRS", "projection": "SIN", "projection_parameters": [0.0, 0.0],
                "crval": [0.030543261909900768, -0.007853981633974483],
                "crpix": [4.0, 3.0],
                "cdelt": [-4.363323129985824e-07, 5.235987755982988e-07],
                "pc": [[1.0, 0.0], [0.0, 1.0]],
                "axes": ["Right Ascension", "Declination"],
                "units": ["rad", "rad"], "conversionSystem": "ICRS",
                "longpole": 180.0, "latpole": -0.45
            },
            "worldmap0": [0, 1], "worldreplace0": [0.0, 0.0],
            "pixelmap0": [0, 1], "pixelreplace0": [0.0, 0.0],
            "stokes1": {"axes": ["Stokes"], "stokes": ["I", "Q"], "crval": [1.0],
                        "crpix": [0.0], "cdelt": [1.0], "pc": [[1.0]]},
            "worldmap1": [2], "worldreplace1": [1.0], "pixelmap1": [2], "pixelreplace1": [1.0],
            "spectral2": {"version": 2, "system": "TOPO", "restfreq": 0.0, "restfreqs": [0.0],
                          "velType": 0, "nativeType": 0, "velUnit": "km/s", "waveUnit": "mm",
                          "formatUnit": "",
                          "wcs": {"crval": 1400000000.0, "crpix": 0.0, "cdelt": 2000000.0,
                                  "pc": 1.0, "ctype": "FREQ"},
                          "unit": "Hz", "name": "Frequency"},
            "worldmap2": [3], "worldreplace2": [1400000000.0],
            "pixelmap2": [3], "pixelreplace2": [0.0]
        }"#;
        let rec = crate::record::parse_json_record(json).unwrap();
        CoordinateSystem::from_record(&rec, 4).unwrap()
    }

    #[test]
    fn direction_toworld_matches_casacore() {
        let cs = probe_system();
        // casacore: image.toworld((0,0,0,0)) = (1.4e9, 1.0, -0.007855552430289408, 0.03054500729300624)
        // pixel tuple is pyrap order (ch, pol, y, x); casa order = reversed.
        let w = cs.to_world(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((w[0] - 0.03054500729300624).abs() < 1e-15, "ra {w:?}");
        assert!((w[1] + 0.007855552430289408).abs() < 1e-15, "dec {w:?}");
        assert!((w[2] - 1.0).abs() < 1e-15, "stokes {w:?}");
        assert!((w[3] - 1.4e9).abs() < 1e-6, "freq {w:?}");
        // image.toworld((1,1,2,3)) = (1402000000.0, 2.0, -0.007854505232749292, 0.030543698255673546)
        let w = cs.to_world(&[3.0, 2.0, 1.0, 1.0]).unwrap();
        assert!((w[0] - 0.030543698255673546).abs() < 1e-15);
        assert!((w[1] + 0.007854505232749292).abs() < 1e-15);
        assert!((w[2] - 2.0).abs() < 1e-15);
        assert!((w[3] - 1.402e9).abs() < 1e-6);
    }

    #[test]
    fn topixel_inverts_toworld() {
        let cs = probe_system();
        for px in [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 2.0, 1.0, 1.0],
            [9.0, 7.0, 0.0, 2.0],
        ] {
            let w = cs.to_world(&px).unwrap();
            let back = cs.to_pixel(&w).unwrap();
            for (b, p) in back.iter().zip(px) {
                assert!((b - p).abs() < 1e-9, "{back:?} vs {px:?}");
            }
        }
    }

    #[test]
    fn off_projection_is_an_error() {
        let cs = probe_system();
        // dec ~1.57 rad is beyond the SIN hemisphere about dec0 = -0.45 deg.
        let world = vec![0.03, 1.57, 1.0, 1.4e9];
        assert!(cs.to_pixel(&world).is_err());
        // A pixel far outside the unit disc likewise.
        assert!(cs.to_world(&[1e7, 1e7, 0.0, 0.0]).is_err());
    }

    #[test]
    fn fit_sourced_system_matches_the_record_one() {
        // The same image expressed as FITS cards must give the same
        // direction coordinate (radians, 0-based crpix).
        let mut hdr = super::super::fits::FitsHeader::default();
        for (k, v) in [
            ("NAXIS", 4.0),
            ("NAXIS1", 10.0),
            ("NAXIS2", 8.0),
            ("NAXIS3", 2.0),
            ("NAXIS4", 3.0),
            ("CRVAL1", 1.75),
            ("CRVAL2", -0.45),
            ("CRVAL3", 1.0),
            ("CRVAL4", 1.4e9),
            ("CDELT1", -2.5e-5),
            ("CDELT2", 3.0e-5),
            ("CDELT3", 1.0),
            ("CDELT4", 2.0e6),
            ("CRPIX1", 5.0),
            ("CRPIX2", 4.0),
            ("CRPIX3", 1.0),
            ("CRPIX4", 1.0),
        ] {
            hdr.cards.push((k.to_string(), Some(CardValue::Float(v))));
        }
        for (k, v) in [
            ("CTYPE1", "RA---SIN"),
            ("CTYPE2", "DEC--SIN"),
            ("CTYPE3", "STOKES"),
            ("CTYPE4", "FREQ"),
        ] {
            hdr.cards
                .push((k.to_string(), Some(CardValue::Str(v.into()))));
        }
        let fits_cs = CoordinateSystem::from_fits(&hdr, 4);
        let rec_cs = probe_system();
        // The direction pixel of pyrap (ch, pol, y, x) = (1, 1, 2, 3):
        // casa pixel (3, 2, 1, 1).
        let wf = fits_cs.to_world(&[3.0, 2.0, 1.0, 1.0]).unwrap();
        let wr = rec_cs.to_world(&[3.0, 2.0, 1.0, 1.0]).unwrap();
        for (a, b) in wf.iter().zip(&wr) {
            assert!((a - b).abs() < 1e-15, "fits {a} vs record {b}");
        }
    }
}
