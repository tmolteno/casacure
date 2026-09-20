//! Units and quantities: a port of casacore's `casa/Quanta` subsystem.
//!
//! Covers the unit system (`Unit`), `Quantity` (a value with a unit) with
//! arithmetic and SI conversion, and the time/angle value semantics
//! (MJD <-> Gregorian calendar, sexagesimal formatting) that casacure's
//! TaQL date/time and `hms`/`dms`/`hdms` functions are built on.
//!
//! This is a deliberately curated subset of casacore's `Quanta`: the SI base
//! dimensions plus the units common in radio astronomy (length, mass, time,
//! frequency, flux, angle, SI derived), SI prefixes, the scale-to-SI
//! machinery, `%.N g`-style value formatting, and MVTime/MVAngle-style
//! calendar and sexagesimal output. The full casacore UnitMap (specialist
//! symbols, offsets like Celsius, logarithmic units) is not included.

use std::fmt;

/// SI base dimensions in casacore's canonical display order.
const BASE_DIMS: [&str; 8] = ["m", "kg", "s", "A", "K", "mol", "cd", "rad"];
const N_DIMS: usize = 8;

/// Exponent vector over the SI base dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims(pub [i32; N_DIMS]);

impl Dims {
    fn zero() -> Dims {
        Dims([0; N_DIMS])
    }
}

#[allow(clippy::too_many_arguments)]
const fn dims(m: i32, kg: i32, s: i32, a: i32, k: i32, mol: i32, cd: i32, rad: i32) -> Dims {
    Dims([m, kg, s, a, k, mol, cd, rad])
}

/// A parsed unit: how it is displayed, its SI dimension vector, and its
/// scale factor to SI base units.
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    pub display: String,
    pub dims: Dims,
    pub scale: f64,
}

/// One entry in the base-unit table: name, dimensions, scale to SI base.
#[derive(Copy, Clone)]
struct UnitDef {
    name: &'static str,
    dims: Dims,
    scale: f64,
}

/// The curated unit table (`UnitMap` subset).
#[allow(clippy::excessive_precision)]
fn unit_table() -> &'static [UnitDef] {
    const DEFS: &[UnitDef] = &[
        // Dimensionless.
        UnitDef {
            name: "pct",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 0),
            scale: 0.01,
        },
        UnitDef {
            name: "%",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 0),
            scale: 0.01,
        },
        UnitDef {
            name: "ppm",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-6,
        },
        // SI base.
        UnitDef {
            name: "m",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "kg",
            dims: dims(0, 1, 0, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "s",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "A",
            dims: dims(0, 0, 0, 1, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "K",
            dims: dims(0, 0, 0, 0, 1, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "mol",
            dims: dims(0, 0, 0, 0, 0, 1, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "cd",
            dims: dims(0, 0, 0, 0, 0, 0, 1, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "rad",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 1),
            scale: 1.0,
        },
        // Length.
        UnitDef {
            name: "km",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e3,
        },
        UnitDef {
            name: "cm",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-2,
        },
        UnitDef {
            name: "mm",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-3,
        },
        UnitDef {
            name: "um",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-6,
        },
        UnitDef {
            name: "µm",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-6,
        },
        UnitDef {
            name: "nm",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-9,
        },
        UnitDef {
            name: "pm",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1e-12,
        },
        UnitDef {
            name: "inch",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 0.0254,
        },
        UnitDef {
            name: "ft",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 0.3048,
        },
        UnitDef {
            name: "mi",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1609.344,
        },
        UnitDef {
            name: "au",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 1.495_978_707e11,
        },
        UnitDef {
            name: "ly",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 9.460_730_472_580_8e15,
        },
        UnitDef {
            name: "pc",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 3.085_677_581_491_367_3e16,
        },
        UnitDef {
            name: "kpc",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 3.085_677_581_491_367_3e19,
        },
        UnitDef {
            name: "Mpc",
            dims: dims(1, 0, 0, 0, 0, 0, 0, 0),
            scale: 3.085_677_581_491_367_3e22,
        },
        // Mass.
        UnitDef {
            name: "g",
            dims: dims(0, 1, 0, 0, 0, 0, 0, 0),
            scale: 1e-3,
        },
        UnitDef {
            name: "mg",
            dims: dims(0, 1, 0, 0, 0, 0, 0, 0),
            scale: 1e-6,
        },
        UnitDef {
            name: "ug",
            dims: dims(0, 1, 0, 0, 0, 0, 0, 0),
            scale: 1e-9,
        },
        UnitDef {
            name: "µg",
            dims: dims(0, 1, 0, 0, 0, 0, 0, 0),
            scale: 1e-9,
        },
        UnitDef {
            name: "t",
            dims: dims(0, 1, 0, 0, 0, 0, 0, 0),
            scale: 1e3,
        },
        // Time.
        UnitDef {
            name: "ms",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 1e-3,
        },
        UnitDef {
            name: "us",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 1e-6,
        },
        UnitDef {
            name: "ns",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 1e-9,
        },
        UnitDef {
            name: "min",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 60.0,
        },
        UnitDef {
            name: "h",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 3600.0,
        },
        UnitDef {
            name: "hr",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 3600.0,
        },
        UnitDef {
            name: "d",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 86_400.0,
        },
        UnitDef {
            name: "day",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 86_400.0,
        },
        UnitDef {
            name: "yr",
            dims: dims(0, 0, 1, 0, 0, 0, 0, 0),
            scale: 3.155_76e7,
        },
        // Frequency.
        UnitDef {
            name: "Hz",
            dims: dims(0, 0, -1, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        // SI derived.
        UnitDef {
            name: "N",
            dims: dims(1, 1, -2, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "J",
            dims: dims(2, 1, -2, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "W",
            dims: dims(2, 1, -3, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "Pa",
            dims: dims(-1, 1, -2, 0, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "T",
            dims: dims(0, 1, -2, -1, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "V",
            dims: dims(2, 1, -3, -1, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "Wb",
            dims: dims(2, 1, -2, -1, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "C",
            dims: dims(0, 0, 1, 1, 0, 0, 0, 0),
            scale: 1.0,
        },
        UnitDef {
            name: "F",
            dims: dims(-2, -1, 4, 2, 0, 0, 0, 0),
            scale: 1.0,
        },
        // Flux (radio astronomy).
        UnitDef {
            name: "Jy",
            dims: dims(0, 1, -2, 0, 0, 0, 0, 0),
            scale: 1e-26,
        },
        // Angles.
        UnitDef {
            name: "deg",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 1),
            scale: std::f64::consts::PI / 180.0,
        },
        UnitDef {
            name: "arcmin",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 1),
            scale: std::f64::consts::PI / 10_800.0,
        },
        UnitDef {
            name: "arcsec",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 1),
            scale: std::f64::consts::PI / 648_000.0,
        },
        UnitDef {
            name: "mas",
            dims: dims(0, 0, 0, 0, 0, 0, 0, 1),
            scale: std::f64::consts::PI / 648_000_000.0,
        },
    ];
    DEFS
}

/// The SI prefixes (`Unit`'s `getPrefix` table): (symbol, long name, factor).
pub const PREFIXES: &[(&str, &str, f64)] = &[
    ("da", "deca", 1e1),
    ("Y", "yotta", 1e24),
    ("Z", "zetta", 1e21),
    ("E", "exa", 1e18),
    ("P", "peta", 1e15),
    ("T", "tera", 1e12),
    ("G", "giga", 1e9),
    ("M", "mega", 1e6),
    ("k", "kilo", 1e3),
    ("h", "hecto", 1e2),
    ("d", "deci", 1e-1),
    ("c", "centi", 1e-2),
    ("m", "milli", 1e-3),
    ("u", "micro", 1e-6),
    ("µ", "micro", 1e-6),
    ("n", "nano", 1e-9),
    ("p", "pico", 1e-12),
    ("f", "femto", 1e-15),
    ("a", "atto", 1e-18),
    ("z", "zepto", 1e-21),
    ("y", "yocto", 1e-24),
];

/// The long (English) name of a unit for the `units` metadata table
/// (defaults to the symbol).
pub fn unit_long_name(name: &str) -> &str {
    match name {
        "m" => "metre",
        "kg" => "kilogram",
        "s" => "second",
        "A" => "ampere",
        "K" => "kelvin",
        "mol" => "mole",
        "cd" => "candela",
        "rad" => "radian",
        "Hz" => "hertz",
        "deg" => "degree",
        "arcmin" => "arcmin",
        "arcsec" => "arcsecond",
        "mas" => "milliarcsecond",
        "Jy" => "jansky",
        "N" => "newton",
        "J" => "joule",
        "W" => "watt",
        "Pa" => "pascal",
        "T" => "tesla",
        "V" => "volt",
        "Wb" => "weber",
        "C" => "coulomb",
        "F" => "farad",
        "km" => "kilometer",
        "cm" => "centimeter",
        "mm" => "millimeter",
        "um" => "micrometer",
        "nm" => "nanometer",
        "mi" => "mile",
        "inch" => "inch",
        "ft" => "foot",
        "au" => "astronomical unit",
        "ly" => "light year",
        "pc" => "parsec",
        "g" => "gram",
        "t" => "tonne",
        "min" => "minute",
        "h" => "hour",
        "hr" => "hour",
        "d" => "day",
        "yr" => "year",
        "%" => "percent",
        _ => name,
    }
}

/// Look up a plain unit name (no prefix) in the table.
fn lookup(name: &str) -> Option<UnitDef> {
    unit_table().iter().find(|u| u.name == name).copied()
}

/// Parse a unit string (`"m"`, `"km.m/s"`, `"mJy"`, ...) into a `Unit`.
///
/// The grammar is the casacore `Unit` subset: identifiers (with optional
/// case-sensitive SI prefixes) combined by `.`/`*` (product) and `/`
/// (quotient), parenthesised groups, and integer exponents (`m2`, `s-2`,
/// `^(2)`).
pub fn parse_unit(s: &str) -> Result<Unit, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Unit {
            display: String::new(),
            dims: Dims::zero(),
            scale: 1.0,
        });
    }
    let mut p = Parser {
        chars: s.chars().collect(),
        pos: 0,
    };
    let t = p.parse_expr()?;
    if p.pos != p.chars.len() {
        return Err(format!("invalid unit string {s:?}"));
    }
    Ok(Unit {
        display: s.to_string(),
        dims: t.dims,
        scale: t.scale,
    })
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// A product/quotient of factors, stopping at `)` or end of input.
    fn parse_expr(&mut self) -> Result<Term, String> {
        let mut t = self.parse_factor()?;
        loop {
            match self.peek() {
                Some('*') | Some('.') => {
                    self.next();
                    let f = self.parse_factor()?;
                    t = Term {
                        scale: t.scale * f.scale,
                        dims: add_dims(t.dims, f.dims),
                    };
                }
                Some('/') => {
                    self.next();
                    let f = self.parse_factor()?;
                    t = Term {
                        scale: t.scale / f.scale,
                        dims: sub_dims(t.dims, f.dims),
                    };
                }
                _ => break,
            }
        }
        Ok(t)
    }

    /// A factor: an atom (`m`, `kHz`, `s-2`) or a parenthesised group with an
    /// optional trailing exponent.
    fn parse_factor(&mut self) -> Result<Term, String> {
        if self.peek() == Some('(') {
            self.next();
            let inner = self.parse_expr()?;
            if self.next() != Some(')') {
                return Err("unbalanced parentheses in unit".into());
            }
            let e = self.read_exponent()? as i32;
            return Ok(Term {
                scale: inner.scale.powi(e),
                dims: inner.dims.scaled(e),
            });
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Term, String> {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphabetic() || c == '%' {
                name.push(c);
                self.next();
            } else {
                break;
            }
        }
        if name.is_empty() {
            let c = self.peek().map_or_else(String::new, |c| c.to_string());
            return Err(format!("expected a unit name, found {c:?}"));
        }
        let e = self.read_exponent()? as i32;
        let (dims, scale) = resolve_name(&name)?;
        Ok(Term {
            scale: scale.powi(e),
            dims: dims.scaled(e),
        })
    }

    /// Optional trailing exponent: `m2`, `s-2`, `(km)^2` (default 1).
    fn read_exponent(&mut self) -> Result<i64, String> {
        if self.peek() == Some('^') {
            self.next();
        }
        let mut neg = false;
        if self.peek() == Some('-') {
            neg = true;
            self.next();
        }
        let mut n = 0i64;
        let mut any = false;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                any = true;
                n = n * 10 + i64::from(d);
                self.next();
            } else {
                break;
            }
        }
        Ok(if any {
            if neg {
                -n
            } else {
                n
            }
        } else {
            1
        })
    }
}

/// Resolve a possibly-prefixed unit name.
fn resolve_name(name: &str) -> Result<(Dims, f64), String> {
    if let Some(u) = lookup(name) {
        return Ok((u.dims, u.scale));
    }
    // Prefix + unit, longest prefix first.
    for &(pre, _long, f) in PREFIXES {
        if let Some(rest) = name.strip_prefix(pre) {
            if let Some(u) = lookup(rest) {
                return Ok((u.dims, u.scale * f));
            }
        }
    }
    Err(format!("unknown unit {name:?}"))
}

/// A parsed unit factor with accumulated dimensions and scale.
#[derive(Debug, Clone, Copy)]
struct Term {
    scale: f64,
    dims: Dims,
}

fn add_dims(a: Dims, b: Dims) -> Dims {
    let mut o = a;
    for (s, t) in o.0.iter_mut().zip(b.0.iter()) {
        *s += *t;
    }
    o
}

fn sub_dims(a: Dims, b: Dims) -> Dims {
    let mut o = a;
    for (s, t) in o.0.iter_mut().zip(b.0.iter()) {
        *s -= *t;
    }
    o
}

/// A value with an attached unit (casacore `Quantum<Double>`).
#[derive(Clone)]
pub struct Quantity {
    pub value: f64,
    pub unit: Unit,
}

impl Quantity {
    /// New quantity; `unit` is parsed (empty string = dimensionless).
    pub fn new(value: f64, unit: &str) -> Result<Quantity, String> {
        Ok(Quantity {
            value,
            unit: parse_unit(unit)?,
        })
    }

    /// The value converted into `unit` (an SI-dimension-conforming unit
    /// string). Errors when the dimensions differ.
    pub fn value_in(&self, unit: &str) -> Result<f64, String> {
        let u = parse_unit(unit)?;
        if self.unit.dims != u.dims {
            return Err(format!(
                "unit {} is not conforming to {}",
                u.display, self.unit.display
            ));
        }
        Ok(self.value * self.unit.scale / u.scale)
    }

    /// Whether `other` has the same dimensions.
    pub fn conforms(&self, other: &Quantity) -> bool {
        self.unit.dims == other.unit.dims
    }

    /// `self + other` (in self's unit; error unless same dimensions).
    pub fn add(&self, other: &Quantity) -> Result<Quantity, String> {
        if self.unit.dims != other.unit.dims {
            return Err("cannot add quantities of different dimensions".into());
        }
        let v = self.value + other.value * other.unit.scale / self.unit.scale;
        Ok(Quantity {
            value: v,
            unit: self.unit.clone(),
        })
    }

    pub fn sub(&self, other: &Quantity) -> Result<Quantity, String> {
        if self.unit.dims != other.unit.dims {
            return Err("cannot subtract quantities of different dimensions".into());
        }
        let v = self.value - other.value * other.unit.scale / self.unit.scale;
        Ok(Quantity {
            value: v,
            unit: self.unit.clone(),
        })
    }

    /// `self * other`: values multiply; the display unit is `a.b`.
    pub fn mul(&self, other: &Quantity) -> Quantity {
        let unit = multiplied_unit(&self.unit, &other.unit);
        Quantity {
            value: self.value * other.value,
            unit,
        }
    }

    /// `self / other`: display `a/(b)`.
    pub fn div(&self, other: &Quantity) -> Quantity {
        let unit = divided_unit(&self.unit, &other.unit);
        Quantity {
            value: self.value / other.value,
            unit,
        }
    }

    /// Integer power; display `(unit)^n` (casacore `pow`).
    pub fn pow(&self, n: i32) -> Quantity {
        let mut unit = self.unit.clone();
        unit.dims = unit.dims.scaled(n);
        unit.scale = unit.scale.powi(n);
        unit.display = format!("({}){}", self.unit.display, n);
        Quantity {
            value: self.value.powi(n),
            unit,
        }
    }

    /// `n`th root (casacore `root`); the display unit is rebuilt from the
    /// dimensions (SI base form, like `sqrt(9 m2) -> 3 m`).
    pub fn root(&self, n: i32) -> Quantity {
        let mut unit = self.unit.clone();
        for e in unit.dims.0.iter_mut() {
            *e = e.div_euclid(n);
        }
        unit.scale = unit.scale.powf(1.0 / n as f64);
        unit.display = unit.canonical_string();
        Quantity {
            value: self.value.powf(1.0 / n as f64),
            unit,
        }
    }

    pub fn sqrt(&self) -> Quantity {
        self.root(2)
    }

    /// The SI (canonical) value and unit string (`get()` / `canonical()`).
    pub fn canonical(&self) -> (f64, String) {
        (self.value * self.unit.scale, self.unit.canonical_string())
    }

    /// Seconds since the Unix epoch (1970-01-01T00:00:00Z) of an MJD
    /// quantity (a time-dimensioned value in days); `casacore
    /// Quantum::toUnixTime`.
    pub fn to_unix_time(&self) -> Result<f64, String> {
        let mjd = self.value_in("d")?;
        Ok((mjd - 40_587.0) * 86_400.0)
    }

    /// The angle represented as whole degrees, minutes-of-degree, and
    /// seconds-of-degree (rounded to whole arcseconds), signed.
    pub fn angle_hms(&self) -> Result<(i64, i64, i64, bool), String> {
        let deg = self.value_in("deg")?;
        let neg = deg < 0.0;
        let x = deg.abs();
        let d = x.floor() as i64;
        let m = ((x - d as f64) * 60.0).floor() as i64;
        let s = (x - d as f64) * 3600.0 - m as f64 * 60.0;
        let sec = s.round() as i64;
        Ok((d, m, sec, neg))
    }
}

fn multiplied_unit(a: &Unit, b: &Unit) -> Unit {
    let mut dims = a.dims;
    for (s, o) in dims.0.iter_mut().zip(b.dims.0.iter()) {
        *s += *o;
    }
    let display = match (a.display.as_str(), b.display.as_str()) {
        ("", x) => x.to_string(),
        (x, "") => x.to_string(),
        (x, y) => format!("{x}.{y}"),
    };
    Unit {
        display,
        dims,
        scale: a.scale * b.scale,
    }
}

fn divided_unit(a: &Unit, b: &Unit) -> Unit {
    let mut dims = a.dims;
    for (s, o) in dims.0.iter_mut().zip(b.dims.0.iter()) {
        *s -= *o;
    }
    let display = if b.display.is_empty() {
        a.display.clone()
    } else {
        format!("{}/({})", a.display, b.display)
    };
    Unit {
        display,
        dims,
        scale: a.scale / b.scale,
    }
}

impl Unit {
    /// The SI base unit string (`"m"`, `"kg.s-2"`, `"m2"`, ...).
    pub fn canonical_string(&self) -> String {
        let mut parts = Vec::new();
        for (i, name) in BASE_DIMS.iter().enumerate() {
            let e = self.dims.0[i];
            if e != 0 {
                if e == 1 {
                    parts.push((*name).to_string());
                } else {
                    parts.push(format!("{name}{e}"));
                }
            }
        }
        parts.join(".")
    }

    /// Whether this is exactly a time dimension (for `HH:MM:SS` display).
    pub fn is_time(&self) -> bool {
        self.dims == dims(0, 0, 1, 0, 0, 0, 0, 0)
    }

    /// Whether this is exactly an angle dimension (for sexagesimal display).
    pub fn is_angle(&self) -> bool {
        self.dims == dims(0, 0, 0, 0, 0, 0, 0, 1)
    }
}

impl Dims {
    /// Multiply every exponent by `e` (integer power).
    fn scaled(&self, e: i32) -> Dims {
        Dims([
            self.0[0] * e,
            self.0[1] * e,
            self.0[2] * e,
            self.0[3] * e,
            self.0[4] * e,
            self.0[5] * e,
            self.0[6] * e,
            self.0[7] * e,
        ])
    }
}

// ---------------------------------------------------------------------------
// Calendar / time-of-day / sexagesimal primitives (shared with taql.rs)
// ---------------------------------------------------------------------------

pub fn mjd_floor(mjd: f64) -> i64 {
    mjd.floor() as i64
}

/// casacore `MVTime::ymd`: (year, month, day) of an MJD (days since
/// 1858-11-17). Verbatim translation of `MVTime.cc`.
pub fn mjd_ymd(mjd: f64) -> (i64, i64, i64) {
    let z = mjd_floor(mjd) + 2_400_001;
    let mut dd = z;
    if z >= 2_299_161 {
        let al = (((z as f64 - 1_867_216.25) / 36_524.25).floor()) as i64;
        dd = z + 1 + al - al / 4;
    }
    dd += 1524;
    let yyyy = ((dd as f64 - 122.1) / 365.25).floor() as i64;
    let d0 = (365.25 * yyyy as f64).floor() as i64;
    let tmp = ((dd as f64 - d0 as f64) / 30.6001).floor() as i64;
    let day = dd - d0 - (30.6001 * tmp as f64).floor() as i64;
    let mm = if tmp < 14 { tmp - 1 } else { tmp - 13 };
    let yyyy = if mm > 2 { yyyy - 4715 - 1 } else { yyyy - 4715 };
    (yyyy, mm, day)
}

/// casacore `MVTime::weekday`: 1=Monday .. 7=Sunday.
pub fn mjd_weekday(mjd: f64) -> i64 {
    (((mjd_floor(mjd) + 2) % 7 + 7) % 7) + 1
}

/// casacore `MVTime::yearday` (1..366).
pub fn mjd_yearday(mjd: f64) -> i64 {
    let (y, m, d) = mjd_ymd(mjd);
    let c = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
        (m + 9) / 12
    } else {
        2 * ((m + 9) / 12)
    };
    (275 * m) / 9 - c + d - 30
}

/// casacore `MVTime::yearweek` (ISO-style week of year; can be 0).
pub fn mjd_yearweek(mjd: f64) -> i64 {
    let mut yd = mjd_yearday(mjd) - 4;
    let yw = (yd + 7) / 7;
    yd %= 7;
    if yd >= 0 {
        if yd >= mjd_weekday(mjd) {
            yw + 1
        } else {
            yw
        }
    } else if yd + 7 >= mjd_weekday(mjd) {
        yw + 1
    } else {
        yw
    }
}

/// Days since 1970-01-01 of a (proleptic Gregorian) calendar date.
pub fn civil_days(year: i64, month: i64, day: i64) -> i64 {
    let mut y = year;
    if month <= 2 {
        y -= 1;
    }
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// (hours, minutes, seconds-of-day) of the fractional day of an MJD.
pub fn mjd_hms(mjd: f64) -> (i64, i64, f64) {
    let total = mjd.fract().rem_euclid(1.0) * 86_400.0;
    let h = total.div_euclid(3600.0) as i64;
    let mi = ((total - h as f64 * 3600.0) / 60.0).floor() as i64;
    let sec = total - h as f64 * 3600.0 - mi as f64 * 60.0;
    (h, mi, sec)
}

/// Time-of-day of an MJD as `HH:MM:SS` (integer seconds, casacore's
/// `MVTime::string` `YMD`-less form).
pub fn format_time(mjd: f64) -> String {
    let (h, mi, s) = mjd_hms(mjd);
    format!("{h:02}:{mi:02}:{:02}", s.round() as i64)
}

/// Angle in radians as `+DDD.MM.SS` (whole arcseconds, casacore's
/// `MVAngle::string` `DM` form used by `Quantum::print` for angle units).
pub fn format_angle(rad: f64) -> String {
    let deg = rad * 180.0 / std::f64::consts::PI;
    let neg = deg < 0.0;
    let x = deg.abs();
    let d = x.floor() as i64;
    let m = ((x - d as f64) * 60.0).floor() as i64;
    let sec = ((x - d as f64) * 3600.0 - m as f64 * 60.0).round() as i64;
    let sign = if neg { "-" } else { "+" };
    format!("{sign}{d:03}.{m:02}.{sec:02}")
}

/// Generic sexagesimal formatter (casacore `MVAngle::string`): degrees-/
/// hours-units with fixed-width minutes and millisecond-precision seconds —
/// `06h00m00.000`, `+090d00m00.000` (the TaQL `hms`/`dms`/`hdms` output).
pub fn sexa_str(
    v: f64,
    s1: &str,
    s2: &str,
    w1: usize,
    w2: usize,
    signed: bool,
    neg: bool,
) -> String {
    let d = v.floor() as i64;
    let m = ((v - d as f64) * 60.0).floor() as i64;
    let sec = (v - d as f64) * 3600.0 - m as f64 * 60.0;
    if signed {
        let sign = if neg { "-" } else { "+" };
        format!("{sign}{d:0w1$}{s1}{m:0w2$}{s2}{sec:06.3}")
    } else {
        format!("{d:0w1$}{s1}{m:0w2$}{s2}{sec:06.3}")
    }
}

/// C-style `%.<prec>g` number formatting: `prec` significant digits,
/// fixed notation when the magnitude's exponent is in `[-4, prec)`,
/// otherwise scientific with a sign-padded two-digit exponent
/// (`"1.5e-26"`, `"1e+05"`, `"0.006"`, `"166.67"`).
pub fn format_g(v: f64, prec: usize) -> String {
    if v == 0.0 || prec == 0 {
        return "0".to_string();
    }
    let neg = v < 0.0;
    let x = v.abs();
    let mut exp = x.log10().floor() as i64;
    // Round the mantissa to `prec` significant digits; carry may bump the
    // exponent by one (999.9 -> 1000).
    let scale_m = 10f64.powi(prec as i32 - 1);
    let mut mant = (x / 10f64.powi(exp as i32) * scale_m).round() / scale_m;
    if mant >= 10.0 {
        mant /= 10.0;
        exp += 1;
    }
    let sign_prefix = if neg { "-" } else { "" };
    if exp >= -(prec as i64) && exp < prec as i64 {
        // Fixed notation: value = mant * 10^exp, trimmed to no trailing
        // zeros or dot.
        let v = mant * 10f64.powi(exp as i32);
        let decimals = (prec as i64 - 1 - exp).max(0) as usize;
        let mut s = format!("{v:.decimals$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        format!("{sign_prefix}{s}")
    } else {
        // Scientific: mantissa with trailing zeros trimmed.
        let dec = prec.saturating_sub(1).min(14);
        let mut m = format!("{mant:.dec$}");
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        let esign = if exp < 0 { "-" } else { "+" };
        format!("{sign_prefix}{m}e{esign}{:02}", exp.abs())
    }
}

impl fmt::Display for Quantity {
    /// `str(q)`: always the plain `%.5g value` + unit form (casacore's
    /// default `str` precision); angle/time units print plainly too.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unit.display.is_empty() {
            write!(f, "{} ", format_g(self.value, 5))
        } else {
            write!(f, "{} {}", format_g(self.value, 5), self.unit.display)
        }
    }
}

/// `repr(q)`: sexagesimal for angle (DMS) / time (HH:MM:SS) units, else the
/// 6-significant-digit value + unit (casacure's `Quantum::print`).
impl std::fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unit.is_time() {
            let dayfrac = self.value_in("d").unwrap_or(self.value);
            write!(f, "{}", format_time(dayfrac))
        } else if self.unit.is_angle() {
            let rad = self.value_in("rad").unwrap_or(self.value);
            write!(f, "{}", format_angle(rad))
        } else if self.unit.display.is_empty() {
            write!(f, "{} ", format_g(self.value, 6))
        } else {
            write!(f, "{} {}", format_g(self.value, 6), self.unit.display)
        }
    }
}

/// Parse a leading numeric prefix of a quantity string (``"1.5 Jy"` /
/// `"1.5Jy"` / `"2deg"`) into (value, unit-rest).
pub fn split_quantity_string(s: &str) -> Option<(f64, &str)> {
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        i += 1;
    }
    let start = i;
    while i < bytes.len()
        && (bytes[i].is_ascii_digit()
            || bytes[i] == b'.'
            || bytes[i] == b'e'
            || bytes[i] == b'E'
            || bytes[i] == b'x')
    {
        i += 1;
    }
    if i == start {
        return None;
    }
    let num = s[..i].parse::<f64>().ok()?;
    let rest = s[i..].trim();
    Some((num, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: f64, u: &str) -> Quantity {
        Quantity::new(v, u).unwrap()
    }

    #[test]
    fn parse_dimensions_and_scale() {
        let m = parse_unit("km").unwrap();
        assert_eq!(m.dims, dims(1, 0, 0, 0, 0, 0, 0, 0));
        assert!((m.scale - 1e3).abs() < 1e-9);
        let ms = parse_unit("mJy").unwrap();
        assert_eq!(ms.dims, dims(0, 1, -2, 0, 0, 0, 0, 0));
        assert!((ms.scale - 1e-29).abs() < 1e-40);
        let compound = parse_unit("kg.m.s-2").unwrap();
        assert_eq!(compound.dims, dims(1, 1, -2, 0, 0, 0, 0, 0));
        assert!((compound.scale - 1.0).abs() < 1e-12);
        let wm2 = parse_unit("W/m2").unwrap();
        assert_eq!(wm2.dims, dims(0, 1, -3, 0, 0, 0, 0, 0));
        assert!((wm2.scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn conversion_matches_casacore() {
        let q = q(1.5, "Jy");
        assert!((q.value_in("mJy").unwrap() - 1500.0).abs() < 1e-9);
        assert!(q.value_in("Jy").unwrap() == 1.5);
        assert!(q.value_in("W").is_err());
        let (val, unit) = q.canonical();
        assert!((val - 1.5e-26).abs() < 1e-34);
        assert_eq!(unit, "kg.s-2");
    }

    #[test]
    fn arithmetic_display_matches_casacore() {
        let a = q(3.0, "km");
        let b = q(500.0, "m");
        assert_eq!(a.add(&b).unwrap().to_string(), "3.5 km");
        assert_eq!(a.mul(&b).to_string(), "1500 km.m");
        assert_eq!(a.div(&b).to_string(), "0.006 km/(m)");
        assert_eq!(b.div(&a).to_string(), "166.67 m/(km)");
        assert_eq!(a.pow(2).to_string(), "9 (km)2");
        assert_eq!(q(9.0, "m2").sqrt().to_string(), "3 m");
        assert_eq!(q(8.0, "m3").root(3).to_string(), "2 m");
        assert_eq!(q(100000.0, "m").to_string(), "1e+05 m");
    }

    #[test]
    fn time_and_angle_display_matches_casacore() {
        // str() is always plain; repr() uses sexagesimal for angle/time.
        assert_eq!(q(2.0, "h").to_string(), "2 h");
        assert_eq!(q(45.0, "deg").to_string(), "45 deg");
        assert_eq!(q(1.5, "h").to_string(), "1.5 h");
        assert_eq!(format!("{:?}", q(2.0, "h")), "02:00:00");
        assert_eq!(format!("{:?}", q(1.5, "h")), "01:30:00");
        assert_eq!(format!("{:?}", q(51544.0, "d")), "00:00:00");
        assert_eq!(format!("{:?}", q(45.0, "deg")), "+045.00.00");
        assert_eq!(format!("{:?}", q(2.0, "rad")), "+114.35.30");
    }

    #[test]
    fn unix_time_matches_casacore() {
        assert!((q(51544.0, "d").to_unix_time().unwrap() - 946_684_800.0).abs() < 1.0);
    }

    #[test]
    fn calendar_roundtrip_anchors() {
        // 2000-01-01 == MJD 51544.
        assert_eq!(mjd_ymd(51544.0), (2000, 1, 1));
        assert_eq!(mjd_weekday(51544.0), 6); // Saturday
        assert_eq!(civil_days(2000, 1, 1), 10_957);
        // 1858-11-17 == MJD 0.
        assert_eq!(mjd_ymd(0.0), (1858, 11, 17));
    }

    #[test]
    fn split_string_forms() {
        assert_eq!(split_quantity_string("1.5 Jy"), Some((1.5, "Jy")));
        assert_eq!(split_quantity_string("1.5Jy"), Some((1.5, "Jy")));
        assert_eq!(split_quantity_string("2 deg"), Some((2.0, "deg")));
        assert_eq!(split_quantity_string("-2.5 Jy"), Some((-2.5, "Jy")));
    }
}
