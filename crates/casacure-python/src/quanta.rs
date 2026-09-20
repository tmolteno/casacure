//! The `casacure.quanta` module: a drop-in for the python-casacore
//! `casacore.quanta` surface (Quantity + units + arithmetic + the value
//! classes), backed by `crates/casacore::quanta`.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use ::casacure::quanta::{format_g, parse_unit, Quantity as CoreQuantity};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A value with an attached unit (`casacore.quanta.Quantity`).
#[pyclass(name = "Quantity")]
#[derive(Clone)]
pub struct Quantity {
    q: CoreQuantity,
}

impl Quantity {
    fn new(core: CoreQuantity) -> Quantity {
        Quantity { q: core }
    }
}

fn get_core(obj: &Bound<'_, PyAny>) -> PyResult<CoreQuantity> {
    let r: PyRef<'_, Quantity> = obj.extract()?;
    Ok(r.q.clone())
}

/// Build a `Quantity` from a value and unit, or from a combined string
/// (`quantity("1.5Jy")`).
#[pyfunction]
#[pyo3(signature = (value, unit = None))]
fn quantity(value: &Bound<'_, PyAny>, unit: Option<&Bound<'_, PyAny>>) -> PyResult<Quantity> {
    // Combined-string form: `quantity("1.5 Jy")`.
    if unit.is_none() {
        if let Ok(s) = value.extract::<String>() {
            let (v, u) = ::casacure::quanta::split_quantity_string(&s)
                .ok_or_else(|| PyValueError::new_err(format!("invalid quantity string {s:?}")))?;
            return CoreQuantity::new(v, u).map(Quantity::new).map_err(err);
        }
    }
    let v = value
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err("quantity: value must be a number or string"))?;
    let u: String = match unit {
        None => return Err(PyTypeError::new_err("quantity: a unit is required")),
        Some(u) if u.is_none() => String::new(),
        Some(u) => u
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("quantity: unit must be a string"))?,
    };
    CoreQuantity::new(v, &u).map(Quantity::new).map_err(err)
}

fn quantity_from(v: f64, u: &str) -> PyResult<Quantity> {
    CoreQuantity::new(v, u).map(Quantity::new).map_err(err)
}

#[pymethods]
impl Quantity {
    /// `get_value(unit=None)` — the value in its own unit, or converted.
    #[pyo3(signature = (target = None))]
    fn get_value(&self, target: Option<Bound<'_, PyAny>>) -> PyResult<f64> {
        let core = &self.q;
        let Some(t) = target else {
            return Ok(core.value);
        };
        if t.is_none() {
            return Ok(core.value);
        }
        if let Ok(u) = t.extract::<String>() {
            return core.value_in(&u).map_err(err);
        }
        let other = get_core(&t)?;
        if core.unit.dims != other.unit.dims {
            return Err(err("non-conforming unit type"));
        }
        Ok(core.value * core.unit.scale / other.unit.scale)
    }

    /// `get_unit()` — the unit string.
    fn get_unit(&self) -> String {
        self.q.unit.display.clone()
    }

    /// `set_value(other)` — replace the value (converting `other` into this
    /// quantity's unit).
    fn set_value(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = get_core(other)?;
        let core = &mut self.q;
        if core.unit.dims != other.unit.dims {
            return Err(err("Quantum::assure non-conforming unit type"));
        }
        core.value = other.value * other.unit.scale / core.unit.scale;
        Ok(())
    }

    /// `to_string()` — `"<value> <unit>"` (plain form, no sexagesimal).
    #[allow(clippy::inherent_to_string, clippy::should_implement_trait)]
    fn to_string(&self) -> String {
        let core = &self.q;
        if core.unit.display.is_empty() {
            format!("{} ", format_g(core.value, 6))
        } else {
            format!("{} {}", format_g(core.value, 6), core.unit.display)
        }
    }

    /// `formatted()` — the display form (sexagesimal for angle/time units).
    fn formatted(&self) -> String {
        if self.q.unit.is_time() || self.q.unit.is_angle() {
            format!("{:?}", self.q)
        } else {
            format!("{}", self.q)
        }
    }

    /// `canonical()` / `get()` — the SI value and unit string.
    fn canonical(&self) -> String {
        let (v, u) = self.q.canonical();
        if u.is_empty() {
            format!("{} ", format_g(v, 5))
        } else {
            format!("{} {}", format_g(v, 5), u)
        }
    }

    fn get(&self) -> String {
        self.canonical()
    }

    /// `conforms(other)` — same dimensions.
    fn conforms(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = get_core(other)?;
        Ok(self.q.unit.dims == other.unit.dims)
    }

    /// `convert(other)` — convert this quantity in place to `other`'s unit;
    /// returns `None` (casacore's in-place `Quantum::convert`).
    fn convert(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = get_core(other)?;
        let core = &mut self.q;
        if core.unit.dims != other.unit.dims {
            return Err(err("Quantum::assure non-conforming unit type"));
        }
        let u = parse_unit(&other.unit.display).map_err(err)?;
        core.value = core.value * core.unit.scale / other.unit.scale;
        core.unit = u;
        Ok(())
    }

    /// `to_dict()` — `{"value": ..., "unit": ...}`.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("value", self.q.value)?;
        d.set_item("unit", self.q.unit.display.clone())?;
        Ok(d)
    }

    /// `to_unix_time()` — seconds since the Unix epoch (MJD quantity).
    fn to_unix_time(&self) -> PyResult<f64> {
        self.q.to_unix_time().map_err(err)
    }

    /// `to_angle()` — the angle as a `+DDD.MM.SS` string.
    fn to_angle(&self) -> PyResult<String> {
        let rad = self.q.value_in("rad").map_err(err)?;
        Ok(::casacure::quanta::format_angle(rad))
    }

    /// `to_time()` — the time-of-day as an `HH:MM:SS` string.
    fn to_time(&self) -> PyResult<String> {
        let d = self.q.value_in("d").map_err(err)?;
        Ok(::casacure::quanta::format_time(d))
    }

    /// `norm()` — the absolute value (for a real quantity).
    fn norm(&self) -> f64 {
        self.q.value.abs()
    }

    fn __str__(&self) -> String {
        format!("{}", self.q)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.q)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = get_core(other)?;
        if self.q.unit.dims != other.unit.dims {
            return Ok(false);
        }
        let a = self.q.value * self.q.unit.scale;
        let b = other.value * other.unit.scale;
        Ok((a - b).abs() <= 1e-12 * a.abs().max(1.0))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Quantity> {
        let other = get_core(other)?;
        self.q.add(&other).map(Quantity::new).map_err(err)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Quantity> {
        let other = get_core(other)?;
        self.q.sub(&other).map(Quantity::new).map_err(err)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Quantity> {
        let other = get_core(other)?;
        Ok(Quantity::new(self.q.mul(&other)))
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Quantity> {
        let other = get_core(other)?;
        Ok(Quantity::new(self.q.div(&other)))
    }

    fn __neg__(&self) -> Quantity {
        let mut q = self.q.clone();
        q.value = -q.value;
        Quantity::new(q)
    }
}

/// `is_quantity(x)`.
#[pyfunction]
fn is_quantity(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<Quantity>()
}

/// `near(q1, q2, prec=None)` — relative closeness (default tolerance
/// 1e-13, matching casacore `Quantum::near`).
#[pyfunction]
#[pyo3(signature = (a, b, prec = None))]
fn near(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, prec: Option<f64>) -> PyResult<bool> {
    let a = get_core(a)?;
    let b = get_core(b)?;
    if a.unit.dims != b.unit.dims {
        return Ok(false);
    }
    let tol = prec.unwrap_or(1e-13);
    let av = a.value * a.unit.scale;
    let bv = b.value * b.unit.scale;
    Ok((av - bv).abs() <= tol * av.abs().max(bv.abs()).max(1e-300))
}

/// `nearabs(q1, q2, prec)` — absolute closeness.
#[pyfunction]
fn nearabs(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, prec: f64) -> PyResult<bool> {
    let a = get_core(a)?;
    let b = get_core(b)?;
    let av = a.value * a.unit.scale;
    let bv = b.value * b.unit.scale;
    Ok((av - bv).abs() <= prec)
}

/// `from_dict({"value": v, "unit": u})`.
#[pyfunction]
fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Quantity> {
    let value = d
        .get_item("value")?
        .ok_or_else(|| PyValueError::new_err("from_dict: missing 'value'"))?
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err("from_dict: 'value' must be a number"))?;
    let unit = d
        .get_item("unit")?
        .ok_or_else(|| PyValueError::new_err("from_dict: missing 'unit'"))?
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err("from_dict: 'unit' must be a string"))?;
    quantity_from(value, &unit)
}

/// `pow(q, n)` — integer power.
#[pyfunction]
fn pow(q: &Bound<'_, PyAny>, n: i64) -> PyResult<Quantity> {
    let q = get_core(q)?;
    Ok(Quantity::new(q.pow(n as i32)))
}

/// `root(q, n)` — integer root.
#[pyfunction]
fn root(q: &Bound<'_, PyAny>, n: i64) -> PyResult<Quantity> {
    if n == 0 {
        return Err(PyValueError::new_err("root: zero index"));
    }
    let q = get_core(q)?;
    Ok(Quantity::new(q.root(n as i32)))
}

/// `sqrt(q)`.
#[pyfunction]
fn sqrt(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let q = get_core(q)?;
    Ok(Quantity::new(q.sqrt()))
}

/// `abs(q)`.
#[pyfunction]
#[pyo3(name = "abs")]
fn qabs(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let q = get_core(q)?;
    let mut qq = q.clone();
    qq.value = qq.value.abs();
    Ok(Quantity::new(qq))
}

/// `ceil(q)` — ceiling of the value (same unit).
#[pyfunction]
#[pyo3(name = "ceil")]
fn qceil(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let c = get_core(q)?;
    Ok(Quantity::new(
        CoreQuantity::new(c.value.ceil(), &c.unit.display).map_err(err)?,
    ))
}

/// `floor(q)` — floor of the value (same unit).
#[pyfunction]
#[pyo3(name = "floor")]
fn qfloor(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let c = get_core(q)?;
    Ok(Quantity::new(
        CoreQuantity::new(c.value.floor(), &c.unit.display).map_err(err)?,
    ))
}

/// Require a dimensionless quantity and return its value.
fn require_dimensionless(q: &CoreQuantity, what: &str) -> PyResult<f64> {
    if !q.unit.dims.0.iter().any(|&e| e != 0) {
        return Ok(q.value);
    }
    Err(err(format!(
        "Quantum::{what} illegal unit type '{}'",
        q.unit.display
    )))
}

/// Trig on an angle/dimensionless quantity; the result is dimensionless.
fn trig_impl(q: &CoreQuantity, what: &str, f: impl Fn(f64) -> f64) -> PyResult<Quantity> {
    let angle = if q.unit.dims.0[7] != 0 {
        q.value_in("rad").map_err(err)?
    } else {
        require_dimensionless(q, what)?
    };
    Ok(Quantity::new(CoreQuantity::new(f(angle), "").map_err(err)?))
}

#[pyfunction]
#[pyo3(name = "sin")]
fn qsin(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    trig_impl(&get_core(q)?, "sin", f64::sin)
}

#[pyfunction]
#[pyo3(name = "cos")]
fn qcos(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    trig_impl(&get_core(q)?, "cos", f64::cos)
}

#[pyfunction]
#[pyo3(name = "tan")]
fn qtan(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    trig_impl(&get_core(q)?, "tan", f64::tan)
}

#[pyfunction]
#[pyo3(name = "asin")]
fn qasin(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    trig_impl(&get_core(q)?, "asin", f64::asin)
}

#[pyfunction]
#[pyo3(name = "acos")]
fn qacos(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    trig_impl(&get_core(q)?, "acos", f64::acos)
}

#[pyfunction]
#[pyo3(name = "atan")]
fn qatan(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    trig_impl(&get_core(q)?, "atan", f64::atan)
}

#[pyfunction]
#[pyo3(name = "atan2")]
fn qatan2(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let a = get_core(a)?;
    let b = get_core(b)?;
    if a.unit.dims != b.unit.dims {
        return Err(err("atan2: quantities must conform"));
    }
    let av = if a.unit.dims.0[7] != 0 {
        a.value_in("rad").map_err(err)?
    } else {
        require_dimensionless(&a, "atan2")?
    };
    let bv = if b.unit.dims.0[7] != 0 {
        b.value_in("rad").map_err(err)?
    } else {
        require_dimensionless(&b, "atan2")?
    };
    Ok(Quantity::new(
        CoreQuantity::new(av.atan2(bv), "").map_err(err)?,
    ))
}

/// `exp(q)` — dimensionless only.
#[pyfunction]
#[pyo3(name = "exp")]
fn qexp(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let q = get_core(q)?;
    let v = require_dimensionless(&q, "exp")?;
    Ok(Quantity::new(CoreQuantity::new(v.exp(), "").map_err(err)?))
}

/// `log(q)` — dimensionless only.
#[pyfunction]
#[pyo3(name = "log")]
fn qlog(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let q = get_core(q)?;
    let v = require_dimensionless(&q, "log")?;
    Ok(Quantity::new(CoreQuantity::new(v.ln(), "").map_err(err)?))
}

/// `log10(q)` — dimensionless only.
#[pyfunction]
#[pyo3(name = "log10")]
fn qlog10(q: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    let q = get_core(q)?;
    let v = require_dimensionless(&q, "log10")?;
    Ok(Quantity::new(
        CoreQuantity::new(v.log10(), "").map_err(err)?,
    ))
}

/// The common physical constants, as dimensions-quantities.
fn constants_table() -> Vec<(&'static str, f64, &'static str)> {
    vec![
        ("pi", std::f64::consts::PI, ""),
        ("ee", std::f64::consts::E, ""),
        ("c", 2.997_924_58e8, "m/s"),
        ("G", 6.674_08e-11, "m3/(kg.s2)"),
        ("h", 6.626_070_15e-34, "J.s"),
        ("k", 1.380_649e-23, "J/K"),
        ("NA", 6.022_140_76e23, "mol-1"),
        ("R", 8.314_462_618e0, "J/(mol.K)"),
        ("e", 1.602_176_634e-19, "C"),
        ("me", 9.109_383_701_5e-31, "kg"),
        ("mp", 1.672_621_923_69e-27, "kg"),
        ("mu0", 1.256_637_062_12e-6, "N/A2"),
        ("eps0", 8.854_187_812_8e-12, "F/m"),
    ]
}

#[pyfunction]
fn constants_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for (name, value, unit) in constants_table() {
        d.set_item(
            name,
            Quantity::new(CoreQuantity::new(value, unit).map_err(err)?),
        )?;
    }
    Ok(d)
}

fn prefixes_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for (name, long, f) in ::casacure::quanta::PREFIXES {
        let entry = PyList::empty(py);
        entry.append(PyString::new(py, long))?;
        entry.append(Quantity::new(CoreQuantity::new(*f, "").map_err(err)?))?;
        d.set_item(*name, entry)?;
    }
    Ok(d)
}

fn units_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let names = [
        "m", "km", "cm", "mm", "um", "nm", "pc", "kpc", "Mpc", "au", "ly", "kg", "g", "t", "s",
        "min", "h", "d", "yr", "Hz", "N", "J", "W", "Pa", "T", "V", "Wb", "Jy", "rad", "deg",
        "arcmin", "arcsec", "mas", "K", "A", "mol", "cd", "%",
    ];
    let d = PyDict::new(py);
    for n in names {
        if let Ok(q) = CoreQuantity::new(1.0, n) {
            // casacore stores the canonical (SI) quantity; the repr picks up
            // the sexagesimal forms because the SI base units `s` (time) and
            // `rad` (angle) are the special display units.
            let (v, u) = q.canonical();
            let store = CoreQuantity::new(v, &u);
            let entry = PyList::empty(py);
            entry.append(PyString::new(py, ::casacure::quanta::unit_long_name(n)))?;
            entry.append(Quantity::new(store.map_err(err)?))?;
            d.set_item(n, entry)?;
        }
    }
    Ok(d)
}

/// Build the `quanta` submodule.
pub fn quanta_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "quanta")?;
    m.add_class::<Quantity>()?;
    m.add_function(wrap_pyfunction!(quantity, &m)?)?;
    m.add_function(wrap_pyfunction!(is_quantity, &m)?)?;
    m.add_function(wrap_pyfunction!(near, &m)?)?;
    m.add_function(wrap_pyfunction!(nearabs, &m)?)?;
    m.add_function(wrap_pyfunction!(from_dict, &m)?)?;
    m.add_function(wrap_pyfunction!(pow, &m)?)?;
    m.add_function(wrap_pyfunction!(root, &m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, &m)?)?;
    m.add_function(wrap_pyfunction!(qabs, &m)?)?;
    m.add_function(wrap_pyfunction!(qsin, &m)?)?;
    m.add_function(wrap_pyfunction!(qcos, &m)?)?;
    m.add_function(wrap_pyfunction!(qtan, &m)?)?;
    m.add_function(wrap_pyfunction!(qasin, &m)?)?;
    m.add_function(wrap_pyfunction!(qacos, &m)?)?;
    m.add_function(wrap_pyfunction!(qatan, &m)?)?;
    m.add_function(wrap_pyfunction!(qatan2, &m)?)?;
    m.add_function(wrap_pyfunction!(qexp, &m)?)?;
    m.add_function(wrap_pyfunction!(qlog, &m)?)?;
    m.add_function(wrap_pyfunction!(qlog10, &m)?)?;
    m.add_function(wrap_pyfunction!(qceil, &m)?)?;
    m.add_function(wrap_pyfunction!(qfloor, &m)?)?;
    m.add("constants", constants_dict(parent.py())?)?;
    m.add("prefixes", prefixes_dict(parent.py())?)?;
    m.add("units", units_dict(parent.py())?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("casacure.quanta", &m)?;
    Ok(())
}
