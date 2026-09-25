//! `casacure-test` / `casacure-bench`: a self-test battery and a speed
//! benchmark, both run over the *public* Python API (`casacure.tables`).
//!
//! Console scripts:
//!   casacure-test  -> `casacure:run_tests`     (exit code = failures)
//!   casacure-bench -> `casacure:run_benchmark`

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;

type FnCheck = fn(&mut Runner<'_>) -> PyResult<bool>;

/// Run the self-test battery; returns the number of failed checks (0 = all
/// passed), which the console script turns into the exit code.
#[pyfunction]
pub fn run_tests(py: Python<'_>) -> PyResult<usize> {
    let mut r = Runner {
        py,
        np: py.import("numpy")?.into_any(),
        tables: py.import("casacure.tables")?.into_any(),
        passed: 0,
        failed: 0,
    };
    let checks: Vec<(&str, FnCheck)> = vec![
        ("scalar double round-trip", scalar_roundtrip),
        ("scalar int round-trip", scalar_int),
        ("string round-trip (long + empty)", string_roundtrip),
        ("fixed-shape array round-trip", fixed_array),
        (
            "dcomplex coercion (complex64 -> complex128)",
            dcomplex_coerce,
        ),
        ("variable array putvarcol/getvarcol", varcol_roundtrip),
        ("record column create + reopen", record_column),
        ("taql SELECT WHERE", taql_where),
        ("table.query() then sort()", query_sort),
        ("getkeyword(MS_VERSION) == 2.0", getkeyword_ms),
        ("default_ms subtables + TpTable keyword", default_ms_ok),
        ("multidim string array write/read", multidim_strings),
        ("int16 -> int32 coercion", int16_coerce),
        ("addrows + flush persistence", persistence),
    ];
    for (name, f) in checks {
        r.check(name, f);
    }
    println!(
        "casacure self-test: {} passed, {} failed",
        r.passed, r.failed
    );
    Ok(r.failed)
}

struct Runner<'py> {
    py: Python<'py>,
    np: Bound<'py, PyAny>,
    tables: Bound<'py, PyAny>,
    passed: usize,
    failed: usize,
}

impl<'py> Runner<'py> {
    fn check(&mut self, name: &str, f: FnCheck) {
        match f(self) {
            Ok(true) => {
                self.passed += 1;
                println!("ok   {name}");
            }
            Ok(false) => {
                self.failed += 1;
                println!("FAIL {name}: check returned false");
            }
            Err(e) => {
                self.failed += 1;
                println!("FAIL {name}: {e}");
            }
        }
    }

    fn tmpdir(&self, tag: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "casacure-selftest-{}-{}-{tag}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0)
            ))
            .display()
            .to_string()
    }

    /// `casacure.tables.table(path, desc, nrow)` — writable create.
    fn create(
        &self,
        path: &str,
        desc: &Bound<'py, PyDict>,
        nrow: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.tables
            .getattr("table")?
            .call1((path, desc, nrow as i64))
    }

    /// `casacure.tables.table(path, readonly=True)` — read-only open.
    fn open(&self, path: &str) -> PyResult<Bound<'py, PyAny>> {
        let kw = PyDict::new(self.py);
        kw.set_item("readonly", true)?;
        self.tables.getattr("table")?.call((path,), Some(&kw))
    }

    fn keywords(&self) -> Bound<'py, PyDict> {
        PyDict::new(self.py)
    }

    fn scalar_desc(&self, name: &str, vt: &str) -> PyResult<Bound<'py, PyDict>> {
        let c = PyDict::new(self.py);
        c.set_item("valueType", vt)?;
        c.set_item("option", 0)?;
        c.set_item("comment", "")?;
        c.set_item("keywords", self.keywords())?;
        let outer = PyDict::new(self.py);
        outer.set_item(name, c)?;
        Ok(outer)
    }

    fn array_desc(&self, name: &str, vt: &str, logical: &[i64]) -> PyResult<Bound<'py, PyDict>> {
        let c = PyDict::new(self.py);
        c.set_item("valueType", vt)?;
        c.set_item("option", 0)?;
        c.set_item("comment", "")?;
        c.set_item("ndim", logical.len() as i64)?;
        c.set_item("shape", PyList::new(self.py, logical)?)?;
        c.set_item("keywords", self.keywords())?;
        let outer = PyDict::new(self.py);
        outer.set_item(name, c)?;
        Ok(outer)
    }

    fn getcol(&self, t: &Bound<'py, PyAny>, col: &str) -> PyResult<Bound<'py, PyAny>> {
        t.call_method1("getcol", (col,))
    }

    fn allclose(&self, a: &Bound<'py, PyAny>, b: Bound<'py, PyAny>) -> PyResult<bool> {
        self.np.call_method1("allclose", (a, b))?.extract()
    }
}

fn scalar_roundtrip(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("scalar");
    let t = r.create(&path, &r.scalar_desc("V", "double")?, 3)?;
    t.call_method1("putcol", ("V", PyList::new(r.py, [1.5_f64, 2.0, 3.5])?))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let n = ro.call_method0("nrows")?.extract::<u64>()?;
    let got = r.getcol(&ro, "V")?;
    let exp =
        r.np.call_method1("array", (PyList::new(r.py, [1.5_f64, 2.0, 3.5])?,))?;
    Ok(n == 3 && r.allclose(&got, exp)?)
}

fn scalar_int(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("int");
    let t = r.create(&path, &r.scalar_desc("V", "int")?, 4)?;
    t.call_method1("putcol", ("V", r.np.call_method1("arange", (4,))?))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let got = r.getcol(&ro, "V")?;
    let exp = r.np.call_method1("arange", (4,))?;
    r.allclose(&got, exp)
}

fn string_roundtrip(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("strings");
    let t = r.create(&path, &r.scalar_desc("S", "string")?, 3)?;
    t.call_method1(
        "putcol",
        (
            "S",
            PyList::new(r.py, ["a", "a label far longer than eight chars", ""])?,
        ),
    )?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let got: Vec<String> = ro.call_method1("getcol", ("S",))?.extract()?;
    Ok(got == ["a", "a label far longer than eight chars", ""].to_vec())
}

fn fixed_array(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("arr");
    let t = r.create(&path, &r.array_desc("V", "double", &[2, 3])?, 2)?;
    let arr =
        r.np.call_method1("arange", (12i64,))?
            .call_method1("reshape", (PyList::new(r.py, [2usize, 2, 3])?,))?;
    t.call_method1("putcol", ("V", arr))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let got = r.getcol(&ro, "V")?;
    let shape: Vec<usize> = got.getattr("shape")?.extract()?;
    let exp =
        r.np.call_method1("arange", (12i64,))?
            .call_method1("reshape", (PyList::new(r.py, [2usize, 2, 3])?,))?;
    Ok(shape == vec![2, 2, 3] && r.allclose(&got, exp)?)
}

fn dcomplex_coerce(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("c64");
    let t = r.create(&path, &r.array_desc("DATA", "dcomplex", &[8, 4])?, 2)?;
    let arr =
        r.np.call_method1("arange", (64i64,))?
            .call_method1("reshape", (PyList::new(r.py, [2usize, 8, 4])?,))?
            .call_method1("astype", ("complex64",))?;
    t.call_method1("putcol", ("DATA", arr))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let got = r.getcol(&ro, "DATA")?;
    let dt: String = got.getattr("dtype")?.str()?.to_string();
    let exp =
        r.np.call_method1("arange", (64i64,))?
            .call_method1("reshape", (PyList::new(r.py, [2usize, 8, 4])?,))?
            .call_method1("astype", ("complex128",))?;
    Ok(dt.starts_with("complex128") && r.allclose(&got, exp)?)
}

fn varcol_roundtrip(r: &mut Runner<'_>) -> PyResult<bool> {
    let c = PyDict::new(r.py);
    c.set_item("valueType", "double")?;
    c.set_item("option", 0)?;
    c.set_item("comment", "")?;
    c.set_item("ndim", 1)?;
    c.set_item("keywords", r.keywords())?;
    let d = PyDict::new(r.py);
    d.set_item("V", c)?;
    let path = r.tmpdir("varcol");
    let t = r.create(&path, &d, 2)?;
    let rows = PyDict::new(r.py);
    rows.set_item(
        "r1",
        r.np.call_method1("array", (PyList::new(r.py, [1.0f64, 2.0, 3.0])?,))?,
    )?;
    rows.set_item(
        "r2",
        r.np.call_method1("array", (PyList::new(r.py, [4.0f64, 5.0])?,))?,
    )?;
    t.call_method1("putvarcol", ("V", rows))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let vc = ro.call_method1("getvarcol", ("V",))?;
    let s1: Vec<f64> = vc.get_item("r1")?.call_method0("tolist")?.extract()?;
    let s2: Vec<f64> = vc.get_item("r2")?.call_method0("tolist")?.extract()?;
    Ok(s1 == vec![1.0, 2.0, 3.0] && s2 == vec![4.0, 5.0])
}

fn record_column(r: &mut Runner<'_>) -> PyResult<bool> {
    let c = PyDict::new(r.py);
    c.set_item("valueType", "record")?;
    c.set_item("option", 0)?;
    c.set_item("comment", "")?;
    c.set_item("keywords", r.keywords())?;
    let d = PyDict::new(r.py);
    d.set_item("SOURCE_MODEL", c)?;
    let path = r.tmpdir("rec");
    let t = r.create(&path, &d, 2)?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let cols: Vec<String> = ro.call_method0("colnames")?.extract()?;
    Ok(cols == ["SOURCE_MODEL"])
}

fn taql_where(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("taql");
    let t = r.create(&path, &r.scalar_desc("TIME", "double")?, 3)?;
    t.call_method1(
        "putcol",
        (
            "TIME",
            r.np.call_method1("array", (PyList::new(r.py, [3.0f64, 1.0, 2.0])?,))?,
        ),
    )?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let q = crate::table::taql(
        r.py,
        "SELECT * FROM $1 WHERE TIME > 1.5",
        Some(&PyList::new(r.py, [ro.clone()])?),
        "Python",
        None,
    )?;
    let got: Vec<f64> = q.bind(r.py).call_method1("getcol", ("TIME",))?.extract()?;
    Ok(got == vec![3.0, 2.0])
}

fn query_sort(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("qsort");
    let t = r.create(&path, &r.scalar_desc("TIME", "double")?, 3)?;
    t.call_method1(
        "putcol",
        (
            "TIME",
            r.np.call_method1("array", (PyList::new(r.py, [3.0f64, 1.0, 2.0])?,))?,
        ),
    )?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let q = ro.call_method1("query", ("TIME > 0.0",))?;
    let s = q.call_method1("sort", ("TIME",))?;
    let got: Vec<f64> = s.call_method1("getcol", ("TIME",))?.extract()?;
    Ok(got == vec![1.0, 2.0, 3.0])
}

fn getkeyword_ms(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("kw");
    let _ms = r.tables.getattr("default_ms")?.call1((&path,))?;
    let ro = r.open(&path)?;
    let v: f64 = ro.call_method1("getkeyword", ("MS_VERSION",))?.extract()?;
    Ok(v == 2.0)
}

fn default_ms_ok(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("ms");
    let ms = r.tables.getattr("default_ms")?.call1((&path,))?;
    let cols: Vec<String> = ms.call_method0("colnames")?.extract()?;
    let ro = r.open(&path)?;
    let s: String = ro.call_method1("getkeyword", ("ANTENNA",))?.extract()?;
    Ok(cols.len() >= 20 && s.starts_with("Table: "))
}

fn multidim_strings(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("mstr");
    let t = r.create(&path, &r.array_desc("S", "string", &[2, 2])?, 2)?;
    let dd = PyDict::new(r.py);
    dd.set_item("shape", PyList::new(r.py, [2usize, 2, 2])?)?;
    dd.set_item(
        "array",
        PyList::new(r.py, ["a", "b", "c", "d", "e", "f", "g", "h"])?,
    )?;
    t.call_method1("putcol", ("S", dd))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    // Multidim strings come back in the {"shape":..,"array":..} dict form.
    let got = r.getcol(&ro, "S")?;
    let shape: Vec<usize> = got.get_item("shape")?.extract()?;
    let arr: Vec<String> = got.get_item("array")?.extract()?;
    Ok(shape == vec![2, 2, 2] && arr == ["a", "b", "c", "d", "e", "f", "g", "h"].to_vec())
}

fn int16_coerce(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("i16");
    let t = r.create(&path, &r.scalar_desc("V", "int")?, 3)?;
    let arr =
        r.np.call_method1("array", (PyList::new(r.py, [1i64, 2, 3])?,))?
            .call_method1("astype", ("int16",))?;
    t.call_method1("putcol", ("V", arr))?;
    t.call_method0("flush")?;
    let ro = r.open(&path)?;
    let v: Vec<i64> = ro
        .call_method1("getcol", ("V",))?
        .call_method0("tolist")?
        .extract()?;
    Ok(v == vec![1, 2, 3])
}

fn persistence(r: &mut Runner<'_>) -> PyResult<bool> {
    let path = r.tmpdir("persist");
    let t = r.create(&path, &r.scalar_desc("V", "int")?, 0)?;
    t.call_method1("addrows", (3u64,))?;
    t.call_method1(
        "putcol",
        (
            "V",
            r.np.call_method1("array", (PyList::new(r.py, [10i64, 20, 30])?,))?,
        ),
    )?;
    t.call_method0("flush")?;
    t.call_method0("close")?;
    let ro = r.open(&path)?;
    let n = ro.call_method0("nrows")?.extract::<u64>()?;
    let v: Vec<i64> = ro
        .call_method1("getcol", ("V",))?
        .call_method0("tolist")?
        .extract()?;
    Ok(n == 3 && v == vec![10, 20, 30])
}

// ---------------------------------------------------------------------------
// Benchmark: casacure vs real casacore (when importable as a distinct module)
// ---------------------------------------------------------------------------

type FnBench<'py> =
    fn(&Bound<'py, PyAny>, &Bound<'py, PyAny>, &Bound<'py, PyAny>, usize) -> PyResult<()>;

#[pyfunction]
pub fn run_benchmark<'py>(py: Python<'py>) -> PyResult<()> {
    let n: usize = 20_000;
    let np = py.import("numpy")?.into_any();
    let ck = py.import("casacure.tables")?.into_any();

    let casacore: Option<Bound<'_, PyAny>> =
        py.import("casacore.tables").ok().map(|m| m.into_any());
    // "Distinct" means a different `table` implementation (real
    // python-casacore), not a wrapper re-exporting casacure's functions
    // (the casacore shim).
    let distinct = if let Some(cg) = &casacore {
        match (ck.getattr("table").ok(), cg.getattr("table").ok()) {
            (Some(a), Some(b)) => a.as_ptr() != b.as_ptr(),
            _ => true,
        }
    } else {
        false
    };

    let mut rows: Vec<(String, f64, Option<f64>)> = Vec::new();
    let ops: Vec<(&str, FnBench<'py>)> = vec![
        ("putcol", bench_putcol),
        ("getcol", bench_getcol),
        ("taql WHERE+ORDERBY", bench_taql),
    ];

    let (t, path) = bench_setup_base(py, &np, &ck, "casacure", n)?;
    for (name, f) in &ops {
        let t0 = std::time::Instant::now();
        f(&t, &np, &ck, n)?;
        rows.push((name.to_string(), t0.elapsed().as_secs_f64() * 1e3, None));
    }
    // Close the table before deleting its directory: a later drop of the
    // object (here or real python-casacore) releases the table lock, whose
    // callback re-touches table.dat_tmp; if the directory is already gone
    // that throws from a destructor and aborts the process (SIGABRT).
    t.call_method0("close")?;
    let _ = std::fs::remove_dir_all(&path);

    if distinct {
        let cg = casacore.unwrap();
        let (t, path) = bench_setup_base(py, &np, &cg, "casacore", n)?;
        for (name, f) in &ops {
            let t0 = std::time::Instant::now();
            f(&t, &np, &cg, n)?;
            if let Some(e) = rows.iter_mut().find(|(k, _, _)| *k == *name) {
                e.2 = Some(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        t.call_method0("close")?;
        let _ = std::fs::remove_dir_all(&path);
    }

    println!("\ncasacure benchmark (n = {n} rows, double scalar columns TIME and WEIGHT):");
    println!(
        "{:<22} {:>11} {:>11} {:>11}",
        "op", "casacure ms", "casacore ms", "cure/core"
    );
    for (name, mc, mr) in &rows {
        match mr {
            Some(rr) => println!(
                "{:<22} {:>11.2} {:>11.2} {:>11.2}",
                name,
                mc,
                rr,
                mc / rr.max(1e-6)
            ),
            None => println!("{:<22} {:>11.2} {:>11} {:>11}", name, mc, "n/a", "n/a"),
        }
    }
    if !rows.iter().any(|(_, _, r)| r.is_some()) {
        println!(
            "\n(real casacore is not importable as a distinct module here — the shim aliases it; \
                  run with python-casacore installed alongside casacure for the ratio)"
        );
    }
    Ok(())
}

fn bench_setup_base<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    tables: &Bound<'py, PyAny>,
    tag: &str,
    n: usize,
) -> PyResult<(Bound<'py, PyAny>, String)> {
    let path = std::env::temp_dir()
        .join(format!(
            "casacure-bench-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ))
        .display()
        .to_string();
    let d = PyDict::new(py);
    for col in ["TIME", "WEIGHT"] {
        let c = PyDict::new(py);
        c.set_item("valueType", "double")?;
        c.set_item("option", 0)?;
        c.set_item("comment", "")?;
        c.set_item("keywords", PyDict::new(py))?;
        d.set_item(col, c)?;
    }
    let t = tables.getattr("table")?.call1((&path, d, n as i64))?;
    t.call_method1(
        "putcol",
        ("TIME", np.getattr("arange")?.call1((n as i64,))?),
    )?;
    t.call_method1("putcol", ("WEIGHT", np.call_method1("ones", (n as i64,))?))?;
    t.call_method0("flush")?;
    Ok((t, path))
}

fn bench_putcol<'py>(
    t: &Bound<'py, PyAny>,
    np: &Bound<'py, PyAny>,
    _tables: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<()> {
    t.call_method1("putcol", ("WEIGHT", np.call_method1("ones", (n as i64,))?))?;
    Ok(())
}

fn bench_getcol<'py>(
    t: &Bound<'py, PyAny>,
    _np: &Bound<'py, PyAny>,
    _tables: &Bound<'py, PyAny>,
    _n: usize,
) -> PyResult<()> {
    let data = t.call_method1("getcol", ("TIME",))?;
    let _ = data.call_method0("nbytes");
    Ok(())
}

fn bench_taql<'py>(
    t: &Bound<'py, PyAny>,
    _np: &Bound<'py, PyAny>,
    tables: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<()> {
    let half = (n / 2) as i64;
    let py = t.py();
    let kw = PyDict::new(py);
    kw.set_item("tables", PyList::new(py, [t.clone()])?)?;
    let q = tables.getattr("taql")?.call(
        (format!("SELECT * FROM $1 WHERE TIME > {half} ORDERBY TIME"),),
        Some(&kw),
    )?;
    let _ = q.call_method0("nrows")?;
    Ok(())
}
