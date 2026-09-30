//! A test's scratch directory under the temp dir, removed when the test drops
//! it.  The unit tests' `temp_dir()` helpers used to return a bare `PathBuf`,
//! so every `cargo test` left ~90 `casacure-test-*` / `casacure-taql-probe-*` /
//! `casacure-mssub-*` directories behind.  `Deref<Target = PathBuf>` keeps the
//! call sites as they were (`&dir`, `dir.join(..)`, `dir.clone()` -> PathBuf).

pub(crate) struct TestDir(pub(crate) std::path::PathBuf);

impl TestDir {
    /// `<temp>/<name>`, removed first so a leftover of an earlier run is not reused.
    pub(crate) fn new(name: String) -> TestDir {
        let p = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&p);
        TestDir(p)
    }
}

impl std::ops::Deref for TestDir {
    type Target = std::path::PathBuf;
    fn deref(&self) -> &std::path::PathBuf {
        &self.0
    }
}

impl AsRef<std::path::Path> for TestDir {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The core APIs take `impl Into<PathBuf>`.
impl From<&TestDir> for std::path::PathBuf {
    fn from(d: &TestDir) -> std::path::PathBuf {
        d.0.clone()
    }
}

thread_local! {
    /// Guards handed over by value (`create(temp_dir("x"), ..)`): kept until the
    /// test's thread exits, so the directory outlives the statement that made it.
    static DEFERRED: std::cell::RefCell<Vec<TestDir>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl From<TestDir> for std::path::PathBuf {
    fn from(d: TestDir) -> std::path::PathBuf {
        let p = d.0.clone();
        DEFERRED.with(|v| v.borrow_mut().push(d));
        p
    }
}
