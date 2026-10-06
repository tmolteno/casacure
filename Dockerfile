# Pre-release test image for casacure.
#
# `invoke test` builds this and runs the same gates CI runs (`.github/workflows/
# ci.yml`'s `test` job), so a release is not tagged until the suite is green
# in a clean environment with real python-casacore available.  The `tsan` and
# `freethreaded` CI jobs are reproduced as `invoke test --sanitizers` /
# `--freethreaded` (they need nightly toolchains / no-GIL interpreters, so
# they are opt-in here and always run in CI).
#
# Build context is the repo root (the .dockerignore excludes target/, .venv/,
# tests/fixtures/).  The tests are COPIED in at build time so the image is
# self-contained and the host tree is never touched.

# CI's `test` job runs on ubuntu-latest + setup-python 3.12 with
# python-casacore + pytest + maturin.  python-casacore (pip) bundles its own
# casacore libs, so no system casacore is needed.  daskms/xarray are optional
# (the daskms tests are importorskip-ed) and pin Python >= 3.12.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# rustup for the cargo test / fmt / clippy gate (CI's dtolnay/rust-toolchain@stable).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --component rustfmt,clippy
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir \
        maturin numpy pytest python-casacore dask-ms xarray astropy

WORKDIR /src

# Copy the whole tree (the .dockerignore excludes build artifacts).
COPY . /src

# Generate the casacore fixture tables (CI does this before the Rust tests).
# This needs REAL casacore (the shim would redirect to casacure, which is not
# installed yet), so it runs before `pip install .` and before PYTHONPATH is set.
RUN python3 tests/make_fixtures.py

# Build + install the wheel (CI's `pip install .` step).
RUN python3 -m pip install --no-cache-dir .

# The shim redirects `casacore.*` to `casacure.*` for the comparison tests.
ENV PYTHONPATH=/src/tests/shim

# CI's `test` job:
#   1. cargo test --workspace
#   2. cargo fmt --check
#   3. cargo clippy --workspace --all-targets -- -D warnings
#   4. build + install the wheel (done above)
#   5. PYTHONPATH=tests/shim pytest tests/
# Steps 1-3 are the build-time gate; 5 is the runtime command (so `invoke
# test` can re-run it against a different wheel).
RUN cargo test --workspace \
    && cargo fmt --check \
    && cargo clippy --workspace --all-targets -- -D warnings

# A non-root user: the permission tests (a_read_only_block_is_named_in_the
#_storage_error, ...rely on chmod 0444 denying writes, which root ignores.
RUN useradd --create-home --uid 1000 tester \
    && chown -R tester:tester /src
USER tester

# Default command: the Python comparison suite (CI's final step).
# `PYTHONPATH=tests/shim` redirects `casacore.*` to `casacure.*` (CI's
# `casacore comparison tests` step).
CMD ["python3", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"]
