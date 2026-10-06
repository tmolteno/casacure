"""Release orchestration for casacure.

Following the example of ../meerkat_imaging's tasks.py: `invoke release` runs
the full pre-release gate locally in Docker before any tag is pushed to
GitHub, so a CI failure on a release tag becomes a local failure the operator
sees first.

Usage (from the casacure repo root, with a venv that has invoke + plumbum):

    invoke version                      # show the current + next versions
    invoke test                         # the pre-release tests (Docker build + suite)
    invoke release                      # bump version, run tests, tag + push
    invoke release --version 3.8.21     # explicit version (skips the auto-bump)

The three CI jobs (`.github/workflows/ci.yml`) are reproduced locally:
  * test         — cargo test / fmt / clippy / maturin build / pytest tests/
  * tsan         — `invoke test --sanitizers` (nightly + ThreadSanitizer)
  * freethreaded — `invoke test --freethreaded` (no-GIL CPython 3.14t)

`invoke release` runs the `test` job by default and the other two with
`--sanitizers` / `--freethreaded` (or `--all-jobs` for all three).  The
publish workflows (`publish-python.yml`, `publish-rust.yml`) are triggered
by the tag push, exactly as before.

Requires: invoke, plumbum, docker, gh (authenticated), cargo, rustup (for
the --sanitizers job).
"""
import json
import re
import subprocess
import time
from pathlib import Path

from invoke import task
from plumbum import FG, local

REPO = Path(__file__).parent
IMAGE = "casacure-pre-release"


# ---------------------------------------------------------------------------
# version + git helpers
# ---------------------------------------------------------------------------

def _read_version() -> str:
    text = (REPO / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else "?"


def _bump_version(new: str) -> None:
    """Rewrite the version in pyproject.toml and Cargo.toml (workspace +
    the casacure-python dependency)."""
    for rel in ("pyproject.toml", "Cargo.toml"):
        p = REPO / rel
        s = p.read_text()
        s = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{new}"', s, count=1, flags=re.MULTILINE)
        s = re.sub(
            r'casacure = \{ path = "crates/casacure", version = "[^"]+" \}',
            f'casacure = {{ path = "crates/casacure", version = "{new}" }}',
            s,
        )
        p.write_text(s)


def _bump_patch(version: str) -> str:
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def _git(*args: str) -> str:
    return local["git"]("-C", str(REPO), *args)()


def _tag_exists(tag: str) -> bool:
    out = _git("ls-remote", "--tags", "origin", f"refs/tags/{tag}")
    return bool(out.strip())


def _wait_for_ci(ref: str, timeout_s: int = 3600) -> None:
    """Wait for the CI + Publish workflows of `ref` (the tag) to go green.

    Filters by workflow + headBranch: a bare `--limit 1` sees whatever ran
    last (often the previous tag's run, still green) and would return before
    the new build even registers.
    """
    start = time.time()
    run_ids: dict[str, int] = {}
    while len(run_ids) < 2:  # CI + Publish Python package
        runs = json.loads(local["gh"]["run", "list", "-R", "tmolteno/casacure",
                                     "--workflow", "ci.yml", "--workflow",
                                     "publish-python.yml", "--limit", "10",
                                     "--json", "databaseId,headBranch,name"]())
        for r in runs:
            if r["headBranch"] == ref:
                run_ids.setdefault(r["name"], r["databaseId"])
        if len(run_ids) >= 2:
            break
        if time.time() - start > 300:
            msg = f"no CI/publish run appeared for {ref} in 300 s"
            raise RuntimeError(msg)
        print("  waiting for the workflow runs of", ref, "to register...")
        time.sleep(20)

    for name, run_id in run_ids.items():
        print(f"  {name} (run {run_id}):")
        while True:
            result = local["gh"]["run", "view", str(run_id), "-R", "tmolteno/casacure",
                                 "--json", "status,conclusion",
                                 "--jq", '"\\(.status) \\(.conclusion)"']()
            line = result.strip()
            if "success" in line:
                print(f"    {name}: {line}")
                break
            if "failure" in line:
                msg = (f"{name} CI failed (run {run_id}); see "
                       f"gh run view {run_id} -R tmolteno/casacure --log-failed")
                raise RuntimeError(msg)
            print(f"    {name}: {line}")
            time.sleep(60)
            if time.time() - start > timeout_s:
                msg = f"{name} CI timed out after {timeout_s}s"
                raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# pre-release tests (the CI gate, reproduced locally in Docker)
# ---------------------------------------------------------------------------

def _build_image() -> None:
    """Build the pre-release test image (rust:1.90 + python-casacore + the
    whole suite).  The .dockerignore excludes target/, .venv/, tests/fixtures/.
    """
    with local.cwd(str(REPO)):
        local["docker"]["build", "-t", IMAGE, "."] & FG


def _run_pytest(extra_args: list[str] | None = None) -> None:
    """Run the Python comparison suite in the built image (CI's final step:
    `PYTHONPATH=tests/shim pytest tests/`)."""
    args = ["run", "--rm", IMAGE, "python3", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"]
    if extra_args:
        args.extend(extra_args)
    local["docker"](*args) & FG


@task
def version(c) -> None:
    """Show the versions the release would tag (the tree's pyproject version)."""
    v = _read_version()
    print(f"casacure: pyproject {v}  ->  would tag v{v} (or v{_bump_patch(v)} with --bump)")
    print("(the convention is to bump CHANGELOG.md + pyproject in the tree first; "
          "--version overrides the tag)")


@task
def test(c,
         sanitizers: bool = False,
         freethreaded: bool = False,
         all_jobs: bool = False) -> None:
    """Run the pre-release tests in Docker.

    Default: the `test` CI job (cargo test / fmt / clippy / build + pytest
    tests/ with the shim).  --sanitizers adds the `tsan` job (nightly +
    ThreadSanitizer); --freethreaded adds the `freethreaded` job (no-GIL
    CPython 3.14t).  --all-jobs runs all three (the full CI gate).
    """
    sanitizers = sanitizers or all_jobs
    freethreaded = freethreaded or all_jobs

    _build_image()
    _run_pytest()
    print("=== test job PASS")

    if sanitizers:
        # CI's `tsan` job: nightly + -Zsanitizer=thread on the core crate.
        local["docker"]["run", "--rm", IMAGE, "bash", "-c",
               "rustup toolchain install nightly --component rust-src && "
               "RUSTFLAGS='-Zsanitizer=thread' cargo +nightly -Zbuild-std test "
               "-p casacure --tests --target x86_64-unknown-linux-gnu"] & FG
        print("=== tsan job PASS")

    if freethreaded:
        # CI's `freethreaded` job: no-GIL CPython 3.14t + maturin develop +
        # the suite with the GIL off.
        local["docker"]["run", "--rm", IMAGE, "bash", "-c",
               "pip install uv && uv python install 3.14t && "
               "uv venv --python 3.14t /opt/venv-ft && "
               "uv pip install --python /opt/venv-ft/bin/python numpy pytest maturin && "
               "/opt/venv-ft/bin/maturin develop && "
               "/opt/venv-ft/bin/python -c \"import casacure, casacure.tables, "
               "casacure.quanta, sys; assert not sys._is_gil_enabled()\" && "
               "PYTHONPATH=tests/shim /opt/venv-ft/bin/python -m pytest tests/ -q"] & FG
        print("=== freethreaded job PASS")


@task(pre=[test])
def release(c,
            version: str | None = None,
            bump: bool = False,
            sanitizers: bool = False,
            freethreaded: bool = False,
            all_jobs: bool = False) -> None:
    """Run the full casacure release chain.

    1. (optional) bump the patch version in pyproject/Cargo and commit.
    2. run the pre-release tests in Docker (the `test` CI gate; --sanitizers
       / --freethreaded / --all-jobs add the other CI jobs).
    3. commit CHANGELOG.md + pyproject/Cargo (the release commit).
    4. tag vX.Y.Z and push, which triggers `publish-python.yml` and
       `publish-rust.yml`.
    5. wait for the publish workflows to go green.

    Idempotent: a tag already on origin is verified and skipped, not re-pushed.
    """
    if bump:
        new = _bump_patch(_read_version())
        print(f"=== bumping version to {new}")
        _bump_version(new)

    v = version or _read_version()
    tag = f"v{v}"

    status = _git("status", "--porcelain").strip()
    if status:
        msg = ("tree is dirty; commit the release (CHANGELOG.md, pyproject.toml, "
               "Cargo.toml) first\n" + status)
        raise RuntimeError(msg)

    commit = _git("rev-parse", "HEAD").strip()
    if _tag_exists(tag):
        print(f"=== {tag} already on origin -- verified, skipping")
        return

    print(f"=== tagging {tag} (commit {commit[:12]})")
    _git("tag", "-a", tag, "-m", tag)
    local["git"]("-C", str(REPO), "push", "origin", "main", tag) & FG

    print("  waiting for the publish workflows...")
    _wait_for_ci(tag)
    print(f"\n=== RELEASE COMPLETE: casacure {tag}")
