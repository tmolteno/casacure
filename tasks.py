"""Release orchestration for casacure.

Following the example of ../meerkat_imaging's tasks.py: `invoke release` runs
the full pre-release gate locally in Docker before any tag is pushed to
GitHub, so a CI failure on a release tag becomes a local failure the operator
sees first.

Usage (from the casacure repo root, with a venv that has invoke + plumbum):

    invoke version                      # show the current + next versions
    invoke test                         # the pre-release tests (Docker build + suite)
    invoke release                      # bump patch, stamp CHANGELOG, tag + push
    invoke release --no-bump            # tag the current version as-is
    invoke release --version 3.8.21     # explicit version (implies --no-bump)

`invoke release` writes the release commit for you: the patch version in
`pyproject.toml`/`Cargo.toml` and the `CHANGELOG.md` heading (`[Unreleased]`
becomes `## [X.Y.Z] - <today>`, with a fresh empty `[Unreleased]` left on
top), so the changelog no longer has to be edited by hand at release time.

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
from datetime import date
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


def _changelog_has(version: str) -> bool:
    """True when CHANGELOG.md already has a `## [version]` section."""
    text = (REPO / "CHANGELOG.md").read_text()
    return bool(re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE))


def _stamp_changelog(new: str) -> bool:
    """Move the `[Unreleased]` changelog entries under `## [new] - <today>`.

    CHANGELOG.md is written newest-first, so a release turns the entries
    accumulated under `## [Unreleased]` into the released version's own
    section and leaves a fresh, empty `## [Unreleased]` on top of it.

    Idempotent and respectful of hand-written notes: if a `## [new]` section
    is already present (written by hand, or a re-run of the release), the
    file is left untouched and `False` is returned; if there is no
    `## [Unreleased]` section at all, the dated section is inserted above the
    first existing section instead.
    """
    p = REPO / "CHANGELOG.md"
    s = p.read_text()
    if re.search(rf"^## \[{re.escape(new)}\]", s, re.MULTILINE):
        return False
    heading = f"## [{new}] - {date.today().isoformat()}\n"
    unreleased = re.search(r"^## \[Unreleased\][ \t]*\n", s, re.MULTILINE)
    if unreleased:
        s = s[: unreleased.end()] + f"\n{heading}" + s[unreleased.end():]
    else:
        first = re.search(r"^## \[", s, re.MULTILINE)
        if not first:
            return False
        s = s[: first.start()] + f"{heading}\n" + s[first.start():]
    p.write_text(s)
    return True


def _git(*args: str) -> str:
    return local["git"]["-C", str(REPO), *args]()


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
    local["docker"][*args] & FG


def _run_pytest_casacore_parity() -> None:
    """Re-run the measures accuracy contract with real casacore visible.

    The image exports `PYTHONPATH=tests/shim` globally, which hides real
    python-casacore; an empty PYTHONPATH override runs the casacore-parity
    half of tests/test_measures.py (the counterpart of CI's "measures
    accuracy contract and casacore parity" step) against the image's own
    python-casacore.  The tests skip themselves when casacore is importable
    but its measures data files are missing (ratt-ru/QuartiCal#330).
    """
    local["docker"][
        "run", "--rm", "-e", "PYTHONPATH=", IMAGE, "python3", "-m", "pytest",
        "tests/test_measures.py", "-q", "-p", "no:cacheprovider",
    ] & FG


@task
def version(c) -> None:
    """Show the versions the release would tag (the tree's pyproject version)."""
    v = _read_version()
    print(f"casacure: pyproject {v}  ->  would bump to {_bump_patch(v)} and tag v{_bump_patch(v)}")
    print(f"           CHANGELOG.md: [Unreleased] -> [{_bump_patch(v)}] - {date.today().isoformat()}")
    print(f"           (or tag v{v} as-is with --no-bump; "
          f"--version X.Y.Z tags that exact version)")


@task
def test(c,
         sanitizers: bool = False,
         freethreaded: bool = False,
         all_jobs: bool = False) -> None:
    """Run the pre-release tests in Docker.

    Default: the `test` CI job (cargo test / fmt / clippy / build + pytest
    tests/ with the shim, plus the measures accuracy contract against the
    image's own python-casacore).  --sanitizers adds the `tsan` job (nightly +
    ThreadSanitizer); --freethreaded adds the `freethreaded` job (no-GIL
    CPython 3.14t).  --all-jobs runs all three (the full CI gate).
    """
    sanitizers = sanitizers or all_jobs
    freethreaded = freethreaded or all_jobs

    _build_image()
    _run_pytest()
    _run_pytest_casacore_parity()
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
            bump: bool = True,
            sanitizers: bool = False,
            freethreaded: bool = False,
            all_jobs: bool = False) -> None:
    """Run the full casacure release chain.

    1. run the pre-release tests in Docker (the `test` CI gate, via this
       task's `pre=[test]`; --sanitizers / --freethreaded / --all-jobs add the
       other CI jobs).
    2. bump the patch version in pyproject/Cargo and CHANGELOG.md (the
       `[Unreleased]` entries move under `## [X.Y.Z] - <today>`) and commit
       them as `release: bump to X.Y.Z` (default; skip with --no-bump, or
       override with --version X.Y.Z which implies --no-bump since the tag is
       given explicitly).
    3. tag vX.Y.Z and push, which triggers `publish-python.yml` and
       `publish-rust.yml`.
    4. wait for the publish workflows to go green.

    Idempotent: a tag already on origin is verified and skipped, not re-pushed.
    """
    # An explicit --version means "tag exactly this"; the tree's version must
    # not be touched or the tag and tree would drift apart.
    if version is not None:
        bump = False

    if bump:
        new = _bump_patch(_read_version())
        print(f"=== bumping version to {new}")
        _bump_version(new)
        if _stamp_changelog(new):
            print(f"=== CHANGELOG.md: [Unreleased] -> [{new}]")
        _git("add", "pyproject.toml", "Cargo.toml", "CHANGELOG.md")
        _git("commit", "-m", f"release: bump to {new}")

    v = version or _read_version()
    tag = f"v{v}"
    if not _changelog_has(v):
        print(f"    warning: CHANGELOG.md has no '## [{v}]' section, so the "
              f"tag will be published without release notes (add one, or let "
              f"the default bump stamp it)")

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
    local["git"]["-C", str(REPO), "push", "origin", "main", tag] & FG

    print("  waiting for the publish workflows...")
    _wait_for_ci(tag)
    print(f"\n=== RELEASE COMPLETE: casacure {tag}")
