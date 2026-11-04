"""The published-wheel matrix must cover every platform and interpreter the
README promises (issue #35: `publish-python.yml`'s macOS matrix started at
3.11, so `pip install casacure` on macOS/Python 3.10 silently found no wheel
while the README advertised 3.10 – 3.14).

Since issue #39 the scheme is one **abi3** wheel per platform (pyo3's
`abi3-py39` limited-API build), which covers every CPython >= 3.9 on every
platform — a per-version matrix (and the #35 drift class) no longer exists.
Linux additionally builds the free-threaded `cp314t` wheel by explicit
interpreter; PyPy is deliberately not built. Three artifacts still have to
agree, and each is edited by a different habit of mind:

* the README's "prebuilt wheels" table (what users are told),
* `.github/workflows/publish-python.yml`'s matrix + build args (what is
  actually built),
* `pyproject.toml`'s `requires-python` (what pip will even attempt).

The abi3 scheme itself can rot in a new way — the feature being dropped
from the build args, or set unconditionally in Cargo.toml (which makes
maturin tag the free-threaded build cp39-abi3 instead of cp314-cp314t) —
so the tests pin that too.

These tests are pure text checks — no network, no wheel build.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# A data row of the README wheel table, e.g.
#   | macOS | `aarch64` (Apple Silicon) | 3.9 – 3.14 |
# (the version cell may use an en dash or a hyphen).
README_ROW = re.compile(
    r"^\|\s*(?P<platform>[^|]+?)\s*\|[^|]+\|\s*"
    r"(?P<first>\d+)\.(?P<first_minor>\d+)\s*[–-]\s*"
    r"(?P<last>\d+)\.(?P<last_minor>\d+)\s*\|$"
)

# README platform name -> the workflow matrix `target` that builds it.
TARGETS = {
    "macOS": "aarch64-apple-darwin",
    "Windows (MSVC)": "x86_64-pc-windows-msvc",
}
LINUX_TARGETS = {"x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"}

# The interpreters each Linux architecture builds: the abi3 wheel (one per
# arch, covering every CPython >= 3.9) and the free-threaded wheel. The
# manylinux image also carries PyPy — deliberately not built.
LINUX_INTERPRETERS = ["python3.14", "python3.14t"]


def readme_wheel_table():
    """{README platform name: ((first_major, first_minor), (last_major, last_minor))}."""
    promised = {}
    for line in (REPO / "README.md").read_text().splitlines():
        m = README_ROW.match(line)
        if m:
            promised[m["platform"].strip()] = (
                (int(m["first"]), int(m["first_minor"])),
                (int(m["last"]), int(m["last_minor"])),
            )
    assert len(promised) == 3, (
        f"README wheel table not recognised (found {sorted(promised)}): it "
        "must keep its three `| Platform | Architectures | CPython versions |` "
        "rows, or this guard needs the new shape"
    )
    return promised


def versions(first, last):
    """Every `major.minor` from `first` to `last`, inclusive."""
    out = set()
    v = first
    while v <= last:
        out.add(f"{v[0]}.{v[1]}")
        v = (v[0], v[1] + 1) if v[1] < 99 else (v[0] + 1, 0)
    return out


def build_matrix():
    """`publish-python.yml`'s `jobs.build.strategy.matrix.include` entries."""
    workflow = yaml.safe_load((REPO / ".github/workflows/publish-python.yml").read_text())
    return workflow["jobs"]["build"]["strategy"]["matrix"]["include"]


def build_wheel_args():
    """The `Build wheel` step's maturin args (a templated GitHub expression)."""
    workflow = yaml.safe_load((REPO / ".github/workflows/publish-python.yml").read_text())
    for step in workflow["jobs"]["build"]["steps"]:
        if step.get("name") == "Build wheel":
            return step["with"]["args"]
    raise AssertionError("publish-python.yml lost its 'Build wheel' step")


def test_macos_and_windows_build_one_abi3_wheel_matching_the_readme():
    # Exactly ONE matrix entry per non-Linux target, with no per-CPython
    # pin: the abi3-py39 wheel covers the whole promised range.
    promised = readme_wheel_table()
    matrix = build_matrix()
    for name, target in TARGETS.items():
        entries = [e for e in matrix if str(e.get("target")) == target]
        assert len(entries) == 1, f"{name} ({target}): expected one abi3 entry, got {entries}"
        assert "python" not in entries[0], (
            f"{name}: a per-version `python` pin alongside the abi3 scheme "
            f"drifts exactly like issue #35: {entries[0]}"
        )


def test_abi3_feature_wiring():
    # The single-wheel-per-platform scheme rests on the abi3-py39 feature
    # reaching every CPython wheel build — and NOT reaching the
    # free-threaded one, and NOT being set unconditionally in Cargo.toml:
    # with the feature on, maturin tags a 3.14t build cp39-abi3 instead of
    # cp314-cp314t (pyo3 compiles a correct version-specific artifact, the
    # wheel tag is wrong), so the cp314t build must not carry it.
    args = build_wheel_args()
    assert "matrix.abi3 && '--features pyo3/abi3-py39'" in args, args
    assert "--find-interpreter" in args, args  # the macOS/Windows default
    cargo = (REPO / "crates/casacure-python/Cargo.toml").read_text()
    assert 'features = ["extension-module", "abi3-py39"]' not in cargo, (
        "abi3-py39 must be passed per build in publish-python.yml, not set "
        "unconditionally in Cargo.toml (it mis-tags the 3.14t wheel)"
    )


def test_linux_builds_both_arches_abi3_and_freethreaded_no_pypy():
    # Each Linux architecture builds exactly two wheels by explicit
    # interpreter: the abi3 wheel (python3.14) and the free-threaded one
    # (python3.14t). Explicit interpreters keep the manylinux image's PyPy
    # out — PyPy does not support abi3 and is deliberately not built.
    matrix = build_matrix()
    linux = [e for e in matrix if str(e.get("target")) in LINUX_TARGETS]
    assert {str(e["target"]) for e in linux} == LINUX_TARGETS, linux
    assert all("manylinux" in e for e in linux), linux
    for target in LINUX_TARGETS:
        entries = [e for e in linux if e["target"] == target]
        interpreters = sorted(str(e["interpreters"]) for e in entries)
        assert interpreters == sorted(LINUX_INTERPRETERS), (target, entries)
        assert {bool(e.get("abi3")) for e in entries} == {True, False}, entries
    # The sdist is produced exactly once, from the x86_64 Linux abi3 job.
    sdist = [e for e in linux if e.get("sdist")]
    assert [e["target"] for e in sdist] == ["x86_64-unknown-linux-gnu"], sdist
    assert all(e.get("abi3") for e in sdist), sdist


def test_requires_python_brackets_every_promised_wheel():
    # pip never attempts an interpreter outside requires-python, so the
    # README's full range must be installable, and every platform's promise
    # must sit inside it.
    promised = readme_wheel_table()
    spec = re.search(
        r'requires-python\s*=\s*"(?P<spec>[^"]+)"', (REPO / "pyproject.toml").read_text()
    )
    assert spec, "pyproject.toml lost its requires-python"
    lower_m = re.search(r">=\s*(\d+)\.(\d+)", spec["spec"])
    upper_m = re.search(r"<\s*(\d+)\.(\d+)", spec["spec"])
    assert lower_m and upper_m, spec["spec"]
    lower = (int(lower_m.group(1)), int(lower_m.group(2)))
    upper = (int(upper_m.group(1)), int(upper_m.group(2)))

    # Every interpreter pip will attempt: lower <= v < upper.
    installable = {
        v
        for v in versions(lower, upper)
        if tuple(map(int, v.split("."))) < upper
    }

    # Every platform row is the full wheel range: it must equal
    # requires-python (both directions — a version pip installs must get a
    # wheel everywhere, and a built wheel must be installable). With the
    # abi3 scheme the rows are uniform, so all three must agree.
    for name, (first, last) in promised.items():
        assert versions(first, last) == installable, (
            f"requires-python {spec['spec']} vs README {name} {first} – {last}"
        )
