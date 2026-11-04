"""The published-wheel matrix must cover every platform and interpreter the
README promises (issue #35: `publish-python.yml`'s macOS matrix started at
3.11, so `pip install casacure` on macOS/Python 3.10 silently found no wheel
while the README advertised 3.10 – 3.14).

Three artifacts have to agree, and each is edited by a different habit of
mind:

* the README's "prebuilt wheels" table (what users are told),
* `.github/workflows/publish-python.yml`'s matrix (what is actually built),
* `pyproject.toml`'s `requires-python` (what pip will even attempt).

Issue #35 was exactly this drifting apart unnoticed.  These tests are pure
text checks — no network, no wheel build.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# A data row of the README wheel table, e.g.
#   | macOS | `aarch64` (Apple Silicon) | 3.10 – 3.14 |
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


def test_macos_and_windows_build_exactly_what_the_readme_promises():
    # The exact regression of issue #35: a non-Linux matrix entry quietly
    # missing a CPython the README names (macOS started at 3.11 there).
    promised = readme_wheel_table()
    matrix = build_matrix()
    for name, target in TARGETS.items():
        first, last = promised[name]
        want = versions(first, last)
        got = {str(e.get("python")) for e in matrix if e.get("target") == target}
        assert got == want, (
            f"{name} ({target}): README promises {sorted(want)} but "
            f"publish-python.yml builds {sorted(got)}"
        )


def test_linux_builds_both_readme_architectures_through_manylinux():
    # Linux versions are not in the matrix: the manylinux image's
    # `--find-interpreter` pass covers the whole range in one job per
    # architecture (README's note), so guard the targets and that pass.
    promised = readme_wheel_table()
    matrix = build_matrix()
    linux = [e for e in matrix if str(e.get("target")) in LINUX_TARGETS]
    assert {str(e["target"]) for e in linux} == LINUX_TARGETS, (
        f"README promises {sorted(LINUX_TARGETS)}: {linux}"
    )
    assert all("manylinux" in e for e in linux), linux
    # The sdist is produced exactly once, from the x86_64 Linux job.
    sdist = [e for e in linux if e.get("sdist")]
    assert [e["target"] for e in sdist] == ["x86_64-unknown-linux-gnu"], sdist


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

    # The Linux row is the full wheel range: it must equal requires-python
    # (both directions — a version pip installs must get a wheel somewhere,
    # and a built wheel must be installable).
    first, last = promised["Linux (manylinux)"]
    assert versions(first, last) == installable, (
        f"requires-python {spec['spec']} vs README Linux {first} – {last}"
    )

    # Every platform promise lies inside the installable range.
    for name, (first, last) in promised.items():
        outside = sorted(versions(first, last) - installable)
        assert not outside, f"{name} promises {outside}, outside requires-python {spec['spec']}"
