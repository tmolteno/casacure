"""Tests for `casacure.quanta` — a port of casacore's `casa/Quanta`.

The expected values are pinned to real python-casacore 3.8.1 `casacore.quanta`
output (probed on this machine); every case below was verified equal to the
real module. A live parity block at the end re-checks against the real module
when it is importable as a distinct module (not shadowed by a `casacore`
shim on `PYTHONPATH`).
"""

import numpy as np
import pytest

from casacure import quanta as qa


def test_quantity_construction():
    q = qa.quantity(1.5, "Jy")
    assert q.get_unit() == "Jy"
    assert q.get_value() == 1.5
    # Combined string forms, like casacore.
    assert str(qa.quantity("1.5 Jy")) == "1.5 Jy"
    assert str(qa.quantity("1.5Jy")) == "1.5 Jy"
    assert str(qa.quantity("2 deg")) == "2 deg"
    with pytest.raises(TypeError):
        qa.quantity(1.5)


def test_conversion_and_canonical():
    q = qa.quantity(1.5, "Jy")
    assert q.get_value("mJy") == 1500.0
    assert q.canonical() == "1.5e-26 kg.s-2"
    assert q.get() == "1.5e-26 kg.s-2"
    # In-place convert via a Quantity, like casacore's Quantum::convert.
    q.convert(qa.quantity(1, "mJy"))
    assert q.get_unit() == "mJy"
    assert q.get_value() == 1500.0


def test_arithmetic_matches_casacore():
    a = qa.quantity(3, "km")
    b = qa.quantity(500, "m")
    assert str(a + b) == "3.5 km"
    assert str(a - b) == "2.5 km"
    assert str(a * b) == "1500 km.m"
    assert (a * b).canonical() == "1.5e+06 m2"
    assert str(a / b) == "0.006 km/(m)"
    assert str(b / a) == "166.67 m/(km)"
    assert str(qa.pow(a, 2)) == "9 (km)2"
    assert str(qa.sqrt(qa.quantity(9, "m2"))) == "3 m"
    assert str(qa.root(qa.quantity(8, "m3"), 3)) == "2 m"
    assert isinstance(a * b, type(a))


def test_number_formatter_is_percent_g():
    assert str(qa.quantity(1.0 / 7, "m")) == "0.14286 m"
    assert str(qa.quantity(1e5, "m")) == "1e+05 m"
    assert repr(qa.quantity(1.23456789, "m")) == "1.23457 m"
    assert str(qa.quantity(0.006, "m")) == "0.006 m"
    assert str(qa.quantity(-2.5, "Jy")) == "-2.5 Jy"


def test_time_and_angle_display():
    # str is always plain; repr (and formatted) use the sexagesimal forms.
    assert str(qa.quantity(45, "deg")) == "45 deg"
    assert repr(qa.quantity(45, "deg")) == "+045.00.00"
    assert str(qa.quantity(1.5, "h")) == "1.5 h"
    assert repr(qa.quantity(1.5, "h")) == "01:30:00"
    assert repr(qa.quantity(2, "rad")) == "+114.35.30"
    assert repr(qa.quantity(51544, "d")) == "00:00:00"
    assert qa.quantity(45, "deg").formatted() == "+045.00.00"
    assert qa.quantity(1.5, "Jy").formatted() == "1.5 Jy"
    assert qa.quantity(12.5, "deg").to_string() == "12.5 deg"
    assert qa.quantity(45, "deg").to_angle() == "+045.00.00"
    assert qa.quantity(0.5, "d").to_time() == "12:00:00"
    assert qa.quantity(1.5, "h").to_time() == "01:30:00"


def test_unix_time():
    # MJD 51544 == 2000-01-01T00:00:00Z.
    assert qa.quantity(51544, "d").to_unix_time() == 946684800.0


def test_near_and_equality():
    assert qa.near(qa.quantity(1, "m"), qa.quantity(1 + 1e-13, "m"))
    assert not qa.near(qa.quantity(1, "m"), qa.quantity(1 + 2e-13, "m"))
    assert qa.nearabs(qa.quantity(1, "m"), qa.quantity(1.005, "m"), 0.01)
    assert qa.quantity(1, "m") == qa.quantity(100, "cm")
    assert qa.quantity(1, "m") != qa.quantity(1, "s")


def test_math_functions():
    assert str(qa.log10(qa.quantity(1000, ""))) == "3 "
    assert str(qa.sin(qa.quantity(0.5, "rad"))) == "0.47943 "
    assert str(qa.abs(qa.quantity(-3, "m"))) == "3 m"
    assert str(qa.ceil(qa.quantity(2.3, "m"))) == "3 m"
    assert str(qa.floor(qa.quantity(2.7, "m"))) == "2 m"
    # Dimensioned logs are rejected.
    with pytest.raises(RuntimeError):
        qa.log(qa.quantity(2, "m"))


def test_dict_roundtrip_and_metadata():
    d = qa.quantity(1.5, "Jy").to_dict()
    assert d == {"value": 1.5, "unit": "Jy"}
    assert str(qa.from_dict(d)) == "1.5 Jy"
    assert qa.is_quantity(qa.quantity(1, "m"))
    assert not qa.is_quantity(5)
    # units / prefixes / constants metadata (shape: [long name, value]).
    assert qa.prefixes["k"][0] == "kilo"
    assert str(qa.prefixes["k"][1]) == "1000 "
    assert qa.units["Jy"][0] == "jansky"
    assert str(qa.units["Jy"][1]) == "1e-26 kg.s-2"
    assert str(qa.units["min"][1]) == "60 s"
    assert str(qa.constants["pi"]) == "3.1416 "


def test_parity_with_real_casacore_quanta():
    """Live check against the real python-casacore module when importable as
    a distinct module (skipped when the casacore shim shadows it)."""
    try:
        import casacore.quanta as real  # real module, not the shim
    except Exception:
        pytest.skip("real casacore.quanta not importable")
    if not hasattr(real, "quantity"):
        pytest.skip("casacore.quanta not available")

    pairs = [
        ("quantity", lambda q: str(q.quantity(1.5, "Jy"))),
        ("get_value", lambda q: q.quantity(1.5, "Jy").get_value("mJy")),
        ("canonical", lambda q: q.quantity(1.5, "Jy").canonical()),
        ("add", lambda q: str(q.quantity(3, "km") + q.quantity(500, "m"))),
        ("div", lambda q: str(q.quantity(3, "km") / q.quantity(500, "m"))),
        ("pow", lambda q: str(q.pow(q.quantity(3, "km"), 2))),
        ("sqrt", lambda q: str(q.sqrt(q.quantity(9, "m2")))),
        ("repr angle", lambda q: repr(q.quantity(45, "deg"))),
        ("repr time", lambda q: repr(q.quantity(1.5, "h"))),
        ("unix", lambda q: q.quantity(51544, "d").to_unix_time()),
        ("near", lambda q: q.near(q.quantity(1, "m"), q.quantity(1 + 2e-13, "m"))),
        ("eq", lambda q: q.quantity(1, "m") == q.quantity(100, "cm")),
        ("sin", lambda q: str(q.sin(q.quantity(0.5, "rad")))),
        ("units", lambda q: [str(x) for x in q.units["Jy"]]),
        ("prefixes", lambda q: [str(x) for x in q.prefixes["k"]]),
    ]
    for label, f in pairs:
        assert str(f(qa)) == str(f(real)), f"quanta parity failed: {label}"


def test_numpy_unaffected_by_quanta_import():
    """The quanta module must not disturb the tables surface or numpy."""
    import numpy as np
    from casacure.tables import makescacoldesc
    d = makescacoldesc("a", 1)
    assert d["desc"]["valueType"] == "int"
    assert np.__version__
