"""`casacure.images` coordinate systems: the default template, the
`toworld`/`topixel` projections, and the `coordinates` object DDFacet
mutates before re-creating an image.

Consumer call-sites guarded here:

* DDFacet ``ClassCasaImage.py`` — ``coordsys = image.coordinates()``, then
  ``get_increment``/``set_increment`` and ``get_referencevalue``/
  ``set_referencevalue`` to move a scratch image onto a target grid, and
  ``image(imagename=..., shape=..., coordsys=coordsys)`` to persist it.
* DDFacet ``Restore.py`` / ``casapy2bbs.py`` — the per-pixel
  ``for y, x: toworld((0, 0, y, x))`` loop and its inverse ``topixel``.
* DDFacet ``MyCasapy2bbs.py`` — ``coordinates().__dict__["_csys"]``.
* killMS ``MakeModelImage.py`` — ``toworld``/``topixel`` on a model cube.

The casacore reference values in the constants below were measured against
python-casacore 3.8 against the same shapes; ``needs_casacore`` re-measures
them live so the table cannot silently drift.
"""

import numpy as np
import pytest

from casacore.tables import table  # noqa: F401  (the shim: casacure.tables)

ct = pytest.importorskip("casacure.images")

CASACORE_AVAILABLE = False
try:
    from casacore.images import image as casacore_image

    CASACORE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    casacore_image = None

needs_casacore = pytest.mark.skipif(
    not CASACORE_AVAILABLE, reason="real python-casacore not installed"
)

FIXTURES = "tests/fixtures"


def make(tmp_path, shape, name="scratch.image"):
    """A default-coordinate image of `shape` (numpy order)."""
    path = str(tmp_path / name)
    ct.image(imagename=path, shape=shape)
    return ct.image(path)


def coords_of(tmp_path, shape, name="scratch.image"):
    return make(tmp_path, shape, name).coordinates()


def as_list(value):
    """Dict values arrive as numpy arrays (numbers) or lists (strings)."""
    return value.tolist() if hasattr(value, "tolist") else list(value)


# ---------------------------------------------------------------------------
# The default coordinate template: casacore's CoordinateUtil::defaultCoords
# ---------------------------------------------------------------------------

# shape -> [(coordinate name, numpy-order image axes)], measured live from
# casacore's `dict()[name]["_image_axes"]`.  These are the *numpy* axes the
# coordinate spans (casa pixel axis p is numpy axis ndim-1-p).
DEFAULT_LAYOUT = [
    ((9,), [("spectral0", [0])]),
    ((9, 8), [("direction0", [0, 1])]),
    ((2, 9, 8), [("direction0", [1, 2]), ("stokes1", [0])]),
    (
        (3, 2, 9, 8),
        [("direction0", [2, 3]), ("stokes1", [1]), ("spectral2", [0])],
    ),
    (
        (4, 3, 2, 9, 8),
        [
            ("direction0", [3, 4]),
            ("stokes1", [2]),
            ("spectral2", [1]),
            ("linear3", [0]),
        ],
    ),
    (
        (3, 2, 1, 9, 8),
        [
            ("direction0", [3, 4]),
            ("stokes1", [2]),
            ("spectral2", [1]),
            ("linear3", [0]),
        ],
    ),
]

# shape -> {coordinate name: casa pixel axes}, i.e. the stored `pixelmapN`.
CASA_PIXEL_AXES = {
    (9,): {"spectral0": [0]},
    (9, 8): {"direction0": [0, 1]},
    (2, 9, 8): {"direction0": [0, 1], "stokes1": [2]},
    (3, 2, 9, 8): {"direction0": [0, 1], "stokes1": [2], "spectral2": [3]},
    (4, 3, 2, 9, 8): {
        "direction0": [0, 1],
        "stokes1": [2],
        "spectral2": [3],
        "linear3": [4],
    },
}


def test_create_requires_an_imagename_even_with_a_shape():
    """pyrap's `image(shape=...)` alone is not casacure's contract: the
    create form needs an explicit destination."""
    with pytest.raises(ValueError, match="image\\(\\) needs a path"):
        ct.image(shape=(5, 7))


@pytest.mark.parametrize("shape,layout", DEFAULT_LAYOUT)
def test_default_coordinate_layout(tmp_path, shape, layout):
    """One coordinate per axis group, annotated with the numpy axes it spans.

    A 2-D image gets a bare direction coordinate: emitting stokes/spectral
    unconditionally put three coordinates on pixel axes 0/1, so `tofits`
    wrote the spectral and stokes world values over the direction ones and a
    default image did not round-trip through `open`.
    """
    d = coords_of(tmp_path, shape).dict()
    prefix = ("direction", "stokes", "spectral", "linear")
    present = sorted(k for k in d if k.startswith(prefix))
    assert present == sorted(name for name, _ in layout)
    for name, axes in layout:
        assert as_list(d[name]["_image_axes"]) == axes, name


@pytest.mark.parametrize("shape,pixel_axes", sorted(CASA_PIXEL_AXES.items()))
def test_default_coordinate_casa_pixel_axes(tmp_path, shape, pixel_axes):
    """The stored `pixelmapN` maps each coordinate onto casa pixel axes; a
    record that fails to parse here silently collapses every non-direction
    coordinate onto casa axis 0."""
    d = coords_of(tmp_path, shape).dict()
    for name, axes in pixel_axes.items():
        assert as_list(d[f"pixelmap{name[-1]}"]) == axes, name
        assert as_list(d[f"worldmap{name[-1]}"]) == axes, name


def test_default_2d_has_no_spectral_or_stokes(tmp_path):
    """The regression that motivated the layout rule."""
    d = coords_of(tmp_path, (5, 7)).dict()
    assert "spectral2" not in d and "stokes1" not in d
    assert sorted(k for k in d if k.startswith("direction")) == ["direction0"]


def test_default_1d_is_a_bare_spectral_coordinate(tmp_path):
    """casacore's defaultCoords puts a lone spectral coordinate on a 1-axis
    image (there is no axis pair for a direction coordinate)."""
    d = coords_of(tmp_path, (9,)).dict()
    assert "direction0" not in d
    assert as_list(d["spectral0"]["_image_axes"]) == [0]


@pytest.mark.parametrize(
    "shape,crpix",
    [
        ((5, 7), [3.0, 2.0]),  # nx=7, ny=5
        ((9, 8), [4.0, 4.0]),  # nx=8, ny=9
        ((2, 5), [2.0, 1.0]),  # nx=5, ny=2
        ((3, 2, 9, 8), [4.0, 4.0]),
    ],
)
def test_default_direction_reference_pixel_is_integer_half_shape(tmp_path, shape, crpix):
    """`crpix = shape/2` in integer arithmetic, not a half-pixel centre.

    With `3.5/2.5` a 5x7 default image emitted `CRPIX1=4.5 CRPIX2=3.5`
    instead of casacore's `4.0/3.0`, so the direction reference moved by
    half a pixel on every `tofits`.
    """
    d = coords_of(tmp_path, shape).dict()
    assert as_list(d["direction0"]["crpix"]) == crpix


def test_default_direction_world_parameters(tmp_path):
    """J2000/SIN in arcmin, `crval = [0, 0]`, `cdelt = [-1, 1]`."""
    d = coords_of(tmp_path, (5, 7)).dict()
    d0 = d["direction0"]
    assert as_list(d0["crval"]) == [0.0, 0.0]
    assert as_list(d0["cdelt"]) == [-1.0, 1.0]
    assert as_list(d0["units"]) == ["'", "'"]
    assert d0["projection"] == "SIN"
    assert d0["system"] == "J2000"
    assert as_list(d0["axes"]) == ["Right Ascension", "Declination"]


def test_default_stokes_coordinate(tmp_path):
    """The third axis is Stokes, named `I` at pixel 0 with `cdelt = 1`."""
    d = coords_of(tmp_path, (2, 9, 8)).dict()
    s = d["stokes1"]
    assert as_list(s["crval"]) == [1.0]
    assert as_list(s["crpix"]) == [0.0]
    assert as_list(s["cdelt"]) == [1.0]
    assert as_list(s["_image_axes"]) == [0]  # numpy axis 0 = polarisation


def test_default_spectral_coordinate_is_frequency(tmp_path):
    """The fourth axis is frequency, in Hz, on numpy axis 0 (channel)."""
    d = coords_of(tmp_path, (3, 2, 9, 8)).dict()
    s = d["spectral2"]
    assert as_list(s["_image_axes"]) == [0]
    # casacore nests the WCS parameters under `wcs`; casacure also exposes
    # them flat (a superset, so both consumer spellings work).
    assert s["wcs"]["crval"] == pytest.approx(1.415e9)
    assert s["wcs"]["cdelt"] == pytest.approx(1000.0)
    assert s["wcs"]["ctype"].startswith("FREQ")
    assert s["unit"] == "Hz"


def test_default_linear_coordinate_reference_pixel(tmp_path):
    """A 5th casa axis gets `linear{n}` numbered by coordinate position (3),
    not by its pixel axis (4), with `crpix = axis_length/2`."""
    for shape, crpix in (((4, 3, 2, 9, 8), 2.0), ((3, 2, 1, 9, 8), 1.0)):
        d = coords_of(tmp_path, shape).dict()
        assert "linear3" in d and "linear4" not in d
        assert as_list(d["linear3"]["crpix"]) == [crpix]
        assert as_list(d["linear3"]["_image_axes"]) == [0]


def test_default_coordinates_round_trip_through_the_record(tmp_path):
    """`create_casa_image` stores the record and `open` parses it back; the
    pixel-axis assignment must survive, or every re-opened image collapses
    its non-direction coordinates onto casa axis 0."""
    path = str(tmp_path / "rt.image")
    source = ct.image(imagename=path, shape=(3, 2, 9, 8))
    reopened = ct.image(path)
    for got, want in (
        (reopened.coordinates().dict(), source.coordinates().dict()),
    ):
        for name in ("direction0", "stokes1", "spectral2"):
            assert as_list(got[name]["_image_axes"]) == as_list(want[name]["_image_axes"])


def test_default_coordinates_survive_a_saveas(tmp_path):
    """saveas re-creates the table; the layout must not shift."""
    src = make(tmp_path, (3, 2, 6, 7), "src.image")
    out = str(tmp_path / "copy.image")
    src.saveas(out)
    d = ct.image(out).coordinates().dict()
    assert as_list(d["direction0"]["_image_axes"]) == [2, 3]
    assert as_list(d["stokes1"]["_image_axes"]) == [1]
    assert as_list(d["spectral2"]["_image_axes"]) == [0]


# ---------------------------------------------------------------------------
# toworld / topixel
# ---------------------------------------------------------------------------


def test_default_reference_pixel_maps_to_crval(tmp_path):
    """Pixel == crpix is the reference; casacore returns [0, 0] here too."""
    im = make(tmp_path, (5, 7))
    assert im.toworld((2, 3)) == pytest.approx((0.0, 0.0), abs=1e-15)
    assert im.topixel((0.0, 0.0)) == pytest.approx((2.0, 3.0), abs=1e-12)


def test_default_toworld_topixel_round_trip(tmp_path):
    """Restore.py's per-pixel loop feeds toworld output straight into
    topixel; the round trip must land back on the pixel."""
    im = make(tmp_path, (5, 7))
    for px in [(2.0, 3.0), (2.4, 3.0), (2.0, 3.6), (1.7, 2.2)]:
        back = im.topixel(im.toworld(px))
        assert back == pytest.approx(px, abs=1e-9)


def test_toworld_topixel_round_trip_on_the_fixture_cube():
    """A real 4-D cube (radian direction units), the DDFacet case."""
    im = ct.image(f"{FIXTURES}/image.image")
    for px in [(0, 0, 0, 0), (1, 1, 2, 3), (2, 0, 7, 9), (1, 1, 4, 5)]:
        back = im.topixel(im.toworld(px))
        assert back == pytest.approx(px, abs=1e-6)


def test_toworld_returns_frequency_stokes_dec_ra():
    """The positional unpacking DDFacet's Restore.py relies on:
    `freq, stokes, dec, ra = im.toworld((0, 0, y, x))`."""
    im = ct.image(f"{FIXTURES}/image.image")
    freq, stokes, dec, ra = im.toworld((0, 0, 0, 0))
    assert freq == pytest.approx(1.4e9)
    assert stokes == pytest.approx(1.0)
    assert dec == pytest.approx(-0.007855552430289316, abs=1e-15)
    assert ra == pytest.approx(0.030545007293006025, abs=1e-15)


def test_toworld_tracks_the_channel_axis():
    """The frequency must move with the channel index (numpy axis 0)."""
    im = ct.image(f"{FIXTURES}/image.image")
    freq0 = im.toworld((0, 0, 4, 5))[0]
    freq1 = im.toworld((1, 0, 4, 5))[0]
    assert freq1 - freq0 == pytest.approx(2.0e6)


def test_toworld_tracks_the_stokes_axis():
    im = ct.image(f"{FIXTURES}/image.image")
    assert im.toworld((0, 0, 4, 5))[1] == pytest.approx(1.0)
    assert im.toworld((0, 1, 4, 5))[1] == pytest.approx(2.0)


def test_topixel_outside_the_sin_hemisphere_raises():
    """The projection has no solution past the tangent plane's horizon; the
    error must be a message, not a NaN."""
    im = ct.image(f"{FIXTURES}/image.image")
    with pytest.raises(RuntimeError, match="outside the projection hemisphere"):
        im.topixel((1.4e9, 1.0, 1.57, 0.03))


def test_toworld_rejects_a_wrong_length_pixel(tmp_path):
    im = make(tmp_path, (3, 2, 8, 10))
    with pytest.raises(RuntimeError, match="pixel/world vector length 2, image has 4 axes"):
        im.toworld((0, 0))
    with pytest.raises(RuntimeError, match="pixel/world vector length 3, image has 4 axes"):
        im.topixel((1.4e9, 1.0, 0.0))


def test_toworld_requires_a_tuple(tmp_path):
    """pyrap's C++ signature takes an IPosition; a list is not accepted."""
    im = make(tmp_path, (5, 7))
    with pytest.raises(TypeError, match="cannot be cast as 'tuple'"):
        im.toworld([2, 3])
    with pytest.raises(TypeError, match="must be real number"):
        im.toworld(("a", "b"))


def test_toworld_on_an_image_without_coordinates(tmp_path):
    """A FITS file with no WCS cards has an empty coordinate system: there
    is no world value for any axis, so `toworld` reports zeros rather than
    raising or echoing the pixel."""
    import struct

    path = tmp_path / "bare.fits"

    def card(text):
        return text.ljust(80).encode()[:80]

    def num(key, value):
        return card(f"{key:<8}= {value:>20}")

    body = b"".join(
        [
            num("SIMPLE", "T"),
            num("BITPIX", -32),
            num("NAXIS", 2),
            num("NAXIS1", 2),
            num("NAXIS2", 2),
            card("END"),
        ]
    )
    path.write_bytes(body + b" " * ((-len(body)) % 2880) + struct.pack(">4f", 1, 2, 3, 4))
    im = ct.image(str(path))
    d = im.coordinates().dict()
    assert not [k for k in d if k.startswith(("direction", "stokes", "spectral"))]
    assert im.toworld((1.0, 0.0)) == pytest.approx((0.0, 0.0))


# ---------------------------------------------------------------------------
# The `coordinates` object
# ---------------------------------------------------------------------------


def test_coordinates_cannot_be_constructed_directly(tmp_path):
    c = coords_of(tmp_path, (5, 7))
    with pytest.raises(TypeError):
        type(c)()


def test_coordinates_unknown_attribute_raises(tmp_path):
    c = coords_of(tmp_path, (5, 7))
    with pytest.raises(AttributeError):
        c.no_such_method


def test_coordinates_dict_is_a_deep_copy(tmp_path):
    """Callers mutate the dict they get back; the image must not change."""
    im = make(tmp_path, (5, 7))
    d = im.coordinates().dict()
    d["direction0"]["cdelt"][0] = 99.0
    assert im.coordinates().dict()["direction0"]["cdelt"][0] == pytest.approx(-1.0)


def test_coordinates_snapshots_are_independent_of_the_image(tmp_path):
    """DDFacet builds a target coordsys from one snapshot, rewrites it, and
    persists it with `image(coordsys=...)`; the source image must keep its
    own values until `image(...)` is called."""
    im = make(tmp_path, (3, 2, 8, 10))
    edited = im.coordinates()
    edited.set_increment([[1.0e6], [1.0], [0.5, 0.5]])
    assert edited.get_increment()[0] == pytest.approx(1.0e6)
    assert im.coordinates().get_increment()[0] == pytest.approx(1000.0)


def test_csys_private_dict_is_exposed(tmp_path):
    """`MyCasapy2bbs.py` reads `coordinates().__dict__["_csys"]`."""
    im = make(tmp_path, (5, 7))
    csys = im.coordinates()._csys
    assert isinstance(csys, dict)
    assert as_list(csys["direction0"]["crpix"]) == [3.0, 2.0]


def test_csys_private_dict_is_a_copy(tmp_path):
    im = make(tmp_path, (5, 7))
    im.coordinates()._csys["direction0"]["crval"][0] = 5.0
    assert as_list(im.coordinates()._csys["direction0"]["crval"]) == [0.0, 0.0]


def test_get_increment_layout_is_reverse_coordinate_order():
    """pyrap's layout for `incr[-1]` indexing: one entry per coordinate in
    reverse CS order — spectral (scalar), stokes, direction."""
    im = ct.image(f"{FIXTURES}/image.image")
    inc = im.coordinates().get_increment()
    assert len(inc) == 3
    assert float(np.asarray(inc[0])) == pytest.approx(2.0e6)
    assert as_list(inc[1]) == [1.0]
    assert len(inc[2]) == 2


def test_set_increment_round_trips_and_moves_the_world_grid():
    """ClassCasaImage retargets a scratch image by setting the increments."""
    im = ct.image(f"{FIXTURES}/image.image")
    c = im.coordinates()
    original = c.get_increment()
    c.set_increment([[2.5e6], [1.0], list(np.asarray(original[2]))])
    assert c.get_increment()[0] == pytest.approx(2.5e6)
    # Restore the fixture's own value.
    c.set_increment([[2.0e6], [1.0], list(np.asarray(original[2]))])
    assert c.get_increment()[0] == pytest.approx(2.0e6)


def test_get_referencevalue_and_referencepixel_layout():
    im = ct.image(f"{FIXTURES}/image.image")
    c = im.coordinates()
    refval = c.get_referencevalue()
    refpix = c.get_referencepixel()
    assert len(refval) == len(refpix) == 3
    assert as_list(refval[2]) == pytest.approx([0.030543261909900768, -0.007853981633974483])
    assert as_list(refpix[2]) == pytest.approx([4.0, 3.0])


def test_set_increment_reaches_the_dict_and_the_private_csys():
    """The setters used to move `get_increment()` while `dict()`/`_csys` kept
    the values the record was read with, so the two views of the same
    coordinate disagreed."""
    c = ct.image(f"{FIXTURES}/image.image").coordinates()
    c.set_increment([[2.5e6], [1.0], [-4.0e-7, 5.0e-7]])
    assert float(np.asarray(c.get_increment()[0])) == pytest.approx(2.5e6)
    assert as_list(c.dict()["direction0"]["cdelt"]) == pytest.approx([-4.0e-7, 5.0e-7])
    assert as_list(c._csys["direction0"]["cdelt"]) == pytest.approx([-4.0e-7, 5.0e-7])


def test_set_referencevalue_reaches_the_dict():
    c = ct.image(f"{FIXTURES}/image.image").coordinates()
    c.set_referencevalue([[1.4e9], [1.0], [0.1, 0.2]])
    assert as_list(c.get_referencevalue()[2]) == pytest.approx([0.1, 0.2])
    assert as_list(c.dict()["direction0"]["crval"]) == pytest.approx([0.1, 0.2])
    assert as_list(c._csys["direction0"]["crval"]) == pytest.approx([0.1, 0.2])


def test_set_referencepixel_reaches_the_dict():
    c = ct.image(f"{FIXTURES}/image.image").coordinates()
    c.set_referencepixel([[0.0], [0.0], [1.0, 2.0]])
    assert as_list(c.get_referencepixel()[2]) == pytest.approx([1.0, 2.0])
    assert as_list(c.dict()["direction0"]["crpix"]) == pytest.approx([1.0, 2.0])
    assert as_list(c._csys["direction0"]["crpix"]) == pytest.approx([1.0, 2.0])


def test_a_noop_set_keeps_the_stored_record_intact():
    """A setter that changes nothing must not discard the record's extra
    casacore fields (`conversionSystem`, the frame measures, ...): the
    record is only dropped when a value actually moves."""
    c = ct.image(f"{FIXTURES}/image.image").coordinates()
    before = sorted(c.dict()["direction0"])
    current = [np.asarray(v).ravel().tolist() for v in c.get_increment()]
    c.set_increment(current)
    assert sorted(c.dict()["direction0"]) == before
    assert "conversionSystem" in c.dict()["direction0"]


def test_mutation_flows_into_a_created_image(tmp_path):
    """DDFacet's retarget-and-create: the mutated coordsys must be what gets
    persisted, not the values the record was read with."""
    target = ct.image(f"{FIXTURES}/image.image").coordinates()
    target.set_increment([[1.0e6], [1.0], [-2.0e-7, 4.0e-7]])
    out = str(tmp_path / "retargeted.image")
    ct.image(imagename=out, shape=(2, 1, 6, 7), coordsys=target)
    made = ct.image(out).coordinates()
    assert as_list(made.dict()["direction0"]["cdelt"]) == pytest.approx([-2.0e-7, 4.0e-7])
    assert float(np.asarray(made.get_increment()[0])) == pytest.approx(1.0e6)


def test_coordsys_object_reused_by_image_create(tmp_path):
    """DDFacet's create: build a coordsys from a template, mutate it, and
    hand it to `image(imagename=, shape=, coordsys=)`."""
    src = ct.image(f"{FIXTURES}/image.image")
    c = src.coordinates()
    c.set_increment([[2.5e6], [1.0], [-4.0e-7, 5.0e-7]])
    out = str(tmp_path / "target.image")
    ct.image(imagename=out, shape=(2, 1, 6, 7), coordsys=c)
    made = ct.image(out)
    assert list(made.shape()) == [2, 1, 6, 7]
    assert made.coordinates().dict()["direction0"]["cdelt"] == pytest.approx([-4.0e-7, 5.0e-7])


# ---------------------------------------------------------------------------
# The angle unit in the record
# ---------------------------------------------------------------------------


def test_default_template_units_are_honoured_by_toworld(tmp_path):
    """The default template stores `cdelt = -1'` (arcmin).  Reading it as
    `-1 rad` put 0.4 rad of RA on a 0.4-pixel offset, so a default image's
    `tofits` (which writes degrees) did not round-trip through `open`."""
    im = make(tmp_path, (5, 7))
    ra = im.toworld((2.4, 3.0))[0]
    # 0.4 pixels * 1 arcmin = 0.4/60 degrees, through the SIN projection.
    assert ra == pytest.approx(np.arcsin(np.deg2rad(0.4 / 60.0)), abs=1e-15)
    assert abs(ra) < 1e-3  # not 0.4 rad


def test_tofits_then_open_preserves_the_world_grid(tmp_path):
    """The D3 acceptance property: a default image's `tofits` output, read
    back, must reproduce the original world coordinates."""
    im = make(tmp_path, (3, 2, 9, 8))
    out = str(tmp_path / "rt.fits")
    im.tofits(out)
    back = ct.image(out)
    for px in [(0, 0, 4, 4), (0, 0, 4.4, 4.0), (1, 1, 5.2, 2.3)]:
        assert back.toworld(px) == pytest.approx(im.toworld(px), abs=1e-12)


def test_tofits_then_open_preserves_the_world_grid_2d(tmp_path):
    im = make(tmp_path, (5, 7))
    out = str(tmp_path / "rt2.fits")
    im.tofits(out)
    back = ct.image(out)
    for px in [(2, 3), (2.4, 3.0), (1.8, 2.6)]:
        assert back.toworld(px) == pytest.approx(im.toworld(px), abs=1e-12)


def test_radian_records_are_unaffected_by_the_unit_conversion():
    """A real image's direction record is already radians; converting it
    must be a no-op (the fixture's answers are casacore's)."""
    im = ct.image(f"{FIXTURES}/image.image")
    assert as_list(im.coordinates().dict()["direction0"]["units"]) == ["rad", "rad"]
    assert im.toworld((0, 0, 0, 0))[3] == pytest.approx(0.030545007293006025, abs=1e-15)


# ---------------------------------------------------------------------------
# Parity with real python-casacore
# ---------------------------------------------------------------------------


@needs_casacore
@pytest.mark.parametrize("shape,layout", DEFAULT_LAYOUT)
def test_default_coordinates_match_casacore(tmp_path, shape, layout):
    """The whole default template, re-measured against casacore."""
    reference = casacore_image(imagename=str(tmp_path / "ref.image"), shape=list(shape))
    mine = make(tmp_path, shape)
    want, got = reference.coordinates().dict(), mine.coordinates().dict()
    for name, _ in layout:
        for field in ("crval", "crpix", "cdelt", "_image_axes"):
            w, g = want[name].get(field), got[name].get(field)
            if w is None:  # casacore nests spectral parameters under `wcs`
                continue
            assert as_list(g) == pytest.approx(as_list(w)), (shape, name, field)


@needs_casacore
def test_default_2d_reference_pixel_matches_casacore(tmp_path):
    """At the reference pixel the default template agrees exactly."""
    reference = casacore_image(imagename=str(tmp_path / "ref.image"), shape=[5, 7])
    mine = make(tmp_path, (5, 7))
    assert as_list(reference.toworld((2, 3))) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert as_list(mine.toworld((2, 3))) == pytest.approx(reference.toworld((2, 3)), abs=1e-12)


@needs_casacore
def test_default_2d_toworld_deliberately_diverges_off_reference(tmp_path):
    """Documented divergence on the degenerate default template.

    casacore's `toworld` treats the template's `cdelt = -1'` as radians, so
    a 0.4-pixel offset becomes 0.4 *rad* of RA and 1.6 pixels overflows the
    SIN tangent plane to ~21600 rad.  casacure honours the record's arcmin
    unit, which is what makes `tofits` (degrees) round-trip through `open`.
    The stored records and the emitted FITS cards are identical either way;
    only this meaningless template differs, and no consumer integrates over
    the default grid before replacing its coordinates.
    """
    reference = casacore_image(imagename=str(tmp_path / "ref.image"), shape=[5, 7])
    mine = make(tmp_path, (5, 7))
    assert as_list(reference.toworld((2.4, 3.0))) == pytest.approx([0.4, 0.0], abs=1e-6)
    assert abs(as_list(reference.toworld((2.0, 3.6)))[1]) > 1.0e4
    got = as_list(mine.toworld((2.4, 3.0)))
    assert got[0] == pytest.approx(np.arcsin(np.deg2rad(0.4 / 60.0)), abs=1e-15)
    # numpy order: index 0 is dec (casa axis 1), index 1 is RA (casa axis 0).
    ra = as_list(mine.toworld((2.0, 3.6)))[1]
    assert abs(ra) == pytest.approx(np.arcsin(np.deg2rad(0.6 / 60.0)), abs=1e-15)


@needs_casacore
def test_fixture_coordinates_match_casacore():
    reference = casacore_image(f"{FIXTURES}/image.image")
    mine = ct.image(f"{FIXTURES}/image.image")
    want, got = reference.coordinates().dict(), mine.coordinates().dict()
    for name in ("direction0", "stokes1", "spectral2"):
        for field in ("crval", "crpix", "cdelt", "_image_axes"):
            w, g = want[name].get(field), got[name].get(field)
            if w is None or g is None:
                continue
            assert as_list(g) == pytest.approx(as_list(w)), (name, field)


@needs_casacore
def test_fixture_toworld_and_topixel_match_casacore():
    reference = casacore_image(f"{FIXTURES}/image.image")
    mine = ct.image(f"{FIXTURES}/image.image")
    for px in [(0, 0, 0, 0), (1, 1, 2, 3), (2, 0, 7, 9)]:
        assert as_list(mine.toworld(px)) == pytest.approx(reference.toworld(px), abs=1e-12)
        world = tuple(as_list(reference.toworld(px)))
        assert as_list(mine.topixel(world)) == pytest.approx(reference.topixel(world), abs=1e-9)
