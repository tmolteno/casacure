"""`casacure.images` parity: what matches python-casacore, what deliberately
does not, and the end-to-end flows DDFacet and killMS actually run.

The divergences below are pinned from *both* sides so they cannot rot into
silent breakage: each test asserts casacure's contract, asserts casacore's,
and states why they differ.  Anything not listed here is expected to match,
and the other modules in this suite compare it value-for-value.

Consumer flows reproduced here:

* DDFacet ``ClassCasaImage.createScratch`` / ``MyCasapy2bbs``.
* DDFacet ``Restore.py`` + ``fits2png.py``.
* killMS ``MakeModelImage.py``.
"""

import os
import shutil

import numpy as np
import pytest

from casacore.tables import table  # noqa: F401  (the shim: casacure.tables)

ct = pytest.importorskip("casacure.images")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
IMAGE_FIXTURE = os.path.join(FIXTURES, "image.image")
FITS_FIXTURE = os.path.join(FIXTURES, "image.fits")

CASACORE_AVAILABLE = False
try:
    from casacore.images import image as casacore_image

    CASACORE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    casacore_image = None

needs_casacore = pytest.mark.skipif(
    not CASACORE_AVAILABLE, reason="real python-casacore not installed"
)


def as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def both_flavours():
    """(casacure, casacore) handles on the 4-D fixture table."""
    mine = ct.image(IMAGE_FIXTURE)
    want = casacore_image(IMAGE_FIXTURE) if CASACORE_AVAILABLE else None
    return mine, want


def make(tmp_path, shape, name="scratch.image"):
    path = str(tmp_path / name)
    ct.image(imagename=path, shape=shape)
    return ct.image(path)


# ---------------------------------------------------------------------------
# Oracle integrity
# ---------------------------------------------------------------------------


def test_the_image_fixture_raster_is_pristine():
    """The fixture is shared, read-only ground truth.

    Several operations in this suite write (``putdata`` rewrites the table
    on disk), so a test that is handed the fixture path instead of a copy
    silently corrupts every later run.  This is the canary: the generator
    (`tests/make_fixtures.py`) fills the raster with
    ``arange(nch*npol*ny*nx)``, and the FITS cube it was built from must
    still agree.
    """
    expected = np.arange(3 * 2 * 8 * 10, dtype=np.float32).reshape(3, 2, 8, 10)
    from_image = ct.image(IMAGE_FIXTURE).getdata()
    from_fits = ct.image(FITS_FIXTURE).getdata()
    np.testing.assert_array_equal(from_image, expected)
    np.testing.assert_array_equal(from_fits, expected)


@needs_casacore
def test_the_fixture_still_reads_identically_through_casacore():
    expected = np.arange(480, dtype=np.float32).reshape(3, 2, 8, 10)
    np.testing.assert_array_equal(casacore_image(IMAGE_FIXTURE).getdata(), expected)
    np.testing.assert_array_equal(casacore_image(FITS_FIXTURE).getdata(), expected)


# ---------------------------------------------------------------------------
# Documented divergences
# ---------------------------------------------------------------------------


def test_shape_returns_a_tuple():
    """casacore returns a list; casacure returns a tuple.

    `list(im.shape())` is the portable spelling and is what the rest of this
    suite uses."""
    mine, want = both_flavours()
    assert isinstance(mine.shape(), tuple)
    assert list(mine.shape()) == [3, 2, 8, 10]
    if want is not None:
        assert isinstance(want.shape(), list)


@needs_casacore
def test_coordinates_has_no_get_names():
    """casacore's `CoordinateSystem::get_names()` has no casacure
    counterpart; the same information is the coordinate record keys."""
    mine, want = both_flavours()
    assert not hasattr(mine.coordinates(), "get_names")
    assert want.coordinates().get_names() == ["spectral", "stokes", "direction"]
    names = [
        k
        for k in mine.coordinates().dict()
        if k.startswith(("direction", "stokes", "spectral", "linear"))
    ]
    assert sorted(names) == ["direction0", "spectral2", "stokes1"]


@needs_casacore
def test_direction_increment_axis_order_is_long_then_lat():
    """casacore reports the direction entry as (lat, long); casacure reports
    (long, lat), matching the order of the direction record's own
    `cdelt`/`crval`/`crpix` arrays and the `crval[k]`/`cdelt[k]` indexing the
    rest of the API uses.

    Code that indexes `incr[-1][0]` must therefore not assume one order:
    index it against `dict()["direction0"]["cdelt"]`."""
    mine, want = both_flavours()
    assert as_list(mine.coordinates().get_increment()[2]) == pytest.approx(
        as_list(mine.coordinates().dict()["direction0"]["cdelt"])
    )
    assert as_list(want.coordinates().get_increment()[2]) == pytest.approx(
        as_list(want.coordinates().dict()["direction0"]["cdelt"])[::-1]
    )


@needs_casacore
def test_spectral_parameters_are_exposed_flat_as_well_as_nested():
    """A spectral coordinate's `crval`/`crpix`/`cdelt` live under `wcs`.

    A *stored* CASA record is served back verbatim, so casacure matches
    casacore exactly there; a synthesised record (a FITS-sourced system, or
    the default template) additionally exposes the same numbers flat, so
    `dict()["spectral2"]["crval"]` happens to work as well."""
    mine, want = both_flavours()
    for label, handle in (("casacure", mine), ("casacore", want)):
        spectral = handle.coordinates().dict()["spectral2"]
        assert "crval" not in spectral, label
        assert spectral["wcs"]["crval"] == pytest.approx(1.4e9), label

    from_fits = ct.image(FITS_FIXTURE).coordinates().dict()["spectral2"]
    assert as_list(from_fits["crval"]) == pytest.approx([1.4e9])
    assert from_fits["wcs"]["crval"] == pytest.approx(1.4e9)


@needs_casacore
def test_dict_omits_axes_sizes():
    """casacore annotates each coordinate with `_axes_sizes` (the pixel
    extent of the axes it spans); casacure only provides `_image_axes`."""
    mine, want = both_flavours()
    got, reference = mine.coordinates().dict(), want.coordinates().dict()
    for name in ("direction0", "stokes1", "spectral2"):
        assert "_axes_sizes" not in got[name]
        assert "_axes_sizes" in reference[name]
    assert as_list(reference["direction0"]["_axes_sizes"]) == [10, 8]


@needs_casacore
def test_dict_omits_linear_units(tmp_path):
    """casacore's generic linear coordinate carries `units` (`['km']` in its
    default 5-axis template); casacure's does not — nothing in the images
    API converts a linear axis' world value, so the unit is not load-bearing.
    """
    reference = casacore_image(imagename=str(tmp_path / "ref.image"), shape=[4, 3, 2, 9, 8])
    mine = make(tmp_path, (4, 3, 2, 9, 8))
    assert as_list(reference.coordinates().dict()["linear3"]["units"]) == ["km"]
    assert "units" not in mine.coordinates().dict()["linear3"]


@needs_casacore
def test_fits_sourced_spectral_system_is_lsrk_not_topo():
    """casacore maps the FITS `SPECSYS` card onto its spectral frame; the
    synthesised record casacure builds for a FITS cube takes
    `SpectralCoordinate`'s own default (`LSRK`).

    A CASA table's stored `coords` record is read verbatim, so only an image
    opened straight from FITS is affected."""
    from_fits = ct.image(FITS_FIXTURE).coordinates().dict()["spectral2"]["system"]
    from_casa = ct.image(IMAGE_FIXTURE).coordinates().dict()["spectral2"]["system"]
    assert from_fits == "LSRK"
    assert from_casa == "TOPO"
    assert (
        casacore_image(FITS_FIXTURE).coordinates().dict()["spectral2"]["system"] == "TOPO"
    )


def test_stokes_letters_come_from_the_crval_code():
    """A FITS `STOKES` axis starts at the Stokes code in `CRVAL` (1 = I) and
    has one letter per plane, so the 2-plane fixture axis is `['I', 'Q']` —
    as casacore reports it, for both flavours."""
    for path in (IMAGE_FIXTURE, FITS_FIXTURE):
        assert as_list(ct.image(path).coordinates().dict()["stokes1"]["stokes"]) == [
            "I",
            "Q",
        ]


@needs_casacore
def test_stokes_letters_match_casacore_for_both_flavours():
    for path in (IMAGE_FIXTURE, FITS_FIXTURE):
        assert as_list(ct.image(path).coordinates().dict()["stokes1"]["stokes"]) == as_list(
            casacore_image(path).coordinates().dict()["stokes1"]["stokes"]
        )


def test_image_without_an_imagename_is_an_error():
    """casacore's `image()` also requires `imagename`; there is no
    unnamed-scratch form, which is why DDFacet builds a path first."""
    with pytest.raises(ValueError, match=r"image\(\) needs a path"):
        ct.image(shape=(5, 7))


# ---------------------------------------------------------------------------
# End-to-end consumer flows
# ---------------------------------------------------------------------------


def test_ddfacet_create_scratch_flow(tmp_path):
    """`ClassCasaImage.createScratch`: take a template's coordinate system,
    retarget it, create a scratch cube, fill it, persist it.

    The mutated coordsys must reach disk — the setters used to leave the
    cached record stale, so the created image came back with the *template's*
    grid and the retargeting silently did nothing.
    """
    template = ct.image(IMAGE_FIXTURE)
    target = template.coordinates()
    target.set_increment([[1.0e6], [1.0], [-2.0e-7, 4.0e-7]])
    target.set_referencevalue([[1.42e9], [1.0], [0.02, -0.01]])

    scratch_path = str(tmp_path / "scratch.image")
    ct.image(imagename=scratch_path, shape=(2, 1, 6, 7), coordsys=target)
    scratch = ct.image(scratch_path)

    assert list(scratch.shape()) == [2, 1, 6, 7]
    assert as_list(scratch.coordinates().get_increment()[2]) == pytest.approx([-2.0e-7, 4.0e-7])
    assert as_list(scratch.coordinates().dict()["direction0"]["crval"]) == pytest.approx(
        [0.02, -0.01]
    )
    # And the persisted world grid follows the mutated reference.  The
    # supplied coordsys keeps the *template's* reference pixel (casa 4, 3 =
    # numpy y=3, x=4), not the new shape's centre.  `toworld` returns numpy
    # order, so the last two entries are (dec, ra) — the reverse of the
    # record's (long, lat) `crval`.
    assert scratch.toworld((0, 0, 3, 4))[2:] == pytest.approx([-0.01, 0.02], abs=1e-12)

    # Fill the scratch cube and keep it.
    scratch.putdata(np.full((2, 1, 6, 7), 3.5, dtype=np.float32))
    final = str(tmp_path / "final.image")
    scratch.saveas(final)
    np.testing.assert_array_equal(ct.image(final).getdata(), np.full((2, 1, 6, 7), 3.5))


def test_killms_make_model_image_flow(tmp_path):
    """`MakeModelImage`: open a model (here the FITS cube), copy it to a
    CASA image, write the model raster, and read it back."""
    model = ct.image(FITS_FIXTURE)
    out = str(tmp_path / "model.image")
    model.saveas(out)

    written = ct.image(out)
    data = np.zeros((3, 2, 8, 10), dtype=np.float32)
    data[1, 0, 4, 5] = 12.5
    written.putdata(data)

    again = ct.image(out)
    np.testing.assert_array_equal(again.getdata(), data)
    assert again.unit() == model.unit()
    assert list(again.shape()) == [3, 2, 8, 10]
    # The model image's world grid is unchanged by the raster write.
    assert again.toworld((0, 0, 0, 0)) == pytest.approx(model.toworld((0, 0, 0, 0)), abs=1e-9)


def test_ddfacet_restore_and_fits2png_flow(tmp_path):
    """`Restore.py` walks the cube pixel by pixel through `toworld` and
    `topixel`; the stack is then exported and re-ingested (`fits2png`)."""
    im = ct.image(IMAGE_FIXTURE)
    for channel in range(3):
        for y in range(0, 8, 3):
            for x in range(0, 10, 4):
                freq, pol, dec, ra = im.toworld((channel, 0, y, x))
                assert freq == pytest.approx(1.4e9 + channel * 2.0e6)
                assert pol == pytest.approx(1.0)
                assert -0.1 < dec < 0.1 and 0.0 < ra < 0.1
                back = im.topixel((freq, pol, dec, ra))
                assert back == pytest.approx((channel, 0, y, x), abs=1e-6)

    exported = str(tmp_path / "restored.fits")
    im.tofits(exported)
    reingested = ct.image(exported)
    np.testing.assert_array_equal(reingested.getdata(), im.getdata())
    for px in [(0, 0, 0, 0), (1, 1, 2, 3), (2, 0, 7, 9)]:
        assert reingested.toworld(px) == pytest.approx(im.toworld(px), abs=1e-9)


def test_ddfacet_mycasapy2bbs_flow():
    """`MyCasapy2bbs.py` reads `coordinates().__dict__["_csys"]` for the
    direction increments; it must agree with `dict()`."""
    im = ct.image(IMAGE_FIXTURE)
    coords = im.coordinates()
    assert as_list(coords._csys["direction0"]["cdelt"]) == pytest.approx(
        as_list(coords.dict()["direction0"]["cdelt"])
    )
    assert coords._csys["direction0"]["projection"] == "SIN"
    assert coords._csys["direction0"]["system"] == "ICRS"


def test_modmosaic_stack_flow(tmp_path):
    """`ModMosaic` regrids facets onto one grid, stacks them, and writes the
    result; the identity regrid must not perturb the stack."""
    facet = ct.image(IMAGE_FIXTURE)
    grid = facet.coordinates()
    stack = None
    for _ in range(2):
        piece = facet.regrid([2, 3], grid, outshape=[3, 2, 8, 10])
        stack = piece if stack is None else stack
    np.testing.assert_array_equal(stack.getdata(), facet.getdata())
    stacked = str(tmp_path / "stacked.image")
    stack.saveas(stacked)
    np.testing.assert_array_equal(ct.image(stacked).getdata(), facet.getdata())


@needs_casacore
def test_a_cube_round_tripped_through_casacure_reads_back_in_casacore(tmp_path):
    """The whole pipeline in one test: create, fill, saveas, export, and
    read every artefact back with casacore."""
    scratch = make(tmp_path, (3, 2, 8, 10))
    data = np.arange(480, dtype=np.float32).reshape(3, 2, 8, 10)
    scratch.putdata(data)

    saved = str(tmp_path / "saved.image")
    scratch.saveas(saved)
    exported = str(tmp_path / "exported.fits")
    ct.image(saved).tofits(exported)

    np.testing.assert_array_equal(casacore_image(saved).getdata(), data)
    np.testing.assert_array_equal(casacore_image(exported).getdata(), data)
    assert list(casacore_image(exported).shape()) == [3, 2, 8, 10]


def test_fixture_copy_is_not_the_fixture(tmp_path):
    """A meta-test for this suite's own hygiene: the helpers that write copy
    the table rather than opening the fixture in place."""
    dst = tmp_path / "copy.image"
    shutil.copytree(IMAGE_FIXTURE, dst)
    copied = ct.image(str(dst))
    copied.putdata(np.zeros((3, 2, 8, 10), dtype=np.float32))
    # The fixture is untouched.
    np.testing.assert_array_equal(
        ct.image(IMAGE_FIXTURE).getdata(),
        np.arange(480, dtype=np.float32).reshape(3, 2, 8, 10),
    )
