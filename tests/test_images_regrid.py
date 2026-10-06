"""`casacure.images.regrid`: resampling a cube onto another coordinate grid.

Consumer call-sites guarded here:

* DDFacet ``ModMosaic.py`` — regrids every facet onto one common grid, so
  the identity regrid has to be bit-exact and the axis list is
  ``[2, 3]`` (the two direction axes of a (ch, pol, y, x) cube).
* DDFacet ``ModFitPSF.py`` / ``MakeMask`` — regrid with an explicit
  ``outshape`` and read the result back through ``getdata``.
* killMS ``MakeModelImage.py`` — regrid a model onto a target grid.

The axes and ``outshape`` are in pyrap/numpy order, the same order
``shape()``/``getdata()`` use.
"""

import os

import numpy as np
import pytest

from casacore.tables import table  # noqa: F401  (the shim: casacure.tables)

ct = pytest.importorskip("casacure.images")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
IMAGE_FIXTURE = os.path.join(FIXTURES, "image.image")
FITS_FIXTURE = os.path.join(FIXTURES, "image.fits")

# Both fixture flavours are opened here; tests/conftest.py skips the module
# when either is absent.
CASACORE_FIXTURES = ("image.image", "image.fits")
SHAPE = [3, 2, 8, 10]


@pytest.fixture
def cube():
    return ct.image(IMAGE_FIXTURE)


def make(tmp_path, shape, name="src.image"):
    path = str(tmp_path / name)
    ct.image(imagename=path, shape=shape)
    return ct.image(path)


def finer_direction(coords, factor):
    """`coords` with only the *direction* increments scaled by `factor`.

    `get_increment()` is in reverse coordinate order, so its last entry is
    the direction coordinate — the grid a facet regrid actually changes.
    Scaling the spectral/stokes entries too would ask for a different
    channel/polarisation grid, which is not what ModMosaic does.
    """
    per_coord = [np.asarray(v, dtype=float) for v in coords.get_increment()]
    per_coord[-1] = per_coord[-1] * factor
    coords.set_increment([v.tolist() for v in per_coord])
    return coords


# ---------------------------------------------------------------------------
# Identity and no-op paths
# ---------------------------------------------------------------------------


def test_identity_regrid_is_bit_exact(cube):
    """Stacked facets share their grid, so the identity regrid must not
    perturb a single pixel."""
    data = cube.getdata()
    out = cube.regrid([2, 3], cube.coordinates(), outshape=list(data.shape))
    assert list(out.shape()) == list(data.shape)
    np.testing.assert_array_equal(out.getdata(), data)


def test_identity_regrid_of_a_fits_cube_is_bit_exact():
    src = ct.image(FITS_FIXTURE)
    data = src.getdata()
    out = src.regrid([2, 3], src.coordinates(), outshape=list(data.shape))
    np.testing.assert_array_equal(out.getdata(), data)


def test_default_outshape_is_the_source_shape(cube):
    out = cube.regrid([2, 3], cube.coordinates())
    assert list(out.shape()) == SHAPE


def test_empty_axis_list_copies_the_source(cube):
    """`regrid([], ...)` resamples nothing: the raster is rounded into the
    target grid, which for an integer-valued cube is an identity copy."""
    out = cube.regrid([], cube.coordinates())
    assert list(out.shape()) == SHAPE
    np.testing.assert_array_equal(out.getdata(), cube.getdata())


def test_all_axes_regridded_identity(cube):
    data = cube.getdata()
    out = cube.regrid([0, 1, 2, 3], cube.coordinates(), outshape=SHAPE)
    np.testing.assert_array_equal(out.getdata(), data)


# ---------------------------------------------------------------------------
# The axis list
# ---------------------------------------------------------------------------


def test_duplicate_axes_behave_as_a_single_axis(cube):
    """A caller that lists an axis twice must not double-weight it."""
    coords = cube.coordinates()
    twice = cube.regrid([2, 2], coords, outshape=[3, 2, 4, 4]).getdata()
    once = cube.regrid([2], coords, outshape=[3, 2, 4, 4]).getdata()
    np.testing.assert_array_equal(twice, once)


def test_duplicate_axes_mixed_with_others(cube):
    coords = cube.coordinates()
    dup = cube.regrid([2, 3, 2, 3], coords, outshape=[3, 2, 4, 4]).getdata()
    plain = cube.regrid([2, 3], coords, outshape=[3, 2, 4, 4]).getdata()
    np.testing.assert_array_equal(dup, plain)


def test_axes_accepts_a_tuple(cube):
    assert list(cube.regrid((2, 3), cube.coordinates(), outshape=SHAPE).shape()) == SHAPE


def test_axis_order_within_the_list_does_not_matter(cube):
    coords = cube.coordinates()
    forward = cube.regrid([2, 3], coords, outshape=[3, 2, 4, 4]).getdata()
    reverse = cube.regrid([3, 2], coords, outshape=[3, 2, 4, 4]).getdata()
    np.testing.assert_array_equal(forward, reverse)


@pytest.mark.parametrize("axis", [4, 5, 9, 1000])
def test_out_of_range_axis_is_an_error_not_a_panic(cube, axis):
    """`src_shape[axis]` used to index out of bounds and surface as a pyo3
    `PanicException`, which no consumer can catch as a message."""
    with pytest.raises(RuntimeError, match=rf"regrid axis {axis} is out of range"):
        cube.regrid([axis], cube.coordinates())


def test_out_of_range_axis_on_a_2d_image(tmp_path):
    """The direction axes of a 2-D image are 0 and 1, so 2 and 3 are out of
    range — the same call that is correct for a 4-D cube."""
    im = make(tmp_path, (5, 7))
    with pytest.raises(RuntimeError, match="regrid axis 2 is out of range"):
        im.regrid([2, 3], im.coordinates())
    assert list(im.regrid([0, 1], im.coordinates(), outshape=[5, 7]).shape()) == [5, 7]


def test_axes_must_be_a_sequence(cube):
    with pytest.raises(TypeError, match="cannot be cast as 'Sequence'"):
        cube.regrid(2, cube.coordinates())


# ---------------------------------------------------------------------------
# outshape and coordsys arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outshape", [[4, 4], [2, 4, 4], [1, 2, 3, 4, 5], []])
def test_outshape_with_the_wrong_rank_is_an_error(cube, outshape):
    with pytest.raises(RuntimeError, match="must have 4 axes"):
        cube.regrid([2, 3], cube.coordinates(), outshape=outshape)


def test_outshape_accepts_a_tuple(cube):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=(1, 1, 4, 4))
    assert list(out.shape()) == [1, 1, 4, 4]


def test_outshape_none_means_the_source_shape(cube):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=None)
    assert list(out.shape()) == SHAPE


def test_outshape_can_enlarge_the_grid(cube):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[3, 2, 16, 20])
    assert list(out.shape()) == [3, 2, 16, 20]


def test_coordsys_must_be_a_coordinates_object(cube):
    with pytest.raises(TypeError, match="cannot be cast as 'coordinates'"):
        cube.regrid([2, 3], None, outshape=SHAPE)
    with pytest.raises(TypeError, match="cannot be cast as 'coordinates'"):
        cube.regrid([2, 3], cube.coordinates().dict(), outshape=SHAPE)


def test_regrid_does_not_modify_the_source(cube):
    before = cube.getdata()
    cube.regrid([2, 3], cube.coordinates(), outshape=[1, 1, 4, 4])
    np.testing.assert_array_equal(cube.getdata(), before)
    assert list(cube.shape()) == SHAPE


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------


def test_result_is_in_memory_and_nameless(cube):
    """A regrid result has no file behind it until it is saved."""
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[1, 1, 4, 4])
    assert out.name() == ""
    assert repr(out) == "<image '' shape [1, 1, 4, 4]>"


def test_result_carries_the_target_coordinates(cube):
    """The result's world grid is the *target* coordsys's grid, not the
    source's — at the shared reference pixel the two must coincide."""
    target = finer_direction(cube.coordinates(), 0.5)
    out = cube.regrid([2, 3], target, outshape=SHAPE)
    assert out.coordinates().dict()["direction0"]["cdelt"] == pytest.approx(
        target.dict()["direction0"]["cdelt"]
    )
    # The fixture's reference pixel is casa (4, 3) = numpy (3, 4).
    assert out.toworld((0, 0, 3, 4)) == pytest.approx(cube.toworld((0, 0, 3, 4)), abs=1e-12)
    assert out.toworld((0, 0, 3, 4))[3] == pytest.approx(0.030543261909900768, abs=1e-12)


def test_result_keeps_the_units_and_beam(cube):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[1, 1, 4, 4])
    assert out.unit() == cube.unit()
    assert out.imageinfo()["restoringbeam"] == cube.imageinfo()["restoringbeam"]


def test_result_can_be_saved(cube, tmp_path):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[1, 1, 4, 4])
    path = str(tmp_path / "stacked.image")
    out.saveas(path)
    again = ct.image(path)
    np.testing.assert_array_equal(again.getdata(), out.getdata())
    assert list(again.shape()) == [1, 1, 4, 4]


def test_result_can_be_exported_to_fits(cube, tmp_path):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[1, 1, 4, 4])
    path = str(tmp_path / "stacked.fits")
    out.tofits(path)
    again = ct.image(path)
    np.testing.assert_array_equal(again.getdata(), out.getdata())


def test_result_of_a_result(cube):
    """ModMosaic regrids repeatedly (facet -> mosaic -> cutout)."""
    first = cube.regrid([2, 3], cube.coordinates(), outshape=[3, 2, 8, 10])
    second = first.regrid([2, 3], first.coordinates(), outshape=[3, 2, 4, 4])
    assert list(second.shape()) == [3, 2, 4, 4]


def test_result_is_float32(cube):
    out = cube.regrid([2, 3], cube.coordinates(), outshape=SHAPE)
    assert out.getdata().dtype == np.float32


# ---------------------------------------------------------------------------
# The interpolation kernel
# ---------------------------------------------------------------------------


def test_identity_regrid_reproduces_a_delta(tmp_path):
    im = make(tmp_path, (5, 7))
    delta = np.zeros((5, 7), dtype=np.float32)
    delta[2, 3] = 1.0
    im.putdata(delta)
    out = im.regrid([0, 1], im.coordinates(), outshape=[5, 7]).getdata()
    np.testing.assert_array_equal(out, delta)


def test_half_size_pixels_interpolate_bilinearly(tmp_path):
    """A target grid with half the increment sits a half source pixel off
    the reference on each side, so a delta spreads with bilinear weights
    1, 1/2, 1/4 — the exact kernel ModMosaic relies on."""
    im = make(tmp_path, (5, 7))
    delta = np.zeros((5, 7), dtype=np.float32)
    delta[2, 3] = 1.0
    im.putdata(delta)
    half = finer_direction(im.coordinates(), 0.5)
    out = im.regrid([0, 1], half, outshape=[5, 7]).getdata()
    expected = np.zeros((5, 7), dtype=np.float32)
    expected[1:4, 2:5] = [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_half_size_pixels_reproduce_a_linear_ramp_exactly(tmp_path):
    """Bilinear interpolation is exact on a linear function, which pins the
    world-to-pixel mapping the kernel is fed.

    The default 5x7 template has casa crpix (x, y) = (3, 2), so a target
    pixel (y, x) samples source (y_s, x_s) with
    y_s = 2 + (y - 2)/2 and x_s = 3 + (x - 3)/2.
    """
    im = make(tmp_path, (5, 7))
    ramp = np.arange(35, dtype=np.float32).reshape(5, 7)
    im.putdata(ramp)
    half = finer_direction(im.coordinates(), 0.5)
    out = im.regrid([0, 1], half, outshape=[5, 7]).getdata()
    for y in range(5):
        for x in range(7):
            y_s = 2.0 + (y - 2.0) / 2.0
            x_s = 3.0 + (x - 3.0) / 2.0
            assert out[y, x] == pytest.approx(7.0 * y_s + x_s, abs=1e-5), (y, x)


def test_coarser_pixels_keep_a_referenced_delta(tmp_path):
    """With the reference pixel on the target grid, a delta stays a delta."""
    im = make(tmp_path, (5, 7))
    delta = np.zeros((5, 7), dtype=np.float32)
    delta[2, 3] = 1.0
    im.putdata(delta)
    coarse = finer_direction(im.coordinates(), 2.0)
    out = im.regrid([0, 1], coarse, outshape=[5, 7]).getdata()
    expected = np.zeros((5, 7), dtype=np.float32)
    expected[2, 3] = 1.0
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_out_of_source_pixels_read_zero(cube):
    """A target grid larger than the tile is padded with zeros; ModMosaic
    relies on that for the mosaic edges."""
    coords = finer_direction(cube.coordinates(), 4.0)
    out = cube.regrid([2, 3], coords, outshape=[3, 2, 8, 10]).getdata()
    # The corners fall outside the source footprint.
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 7, 9] == 0.0


def test_enlarged_outshape_keeps_the_source_in_the_corner(cube):
    """Growing the target grid must not rescale: the source pixels land
    where the shared origin puts them, and the rest is zero."""
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[3, 2, 16, 20]).getdata()
    src = cube.getdata()
    # The reference pixel keeps its value (no shift of the origin).
    assert out[0, 0, 0, 0] == pytest.approx(src[0, 0, 0, 0])


def test_non_direction_axes_are_not_interpolated(tmp_path):
    """A direction-only regrid may average neighbouring *pixels*, never
    neighbouring channels or polarisations: each constant channel/pol plane
    must come back as that same constant."""
    im = make(tmp_path, (3, 2, 5, 7))
    data = np.zeros((3, 2, 5, 7), dtype=np.float32)
    for channel in range(3):
        for pol in range(2):
            data[channel, pol] = 100.0 * (channel + 1) + pol
    im.putdata(data)
    half = finer_direction(im.coordinates(), 0.5)
    out = im.regrid([2, 3], half, outshape=[3, 2, 5, 7]).getdata()
    for channel in range(3):
        for pol in range(2):
            np.testing.assert_allclose(
                out[channel, pol], 100.0 * (channel + 1) + pol, atol=1e-5
            )


def test_regridding_the_stokes_axis_selects_one_polarisation(cube):
    """A target grid with a single polarisation plane reads the first."""
    out = cube.regrid([2, 3], cube.coordinates(), outshape=[3, 1, 8, 10])
    assert list(out.shape()) == [3, 1, 8, 10]


def test_interpolated_values_stay_within_the_source_range(cube):
    """Bilinear interpolation never overshoots: no ringing artefacts."""
    coords = finer_direction(cube.coordinates(), 0.5)
    out = cube.regrid([2, 3], coords, outshape=SHAPE).getdata()
    src = cube.getdata()
    assert out.min() >= src.min() - 1e-6
    assert out.max() <= src.max() + 1e-6


def test_regrid_is_deterministic(cube):
    coords = finer_direction(cube.coordinates(), 0.5)
    first = cube.regrid([2, 3], coords, outshape=SHAPE).getdata()
    second = cube.regrid([2, 3], coords, outshape=SHAPE).getdata()
    np.testing.assert_array_equal(first, second)
