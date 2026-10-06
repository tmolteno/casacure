"""The `casacure.images.image` object surface: opening, creating, raster
access, metadata, and the copy/overwrite rules.

Consumer call-sites guarded here:

* DDFacet ``ClassCasaImage.py`` — ``image(imagename=..., shape=...,
  overwrite=...)`` to create a scratch cube, ``.getdata()``/``.putdata()``
  to fill it, ``.saveas()`` to keep it.
* killMS ``MakeModelImage.py`` — ``saveas`` a template then ``putdata`` the
  model into the copy, including into a cube casacore (rather than casacure)
  created, whose raster column is a ``TiledCellStMan``.
* DDFacet ``fits2png.py`` — ``.getdata()`` dtype and the ``unit()`` string.

The fixture image is read-only for these tests: anything that writes copies
it into ``tmp_path`` first.
"""

import os
import pathlib
import shutil

import numpy as np
import pytest

from casacore.tables import table  # noqa: F401  (the shim: casacure.tables)

ct = pytest.importorskip("casacure.images")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
IMAGE_FIXTURE = os.path.join(FIXTURES, "image.image")
FITS_FIXTURE = os.path.join(FIXTURES, "image.fits")

# Read as a CASA image and as FITS; tests/conftest.py skips the module when
# either fixture is absent.
CASACORE_FIXTURES = ("image.image", "image.fits")
SHAPE = [3, 2, 8, 10]


@pytest.fixture
def casa_image(tmp_path):
    """A private copy of the casacore-written fixture.

    Several tests `putdata` (which rewrites the table on disk), so they must
    never be handed the shared fixture path itself — doing so silently
    zeroes `tests/fixtures/image.image` for every later test in the session.
    """
    dst = tmp_path / "fixture.image"
    shutil.copytree(IMAGE_FIXTURE, dst)
    return ct.image(str(dst))


@pytest.fixture
def fits_image():
    return ct.image(FITS_FIXTURE)


@pytest.fixture
def writable_copy(tmp_path):
    """A private copy of the casacore-written image table."""
    dst = tmp_path / "copy.image"
    shutil.copytree(IMAGE_FIXTURE, dst)
    return str(dst)


def make(tmp_path, shape, name="scratch.image", **kwargs):
    path = str(tmp_path / name)
    ct.image(imagename=path, shape=shape, **kwargs)
    return ct.image(path)


# ---------------------------------------------------------------------------
# Opening and creating
# ---------------------------------------------------------------------------


def test_open_a_casa_image_table(casa_image):
    assert list(casa_image.shape()) == SHAPE
    assert isinstance(casa_image.shape(), tuple)


def test_open_a_fits_cube(fits_image):
    assert list(fits_image.shape()) == SHAPE


def test_open_positional_argument(casa_image):
    assert list(ct.image(IMAGE_FIXTURE).shape()) == SHAPE


def test_open_accepts_a_pathlib_path(tmp_path):
    path = tmp_path / "p.image"
    ct.image(imagename=str(path), shape=(2, 3))
    assert list(ct.image(imagename=path).shape()) == [2, 3]


def test_create_from_shape(tmp_path):
    im = make(tmp_path, (2, 3, 4, 5))
    assert list(im.shape()) == [2, 3, 4, 5]
    # The raster starts zeroed.
    np.testing.assert_array_equal(im.getdata(), np.zeros((2, 3, 4, 5), dtype=np.float32))


def test_create_writes_a_casa_table_with_a_logtable(tmp_path):
    """casacore's image() requires the `logtable` keyword and directory."""
    path = str(tmp_path / "withlog.image")
    ct.image(imagename=path, shape=(2, 2))
    assert os.path.isdir(os.path.join(path, "logtable"))
    assert os.path.isfile(os.path.join(path, "table.info"))


def test_create_without_a_path_is_an_error():
    with pytest.raises(ValueError, match=r"image\(\) needs a path"):
        ct.image()


def test_open_a_path_that_is_not_an_image(tmp_path):
    with pytest.raises(RuntimeError, match="no such image"):
        ct.image(str(tmp_path / "missing.image"))


def test_overwrite_false_refuses_an_existing_image(tmp_path):
    """pyrap's `overwrite=False` must not silently destroy a caller's cube."""
    path = str(tmp_path / "keep.image")
    ct.image(imagename=path, shape=(2, 2), overwrite=False)
    with pytest.raises(RuntimeError, match="already exists and should not be overwritten"):
        ct.image(imagename=path, shape=(3, 3), overwrite=False)
    # The original is intact.
    assert list(ct.image(path).shape()) == [2, 2]


def test_overwrite_true_replaces_an_existing_image(tmp_path):
    path = str(tmp_path / "replace.image")
    ct.image(imagename=path, shape=(2, 2))
    assert list(ct.image(imagename=path, shape=(3, 3), overwrite=True).shape()) == [3, 3]


def test_create_is_the_default_overwrite(tmp_path):
    path = str(tmp_path / "default.image")
    ct.image(imagename=path, shape=(2, 2))
    assert list(ct.image(imagename=path, shape=(4, 4)).shape()) == [4, 4]


# ---------------------------------------------------------------------------
# Raster access
# ---------------------------------------------------------------------------


def test_casa_getdata_shape_and_dtype(casa_image):
    """fits2png budgets memory per cube; the raster must stay float32."""
    data = casa_image.getdata()
    assert list(data.shape) == SHAPE
    assert data.dtype == np.float32


def test_getdata_returns_a_copy(casa_image):
    data = casa_image.getdata()
    first = float(data.flat[0])
    data.fill(-1.0)
    assert float(casa_image.getdata().flat[0]) == first


def test_getdata_is_c_order_contiguous(casa_image):
    """Downstream code reshapes and views the raster; a non-contiguous array
    would silently produce wrong pixels in a reshape."""
    assert casa_image.getdata().flags["C_CONTIGUOUS"]


def test_putdata_round_trips(casa_image):
    data = casa_image.getdata()
    data.fill(0.0)
    data[0, 0, 3, 4] = 42.0
    data[1, 1, 7, 9] = -7.5
    casa_image.putdata(data)
    np.testing.assert_array_equal(casa_image.getdata(), data)


def test_putdata_into_a_casacore_written_image(writable_copy):
    """The raster column of an image casacore wrote is a TiledCellStMan,
    which the write layer used to refuse outright with "cannot
    preserve-rewrite data-manager type TiledCellStMan"."""
    im = ct.image(writable_copy)
    data = np.arange(480, dtype=np.float32).reshape(SHAPE)
    im.putdata(data)
    np.testing.assert_array_equal(ct.image(writable_copy).getdata(), data)
    # The WCS survives the raster rewrite.
    assert im.toworld((0, 0, 0, 0)) == pytest.approx(
        (1.4e9, 1.0, -0.007855552430289316, 0.030545007293006025), abs=1e-15
    )


def test_putdata_into_a_copy_preserves_the_raster_dtype(casa_image):
    casa_image.putdata(casa_image.getdata())
    assert casa_image.getdata().dtype == np.float32


def test_putdata_rejects_a_wrong_shape(casa_image):
    """The caller's mistake must be named, not reported as a storage-layer
    "unsupported tiled element type" complaint."""
    with pytest.raises(RuntimeError, match=r"array shape \[2, 2\] does not match the image"):
        casa_image.putdata(np.zeros((2, 2), dtype=np.float32))


def test_putdata_rejects_a_transposed_shape(casa_image):
    with pytest.raises(RuntimeError, match="does not match the image"):
        casa_image.putdata(np.zeros((10, 8, 2, 3), dtype=np.float32))


def test_putdata_rejects_a_scalar(casa_image):
    with pytest.raises(ValueError, match="putdata expects an array"):
        casa_image.putdata(1.0)


def test_putdata_rejects_a_ragged_list(casa_image):
    """A python list is not a raster; the failure must be a TypeError about
    the value, not a storage error."""
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        casa_image.putdata([[1.0, 2.0], [3.0]])


def test_putdata_does_not_change_the_shape_or_coordinates(casa_image):
    before_shape = casa_image.shape()
    before_world = casa_image.toworld((1, 1, 2, 3))
    casa_image.putdata(casa_image.getdata())
    assert casa_image.shape() == before_shape
    assert casa_image.toworld((1, 1, 2, 3)) == pytest.approx(before_world)


@pytest.mark.parametrize(
    "dtype,value",
    [(np.float64, 2.0), (np.int64, 3), (np.int32, 4), (np.float32, 5.0)],
)
def test_putdata_coerces_every_supported_numeric_raster(tmp_path, dtype, value):
    """casacore converts every numeric raster type to the float storage."""
    im = make(tmp_path, (2, 2))
    im.putdata(np.full((2, 2), value, dtype=dtype))
    got = im.getdata()
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.full((2, 2), value, dtype=np.float32))


def test_putdata_accepts_a_fortran_ordered_array(tmp_path):
    """The values are read in C order regardless of the buffer's strides."""
    im = make(tmp_path, (2, 2))
    arr = np.asfortranarray(np.arange(1, 5, dtype=np.float32).reshape(2, 2))
    im.putdata(arr)
    np.testing.assert_array_equal(im.getdata(), np.arange(1, 5, dtype=np.float32).reshape(2, 2))


def test_putdata_accepts_a_non_contiguous_view(tmp_path):
    im = make(tmp_path, (2, 2))
    view = np.arange(8, dtype=np.float32).reshape(2, 4)[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    im.putdata(view)
    np.testing.assert_array_equal(im.getdata(), [[0.0, 2.0], [4.0, 6.0]])


def test_putdata_rejects_a_bool_raster(tmp_path):
    """casacore rejects `Array<Bool>` for a Float image; the message must
    name the caller's array type, not the column's."""
    im = make(tmp_path, (2, 2))
    with pytest.raises(RuntimeError, match=r"invalid data type Array<Bool>"):
        im.putdata(np.ones((2, 2), dtype=bool))


def test_putdata_rejects_a_complex_raster(tmp_path):
    """casacore rejects `Array<Complex>` too (with an "invalid data type"
    message); casacure refuses the array in the pyo3 converter, so the
    wording differs but the rejection is the same."""
    im = make(tmp_path, (2, 2))
    with pytest.raises((ValueError, RuntimeError)):
        im.putdata(np.ones((2, 2), dtype=np.complex64))


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.uint8])
@pytest.mark.xfail(
    strict=True,
    reason="convert::pyobject_to_record has no i16/u8/u16 numpy arm, so these "
    "raster types are refused before reaching the image code; casacore accepts "
    "every numeric raster for a Float image",
)
def test_putdata_accepts_every_numeric_raster_casacore_accepts(tmp_path, dtype):
    im = make(tmp_path, (2, 2))
    im.putdata(np.full((2, 2), 6, dtype=dtype))
    np.testing.assert_array_equal(im.getdata(), np.full((2, 2), 6.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_name_is_absolute(casa_image):
    assert os.path.isabs(casa_image.name())
    assert os.path.isdir(casa_image.name())


def test_repr_names_the_path_and_shape(casa_image):
    assert repr(casa_image) == f"<image '{casa_image.name()}' shape [3, 2, 8, 10]>"


def test_unit_is_quoted_for_both_flavours(casa_image, fits_image):
    """pyrap's `unit()` wraps the stored units keyword in quotes, whichever
    flavour the image came from (ImageMeta strips them for FITS output)."""
    assert casa_image.unit() == "'Jy/beam'"
    assert fits_image.unit() == "'Jy/beam'"


def test_unit_of_a_created_image_is_empty_and_quoted(tmp_path):
    assert make(tmp_path, (2, 2)).unit() == "''"


def test_miscinfo_is_empty(casa_image, fits_image):
    assert casa_image.miscinfo() == {}
    assert fits_image.miscinfo() == {}


def test_imageinfo_carries_the_restoring_beam(casa_image):
    beam = casa_image.imageinfo()["restoringbeam"]
    assert beam["major"]["value"] == pytest.approx(12.6)
    assert beam["minor"]["value"] == pytest.approx(9.0)
    assert beam["positionangle"]["value"] == pytest.approx(15.0)


def test_imageinfo_of_a_created_image_has_the_defaults(tmp_path):
    info = make(tmp_path, (2, 2)).imageinfo()
    assert "imagetype" in info and "objectname" in info


# ---------------------------------------------------------------------------
# saveas
# ---------------------------------------------------------------------------


def test_saveas_copies_the_raster_and_coordinates(casa_image, tmp_path):
    out = str(tmp_path / "copy.image")
    casa_image.saveas(out)
    copy = ct.image(out)
    np.testing.assert_array_equal(copy.getdata(), casa_image.getdata())
    assert list(copy.shape()) == SHAPE
    for px in [(0, 0, 0, 0), (1, 1, 2, 3)]:
        assert copy.toworld(px) == pytest.approx(casa_image.toworld(px), abs=1e-15)


def test_saveas_copies_the_restoring_beam(casa_image, tmp_path):
    out = str(tmp_path / "beam.image")
    casa_image.saveas(out)
    assert ct.image(out).imageinfo()["restoringbeam"]["major"]["value"] == pytest.approx(12.6)


def test_saveas_of_a_fits_image_produces_a_casa_table(fits_image, tmp_path):
    """killMS converts an ingested FITS model into a CASA image this way."""
    out = str(tmp_path / "fromfits.image")
    fits_image.saveas(out)
    copy = ct.image(out)
    np.testing.assert_array_equal(copy.getdata(), fits_image.getdata())
    assert os.path.isdir(out) and os.path.isfile(os.path.join(out, "table.info"))
    assert copy.unit() == "'Jy/beam'"


def test_saveas_replaces_an_existing_destination(casa_image, tmp_path):
    out = str(tmp_path / "again.image")
    casa_image.saveas(out)
    casa_image.saveas(out)
    np.testing.assert_array_equal(ct.image(out).getdata(), casa_image.getdata())


def test_saveas_then_putdata_is_the_killms_pattern(casa_image, tmp_path):
    """MakeModelImage: copy a template, zero the model, write it back."""
    out = str(tmp_path / "model.image")
    casa_image.saveas(out)
    model = ct.image(out)
    data = np.zeros(SHAPE, dtype=np.float32)
    data[0, 0, 4, 5] = 12.5
    model.putdata(data)
    np.testing.assert_array_equal(ct.image(out).getdata(), data)


def test_saveas_of_a_created_image_keeps_the_default_coordinates(tmp_path):
    src = make(tmp_path, (3, 2, 8, 10), "src.image")
    out = str(tmp_path / "dst.image")
    src.saveas(out)
    assert src.toworld((0, 0, 4, 4)) == pytest.approx(ct.image(out).toworld((0, 0, 4, 4)))


# ---------------------------------------------------------------------------
# Parity with real casacore
# ---------------------------------------------------------------------------

CASACORE_AVAILABLE = False
try:
    from casacore.images import image as casacore_image

    CASACORE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    casacore_image = None

needs_casacore = pytest.mark.skipif(
    not CASACORE_AVAILABLE, reason="real python-casacore not installed"
)


@needs_casacore
def test_casacore_reads_a_casacure_putdata(tmp_path):
    """A cube casacure wrote must be readable, and hold the same pixels."""
    path = str(tmp_path / "written.image")
    im = ct.image(imagename=path, shape=SHAPE)
    data = np.arange(480, dtype=np.float32).reshape(SHAPE)
    im.putdata(data)
    np.testing.assert_array_equal(casacore_image(path).getdata(), data)


@needs_casacore
def test_casacore_reads_a_casacure_putdata_into_its_own_image(writable_copy):
    """The TiledCellStMan in-place rewrite must produce a table casacore
    still considers valid, with the same data manager."""
    im = ct.image(writable_copy)
    data = np.arange(480, dtype=np.float32).reshape(SHAPE)
    im.putdata(data)
    np.testing.assert_array_equal(casacore_image(writable_copy).getdata(), data)
    from casacore.tables import table as casacore_table

    assert casacore_table(writable_copy).getdminfo()["*1"]["TYPE"] == "TiledCellStMan"


@needs_casacore
def test_overwrite_false_matches_casacore(tmp_path):
    path = str(tmp_path / "ow.image")
    casacore_image(imagename=path, shape=[2, 2], overwrite=False)
    with pytest.raises(RuntimeError, match="already exists and should not be overwritten"):
        ct.image(imagename=path, shape=(3, 3), overwrite=False)


@needs_casacore
def test_casacore_reads_a_saveas_of_the_fixture(casa_image, tmp_path):
    out = str(tmp_path / "saved.image")
    casa_image.saveas(out)
    np.testing.assert_array_equal(casacore_image(out).getdata(), casa_image.getdata())
    # casacore returns a list here, casacure a tuple; compare the values.
    assert list(casacore_image(out).shape()) == list(casa_image.shape())
