"""`casacure.images` (issue #14): the `pyrap.images` surface DDFacet and
killMS use — opening CASA image tables and FITS cubes, raster access,
coordinate conversions, and the restoring beam.

The fixtures come from `tests/make_fixtures.py`: a casacore-written CASA
`.image` (TiledCellStMan raster + coords/imageinfo records) and the FITS
cube it was made from, with the manifest recording casacore's answers.
When real python-casacore is installed, the same files are opened through
`casacore.images` and compared value-for-value.
"""

import os

import numpy as np
import pytest

from casacore.tables import table  # the shim: casacure.tables

ct = pytest.importorskip("casacure.images")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MANIFEST = os.path.join(FIXTURES, "manifest.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MANIFEST), reason="fixtures not generated (make_fixtures.py)"
)


def manifest():
    import json

    with open(MANIFEST) as f:
        return json.load(f)


CASACORE_AVAILABLE = False
try:
    from casacore.images import image as casacore_image

    CASACORE_AVAILABLE = True
except ImportError:
    casacore_image = None

needs_casacore = pytest.mark.skipif(
    not CASACORE_AVAILABLE, reason="real python-casacore not installed"
)


@pytest.fixture(scope="module")
def img():
    return ct.image(os.path.join(FIXTURES, manifest()["tables"]["image"]["path"]))


@pytest.fixture(scope="module")
def fits_img():
    return ct.image(os.path.join(FIXTURES, manifest()["tables"]["image"]["fits"]))


def test_casa_image_raster_matches_manifest(img):
    f = manifest()["tables"]["image"]
    d = img.getdata()
    assert list(d.shape) == f["shape"]
    assert d.dtype == np.float32
    assert d.flat[0] == f["first"] and d.flat[-1] == f["last"]


def test_casa_image_raster_matches_casacore(img):
    if not CASACORE_AVAILABLE:
        pytest.skip("real python-casacore not installed")
    want = casacore_image(os.path.join(FIXTURES, manifest()["tables"]["image"]["path"])).getdata()
    np.testing.assert_array_equal(img.getdata(), want)


def test_fits_cube_matches_casacore(fits_img):
    if not CASACORE_AVAILABLE:
        pytest.skip("real python-casacore not installed")
    want = casacore_image(os.path.join(FIXTURES, manifest()["tables"]["image"]["fits"])).getdata()
    got = fits_img.getdata()
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("px", [(0, 0, 0, 0), (1, 1, 2, 3), (0, 1, 7, 9), (2, 0, 4, 5)])
def test_toworld_matches_casacore(img, px):
    f = manifest()["tables"]["image"]
    want = {"0000": f["toworld_0000"], "1123": f["toworld_1123"]}
    if px == (0, 0, 0, 0):
        expected = want["0000"]
    elif px == (1, 1, 2, 3):
        expected = want["1123"]
    else:
        if not CASACORE_AVAILABLE:
            pytest.skip("real python-casacore not installed")
        expected = casacore_image(
            os.path.join(FIXTURES, f["path"])
        ).toworld(px)
    got = img.toworld(px)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-12)


def test_toworld_topixel_roundtrip(img):
    # DDFacet feeds toworld output straight into topixel (fits2png,
    # MakeMask); the roundtrip must land back on the pixel.
    world = img.toworld((1, 0, 5, 7))
    pixel = img.topixel(world)
    assert pixel == pytest.approx((1.0, 0.0, 5.0, 7.0), abs=1e-6)


def test_topixel_off_projection_raises(img):
    # Beyond the SIN hemisphere (dec ~1.57 rad, tangent at -0.45 deg).
    with pytest.raises(RuntimeError):
        img.topixel((1.4e9, 1.0, 1.57, 0.03))


def test_coordinates_dict_matches_record(img):
    f = manifest()["tables"]["image"]
    d = img.coordinates().dict()
    for got, want in zip(d["direction0"]["cdelt"], f["keywords"]["coords"]["direction0"]["cdelt"]):
        assert got == pytest.approx(want, abs=1e-18)
    for got, want in zip(d["direction0"]["crval"], f["keywords"]["coords"]["direction0"]["crval"]):
        assert got == pytest.approx(want, abs=1e-18)
    # The projection and system survive the record round-trip.
    assert d["direction0"]["projection"] == "SIN"


def test_coordinates_private_csys(img):
    # MyCasapy2bbs.py reads coordinates().__dict__["_csys"]["direction0"]["cdelt"].
    csys = img.coordinates()._csys
    assert csys["direction0"]["cdelt"] == pytest.approx(
        manifest()["tables"]["image"]["keywords"]["coords"]["direction0"]["cdelt"],
        abs=1e-18,
    )


def test_get_increment_layout(img):
    # pyrap's per-coordinate layout: [spectral-scalar, stokes-array,
    # direction-array] (reverse CS order; spectral is a scalar).
    inc = img.coordinates().get_increment()
    assert inc[0] == 2.0e6
    assert list(inc[1]) == [1.0]
    assert len(inc[2]) == 2
    # set/get round-trip.
    c = img.coordinates()
    c.set_increment([[2.5e6], [1.0], list(inc[2])])
    assert c.get_increment()[0] == 2.5e6


def test_imageinfo_restoringbeam(img):
    f = manifest()["tables"]["image"]
    beam = img.imageinfo()["restoringbeam"]
    for axis in ("major", "minor", "positionangle"):
        assert beam[axis]["value"] == pytest.approx(
            f["keywords"]["imageinfo"]["restoringbeam"][axis]["value"]
        )
        assert beam[axis]["unit"] == f["keywords"]["imageinfo"]["restoringbeam"][axis]["unit"]


def test_fits_beam_from_cards(fits_img):
    # BMAJ 3.5e-3 deg -> 12.6 arcsec; BPA 15 deg.
    beam = fits_img.imageinfo()["restoringbeam"]
    assert beam["major"] == {"value": 12.6, "unit": "arcsec"}
    assert beam["minor"] == {"value": 9.0, "unit": "arcsec"}
    assert beam["positionangle"]["value"] == 15.0


def test_unit_shape_name_miscinfo(img):
    f = manifest()["tables"]["image"]
    # pyrap's unit() wraps the stored units keyword in quotes (probed).
    assert img.unit() == "'" + f["keywords"]["units"] + "'"
    assert list(img.shape()) == f["shape"]
    assert os.path.basename(img.name()) == f["path"]
    assert img.miscinfo() == {}


def test_fits_toworld_matches_casacore(fits_img):
    if not CASACORE_AVAILABLE:
        pytest.skip("real python-casacore not installed")
    want = casacore_image(os.path.join(FIXTURES, manifest()["tables"]["image"]["fits"])).toworld(
        (1, 1, 2, 3)
    )
    got = fits_img.toworld((1, 1, 2, 3))
    for g, e in zip(got, want):
        assert g == pytest.approx(e, abs=1e-12)


def test_ddfacet_restore_loop(img):
    """The Restore.py pattern: per-pixel toworld((0,0,y,x)) -> (f, p, dec, ra),
    unpacked positionally and fed onward."""
    f, p, dec, ra = img.toworld((0, 0, 0, 0))
    assert f == 1.4e9 and p == 1.0
    assert dec == pytest.approx(
        manifest()["tables"]["image"]["toworld_0000"][2], abs=1e-15
    )
    # A small grid, like casapy2bbs's per-pixel loop.
    for y in range(0, 8, 3):
        for x in range(0, 10, 4):
            _, _, dec, ra = img.toworld((0, 0, y, x))
            assert -0.1 < dec < 0.1 and 0.0 < ra < 0.1
            back = img.topixel((1.4e9, 1.0, dec, ra))
            assert back == pytest.approx((0.0, 0.0, y, x), abs=1e-6)


def test_multi_row_tsmcell_table():
    """The TiledCellStMan cube-index-equals-row mapping, through the table
    surface (the storage behind CASA images)."""
    f = manifest()["tables"]["tsmcell"]
    t = table(os.path.join(FIXTURES, f["path"]), readonly=True)
    for r in range(f["nrows"]):
        cell = np.asarray(t.getcell("DATA", r))
        assert cell.tolist() == f["cells"][r]
    t.close()


def test_create_putdata_roundtrip(tmp_path):
    """killMS MakeModelImage's pattern: saveas a template, fill, putdata."""
    tmpl = ct.image(os.path.join(FIXTURES, manifest()["tables"]["image"]["path"]))
    out = str(tmp_path / "model.image")
    tmpl.saveas(out)
    im = ct.image(out)
    assert list(im.shape()) == manifest()["tables"]["image"]["shape"]
    np.testing.assert_array_equal(im.getdata(), tmpl.getdata())
    data = im.getdata()
    data.fill(0)
    data[0, 0, 3, 4] = 42.0
    data[1, 1, 7, 9] = -7.5
    im.putdata(data)
    np.testing.assert_array_equal(ct.image(out).getdata(), data)
    # Coordinates and beam survive the copy + write.
    tw, twt = im.toworld((0, 0, 3, 4)), tmpl.toworld((0, 0, 3, 4))
    for a, b in zip(tw, twt):
        assert a == pytest.approx(b, abs=1e-15)
    assert os.path.isdir(os.path.join(out, "logtable"))


@needs_casacore
def test_casacore_reads_casacure_created_image(tmp_path):
    from casacore.images import image as cc_image

    tmpl = ct.image(os.path.join(FIXTURES, manifest()["tables"]["image"]["path"]))
    out = str(tmp_path / "c.image")
    tmpl.saveas(out)
    data = tmpl.getdata()
    data.fill(0)
    data[1, 0, 2, 5] = 9.75
    ct.image(out).putdata(data)

    cc = cc_image(out)
    np.testing.assert_array_equal(cc.getdata(), data)
    cc_tw = cc.toworld((0, 0, 2, 5))
    tmpl_tw = tmpl.toworld((0, 0, 2, 5))
    for a, b in zip(cc_tw, tmpl_tw):
        assert a == pytest.approx(b, abs=1e-15)


def test_create_with_coordsys_object(tmp_path):
    """DDFacet ClassCasaImage's create: coordsys from another image,
    optionally mutated through the get/set surface."""
    src = ct.image(os.path.join(FIXTURES, manifest()["tables"]["image"]["path"]))
    c = src.coordinates()
    out = str(tmp_path / "scratch.image")
    ct.image(imagename=out, shape=(2, 1, 6, 7), coordsys=c)
    im = ct.image(out)
    assert list(im.shape()) == [2, 1, 6, 7]
    # The created image carries the passed coordsys (same cdelt).
    assert im.coordinates().dict()["direction0"]["cdelt"] == pytest.approx(
        src.coordinates().dict()["direction0"]["cdelt"]
    )


@needs_casacore
def test_tofits_read_back_by_casacore_and_astropy(tmp_path):
    from astropy.io import fits
    from casacore.images import image as cc_image

    src = ct.image(os.path.join(FIXTURES, manifest()["tables"]["image"]["path"]))
    out = str(tmp_path / "export.fits")
    src.tofits(out)

    cc = cc_image(out)
    np.testing.assert_array_equal(cc.getdata(), src.getdata())
    for a, b in zip(cc.toworld((1, 1, 2, 3)), src.toworld((1, 1, 2, 3))):
        assert a == pytest.approx(b, abs=1e-12)

    with fits.open(out) as hdul:
        h = hdul[0].header
        assert [h[f"CTYPE{i}"] for i in range(1, 5)] == [
            "RA---SIN", "DEC--SIN", "STOKES", "FREQ",
        ]
        assert h["BMAJ"] == pytest.approx(3.5e-3)
        np.testing.assert_array_equal(hdul[0].data, src.getdata())
