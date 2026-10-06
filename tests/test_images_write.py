"""`casacure.images` writers: `tofits` (the FITS primary HDU export) and the
`saveas`/`create` table-writing path.

Consumer call-sites guarded here:

* DDFacet ``fits2png.py`` / ``Restore.py`` — ``image.tofits(<cube>.fits)``
  to hand a restored cube to external FITS tools; the direction cards must
  match what casacore would have written, in the units casacore uses
  (degrees, ``CUNITn = 'deg'``, 1-based ``CRPIX``).
* DDFacet ``ClassCasaImage.py`` — ``saveas`` a scratch image onto disk.
* killMS ``MakeModelImage.py`` — ``saveas`` a template then write a model.

The card expectations are the exact set casacore 3.8 emits for the same
image, measured live; ``needs_casacore`` re-measures them.
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

ASTROPY_AVAILABLE = False
try:
    from astropy.io import fits as astropy_fits

    ASTROPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    astropy_fits = None

CASACORE_AVAILABLE = False
try:
    from casacore.images import image as casacore_image

    CASACORE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    casacore_image = None

needs_astropy = pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not installed")
needs_casacore = pytest.mark.skipif(
    not CASACORE_AVAILABLE, reason="real python-casacore not installed"
)


# ---------------------------------------------------------------------------
# Reading raw cards back
# ---------------------------------------------------------------------------


def raw_cards(path):
    """The header's (keyword, raw value field) pairs, in file order."""
    out = []
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(80)
            if len(raw) < 80:
                break
            line = raw.decode("ascii")
            if line.startswith("END"):
                break
            key = line[:8].strip()
            if key:
                out.append((key, line[10:]))
    return out


def card_value(raw):
    """A card's value with its quotes and trailing comment removed.

    Only unquoted values can carry a `/` comment: a FITS string such as
    `'Jy/beam'` must keep its slash.
    """
    raw = raw.strip()
    if raw.startswith("'"):
        end = raw.find("'", 1)
        return (raw[1:end] if end != -1 else raw[1:]).strip()
    return raw.split("/")[0].strip()


def cards(path):
    """keyword -> value (the last card wins, as FITS readers do)."""
    return {key: card_value(raw) for key, raw in raw_cards(path)}


def number(path, key):
    return float(cards(path)[key])


def string(path, key):
    return cards(path)[key]


def make(tmp_path, shape, name="scratch.image"):
    path = str(tmp_path / name)
    ct.image(imagename=path, shape=shape)
    return ct.image(path)


# ---------------------------------------------------------------------------
# Header structure
# ---------------------------------------------------------------------------


def test_header_is_a_padded_primary_hdu(tmp_path):
    im = make(tmp_path, (3, 2, 8, 10))
    out = str(tmp_path / "x.fits")
    im.tofits(out)
    size = os.path.getsize(out)
    assert size % 2880 == 0
    assert string(out, "SIMPLE") == "T"
    assert number(out, "BITPIX") == -32.0
    assert b"SIMPLE  =" in open(out, "rb").read(80)


def test_naxis_and_axis_sizes_follow_the_fits_order(tmp_path):
    """FITS axis 1 is the fastest (numpy's last); the sizes are reversed."""
    im = make(tmp_path, (3, 2, 8, 10))
    out = str(tmp_path / "x.fits")
    im.tofits(out)
    assert number(out, "NAXIS") == 4.0
    assert [number(out, f"NAXIS{i}") for i in (1, 2, 3, 4)] == [10.0, 8.0, 2.0, 3.0]


def test_exactly_one_wcs_card_set_per_fits_axis(tmp_path):
    """The invariant `open` assumes on read-back: one CTYPE/CRVAL/CDELT/
    CRPIX group per axis.  A coordinate claiming the same axis twice used to
    emit two groups under one number, and the last write silently won."""
    im = make(tmp_path, (3, 2, 8, 10))
    out = str(tmp_path / "x.fits")
    im.tofits(out)
    keys = [k for k, _ in raw_cards(out)]
    naxis = int(number(out, "NAXIS"))
    for prefix in ("CTYPE", "CRVAL", "CDELT", "CRPIX"):
        for axis in range(1, naxis + 1):
            assert keys.count(f"{prefix}{axis}") == 1, (prefix, axis)


def test_the_header_terminates_with_end(tmp_path):
    im = make(tmp_path, (2, 2))
    out = str(tmp_path / "x.fits")
    im.tofits(out)
    header = open(out, "rb").read(2880)
    assert header.startswith(b"SIMPLE  =")
    assert b"END" in header


# ---------------------------------------------------------------------------
# The 4-D fixture: cards and data casacore would write
# ---------------------------------------------------------------------------


def test_fixture_direction_cards(tmp_path):
    ct.image(IMAGE_FIXTURE).tofits(str(tmp_path / "f.fits"))
    out = str(tmp_path / "f.fits")
    assert string(out, "CTYPE1") == "RA---SIN"
    assert string(out, "CTYPE2") == "DEC--SIN"
    assert number(out, "CRVAL1") == pytest.approx(1.75)
    assert number(out, "CRVAL2") == pytest.approx(-0.45)
    assert number(out, "CDELT1") == pytest.approx(-2.5e-5)
    assert number(out, "CDELT2") == pytest.approx(3.0e-5)
    # CRPIX is 1-based in FITS: casa's 0-based 4/3 becomes 5/4.
    assert number(out, "CRPIX1") == pytest.approx(5.0)
    assert number(out, "CRPIX2") == pytest.approx(4.0)
    assert string(out, "CUNIT1") == "deg"
    assert string(out, "CUNIT2") == "deg"


def test_fixture_stokes_and_frequency_cards(tmp_path):
    ct.image(IMAGE_FIXTURE).tofits(str(tmp_path / "f.fits"))
    out = str(tmp_path / "f.fits")
    assert string(out, "CTYPE3") == "STOKES"
    assert number(out, "CRVAL3") == pytest.approx(1.0)
    assert string(out, "CUNIT3") == ""
    assert string(out, "CTYPE4") == "FREQ"
    assert number(out, "CRVAL4") == pytest.approx(1.4e9)
    assert number(out, "CDELT4") == pytest.approx(2.0e6)
    assert string(out, "CUNIT4") == "Hz"


def test_fixture_beam_and_unit_cards(tmp_path):
    """BMAJ/BMIN are degrees in FITS; the arcsec imageinfo values are
    converted back (12.6" -> 3.5e-3 deg)."""
    ct.image(IMAGE_FIXTURE).tofits(str(tmp_path / "f.fits"))
    out = str(tmp_path / "f.fits")
    assert number(out, "BMAJ") == pytest.approx(3.5e-3)
    assert number(out, "BMIN") == pytest.approx(2.5e-3)
    assert number(out, "BPA") == pytest.approx(15.0)
    assert string(out, "BUNIT") == "Jy/beam"


def test_fixture_data_is_written_in_fits_axis_order(tmp_path):
    im = ct.image(IMAGE_FIXTURE)
    out = str(tmp_path / "f.fits")
    im.tofits(out)
    assert ct.image(out).getdata().shape == im.getdata().shape
    np.testing.assert_array_equal(ct.image(out).getdata(), im.getdata())


def test_tofits_replaces_an_existing_file(tmp_path):
    im = make(tmp_path, (2, 2))
    out = str(tmp_path / "x.fits")
    im.tofits(out)
    im.tofits(out)
    assert number(out, "NAXIS") == 2.0


# ---------------------------------------------------------------------------
# The default template: the D3 regression
# ---------------------------------------------------------------------------


def test_default_2d_cards_match_casacore(tmp_path):
    """A 5x7 default image used to emit CDELT1 = -57.29577951 (the template's
    `-1'` treated as radians), CRPIX 4.5/3.5 and no CUNIT at all."""
    make(tmp_path, (5, 7)).tofits(str(tmp_path / "two.fits"))
    out = str(tmp_path / "two.fits")
    assert number(out, "NAXIS1") == 7.0 and number(out, "NAXIS2") == 5.0
    assert string(out, "CTYPE1") == "RA---SIN"
    assert string(out, "CTYPE2") == "DEC--SIN"
    assert number(out, "CRVAL1") == pytest.approx(0.0)
    assert number(out, "CRVAL2") == pytest.approx(0.0)
    assert number(out, "CDELT1") == pytest.approx(-1.0 / 60.0)
    assert number(out, "CDELT2") == pytest.approx(1.0 / 60.0)
    assert number(out, "CRPIX1") == pytest.approx(4.0)
    assert number(out, "CRPIX2") == pytest.approx(3.0)
    assert string(out, "CUNIT1") == "deg"
    assert string(out, "CUNIT2") == "deg"


def test_default_2d_has_no_spectral_or_stokes_cards(tmp_path):
    """The layout regression: a 2-D image used to emit three coordinates,
    and the spectral/stokes world values overwrote the direction ones."""
    make(tmp_path, (5, 7)).tofits(str(tmp_path / "two.fits"))
    out = str(tmp_path / "two.fits")
    assert number(out, "NAXIS") == 2.0
    assert "CTYPE3" not in cards(out)
    assert string(out, "CTYPE1").startswith("RA")


def test_1d_default_is_a_frequency_axis(tmp_path):
    make(tmp_path, (9,)).tofits(str(tmp_path / "one.fits"))
    out = str(tmp_path / "one.fits")
    assert number(out, "NAXIS") == 1.0
    assert string(out, "CTYPE1") == "FREQ"
    assert number(out, "CRVAL1") == pytest.approx(1.415e9)
    assert number(out, "CDELT1") == pytest.approx(1000.0)
    assert number(out, "CRPIX1") == pytest.approx(1.0)
    assert string(out, "CUNIT1") == "Hz"


def test_3d_default_has_a_stokes_axis(tmp_path):
    make(tmp_path, (3, 2, 9)).tofits(str(tmp_path / "three.fits"))
    out = str(tmp_path / "three.fits")
    assert number(out, "NAXIS") == 3.0
    assert string(out, "CTYPE3") == "STOKES"
    assert number(out, "CRVAL3") == pytest.approx(1.0)
    assert string(out, "CUNIT3") == ""


def test_4d_default_card_sets(tmp_path):
    make(tmp_path, (3, 2, 8, 10)).tofits(str(tmp_path / "four.fits"))
    out = str(tmp_path / "four.fits")
    assert [string(out, f"CTYPE{i}") for i in (1, 2, 3, 4)] == [
        "RA---SIN",
        "DEC--SIN",
        "STOKES",
        "FREQ",
    ]
    # casacore's default reference pixel is shape/2, 1-based in FITS.
    assert number(out, "CRPIX1") == pytest.approx(6.0)
    assert number(out, "CRPIX2") == pytest.approx(5.0)
    assert number(out, "CRVAL4") == pytest.approx(1.415e9)


def test_created_image_without_units_has_no_bunit(tmp_path):
    make(tmp_path, (2, 2)).tofits(str(tmp_path / "u.fits"))
    assert "BUNIT" not in cards(str(tmp_path / "u.fits"))


def test_created_image_without_a_beam_has_no_beam_cards(tmp_path):
    make(tmp_path, (2, 2)).tofits(str(tmp_path / "u.fits"))
    out = str(tmp_path / "u.fits")
    assert "BMAJ" not in cards(out) and "BPA" not in cards(out)


def test_round_trip_of_every_default_rank(tmp_path):
    """`tofits` then `open` must reproduce the world grid for 1..4 axes,
    which is only true when the direction angles are unit-converted on the
    way out and back."""
    for shape, probes in (
        ((9,), [(0,), (4,)]),
        ((5, 7), [(2, 3), (2.4, 3.0)]),
        ((3, 2, 9), [(0, 0, 4), (1, 1, 5)]),
        ((3, 2, 9, 8), [(0, 0, 4, 4), (1, 1, 4.4, 4.0)]),
    ):
        im = make(tmp_path, shape, name=f"rt{len(shape)}.image")
        out = str(tmp_path / f"rt{len(shape)}.fits")
        im.tofits(out)
        back = ct.image(out)
        assert list(back.shape()) == list(shape)
        for px in probes:
            assert back.toworld(px) == pytest.approx(im.toworld(px), abs=1e-12), (shape, px)


def test_round_trip_of_the_fixture_cube(tmp_path):
    im = ct.image(IMAGE_FIXTURE)
    out = str(tmp_path / "rt.fits")
    im.tofits(out)
    back = ct.image(out)
    np.testing.assert_array_equal(back.getdata(), im.getdata())
    for px in [(0, 0, 0, 0), (1, 1, 2, 3), (2, 0, 7, 9)]:
        assert back.toworld(px) == pytest.approx(im.toworld(px), abs=1e-9)


def test_tofits_of_an_in_memory_regrid_result(tmp_path):
    """ModMosaic regrids facets in memory and then exports the stack."""
    im = ct.image(IMAGE_FIXTURE)
    out_image = im.regrid([2, 3], im.coordinates(), outshape=[1, 1, 4, 4])
    out = str(tmp_path / "stacked.fits")
    out_image.tofits(out)
    back = ct.image(out)
    assert list(back.shape()) == [1, 1, 4, 4]
    np.testing.assert_array_equal(back.getdata(), out_image.getdata())


def test_tofits_of_a_fits_image_round_trips(tmp_path):
    src = ct.image(FITS_FIXTURE)
    out = str(tmp_path / "again.fits")
    src.tofits(out)
    np.testing.assert_array_equal(ct.image(out).getdata(), src.getdata())


# ---------------------------------------------------------------------------
# saveas
# ---------------------------------------------------------------------------


def test_saveas_writes_a_readable_casa_table(tmp_path):
    src = ct.image(IMAGE_FIXTURE)
    out = str(tmp_path / "saved.image")
    src.saveas(out)
    assert os.path.isfile(os.path.join(out, "table.info"))
    np.testing.assert_array_equal(ct.image(out).getdata(), src.getdata())


def test_saveas_preserves_the_beams_and_units(tmp_path):
    src = ct.image(IMAGE_FIXTURE)
    out = str(tmp_path / "saved.image")
    src.saveas(out)
    copy = ct.image(out)
    assert copy.unit() == src.unit()
    assert copy.imageinfo()["restoringbeam"] == src.imageinfo()["restoringbeam"]


def test_saveas_then_tofits_of_a_created_image(tmp_path):
    src = make(tmp_path, (3, 2, 8, 10), "src.image")
    out = str(tmp_path / "dst.image")
    src.saveas(out)
    fits_out = str(tmp_path / "dst.fits")
    ct.image(out).tofits(fits_out)
    assert number(fits_out, "NAXIS") == 4.0


# ---------------------------------------------------------------------------
# Third-party readers
# ---------------------------------------------------------------------------


@needs_astropy
def test_astropy_reads_the_fixture_export(tmp_path):
    im = ct.image(IMAGE_FIXTURE)
    out = str(tmp_path / "f.fits")
    im.tofits(out)
    with astropy_fits.open(out) as hdul:
        header = hdul[0].header
        assert header["NAXIS"] == 4
        assert [header[f"CTYPE{i}"] for i in range(1, 5)] == [
            "RA---SIN",
            "DEC--SIN",
            "STOKES",
            "FREQ",
        ]
        assert header["BMAJ"] == pytest.approx(3.5e-3)
        assert header["BUNIT"].strip() == "Jy/beam"
        assert header["CRPIX1"] == pytest.approx(5.0)
        np.testing.assert_array_equal(hdul[0].data, im.getdata())


@needs_astropy
def test_astropy_reads_a_default_template_export(tmp_path):
    make(tmp_path, (5, 7)).tofits(str(tmp_path / "two.fits"))
    with astropy_fits.open(str(tmp_path / "two.fits")) as hdul:
        header = hdul[0].header
        assert header["NAXIS"] == 2
        assert header["CDELT1"] == pytest.approx(-1.0 / 60.0)
        assert header["CUNIT1"] == "deg"
        assert header["CRPIX1"] == pytest.approx(4.0)


@needs_casacore
def test_fixture_export_cards_match_casacore(tmp_path):
    """The direction/stokes/frequency/beam cards, against casacore itself."""
    reference = casacore_image(IMAGE_FIXTURE)
    reference.tofits(str(tmp_path / "ref.fits"))
    ct.image(IMAGE_FIXTURE).tofits(str(tmp_path / "got.fits"))
    want, got = cards(str(tmp_path / "ref.fits")), cards(str(tmp_path / "got.fits"))
    for key in (
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "NAXIS3",
        "NAXIS4",
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CTYPE4",
        "CRVAL1",
        "CRVAL2",
        "CRVAL3",
        "CRVAL4",
        "CDELT1",
        "CDELT2",
        "CDELT3",
        "CDELT4",
        "CRPIX1",
        "CRPIX2",
        "CRPIX3",
        "CRPIX4",
        "CUNIT1",
        "CUNIT2",
        "CUNIT4",
        "BMAJ",
        "BMIN",
        "BPA",
    ):
        try:
            assert float(want[key]) == pytest.approx(float(got[key]), rel=1e-9), key
        except ValueError:
            assert want[key].strip("'").strip() == got[key].strip("'").strip(), key
    assert want["BUNIT"].strip("'").strip() == got["BUNIT"].strip("'").strip()


@needs_casacore
def test_default_2d_export_cards_match_casacore(tmp_path):
    """The whole point of the D3 fix: a scratch image casacore would have
    described identically."""
    casacore_image(imagename=str(tmp_path / "ref.image"), shape=[5, 7]).tofits(
        str(tmp_path / "ref.fits")
    )
    make(tmp_path, (5, 7)).tofits(str(tmp_path / "got.fits"))
    want, got = cards(str(tmp_path / "ref.fits")), cards(str(tmp_path / "got.fits"))
    for key in (
        "NAXIS1",
        "NAXIS2",
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CDELT1",
        "CDELT2",
        "CRPIX1",
        "CRPIX2",
        "CUNIT1",
        "CUNIT2",
    ):
        try:
            assert float(want[key]) == pytest.approx(float(got[key]), rel=1e-9), key
        except ValueError:
            assert want[key].strip("'").strip() == got[key].strip("'").strip(), key


@needs_casacore
def test_casacore_reads_a_casacure_export(tmp_path):
    im = ct.image(IMAGE_FIXTURE)
    out = str(tmp_path / "f.fits")
    im.tofits(out)
    reference = casacore_image(out)
    np.testing.assert_array_equal(reference.getdata(), im.getdata())
    for px in [(0, 0, 0, 0), (1, 1, 2, 3)]:
        assert list(reference.toworld(px)) == pytest.approx(im.toworld(px), abs=1e-9)


# ---------------------------------------------------------------------------
# A documented divergence
# ---------------------------------------------------------------------------


def test_synthesised_spectral_axis_is_written_as_frequency(tmp_path):
    """Known divergence: for a *synthesised* spectral coordinate casacore's
    `tofits` converts the axis to optical velocity (``CTYPE4 = 'VOPT'``,
    m/s), because its default template carries a rest frequency.  casacure
    writes the frequency it actually stores.  The written axis round-trips
    through `open` either way, and a real image (whose spectral coordinate
    came from a file) is written as ``FREQ`` by both."""
    make(tmp_path, (3, 2, 9, 8)).tofits(str(tmp_path / "d.fits"))
    out = str(tmp_path / "d.fits")
    assert string(out, "CTYPE4") == "FREQ"
    assert string(out, "CUNIT4") == "Hz"
    assert number(out, "CRVAL4") == pytest.approx(1.415e9)
    # and it reads back as the same frequency axis
    assert ct.image(out).toworld((0, 0, 4, 4))[0] == pytest.approx(1.415e9)
