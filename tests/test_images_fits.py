"""`casacure.images` FITS reader: the hand-rolled primary-HDU decoder, its
scaling rules, and the error contract for files it refuses.

Consumer call-sites guarded here:

* DDFacet ``ClassCasaImage.py`` — ``image(imagename=<cube>.fits)`` ingestion
  of externally produced cubes (``BITPIX`` of 8/16/32/64/-32/-64, integer
  rasters with ``BSCALE``/``BZERO``).
* DDFacet ``fits2png.py`` — ``getdata()`` on a FITS image, whose dtype must
  survive (``float32`` cubes must not be widened, so downstream memory
  budgets hold).
* killMS ``MakeModelImage.py`` — opening a FITS model image to saveas into a
  CASA image.

Every file here is built byte-by-byte in the test, so the reader's contract
for malformed input is pinned rather than inferred from a fixture.
"""

import struct

import numpy as np
import pytest

from casacore.tables import table  # noqa: F401  (the shim: casacure.tables)

ct = pytest.importorskip("casacure.images")

FIXTURES = "tests/fixtures"

# The casacore-written FITS cube this module reads; tests/conftest.py skips
# the module when it is absent.
CASACORE_FIXTURES = ("image.fits",)

# A SIMPLE = T primary HDU is 80-column cards, padded to 2880 bytes.
BLOCK = 2880


def card(text):
    """One 80-byte card; `text` may be shorter (it is blank-padded)."""
    raw = text.ljust(80).encode()[:80]
    assert len(raw) == 80
    return raw


def num(key, value):
    """A numeric card in FITS fixed format: `KEY     = <value>`."""
    return card(f"{key:<8}= {value:>20}")


def text(key, value):
    """A quoted string card."""
    return card(f"{key:<8}= '{value}'")


def boolean(key, value):
    return card(f"{key:<8}= {'T' if value else 'F':>20}")


def hdu(cards, data=b""):
    """A complete primary HDU: header + data."""
    body = b"".join(cards) + card("END")
    body += b" " * ((-len(body)) % BLOCK)
    return body + data


def write(tmp_path, name, cards, data=b""):
    path = tmp_path / name
    path.write_bytes(hdu(cards, data))
    return str(path)


def primary(tmp_path, name, extra_cards=(), bitpix=-32, naxis=(), data=b""):
    cards = [boolean("SIMPLE", True), num("BITPIX", bitpix), num("NAXIS", len(naxis))]
    cards += [num(f"NAXIS{i + 1}", n) for i, n in enumerate(naxis)]
    cards += list(extra_cards)
    return write(tmp_path, name, cards, data)


# ---------------------------------------------------------------------------
# BITPIX / dtype contract
# ---------------------------------------------------------------------------


def test_float32_cube_keeps_its_dtype(tmp_path):
    """fits2png budgets memory per cube; a float32 FITS must not be widened
    to float64 on read (casacore's getdata keeps the on-disk type too)."""
    path = primary(
        tmp_path, "f32.fits", bitpix=-32, naxis=(2, 2), data=struct.pack(">4f", 1, 2, 3, 4)
    )
    im = ct.image(path)
    got = im.getdata()
    assert got.dtype == np.float32
    assert got.shape == (2, 2)
    np.testing.assert_array_equal(got, [[1.0, 2.0], [3.0, 4.0]])


def test_float64_cube_keeps_its_dtype(tmp_path):
    path = primary(
        tmp_path, "f64.fits", bitpix=-64, naxis=(2,), data=struct.pack(">2d", 1.5, -2.5)
    )
    im = ct.image(path)
    assert im.getdata().dtype == np.float64
    np.testing.assert_array_equal(im.getdata(), [1.5, -2.5])


@pytest.mark.parametrize(
    "bitpix,fmt,values",
    [
        (8, "B", [0, 1, 2, 255]),
        (16, "h", [-1, 0, 32767]),
        (32, "i", [-2, 0, 70000]),
        (64, "q", [-3, 0, 4294967296]),
    ],
)
def test_integer_rasters_decode_as_float64(tmp_path, bitpix, fmt, values):
    """Integer cubes are scaled to physical values (exactly, in f64), the
    way casacore's getdata reports an integer image."""
    path = primary(
        tmp_path,
        f"i{bitpix}.fits",
        bitpix=bitpix,
        naxis=(len(values),),
        data=struct.pack(f">{len(values)}{fmt}", *values),
    )
    got = ct.image(path).getdata()
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.asarray(values, dtype=np.float64))


def test_bscale_and_bzero_are_applied(tmp_path):
    """Externally produced cubes often store integers with a scale/offset."""
    path = primary(
        tmp_path,
        "scale.fits",
        extra_cards=[num("BSCALE", 2.0), num("BZERO", 100.0)],
        bitpix=16,
        naxis=(3,),
        data=struct.pack(">3h", 1, 2, 3),
    )
    np.testing.assert_array_equal(ct.image(path).getdata(), [102.0, 104.0, 106.0])


def test_unsigned_8bit_via_bzero(tmp_path):
    """The FITS convention for unsigned bytes: BITPIX 8 with BZERO 0.

    (BITPIX 8 has no sign, so the raw byte is the value here.)"""
    path = primary(
        tmp_path, "u8.fits", bitpix=8, naxis=(3,), data=bytes([0, 128, 255])
    )
    np.testing.assert_array_equal(ct.image(path).getdata(), [0.0, 128.0, 255.0])


def test_bscale_defaults_are_identity(tmp_path):
    path = primary(tmp_path, "id.fits", bitpix=16, naxis=(2,), data=struct.pack(">2h", -5, 5))
    np.testing.assert_array_equal(ct.image(path).getdata(), [-5.0, 5.0])


# ---------------------------------------------------------------------------
# The accepted header contract
# ---------------------------------------------------------------------------


def test_axis_shape_and_data_order(tmp_path):
    """NAXIS1 is the fastest axis; the pyrap/numpy shape is the reverse."""
    data = struct.pack(">6f", *range(6))
    path = primary(tmp_path, "order.fits", naxis=(3, 2), data=data)
    im = ct.image(path)
    assert list(im.shape()) == [2, 3]
    np.testing.assert_array_equal(im.getdata(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


def test_missing_naxis_is_treated_as_one(tmp_path):
    """A FITS axis count below NAXIS defaults each missing NAXISn to 1."""
    path = primary(tmp_path, "short.fits", naxis=(3,), data=struct.pack(">3f", 1, 2, 3))
    assert list(ct.image(path).shape()) == [3]


def test_trailing_padding_after_the_data_is_ignored(tmp_path):
    data = struct.pack(">3f", 1, 2, 3) + b"\0" * 8
    path = primary(tmp_path, "pad.fits", naxis=(3,), data=data)
    np.testing.assert_array_equal(ct.image(path).getdata(), [1.0, 2.0, 3.0])


def test_commentary_cards_without_values_are_accepted(tmp_path):
    """COMMENT/HISTORY and blank cards carry no value and must not stop the
    header scan."""
    cards = [
        boolean("SIMPLE", True),
        num("BITPIX", -32),
        num("NAXIS", 1),
        num("NAXIS1", 2),
        card("COMMENT a remark"),
        card("HISTORY produced by something"),
        card(""),
    ]
    path = write(tmp_path, "comment.fits", cards, struct.pack(">2f", 7, 8))
    np.testing.assert_array_equal(ct.image(path).getdata(), [7.0, 8.0])


def test_unknown_keywords_are_ignored(tmp_path):
    path = primary(
        tmp_path,
        "extra.fits",
        extra_cards=[text("BUNIT", "Jy/beam"), text("ORIGIN", "somewhere"), num("EQUINOX", 2000.0)],
        naxis=(2,),
        data=struct.pack(">2f", 1, 2),
    )
    assert list(ct.image(path).shape()) == [2]


def test_repeated_keywords_take_the_last_card(tmp_path):
    """A rewritten header keeps the final occurrence, as FITS readers do."""
    cards = [
        boolean("SIMPLE", True),
        num("BITPIX", -32),
        num("NAXIS", 1),
        num("NAXIS1", 9),
        num("NAXIS1", 2),
    ]
    path = write(tmp_path, "dup.fits", cards, struct.pack(">2f", 4, 5))
    assert list(ct.image(path).shape()) == [2]


# ---------------------------------------------------------------------------
# The rejection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cards,data",
    [
        ("no_simple.fits", [num("BITPIX", -32), num("NAXIS", 0)], b""),
        ("empty.fits", [], b""),
    ],
)
def test_missing_simple_card_is_reported_as_not_fits(tmp_path, name, cards, data):
    path = write(tmp_path, name, cards, data)
    with pytest.raises(RuntimeError, match=r"not a FITS file \(no SIMPLE card\)"):
        ct.image(path)


def test_non_fits_bytes_are_reported_as_not_fits(tmp_path):
    path = tmp_path / "random.bin"
    path.write_bytes(b"\x7fELF" + b"\0" * 4000)
    with pytest.raises(RuntimeError, match="not a FITS file"):
        ct.image(str(path))


def test_simple_false_is_unsupported(tmp_path):
    path = write(tmp_path, "f.fits", [boolean("SIMPLE", False), num("BITPIX", -32), num("NAXIS", 0)])
    with pytest.raises(RuntimeError, match="unsupported FITS feature: SIMPLE is not T"):
        ct.image(path)


def test_random_groups_are_unsupported(tmp_path):
    path = write(
        tmp_path,
        "groups.fits",
        [
            boolean("SIMPLE", True),
            num("BITPIX", -32),
            num("NAXIS", 0),
            boolean("GROUPS", True),
        ],
    )
    with pytest.raises(RuntimeError, match="unsupported FITS feature: random-groups records"):
        ct.image(path)


def test_extension_hdus_are_unsupported(tmp_path):
    """Only the primary HDU is read; a file that declares XTENSION in its
    primary header is refused rather than silently mistreated."""
    path = write(
        tmp_path,
        "ext.fits",
        [
            boolean("SIMPLE", True),
            num("BITPIX", -32),
            num("NAXIS", 0),
            text("XTENSION", "IMAGE"),
        ],
    )
    with pytest.raises(RuntimeError, match="unsupported FITS feature: extension HDUs"):
        ct.image(path)


@pytest.mark.parametrize("bitpix", [-16, -8, 0, 128])
def test_unsupported_bitpix_is_reported_with_its_value(tmp_path, bitpix):
    path = primary(tmp_path, f"bp{bitpix}.fits", bitpix=bitpix, naxis=(1,))
    with pytest.raises(RuntimeError, match=f"unsupported FITS feature: BITPIX {bitpix}"):
        ct.image(path)


@pytest.mark.parametrize("naxis", [-1, 99])
def test_absurd_naxis_is_rejected(tmp_path, naxis):
    path = primary(tmp_path, f"n{naxis}.fits", naxis=())
    # Overwrite the NAXIS card with the absurd value.
    path = write(
        tmp_path,
        f"n{naxis}.fits",
        [boolean("SIMPLE", True), num("BITPIX", -32), num("NAXIS", naxis)],
    )
    with pytest.raises(RuntimeError, match="unsupported FITS feature: missing/absurd NAXIS"):
        ct.image(path)


def test_missing_naxis_card_is_rejected(tmp_path):
    path = write(tmp_path, "nonaxis.fits", [boolean("SIMPLE", True), num("BITPIX", -32)])
    with pytest.raises(RuntimeError, match="missing/absurd NAXIS"):
        ct.image(path)


def test_truncated_data_region_is_rejected_at_open(tmp_path):
    """A cube cut short in transit must fail at open with the byte range
    that is missing, not panic while decoding (a sliced read past the
    mapping)."""
    path = primary(tmp_path, "trunc.fits", naxis=(2, 2), data=struct.pack(">4f", 1, 2, 3, 4))
    full = tmp_path / "trunc.fits"
    full.write_bytes(full.read_bytes()[:-6])  # keep 10 of the 16 data bytes
    with pytest.raises(RuntimeError, match=r"data region 2880\.\.2896 exceeds file length 2890"):
        ct.image(str(full))


def test_zero_data_bytes_are_rejected(tmp_path):
    """A header claiming pixels but no data at all."""
    path = write(tmp_path, "nodata.fits", [boolean("SIMPLE", True), num("BITPIX", -32), num("NAXIS", 0)])
    with pytest.raises(RuntimeError, match="data region"):
        ct.image(path)


def test_malformed_card_is_reported_with_its_offset(tmp_path):
    """An unparsable value must name the byte offset of the bad card."""
    path = write(
        tmp_path,
        "bad.fits",
        [boolean("SIMPLE", True), num("BITPIX", -32), card("NAXIS   =                  abc")],
    )
    with pytest.raises(RuntimeError, match=r"malformed card at byte 160: bad integer"):
        ct.image(path)


def test_header_without_an_end_card_is_reported_as_truncated(tmp_path):
    """The bytes started with SIMPLE, so blaming a missing SIMPLE card would
    send the reader looking for a card that is already there."""
    path = tmp_path / "noend.fits"
    path.write_bytes(card("SIMPLE  =                    T") + b" " * (BLOCK - 80))
    with pytest.raises(RuntimeError, match=r"not a FITS file \(no END card\)"):
        ct.image(str(path))


def test_a_directory_that_is_not_an_image_is_rejected(tmp_path):
    """The generic open error for a path that is neither a CASA image table
    nor a FITS file.  A directory that exists but holds no table must not
    surface the table layer's "No such file or directory"."""
    (tmp_path / "plain").mkdir()
    with pytest.raises(RuntimeError, match="no such image"):
        ct.image(str(tmp_path / "plain"))


def test_a_missing_path_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="no such image"):
        ct.image(str(tmp_path / "absent.image"))


# ---------------------------------------------------------------------------
# Metadata derived from FITS cards
# ---------------------------------------------------------------------------


def test_beam_cards_become_the_restoring_beam(tmp_path):
    """BMAJ/BMIN are in degrees in FITS, arcsec in casacore's imageinfo
    (12.6" = 3.5e-3 deg); BPA is degrees."""
    path = primary(
        tmp_path,
        "beam.fits",
        extra_cards=[
            num("BMAJ", 3.5e-3),
            num("BMIN", 2.5e-3),
            num("BPA", 15.0),
        ],
        naxis=(2,),
        data=struct.pack(">2f", 1, 2),
    )
    beam = ct.image(path).imageinfo()["restoringbeam"]
    assert beam["major"] == {"value": pytest.approx(12.6), "unit": "arcsec"}
    assert beam["minor"] == {"value": pytest.approx(9.0), "unit": "arcsec"}
    assert beam["positionangle"]["value"] == pytest.approx(15.0)


def test_no_beam_cards_means_no_restoring_beam(tmp_path):
    path = primary(tmp_path, "nobeam.fits", naxis=(2,), data=struct.pack(">2f", 1, 2))
    assert "restoringbeam" not in ct.image(path).imageinfo()


def test_unit_comes_from_bunit_and_is_quoted(tmp_path):
    """pyrap's `unit()` wraps the stored units keyword in quotes."""
    path = primary(
        tmp_path,
        "bunit.fits",
        extra_cards=[text("BUNIT", "Jy/beam")],
        naxis=(2,),
        data=struct.pack(">2f", 1, 2),
    )
    assert ct.image(path).unit() == "'Jy/beam'"


def test_missing_bunit_gives_an_empty_quoted_unit(tmp_path):
    path = primary(tmp_path, "nounit.fits", naxis=(2,), data=struct.pack(">2f", 1, 2))
    assert ct.image(path).unit() == "''"


def test_name_is_the_absolute_path(tmp_path):
    path = primary(tmp_path, "named.fits", naxis=(2,), data=struct.pack(">2f", 1, 2))
    assert ct.image(path).name() == str(tmp_path / "named.fits")


def test_fits_images_cannot_be_putdata_into(tmp_path):
    """A FITS file is read-only; the message must say so rather than
    reporting a storage failure."""
    path = primary(tmp_path, "ro.fits", naxis=(2,), data=struct.pack(">2f", 1, 2))
    im = ct.image(path)
    with pytest.raises(RuntimeError, match="cannot putdata into a FITS file"):
        im.putdata(np.zeros((2,), dtype=np.float32))


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
def test_fixture_fits_cube_matches_casacore():
    want = casacore_image(f"{FIXTURES}/image.fits")
    got = ct.image(f"{FIXTURES}/image.fits")
    np.testing.assert_array_equal(got.getdata(), want.getdata())
    assert got.getdata().dtype == want.getdata().dtype
    assert list(got.shape()) == list(want.shape())


@needs_casacore
def test_fixture_fits_beam_matches_casacore():
    want = casacore_image(f"{FIXTURES}/image.fits").imageinfo()["restoringbeam"]
    got = ct.image(f"{FIXTURES}/image.fits").imageinfo()["restoringbeam"]
    assert got == want or got["major"] == want["major"]
