#! /usr/bin/env python3
"""Ground-truth probe for the `casacure.images` port (issue #14).

Runs real python-casacore's `casacore.images.image` through exactly the call
patterns DDFacet/killMS use, on artifacts produced the way they produce them
(an astropy-written 4-D SIN/FREQ/STOKES FITS cube, plus the CASA `.image`
table casacore makes from it), and records every convention the Rust port
must reproduce:

- `getdata()` dtype / shape / axis order
- `coordinates().dict()` layout (and the coordsys get/set surface)
- `toworld` / `topixel` value conventions and round-trip behaviour
- `imageinfo()` restoring-beam structure and units
- the on-disk CASA-image table layout (columns, keywords, dminfo, table.info)
- `topixel` off-projection error type
- the FITS header cards `tofits()` writes

Usage:  python scripts/probe_pyrap_images.py [outdir]
Requires: python-casacore, numpy, astropy (a probe venv; see CHANGELOG notes).
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

NCH, NPOL, NY, NX = 3, 2, 8, 10


def make_fits(path):
    """A 4-D cube exactly the way DDFacet's ClassCasaImage writes one."""
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=4)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    w.wcs.crval = [1.75, -0.45, 1.0, 1.4e9]  # rad, rad, stokes index, Hz
    w.wcs.cdelt = [-2.5e-5, 3.0e-5, 1.0, 2.0e6]  # rad, rad, stokes, Hz
    w.wcs.crpix = [NX / 2.0, NY / 2.0, 1.0, 1.0]  # 1-based FITS
    w.wcs.crota = [0.0, 0.0, 0.0, 0.0]
    w.wcs.cunit = ["deg", "deg", "", "Hz"]  # deg for header; rad inside? probe!

    data = np.arange(NCH * NPOL * NY * NX, dtype=np.float32).reshape(
        NCH, NPOL, NY, NX
    )
    hdu = fits.PrimaryHDU(data)
    for k, v in w.to_header(relax=True).items():
        hdu.header[k] = v
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.header["BMAJ"] = 3.5e-3  # deg
    hdu.header["BMIN"] = 2.5e-3
    hdu.header["BPA"] = 15.0
    hdu.header["SPECSYS"] = "TOPOCENT"
    hdu.writeto(path, overwrite=True)
    return data


def jdefault(o):
    if isinstance(o, (np.ndarray, np.number)):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o.item()
    if isinstance(o, bytes):
        return o.decode()
    return repr(o)


def probe(out):
    from casacore.images import image
    from casacore.tables import table

    report = {}

    fits_path = os.path.join(out, "probe.fits")
    want = make_fits(fits_path)

    im = image(fits_path)
    d = im.getdata()
    report["fits_getdata"] = {
        "shape": list(d.shape),
        "dtype": str(d.dtype),
        "equals_astropy_order": bool(np.array_equal(d, want)),
        "corner_f0p0": float(d[0, 0, 0, 0]),
        "corner_last": float(d[-1, -1, -1, -1]),
    }
    report["fits_shape_method"] = list(im.shape())
    report["fits_name"] = im.name()

    c = im.coordinates()
    report["coordsys_dict"] = json.loads(json.dumps(c.dict(), default=jdefault))
    report["coordsys_dir"] = [n for n in dir(c) if not n.startswith("_")]

    tw00 = im.toworld((0, 0, 0, 0))
    tw11 = im.toworld((1, 1, 2, 3))
    report["toworld_0000"] = jdefault(tw00)
    report["toworld_1123"] = jdefault(tw11)
    report["toworld_tuple_0000"] = [jdefault(x) for x in tuple(tw00)]
    report["topixel_roundtrip_1123"] = [
        jdefault(x) for x in tuple(im.topixel(tuple(tw11)))
    ]

    ii = im.imageinfo()
    report["imageinfo"] = json.loads(json.dumps(ii, default=jdefault))

    report["miscinfo"] = json.loads(json.dumps(im.miscinfo(), default=jdefault))
    report["unit"] = jdefault(im.unit())

    # Off-projection world -> pixel behaviour.
    try:
        im.topixel((1.4e9, 1.0, 10.0, 10.0))
        report["topixel_offsky"] = "no error"
    except Exception as e:  # noqa: BLE001
        report["topixel_offsky"] = f"{type(e).__name__}: {e}"

    # CASA .image table via saveas, then the table-layer layout.
    casa_path = os.path.join(out, "probe.image")
    im.saveas(casa_path)
    t = table(casa_path, readonly=True, lockoptions="auto")
    report["casa_table"] = {
        "colnames": t.colnames(),
        "nrows": t.nrows(),
        "keywordnames": t.keywordnames(),
        "keywords": json.loads(json.dumps(t.getkeywords(), default=jdefault)),
        "dminfo": json.loads(json.dumps(t.getdminfo(), default=jdefault)),
        "coldesc_raster": json.loads(
            json.dumps(t.getcoldesc(t.colnames()[0]), default=jdefault)
        ),
    }
    with open(os.path.join(casa_path, "table.info")) as f:
        report["casa_table_info_file"] = f.read()
    t.close()

    casaim = image(casa_path)
    report["casa_getdata_matches_fits"] = bool(
        np.array_equal(casaim.getdata(), d)
    )
    report["casa_toworld_0000"] = [jdefault(x) for x in tuple(casaim.toworld((0, 0, 0, 0)))]
    report["casa_coordsys_dict"] = json.loads(
        json.dumps(casaim.coordinates().dict(), default=jdefault)
    )
    del casaim

    # tofits from the CASA image: the header our writer must reproduce.
    tofits_path = os.path.join(out, "probe_tofits.fits")
    im.tofits(tofits_path)
    from astropy.io import fits as afits

    with afits.open(tofits_path) as hdul:
        report["tofits_header"] = dict(hdul[0].header)
        report["tofits_data_equal"] = bool(
            np.array_equal(hdul[0].data, d.reshape(d.shape))
        )

    # Create-with-shape + coordsys (the SkyModel/Other/ClassCasaImage path).
    scratch = os.path.join(out, "scratch.image")
    c2 = image(casa_path).coordinates()
    inc = c2.get_increment()
    report["coordsys_get_increment"] = jdefault(inc)
    report["coordsys_get_referencevalue"] = jdefault(c2.get_referencevalue())
    report["coordsys_get_referencepixel"] = jdefault(c2.get_referencepixel())
    inc[-1][0] = -2.6e-5
    c2.set_increment(inc)
    im2 = image(imagename=scratch, shape=(NCH, NPOL, NY, NX), coordsys=c2)
    im2.putdata(d)
    report["created_getdata_roundtrip"] = bool(np.array_equal(im2.getdata(), d))
    report["created_coordsys_cdelt"] = jdefault(
        im2.coordinates().dict()["direction0"]["cdelt"]
    )
    # The private-attribute access MyCasapy2bbs relies on.
    report["coordsys_private_csys_keys"] = sorted(
        image(casa_path).coordinates().__dict__["_csys"].keys()
    )
    del im2
    del im
    return report


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="pyrap-probe-")
    os.makedirs(out, exist_ok=True)
    try:
        report = probe(out)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
    dst = os.path.join(out, "report.json")
    with open(dst, "w") as f:
        json.dump(report, f, indent=1, default=jdefault)
    shutil.copy(dst, "probe_report.json")
    print("wrote", dst, "and ./probe_report.json")
    print(json.dumps(report, indent=1, default=jdefault)[:12000])


if __name__ == "__main__":
    main()
