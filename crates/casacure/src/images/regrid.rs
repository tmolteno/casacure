//! `regrid` (issue #14 phase 3): resample an image's spatial axes onto a
//! target coordinate system — the ModMosaic mosaic-stacking operation
//! (`img.regrid([2, 3], cMain, outshape=(1, 1, N, N))`).
//!
//! The output raster is built in memory: every output pixel is mapped
//! world-ward through the target coordinates and back through the source
//! coordinates, then bilinearly sampled over the regridded axes (the
//! mosaic images being stacked share their projection and increment, so
//! this matches casacore's regrid where it matters; pixels outside the
//! source read as 0, like an unmasked regrid edge).

use crate::images::coordsys::CoordinateSystem;
use crate::images::image::{Image, ImageError};
use crate::images::write::ImageMeta;
use crate::record::{ArrayData, ArrayValue};

/// Regrid `src` onto `target`'s coordinates over the given numpy-order
/// `axes`, with the full output shape `outshape` (numpy order).  Returns
/// an in-memory image carrying the target coordinates.
pub fn regrid(
    src: &Image,
    axes: &[usize],
    target: &CoordinateSystem,
    outshape: &[usize],
) -> Result<Image, ImageError> {
    let src_shape = src.shape().to_vec();
    if outshape.len() != src_shape.len() {
        return Err(ImageError::Other {
            path: src.path().to_path_buf(),
            msg: format!(
                "regrid outshape {:?} must have {} axes",
                outshape,
                src_shape.len()
            ),
        });
    }
    let ndim = src_shape.len();
    // Numpy strides of the source.
    let mut src_strides = vec![1usize; ndim];
    for k in (0..ndim - 1).rev() {
        src_strides[k] = src_strides[k + 1] * src_shape[k + 1];
    }
    let src_flat = match &src.getdata()?.data {
        ArrayData::Float(v) => v.iter().map(|f| *f as f64).collect(),
        ArrayData::Double(v) => v.clone(),
        _ => {
            return Err(ImageError::Other {
                path: src.path().to_path_buf(),
                msg: "only numeric rasters regrid".into(),
            })
        }
    };
    let sample = |idx: &[usize]| -> f64 {
        let mut off = 0usize;
        for (k, &i) in idx.iter().enumerate() {
            if i >= src_shape[k] {
                return 0.0;
            }
            off += i * src_strides[k];
        }
        src_flat[off]
    };

    // Bilinear gather over the regridded axes; the other axes map 1:1.
    let mut out_flat = vec![0f64; outshape.iter().product()];
    let mut out_strides = vec![1usize; ndim];
    for k in (0..ndim - 1).rev() {
        out_strides[k] = out_strides[k + 1] * outshape[k + 1];
    }
    let mut out_index = vec![0usize; ndim];
    let mut src_frac = vec![0f64; ndim];
    for (o, out_val) in out_flat.iter_mut().enumerate() {
        // Decompose the flat output index into numpy axes (axis 0 is the
        // slowest in numpy order).
        for k in 0..ndim {
            out_index[k] = (o / out_strides[k]) % outshape[k];
        }
        // Output pixel (casa order = reversed numpy) -> world.
        let mut out_pixel: Vec<f64> = out_index.iter().map(|&i| i as f64).collect();
        out_pixel.reverse();
        let world = match target.to_world(&out_pixel) {
            Ok(w) => w,
            // Off the target's own projection: no data.
            Err(_) => continue,
        };
        // World -> source pixel (casa order), then back to numpy order.
        let src_pixel = match src.coordinates().to_pixel(&world) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for (k, v) in src_pixel.into_iter().enumerate() {
            src_frac[ndim - 1 - k] = v;
        }
        // Bilinear over the regridded axes; exact (round) on the others.
        let ncorner = 1usize << axes.len();
        let mut corners = vec![0usize; ncorner * ndim];
        let mut weights = vec![1f64; ncorner];
        for slot in 0..ncorner {
            let corner = &mut corners[slot * ndim..][..ndim];
            let mut w = 1f64;
            for (a, &axis) in axes.iter().enumerate() {
                // A world->pixel roundtrip at the grid boundary lands a
                // few ulps outside it; tolerate half a pixel before a
                // sample counts as "no data".
                let limit = (src_shape[axis] - 1) as f64;
                if !(-0.5..=limit + 0.5).contains(&src_frac[axis]) {
                    w = 0.0;
                    continue;
                }
                let base = (src_frac[axis].floor().clamp(0.0, limit)) as usize;
                let frac = (src_frac[axis] - base as f64).clamp(0.0, 1.0);
                let hi = (base + 1).min(src_shape[axis] - 1);
                let take_hi = slot & (1 << a) != 0;
                corner[axis] = if take_hi { hi } else { base };
                if take_hi {
                    w *= frac;
                } else {
                    w *= 1.0 - frac;
                }
            }
            // Non-regridded axes: round-to-nearest identity (their grids
            // match by construction; ModMosaic regrids only the spatial
            // pair and the spectral/stokes grids are shared).
            for k in 0..ndim {
                if !axes.contains(&k) {
                    corner[k] = src_frac[k].round() as usize;
                }
            }
            weights[slot] = w;
        }
        let mut acc = 0f64;
        for (slot, w) in weights.iter().enumerate() {
            if *w == 0.0 {
                continue;
            }
            acc += w * sample(&corners[slot * ndim..][..ndim]);
        }
        *out_val = acc;
    }

    let out_shape32: Vec<u32> = outshape.iter().map(|&d| d as u32).collect();
    Ok(Image::Memory(crate::images::image::MemoryImage {
        data: ArrayValue {
            shape: out_shape32,
            data: ArrayData::Float(out_flat.into_iter().map(|f| f as f32).collect()),
        },
        shape: outshape.to_vec(),
        coords: target.clone(),
        meta: ImageMeta::from_image(src),
    }))
}
