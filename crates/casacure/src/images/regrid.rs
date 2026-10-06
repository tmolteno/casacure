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
    // Every regridded axis indexes `src_shape`/`src_frac` below: reject an
    // out-of-range axis here instead of panicking out of bounds through
    // pyo3.  DDFacet's ModMosaic passes the fixed spatial pair [2, 3], but
    // a caller mistake must surface as an error, not a `PanicException`.
    for &axis in axes {
        if axis >= ndim {
            return Err(ImageError::BadAxis { axis, ndim });
        }
    }
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
        // The interpolation is keyed by axis, so a repeated axis must not
        // double-count its weight (dedup: [2, 2] is [2]).
        let mut uniq_axes: Vec<usize> = axes.to_vec();
        uniq_axes.sort_unstable();
        uniq_axes.dedup();
        let ncorner = 1usize << uniq_axes.len();
        let mut corners = vec![0usize; ncorner * ndim];
        let mut weights = vec![1f64; ncorner];
        for slot in 0..ncorner {
            let corner = &mut corners[slot * ndim..][..ndim];
            let mut w = 1f64;
            for (a, &axis) in uniq_axes.iter().enumerate() {
                // A world->pixel roundtrip at the grid boundary lands a
                // few ulps outside it; tolerate half a pixel before a
                // sample counts as "no data".  `saturating_sub` keeps a
                // zero-length axis from underflowing (everything then
                // samples as no-data, which `sample` already reports).
                let limit = src_shape[axis].saturating_sub(1) as f64;
                if !(-0.5..=limit + 0.5).contains(&src_frac[axis]) {
                    w = 0.0;
                    continue;
                }
                let base = (src_frac[axis].floor().clamp(0.0, limit)) as usize;
                let frac = (src_frac[axis] - base as f64).clamp(0.0, 1.0);
                let hi = (base + 1).min(src_shape[axis].saturating_sub(1));
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
                if !uniq_axes.contains(&k) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An in-memory image over a default coordinate system, filled by `f`.
    fn memory(shape: &[usize], f: impl Fn(&[usize]) -> f32) -> Image {
        let ndim = shape.len();
        let mut data = Vec::with_capacity(shape.iter().product());
        let mut index = vec![0usize; ndim];
        for flat in 0..shape.iter().product::<usize>() {
            let mut rem = flat;
            for k in (0..ndim).rev() {
                index[k] = rem % shape[k];
                rem /= shape[k];
            }
            data.push(f(&index));
        }
        Image::Memory(crate::images::image::MemoryImage {
            data: ArrayValue {
                shape: shape.iter().map(|&d| d as u32).collect(),
                data: ArrayData::Float(data),
            },
            shape: shape.to_vec(),
            coords: CoordinateSystem::default_for(shape),
            meta: ImageMeta::default(),
        })
    }

    /// `Image` has no `Debug`, so `unwrap_err` is unavailable.
    fn regrid_err(
        src: &Image,
        axes: &[usize],
        target: &CoordinateSystem,
        outshape: &[usize],
    ) -> ImageError {
        match regrid(src, axes, target, outshape) {
            Ok(_) => panic!("expected an error for axes {axes:?} outshape {outshape:?}"),
            Err(e) => e,
        }
    }

    fn floats(img: &Image) -> Vec<f32> {
        match &img.getdata().unwrap().data {
            ArrayData::Float(v) => v.clone(),
            other => panic!("expected a float raster, got {other:?}"),
        }
    }

    #[test]
    fn regrid_rejects_an_out_of_range_axis() {
        let src = memory(&[3, 2, 8, 10], |_| 0.0);
        let target = CoordinateSystem::default_for(&[3, 2, 8, 10]);
        // Indexing `src_shape[axis]` here used to panic out through pyo3 as
        // a PanicException no consumer can catch as a message.
        for axis in [4usize, 9, 1000] {
            let err = regrid_err(&src, &[axis], &target, &[3, 2, 8, 10]);
            match err {
                ImageError::BadAxis { axis: got, ndim } => {
                    assert_eq!(got, axis);
                    assert_eq!(ndim, 4);
                    assert!(err.to_string().contains("is out of range"));
                }
                other => panic!("expected BadAxis, got {other:?}"),
            }
        }
    }

    #[test]
    fn regrid_rejects_a_wrong_rank_outshape() {
        let src = memory(&[3, 2, 8, 10], |_| 0.0);
        let target = CoordinateSystem::default_for(&[3, 2, 8, 10]);
        for outshape in [vec![4, 4], vec![2, 4, 4], vec![]] {
            let err = regrid_err(&src, &[2, 3], &target, &outshape);
            assert!(
                err.to_string().contains("must have 4 axes"),
                "{outshape:?}: {err}"
            );
        }
    }

    #[test]
    fn regrid_identity_is_bit_exact() {
        let shape = [5usize, 7];
        let src = memory(&shape, |idx| (idx[0] * 7 + idx[1]) as f32);
        let target = CoordinateSystem::default_for(&shape);
        let out = regrid(&src, &[0, 1], &target, &shape).unwrap();
        assert_eq!(out.shape(), &shape);
        assert_eq!(floats(&out), floats(&src));
    }

    #[test]
    fn regrid_copies_the_axes_it_does_not_touch() {
        // Constant channel/polarisation planes: a direction-only regrid may
        // average neighbouring pixels but must never mix planes.
        let shape = [3usize, 2, 5, 7];
        let src = memory(&shape, |idx| (100 * (idx[0] + 1) + idx[1]) as f32);
        let mut target = CoordinateSystem::default_for(&shape);
        // Halve only the direction increments: `increments()` is in CS
        // order, so entry 0 is the direction coordinate.
        let mut inc = target.increments();
        for v in inc[0].iter_mut() {
            *v *= 0.5;
        }
        target.set_increments(&inc);

        let out = regrid(&src, &[2, 3], &target, &shape).unwrap();
        let got = floats(&out);
        for channel in 0..3 {
            for pol in 0..2 {
                let want = (100 * (channel + 1) + pol) as f32;
                for y in 0..5 {
                    for x in 0..7 {
                        let at = ((channel * 2 + pol) * 5 + y) * 7 + x;
                        assert_eq!(got[at], want, "plane ({channel},{pol}) at ({y},{x})");
                    }
                }
            }
        }
    }

    #[test]
    fn duplicate_axes_do_not_double_count_their_weight() {
        let shape = [5usize, 7];
        let src = memory(&shape, |idx| (idx[0] * 7 + idx[1]) as f32);
        let mut target = CoordinateSystem::default_for(&shape);
        let mut inc = target.increments();
        for v in inc[0].iter_mut() {
            *v *= 0.5;
        }
        target.set_increments(&inc);

        // A repeated axis used to contribute its (1-frac) weight twice, so
        // `[0, 0]` collapsed the result towards zero.
        let once = regrid(&src, &[0], &target, &shape).unwrap();
        let doubled = regrid(&src, &[0, 0], &target, &shape).unwrap();
        let tripled = regrid(&src, &[0, 0, 0], &target, &shape).unwrap();
        assert_eq!(floats(&doubled), floats(&once));
        assert_eq!(floats(&tripled), floats(&once));
    }

    #[test]
    fn pixels_outside_the_source_read_zero() {
        let shape = [5usize, 7];
        let src = memory(&shape, |_| 1.0);
        let mut target = CoordinateSystem::default_for(&shape);
        // A much coarser target grid reaches past the source footprint.
        let mut inc = target.increments();
        for v in inc[0].iter_mut() {
            *v *= 4.0;
        }
        target.set_increments(&inc);

        let out = regrid(&src, &[0, 1], &target, &shape).unwrap();
        let got = floats(&out);
        assert_eq!(got[0], 0.0, "the corner is off the source");
        assert!(got.contains(&1.0), "the centre stays in it");
    }

    #[test]
    fn regrid_of_a_zero_length_axis_does_not_panic() {
        // Not reachable through `image(shape=...)` (casacore rejects a
        // zero-length hypercube) but a hand-built raster can carry one, and
        // `src_shape[axis] - 1` would underflow.
        let src = Image::Memory(crate::images::image::MemoryImage {
            data: ArrayValue {
                shape: vec![2, 0],
                data: ArrayData::Float(Vec::new()),
            },
            shape: vec![2, 0],
            coords: CoordinateSystem::default_for(&[2, 0]),
            meta: ImageMeta::default(),
        });
        let target = CoordinateSystem::default_for(&[2, 0]);
        let out = regrid(&src, &[1], &target, &[2, 0]).unwrap();
        assert!(floats(&out).is_empty());
    }

    #[test]
    fn the_result_carries_the_target_coordinates() {
        let shape = [5usize, 7];
        let src = memory(&shape, |_| 2.0);
        let target = CoordinateSystem::default_for(&shape);
        let out = regrid(&src, &[0, 1], &target, &[7, 5]).unwrap();
        assert_eq!(out.shape(), &[7, 5]);
        assert_eq!(out.path(), std::path::Path::new(""));
        match out {
            Image::Memory(m) => assert_eq!(m.coords.direction_cdelt(), target.direction_cdelt()),
            other => panic!(
                "expected an in-memory image, got {}",
                other.path().display()
            ),
        }
    }
}
