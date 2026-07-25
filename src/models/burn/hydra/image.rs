//! Image preprocessing for Hydra-3.5.

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, Rgb, RgbImage};
use image::imageops::FilterType;
use image::ImageDecoder;
use ndarray::{Array2, Array3, Array4, Axis, s};

use crate::models::image_utils::{apply_exif_orientation, apply_icc_profile};

const PATCH_SIZE: usize = 16;
const POS_EMBED_H: usize = 16;
const POS_EMBED_W: usize = 16;
const EMBED_DIM: usize = 1152;

pub struct PreprocessedImage {
    pub patches: Array2<f32>,
    pub valid: Array2<bool>,
    pub pos_embed: Array2<f32>,
}

pub fn preprocess(
    image_path: &Path,
    pos_embed: &Array4<f32>,
    max_seq_len: usize,
    background: [u8; 3],
    pos_embed_cache: &mut std::collections::HashMap<(usize, usize), Array2<f32>>,
) -> Result<PreprocessedImage> {
    let icc_profile = image::ImageReader::open(image_path)
        .ok()
        .and_then(|r| r.into_decoder().ok())
        .and_then(|mut d| d.icc_profile().ok())
        .flatten();

    let img = image::open(image_path)
        .with_context(|| format!("failed to open image: {}", image_path.display()))?;
    let mut rgb = flatten_alpha(img, background);
    rgb = apply_exif_orientation(image_path, rgb)
        .with_context(|| format!("failed to apply EXIF orientation: {}", image_path.display()))?;

    if let Some(icc) = icc_profile {
        rgb = apply_icc_profile(&icc, rgb)
            .with_context(|| format!("failed to apply ICC profile: {}", image_path.display()))?;
    }

    let (orig_w, orig_h) = (rgb.width() as usize, rgb.height() as usize);
    let (resize_h, resize_w) = compute_resize_for_seq(orig_h, orig_w, PATCH_SIZE, max_seq_len);

    // Hydra metadata requests mks2013-linear, but image crate does not expose it.
    // Use Lanczos3 for the spike; revisit if output diverges significantly.
    let resized = image::imageops::resize(&rgb, resize_w as u32, resize_h as u32, FilterType::Lanczos3);

    let grid_h = resize_h / PATCH_SIZE;
    let grid_w = resize_w / PATCH_SIZE;
    let n_valid = grid_h * grid_w;

    let raw = resized.into_raw();
    let arr = Array3::from_shape_vec((resize_h, resize_w, 3), raw)
        .context("failed to build resized image array")?;
    let patches_view = arr
        .into_shape_with_order((grid_h, PATCH_SIZE, grid_w, PATCH_SIZE, 3))
        .context("failed to reshape into patches")?
        .permuted_axes([0, 2, 1, 3, 4]);
    let patches = patches_view
        .as_standard_layout()
        .into_shape_with_order((n_valid, PATCH_SIZE * PATCH_SIZE * 3))
        .context("failed to flatten patches")?;

    let mut patches_padded = Array2::<f32>::zeros((max_seq_len, PATCH_SIZE * PATCH_SIZE * 3));
    patches_padded.slice_mut(s![..n_valid, ..]).assign(&patches.mapv(|v| v as f32));

    let mut valid = Array2::<bool>::from_elem((1, max_seq_len), false);
    valid.slice_mut(s![0, ..n_valid]).fill(true);

    // The interpolated position embedding depends only on the patch grid
    // size, which the resize search quantizes to a handful of distinct values
    // across a collection — cache it instead of re-interpolating per image.
    let pos_embed_padded = match pos_embed_cache.get(&(grid_h, grid_w)) {
        Some(cached) => cached.clone(),
        None => {
            let interpolated = interpolate_pos_embed(pos_embed, grid_h, grid_w, max_seq_len)
                .context("failed to interpolate position embedding")?;
            pos_embed_cache.insert((grid_h, grid_w), interpolated.clone());
            interpolated
        }
    };

    // Normalize to [-1, 1].
    patches_padded /= 127.5;
    patches_padded -= 1.0;

    Ok(PreprocessedImage {
        patches: patches_padded,
        valid,
        pos_embed: pos_embed_padded,
    })
}

/// Binary-search for the largest resize that keeps the patch count within
/// `max_seq_len`, matching Hydra's `get_image_size_for_seq`.
fn compute_resize_for_seq(
    orig_h: usize,
    orig_w: usize,
    patch_size: usize,
    max_seq_len: usize,
) -> (usize, usize) {
    let max_ratio: f64 = 1.0;
    let eps: f64 = 1e-5;

    let max_py = ((orig_h as f64 * max_ratio) / patch_size as f64).max(1.0) as usize;
    let max_px = ((orig_w as f64 * max_ratio) / patch_size as f64).max(1.0) as usize;

    if max_py * max_px <= max_seq_len {
        return (max_py * patch_size, max_px * patch_size);
    }

    let patchify = |ratio: f64| -> (usize, usize) {
        let py = ((orig_h as f64 * ratio) / patch_size as f64)
            .ceil()
            .min(max_py as f64) as usize;
        let px = ((orig_w as f64 * ratio) / patch_size as f64)
            .ceil()
            .min(max_px as f64) as usize;
        (py.max(1), px.max(1))
    };

    let (mut py, mut px) = patchify(eps);
    if py * px > max_seq_len {
        return (patch_size, patch_size);
    }

    let mut ratio = eps;
    let mut max_ratio = max_ratio;
    while (max_ratio - ratio) >= eps {
        let mid = (ratio + max_ratio) / 2.0;
        let (mpy, mpx) = patchify(mid);
        let seq_len = mpy * mpx;

        if seq_len > max_seq_len {
            max_ratio = mid;
            continue;
        }

        ratio = mid;
        py = mpy;
        px = mpx;

        if seq_len == max_seq_len {
            break;
        }
    }

    (py * patch_size, px * patch_size)
}

/// Flatten an RGBA image against a solid background color.
/// Non-RGBA images are converted to RGB8 as-is.
fn flatten_alpha(img: DynamicImage, background: [u8; 3]) -> RgbImage {
    match img {
        DynamicImage::ImageRgba8(rgba) => {
            let (w, h) = (rgba.width(), rgba.height());
            let mut rgb = RgbImage::new(w, h);
            for (x, y, pixel) in rgba.enumerate_pixels() {
                let a = pixel[3] as f32 / 255.0;
                let r = (a * pixel[0] as f32 + (1.0 - a) * background[0] as f32).round() as u8;
                let g = (a * pixel[1] as f32 + (1.0 - a) * background[1] as f32).round() as u8;
                let b = (a * pixel[2] as f32 + (1.0 - a) * background[2] as f32).round() as u8;
                rgb.put_pixel(x, y, Rgb([r, g, b]));
            }
            rgb
        }
        _ => img.to_rgb8(),
    }
}

/// Bilinearly interpolate the learned 16x16 position embedding.
fn interpolate_pos_embed(
    pos_embed: &Array4<f32>,
    grid_h: usize,
    grid_w: usize,
    max_seq_len: usize,
) -> Result<Array2<f32>> {
    let view = pos_embed.index_axis(Axis(0), 0); // (16, 16, 1152)
    let mut out = Array2::<f32>::zeros((max_seq_len, EMBED_DIM));

    for y_out in 0..grid_h {
        let y_src = (y_out as f64 + 0.5) * (POS_EMBED_H as f64 / grid_h as f64) - 0.5;
        let y0 = y_src.floor() as isize;
        let dy = (y_src - y0 as f64) as f32;

        for x_out in 0..grid_w {
            let x_src = (x_out as f64 + 0.5) * (POS_EMBED_W as f64 / grid_w as f64) - 0.5;
            let x0 = x_src.floor() as isize;
            let dx = (x_src - x0 as f64) as f32;

            let row = y_out * grid_w + x_out;
            for c in 0..EMBED_DIM {
                let v00 = sample(&view, y0, x0, c);
                let v01 = sample(&view, y0, x0 + 1, c);
                let v10 = sample(&view, y0 + 1, x0, c);
                let v11 = sample(&view, y0 + 1, x0 + 1, c);

                let v0 = v00 * (1.0 - dx) + v01 * dx;
                let v1 = v10 * (1.0 - dx) + v11 * dx;
                out[[row, c]] = v0 * (1.0 - dy) + v1 * dy;
            }
        }
    }

    Ok(out)
}

#[inline]
fn sample(view: &ndarray::ArrayView3<f32>, y: isize, x: isize, c: usize) -> f32 {
    let h = view.shape()[0] as isize;
    let w = view.shape()[1] as isize;
    let y = y.clamp(0, h - 1) as usize;
    let x = x.clamp(0, w - 1) as usize;
    view[[y, x, c]]
}
