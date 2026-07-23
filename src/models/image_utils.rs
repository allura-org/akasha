//! Shared image preprocessing utilities used by multiple model backends.

use std::path::Path;

use anyhow::{Context, Result};
use image::RgbImage;

/// Read the EXIF Orientation tag from `image_path` and apply the corresponding
/// transform to `rgb`.  If no orientation tag is present, `rgb` is returned
/// unchanged.  This matches PIL's `ImageOps.exif_transpose` behavior.
pub fn apply_exif_orientation(image_path: &Path, rgb: RgbImage) -> Result<RgbImage> {
    let file = std::fs::File::open(image_path)
        .with_context(|| format!("failed to open image for EXIF: {}", image_path.display()))?;
    let mut bufreader = std::io::BufReader::new(file);
    let exif_reader = exif::Reader::new();
    let exif = match exif_reader.read_from_container(&mut bufreader) {
        Ok(e) => e,
        Err(_) => return Ok(rgb),
    };

    let orientation_value = exif
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .unwrap_or(1);

    let oriented = match orientation_value {
        1 => rgb,
        2 => image::imageops::flip_horizontal(&rgb),
        3 => image::imageops::rotate180(&rgb),
        4 => image::imageops::flip_vertical(&rgb),
        5 => transpose_image(&rgb),
        6 => image::imageops::rotate90(&rgb),
        7 => image::imageops::rotate180(&transpose_image(&rgb)),
        8 => image::imageops::rotate270(&rgb),
        _ => rgb,
    };

    Ok(oriented)
}

/// Convert `rgb` from its embedded ICC profile to sRGB using qcms.
pub fn apply_icc_profile(icc: &[u8], rgb: RgbImage) -> Result<RgbImage> {
    let src_profile = qcms::Profile::new_from_slice(icc, false)
        .context("failed to parse embedded ICC profile")?;

    // Already sRGB; nothing to do.
    if src_profile.is_sRGB() {
        return Ok(rgb);
    }

    let dst_profile = qcms::Profile::new_sRGB();
    let transform = qcms::Transform::new(
        &src_profile,
        &dst_profile,
        qcms::DataType::RGB8,
        qcms::Intent::RelativeColorimetric,
    )
    .context("failed to create ICC transform")?;

    let (w, h) = (rgb.width(), rgb.height());
    let mut data = rgb.into_raw();
    transform.apply(&mut data);

    RgbImage::from_raw(w, h, data).context("failed to rebuild RGB image after ICC conversion")
}

/// Transpose rows and columns (mirror across the top-left to bottom-right
/// diagonal).  Equivalent to PIL's `Image.Transpose.TRANSPOSE`.
fn transpose_image(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(h, w);
    for (x, y, pixel) in img.enumerate_pixels() {
        out.put_pixel(y, x, *pixel);
    }
    out
}
