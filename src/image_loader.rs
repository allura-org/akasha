use std::path::Path;

/// Open an image file, with HEIF/HEIC support when the `heif` feature is enabled.
pub fn open_image(path: &Path) -> anyhow::Result<image::DynamicImage> {
    if is_heif(path) {
        #[cfg(feature = "heif")]
        {
            return open_heif(path);
        }
        #[cfg(not(feature = "heif"))]
        {
            return Err(anyhow::anyhow!(
                "HEIF/HEIC support is not enabled. \
                 Rebuild with --features heif after installing libheif-dev and libde265-dev."
            ));
        }
    }
    Ok(image::open(path)?)
}

/// Get image dimensions, with HEIF/HEIC support when the `heif` feature is enabled.
pub fn image_dimensions(path: &Path) -> anyhow::Result<(u32, u32)> {
    if is_heif(path) {
        #[cfg(feature = "heif")]
        {
            let ctx = libheif_rs::HeifContext::read_from_file(path.to_string_lossy().as_ref())?;
            let handle = ctx.primary_image_handle()?;
            return Ok((handle.width(), handle.height()));
        }
        #[cfg(not(feature = "heif"))]
        {
            return Err(anyhow::anyhow!(
                "HEIF/HEIC support is not enabled. \
                 Rebuild with --features heif after installing libheif-dev and libde265-dev."
            ));
        }
    }
    Ok(image::ImageReader::open(path)?.into_dimensions()?)
}

/// Return a format string for the image (e.g. "png", "heic").
pub fn image_format(path: &Path) -> Option<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    match ext.as_deref() {
        Some("heif") => Some("heif".to_string()),
        Some("heic") => Some("heic".to_string()),
        _ => image::ImageReader::open(path)
            .ok()
            .and_then(|r| r.format())
            .map(|f| format!("{:?}", f).to_lowercase()),
    }
}

fn is_heif(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref(),
        Some("heif") | Some("heic")
    )
}

#[cfg(feature = "heif")]
fn open_heif(path: &Path) -> anyhow::Result<image::DynamicImage> {
    use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

    let lib_heif = LibHeif::new();
    let ctx = HeifContext::read_from_file(path.to_string_lossy().as_ref())?;
    let handle = ctx.primary_image_handle()?;
    let img = lib_heif.decode(
        &handle,
        ColorSpace::Rgb(RgbChroma::Rgba),
        None,
    )?;

    let planes = img.planes();
    let interleaved = planes
        .interleaved
        .ok_or_else(|| anyhow::anyhow!("HEIF image has no interleaved plane"))?;

    let width = interleaved.width;
    let height = interleaved.height;
    let stride = interleaved.stride;
    let data = interleaved.data;

    // Copy row-by-row to handle stride padding
    let mut rgba = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        let row_start = y * stride;
        rgba.extend_from_slice(&data[row_start..row_start + width * 4]);
    }

    let rgba_img = image::RgbaImage::from_raw(width as u32, height as u32, rgba)
        .ok_or_else(|| anyhow::anyhow!("Invalid HEIF dimensions"))?;
    Ok(image::DynamicImage::ImageRgba8(rgba_img))
}
