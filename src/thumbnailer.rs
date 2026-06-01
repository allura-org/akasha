use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub enum CacheMode {
    Disabled,
    Global,
    PerFolder,
    Custom(PathBuf),
}

impl CacheMode {
    pub fn from_config(mode: &str, custom_path: &str) -> Self {
        match mode {
            "disabled" => CacheMode::Disabled,
            "global" => CacheMode::Global,
            "per_folder" => CacheMode::PerFolder,
            "custom" => CacheMode::Custom(PathBuf::from(custom_path)),
            _ => CacheMode::Global,
        }
    }
}

pub struct Thumbnailer {
    pub size: u32,
    pub cache_mode: CacheMode,
}

impl Thumbnailer {
    pub fn new(size: u32, cache_mode: CacheMode) -> Self {
        Self { size, cache_mode }
    }

    pub fn cache_path_for(&self, hash: &str, folder_root: Option<&Path>) -> Option<PathBuf> {
        match &self.cache_mode {
            CacheMode::Disabled => None,
            CacheMode::Global => {
                directories::ProjectDirs::from("", "", "akasha")
                    .map(|dirs| sharded_cache_path(dirs.cache_dir(), hash, self.size))
            }
            CacheMode::PerFolder => {
                folder_root.and_then(|root| {
                    directories::ProjectDirs::from("", "", "akasha").map(|dirs| {
                        let base = dirs.cache_dir().join("per_folder").join(
                            blake3::hash(root.as_os_str().as_encoded_bytes()).to_hex().as_str(),
                        );
                        sharded_cache_path(&base, hash, self.size)
                    })
                })
            }
            CacheMode::Custom(base) => Some(sharded_cache_path(base, hash, self.size)),
        }
    }

    /// Returns encoded thumbnail image bytes.
    /// For cached modes, reads from cache or generates and writes to cache.
    /// For disabled mode, generates in memory and returns bytes directly.
    pub fn load_thumbnail_bytes(
        &self,
        source: &Path,
        hash: &str,
        folder_root: Option<&Path>,
    ) -> anyhow::Result<Vec<u8>> {
        match &self.cache_mode {
            CacheMode::Disabled => generate_in_memory(source, self.size),
            _ => {
                let cache_path = self
                    .cache_path_for(hash, folder_root)
                    .ok_or_else(|| anyhow::anyhow!("No cache path available"))?;

                if !cache_path.exists() {
                    if let Some(parent) = cache_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let bytes = generate_in_memory(source, self.size)?;
                    std::fs::write(&cache_path, &bytes)?;
                    Ok(bytes)
                } else {
                    Ok(std::fs::read(&cache_path)?)
                }
            }
        }
    }
}

fn sharded_cache_path(base: &Path, hash: &str, size: u32) -> PathBuf {
    // 2-level hash prefix: aa/bb/{hash}_{size}.webp
    // Prevents ext4/xfs metadata stress with hundreds of thousands of files.
    let prefix1 = &hash[..2.min(hash.len())];
    let prefix2 = &hash[2..4.min(hash.len())];
    base.join(prefix1).join(prefix2).join(format!("{}_{}.webp", hash, size))
}

#[cfg(feature = "simd-thumbnails")]
fn generate_in_memory(source: &Path, size: u32) -> anyhow::Result<Vec<u8>> {
    use fast_image_resize::images::Image;

    let img = image::open(source)?;
    let (src_w, src_h) = (img.width(), img.height());

    // Convert to RGBA8
    let rgba = img.to_rgba8();
    let src_data = rgba.as_raw();

    let src_image = Image::from_vec_u8(
        src_w,
        src_h,
        src_data.to_vec(),
        fast_image_resize::PixelType::U8x4,
    )?;

    // Compute thumbnail dimensions preserving aspect ratio
    let (thumb_w, thumb_h) = if src_w > src_h {
        (size, (src_h * size / src_w).max(1))
    } else {
        ((src_w * size / src_h).max(1), size)
    };

    let mut dst_image = Image::new(thumb_w, thumb_h, fast_image_resize::PixelType::U8x4);

    let mut resizer = fast_image_resize::Resizer::new();
    resizer.resize(
        &src_image,
        &mut dst_image,
        &fast_image_resize::ResizeOptions::new(),
    )?;

    // Encode WebP via libwebp
    let encoder = webp::Encoder::new(
        dst_image.buffer(),
        webp::PixelLayout::Rgba,
        thumb_w,
        thumb_h,
    );
    let webp_bytes = encoder.encode_lossless();
    Ok(webp_bytes.to_vec())
}

#[cfg(not(feature = "simd-thumbnails"))]
fn generate_in_memory(source: &Path, size: u32) -> anyhow::Result<Vec<u8>> {
    let img = image::open(source)?;
    let thumb = img.resize(size, size, image::imageops::FilterType::Lanczos3);
    let mut bytes = Vec::new();
    // Use PNG as fallback if WebP encoding fails
    if thumb
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::WebP)
        .is_err()
    {
        bytes.clear();
        thumb.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)?;
    }
    Ok(bytes)
}
