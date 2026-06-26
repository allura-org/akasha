use std::path::{Path, PathBuf};
use tracing::warn;

pub const TEMP_CACHE_DIR: &str = "/tmp/.akasha_thumbnails";

pub struct Thumbnailer {
    pub size: u32,
    /// Resolved global cache folder. Always an absolute path.
    pub global_cache_folder: PathBuf,
    /// If true, do not write to any cache.
    pub disable_cache: bool,
    /// If true, write to a temporary cache instead of the configured location.
    pub temporary_cache: bool,
    /// If true, skip all cache reads and always regenerate.
    pub no_cache_read: bool,
}

impl Thumbnailer {
    pub fn new(
        size: u32,
        global_cache_folder: PathBuf,
        disable_cache: bool,
        temporary_cache: bool,
        no_cache_read: bool,
    ) -> Self {
        let global_cache_folder = if global_cache_folder.as_os_str().is_empty() {
            directories::ProjectDirs::from("", "", "akasha")
                .map(|dirs| dirs.cache_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".cache/akasha"))
        } else {
            global_cache_folder
        };

        Self {
            size,
            global_cache_folder,
            disable_cache,
            temporary_cache,
            no_cache_read,
        }
    }

    /// Returns encoded thumbnail image bytes.
    ///
    /// Reads from cache in the configured order unless `no_cache_read` is set.
    /// Generates the thumbnail and writes it to the resolved cache location
    /// unless caching is disabled.
    pub fn load_thumbnail_bytes(
        &self,
        source: &Path,
        hash: &str,
        import_root: Option<&Path>,
        import_cache_mode: Option<&str>,
        import_cache_folder: Option<&str>,
        import_cache_fallback: Option<&str>,
    ) -> anyhow::Result<Vec<u8>> {
        // Determine which cache bases to try for reads, in order.
        let read_bases = self.read_bases(import_root, import_cache_mode, import_cache_folder);

        if !self.no_cache_read {
            for base in &read_bases {
                let cache_path = sharded_cache_path(base, hash, self.size);
                if cache_path.exists() {
                    return Ok(std::fs::read(&cache_path)?);
                }
            }
        }

        let bytes = generate_in_memory(source, self.size)?;

        if let Some(write_path) = self.write_path(
            import_root,
            import_cache_mode,
            import_cache_folder,
            import_cache_fallback,
            hash,
        ) {
            if let Some(parent) = write_path.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    warn!("Failed to create thumbnail cache dir {}: {e}", parent.display());
                }
            }
            if let Err(e) = std::fs::write(&write_path, &bytes) {
                warn!("Failed to write thumbnail cache {}: {e}", write_path.display());
            }
        }

        Ok(bytes)
    }

    /// Cache base directories to check when reading, in priority order.
    fn read_bases(
        &self,
        import_root: Option<&Path>,
        import_cache_mode: Option<&str>,
        import_cache_folder: Option<&str>,
    ) -> Vec<PathBuf> {
        let mut bases = Vec::new();

        if let Some(mode) = import_cache_mode {
            if mode != "global" {
                if let Some(folder) = import_cache_folder.filter(|s| !s.is_empty()) {
                    bases.push(PathBuf::from(folder));
                } else if let Some(root) = import_root {
                    bases.push(root.join(".akasha_thumbnails"));
                }
            }
        }

        bases.push(self.global_cache_folder.clone());
        bases
    }

    /// Full cache path to write to, or `None` if writing is disabled.
    fn write_path(
        &self,
        import_root: Option<&Path>,
        import_cache_mode: Option<&str>,
        import_cache_folder: Option<&str>,
        import_cache_fallback: Option<&str>,
        hash: &str,
    ) -> Option<PathBuf> {
        if self.disable_cache || self.temporary_cache {
            // Temporary cache is handled separately.
            if self.temporary_cache {
                return Some(sharded_cache_path(
                    Path::new(TEMP_CACHE_DIR),
                    hash,
                    self.size,
                ));
            }
            return None;
        }

        let mode = import_cache_mode.unwrap_or("global");
        match mode {
            "disabled" => None,
            "custom" => {
                let base = import_cache_folder
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .or_else(|| import_root.map(|r| r.join(".akasha_thumbnails")))?;

                if Self::dir_writable(&base) {
                    Some(sharded_cache_path(&base, hash, self.size))
                } else {
                    match import_cache_fallback.unwrap_or("disable") {
                        "global" => Some(sharded_cache_path(&self.global_cache_folder, hash, self.size)),
                        _ => None,
                    }
                }
            }
            "global" | _ => Some(sharded_cache_path(&self.global_cache_folder, hash, self.size)),
        }
    }

    fn dir_writable(path: &Path) -> bool {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                // Be optimistic: we'll try to create it later.
                return true;
            }
        }
        if path.exists() {
            return std::fs::metadata(path)
                .map(|m| m.is_dir())
                .unwrap_or(false);
        }
        true
    }
}

fn sharded_cache_path(base: &Path, hash: &str, size: u32) -> PathBuf {
    // 2-level hash prefix: aa/bb/{hash}_{size}.webp
    // Prevents ext4/xfs metadata stress with hundreds of thousands of files.
    let prefix1 = &hash[..2.min(hash.len())];
    let prefix2 = &hash[2..4.min(hash.len())];
    base.join(prefix1)
        .join(prefix2)
        .join(format!("{}_{}.webp", hash, size))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_cache_uses_resolved_default_folder() {
        let thumbnailer = Thumbnailer::new(256, PathBuf::new(), false, false, false);
        let expected = directories::ProjectDirs::from("", "", "akasha")
            .map(|d| d.cache_dir().to_path_buf())
            .unwrap();
        assert_eq!(thumbnailer.global_cache_folder, expected);
    }

    #[test]
    fn custom_global_cache_folder_is_preserved() {
        let thumbnailer = Thumbnailer::new(256, PathBuf::from("/custom/cache"), false, false, false);
        assert_eq!(thumbnailer.global_cache_folder, PathBuf::from("/custom/cache"));
    }

    #[test]
    fn disabled_cache_skips_writes() {
        let thumbnailer = Thumbnailer::new(256, PathBuf::new(), true, false, false);
        assert!(thumbnailer
            .write_path(
                Some(Path::new("/import")),
                Some("global"),
                None,
                None,
                "hash",
            )
            .is_none());
    }

    #[test]
    fn custom_import_mode_without_folder_uses_dot_akasha_thumbnails() {
        let thumbnailer = Thumbnailer::new(256, PathBuf::new(), false, false, false);
        let path = thumbnailer
            .write_path(
                Some(Path::new("/import")),
                Some("custom"),
                None,
                Some("disable"),
                "deadbeef",
            )
            .unwrap();
        assert!(path.starts_with("/import/.akasha_thumbnails"));
        assert!(path.to_string_lossy().contains("deadbeef"));
    }

    #[test]
    fn temporary_cache_redirects_writes_to_tmp() {
        let thumbnailer = Thumbnailer::new(256, PathBuf::new(), false, true, false);
        let path = thumbnailer
            .write_path(None, Some("global"), None, None, "hash")
            .unwrap();
        assert!(path.starts_with(TEMP_CACHE_DIR));
    }
}