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

    pub fn cache_path_for(&self, hash: &str, _folder_root: Option<&Path>) -> Option<PathBuf> {
        match &self.cache_mode {
            CacheMode::Disabled => None,
            CacheMode::Global => {
                directories::ProjectDirs::from("", "", "akasha")
                    .map(|dirs| dirs.cache_dir().join(format!("{}_{}.webp", hash, self.size)))
            }
            CacheMode::PerFolder => {
                // TODO: implement per-folder cache paths
                None
            }
            CacheMode::Custom(base) => Some(base.join(format!("{}_{}.webp", hash, self.size))),
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
