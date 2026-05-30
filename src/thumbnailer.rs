use std::path::{Path, PathBuf};

pub enum CacheMode {
    Disabled,
    Global,
    PerFolder,
    Custom(PathBuf),
}

pub struct Thumbnailer {
    pub size: u32,
    pub cache_mode: CacheMode,
}

impl Thumbnailer {
    pub fn new(size: u32, cache_mode: CacheMode) -> Self {
        Self { size, cache_mode }
    }

    pub fn cache_path_for(&self, _hash: &str, _folder_root: Option<&Path>) -> Option<PathBuf> {
        match &self.cache_mode {
            CacheMode::Disabled => None,
            CacheMode::Global => {
                let base = directories::ProjectDirs::from("", "", "akasha")?
                    .cache_dir()
                    .to_path_buf();
                Some(base)
            }
            CacheMode::PerFolder => {
                // TODO: derive from folder root
                None
            }
            CacheMode::Custom(path) => Some(path.clone()),
        }
    }

    pub async fn generate(&self, _source: &Path) -> anyhow::Result<Option<Vec<u8>>> {
        // TODO: resize image and encode as WebP
        Ok(None)
    }
}
