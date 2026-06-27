use std::path::{Path, PathBuf};
use anyhow::Result;

pub enum ModelSource {
    HfSlug(String),
    LocalPath(PathBuf),
}

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
    pub labels_path: PathBuf,
}

pub fn resolve_source(path: &str) -> Result<ModelSource> {
    let p = Path::new(path);
    if p.exists() {
        Ok(ModelSource::LocalPath(p.to_path_buf()))
    } else {
        Ok(ModelSource::HfSlug(path.to_string()))
    }
}

#[cfg(feature = "candle")]
pub fn load_model_files(source: &ModelSource) -> Result<ModelFiles> {
    match source {
        ModelSource::HfSlug(slug) => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(slug.clone());
            Ok(ModelFiles {
                config_path: repo.get("config.json")?,
                weights_path: repo.get("model.safetensors")?,
                labels_path: repo.get("selected_tags.csv")?,
            })
        }
        ModelSource::LocalPath(dir) => {
            Ok(ModelFiles {
                config_path: dir.join("config.json"),
                weights_path: dir.join("model.safetensors"),
                labels_path: dir.join("selected_tags.csv"),
            })
        }
    }
}

#[cfg(not(feature = "candle"))]
pub fn load_model_files(_source: &ModelSource) -> Result<ModelFiles> {
    anyhow::bail!("candle feature not enabled")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_local_path() {
        let src = resolve_source("/tmp").unwrap();
        match src {
            ModelSource::LocalPath(p) => assert_eq!(p, PathBuf::from("/tmp")),
            _ => panic!("expected local"),
        }
    }

    #[test]
    fn resolve_hf_slug() {
        let src = resolve_source("SmilingWolf/wd-vit-tagger-v3").unwrap();
        match src {
            ModelSource::HfSlug(s) => assert_eq!(s, "SmilingWolf/wd-vit-tagger-v3"),
            _ => panic!("expected hf slug"),
        }
    }
}
