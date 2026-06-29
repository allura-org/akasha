use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

pub enum ModelSource {
    HfSlug(String),
    LocalPath(PathBuf),
}

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub weights_paths: Vec<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    /// Optional per-model label list. Some models (e.g. standard HF ViT classifiers)
    /// store labels in `config.json` instead of a separate file.
    pub labels_path: Option<PathBuf>,
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
fn parse_safetensors_index(index_path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(index_path)
        .with_context(|| format!("failed to open {index_path:?}"))?;
    let json: serde_json::Value = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {index_path:?}"))?;
    let weight_map = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("no weight_map object in safetensors index")?;

    let mut files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(name) = value.as_str() {
            files.insert(name.to_string());
        }
    }
    let mut files: Vec<_> = files.into_iter().collect();
    files.sort();
    Ok(files)
}

#[cfg(feature = "candle")]
pub fn load_safetensors_paths(
    repo: Option<&hf_hub::api::sync::ApiRepo>,
    local_dir: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    let index_path = match (repo, local_dir) {
        (Some(repo), None) => repo.get("model.safetensors.index.json").ok(),
        (None, Some(dir)) => {
            let p = dir.join("model.safetensors.index.json");
            p.exists().then_some(p)
        }
        _ => None,
    };

    if let Some(index_path) = index_path {
        let names = parse_safetensors_index(&index_path)?;
        match repo {
            Some(repo) => names
                .into_iter()
                .map(|n| repo.get(&n).with_context(|| format!("failed to fetch {n}")))
                .collect(),
            None => {
                let dir = local_dir.unwrap();
                Ok(names.into_iter().map(|n| dir.join(n)).collect())
            }
        }
    } else {
        let single = match (repo, local_dir) {
            (Some(repo), None) => repo.get("model.safetensors")?,
            (None, Some(dir)) => {
                let p = dir.join("model.safetensors");
                if p.exists() {
                    p
                } else {
                    anyhow::bail!("model.safetensors not found in {}", dir.display());
                }
            }
            _ => anyhow::bail!("expected repo or local_dir"),
        };
        Ok(vec![single])
    }
}

#[cfg(feature = "candle")]
pub fn load_model_files(source: &ModelSource) -> Result<ModelFiles> {
    match source {
        ModelSource::HfSlug(slug) => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(slug.clone());
            let labels_path = repo.get("selected_tags.csv").ok();
            let tokenizer_path = repo.get("tokenizer.json").ok();
            let weights_paths = load_safetensors_paths(Some(&repo), None)?;
            Ok(ModelFiles {
                config_path: repo.get("config.json")?,
                weights_paths,
                tokenizer_path,
                labels_path,
            })
        }
        ModelSource::LocalPath(dir) => {
            let labels_path = dir.join("selected_tags.csv");
            let labels_path = labels_path.exists().then_some(labels_path);
            let tokenizer_path = dir.join("tokenizer.json");
            let tokenizer_path = tokenizer_path.exists().then_some(tokenizer_path);
            let weights_paths = load_safetensors_paths(None, Some(dir))?;
            Ok(ModelFiles {
                config_path: dir.join("config.json"),
                weights_paths,
                tokenizer_path,
                labels_path,
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

    #[test]
    #[cfg(feature = "candle")]
    fn parse_safetensors_index_returns_unique_sorted_files() {
        use std::io::Write;
        let temp = tempfile::tempdir().unwrap();
        let index = temp.path().join("model.safetensors.index.json");
        let mut f = std::fs::File::create(&index).unwrap();
        writeln!(
            f,
            r#"{{"weight_map":{{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}}}}"#
        ).unwrap();

        let files = super::parse_safetensors_index(&index).unwrap();
        assert_eq!(files, vec![
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]);
    }
}
