//! Local candle-based backend.

use std::sync::Arc;
use anyhow::{Context, Result};
use candle_core::Device;

use crate::config::ModelConfig;

use super::{loader, Backend, Model};

pub struct CandleBackend;

impl Backend for CandleBackend {
    fn id(&self) -> &'static str {
        "candle"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        // Candle only handles local models with a path right now.
        config.kind == crate::config::ModelKind::Local && config.path.is_some()
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let path = config.path.as_deref().context("candle model missing path")?;
        let source = loader::resolve_source(path)?;
        let files = loader::load_model_files(&source)
            .with_context(|| format!("failed to load model files for {}", config.name))?;
        #[cfg(feature = "cuda")]
        let device = {
            match Device::new_cuda(0) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("CUDA device unavailable, falling back to CPU: {e}");
                    Device::Cpu
                }
            }
        };
        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;

        match config.tags.as_ref() {
            Some(options) => {
                let tagger = super::tagger::ViTTagger::load(
                    &config.name,
                    &files,
                    device,
                    options.threshold,
                )?;
                Ok(Arc::new(tagger))
            }
            None => anyhow::bail!("candle backend only supports tags output kind right now"),
        }
    }
}

#[cfg(all(test, feature = "candle"))]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelTagsOptions};

    #[test]
    fn candle_backend_supports_local_path_or_hf_slug() {
        let backend = CandleBackend;
        let cfg = ModelConfig {
            name: "vit-base".into(),
            kind: ModelKind::Local,
            backend: None,
            path: Some("google/vit-base-patch16-224".into()),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(ModelTagsOptions { threshold: 0.1 }),
            description: None,
            classification: None,
            remote: None,
        };
        assert!(backend.supports(&cfg));
    }
}
