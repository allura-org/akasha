//! Local candle-based backend.

pub mod vlm;

use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use candle_core::Device;

use crate::config::ModelConfig;
use crate::models::{loader, Backend, Model, ModelOutput};
use vlm::{VlmArchitectureRegistry, VlmModel};

/// Adapts a stateful `VlmModel` to the `Model` trait used by the worker.
pub struct VlmModelWrapper {
    inner: std::sync::Mutex<Box<dyn VlmModel>>,
    prompt: Option<String>,
}

impl VlmModelWrapper {
    pub fn new(model: Box<dyn VlmModel>, prompt: Option<String>) -> Self {
        Self {
            inner: std::sync::Mutex::new(model),
            prompt,
        }
    }
}

impl Model for VlmModelWrapper {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("VLM model mutex was poisoned"))?;
        let text = inner.generate(image_path, self.prompt.as_deref())?;
        Ok(ModelOutput::Description(text))
    }
}

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
        // Respect an explicit backend choice; don't grab ONNX models.
        if config.kind != crate::config::ModelKind::Local {
            return false;
        }
        if config.base_url.is_some() {
            return false;
        }
        if config.path.is_none() {
            return false;
        }
        match config.backend.as_deref() {
            Some("candle") | None => true,
            _ => false,
        }
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

        if config.description.is_some() {
            let registry = VlmArchitectureRegistry::with_defaults();
            let vlm = registry.select(config, &files, &device)?;
            let prompt = config.description.as_ref().and_then(|d| d.prompt.clone());
            return Ok(Arc::new(VlmModelWrapper::new(vlm, prompt)));
        }

        match config.tags.as_ref() {
            Some(options) => {
                let tagger = super::tagger::ViTTagger::load(&config.name, &files, device, options)?;
                Ok(Arc::new(tagger))
            }
            None => anyhow::bail!("candle backend requires either tags or description configuration"),
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
            tags: Some(ModelTagsOptions { threshold: 0.1, top_k: None }),
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        };
        assert!(backend.supports(&cfg));
    }
}
