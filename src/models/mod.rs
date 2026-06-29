use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use crate::config::{ModelConfig, RemoteConfig};

pub mod loader;
#[cfg(feature = "candle")]
pub mod preprocess;
#[cfg(feature = "candle")]
pub mod tagger;
#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "candle")]
pub mod stub;
#[cfg(feature = "remote")]
pub mod remote;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "mistralrs")]
pub mod mistralrs;

#[derive(Debug, Clone)]
pub enum ModelOutput {
    Tags(HashMap<String, f32>),
    Description(String),
    Classification { label: String, score: f32 },
    Vector(Vec<f32>),
}

pub trait Model: Send + Sync {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput>;
}

pub trait Backend: Send + Sync {
    fn id(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn supports(&self, config: &ModelConfig) -> bool;
    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>>;
}

pub struct BackendRegistry {
    backends: Vec<Arc<dyn Backend>>,
}

impl BackendRegistry {
    pub fn empty() -> Self {
        Self { backends: Vec::new() }
    }

    pub fn register<B: Backend + 'static>(&mut self, backend: B) {
        self.backends.push(Arc::new(backend));
    }

    pub fn with_remote(remote: RemoteConfig) -> Self {
        #[allow(unused_mut)]
        let mut reg = Self::empty();
        #[cfg(feature = "remote")]
        reg.register(remote::RemoteBackend::new(remote));
        #[cfg(feature = "mistralrs")]
        reg.register(mistralrs::MistralRsBackend);
        #[cfg(feature = "candle")]
        reg.register(candle::CandleBackend);
        #[cfg(feature = "onnx")]
        reg.register(onnx::OrtBackend);
        #[cfg(not(feature = "remote"))]
        let _ = remote;
        reg
    }

    pub fn select(&self, config: &ModelConfig) -> Option<Arc<dyn Backend>> {
        if let Some(id) = &config.backend {
            self.backends
                .iter()
                .find(|b| b.id() == id && b.is_available())
                .cloned()
        } else {
            self.backends
                .iter()
                .find(|b| b.is_available() && b.supports(config))
                .cloned()
        }
    }

    pub fn select_with_error(&self, config: &ModelConfig) -> Result<Arc<dyn Backend>> {
        if let Some(id) = &config.backend {
            match self.backends.iter().find(|b| b.id() == id) {
                Some(b) if !b.is_available() => {
                    anyhow::bail!("backend '{}' is not available", id)
                }
                Some(b) if !b.supports(config) => {
                    anyhow::bail!("backend '{}' does not support this model configuration", id)
                }
                Some(b) => Ok(b.clone()),
                None => {
                    anyhow::bail!(
                        "backend '{}' is not compiled in. Rebuild with the matching feature flag",
                        id
                    )
                }
            }
        } else {
            self.backends
                .iter()
                .find(|b| b.is_available() && b.supports(config))
                .cloned()
                .with_context(|| {
                    format!(
                        "no backend available for model {}. Add `backend = \"...\"` or check feature flags",
                        config.name
                    )
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind};

    struct AlwaysBackend;
    impl Backend for AlwaysBackend {
        fn id(&self) -> &'static str { "always" }
        fn is_available(&self) -> bool { true }
        fn supports(&self, _config: &ModelConfig) -> bool { true }
        fn load(&self, _config: &ModelConfig) -> Result<Arc<dyn Model>> {
            unimplemented!()
        }
    }

    #[test]
    fn registry_selects_by_backend_id() {
        let mut reg = BackendRegistry::empty();
        reg.register(AlwaysBackend);
        let config = ModelConfig {
            name: "x".into(),
            kind: ModelKind::Local,
            backend: Some("always".into()),
            path: None,
            base_url: None,
            model_id: None,
            api_key: None,
            tags: None,
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        };
        assert!(reg.select(&config).is_some());
    }

    #[test]
    fn registry_returns_none_for_missing_backend() {
        let reg = BackendRegistry::empty();
        let config = ModelConfig {
            name: "x".into(),
            kind: ModelKind::Local,
            backend: Some("nope".into()),
            path: None,
            base_url: None,
            model_id: None,
            api_key: None,
            tags: None,
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        };
        assert!(reg.select(&config).is_none());
    }
}
