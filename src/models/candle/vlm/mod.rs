use std::path::Path;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::config::ModelConfig;
use crate::models::loader::ModelFiles;

/// Stateful vision-language model. Implementations manage their own KV cache.
pub trait VlmModel: Send + Sync {
    fn generate(&mut self, image_path: &Path, prompt: Option<&str>) -> Result<String>;
}

/// Describes and loads one VLM architecture (e.g. Gemma4, BLIP).
pub trait VlmArchitecture: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, config: &ModelConfig) -> bool;
    fn load(
        &self,
        config: &ModelConfig,
        files: &ModelFiles,
        device: &Device,
    ) -> Result<Box<dyn VlmModel>>;
}

/// Registry of available VLM architectures.
pub struct VlmArchitectureRegistry {
    architectures: Vec<Box<dyn VlmArchitecture>>,
}

impl VlmArchitectureRegistry {
    pub fn empty() -> Self {
        Self { architectures: Vec::new() }
    }

    pub fn register<A: VlmArchitecture + 'static>(&mut self, arch: A) {
        self.architectures.push(Box::new(arch));
    }

    pub fn with_defaults() -> Self {
        let mut reg = Self::empty();
        reg.register(gemma4::Gemma4Architecture);
        reg.register(blip::BlipArchitecture);
        reg
    }

    pub fn select(
        &self,
        config: &ModelConfig,
        files: &ModelFiles,
        device: &Device,
    ) -> Result<Box<dyn VlmModel>> {
        for arch in &self.architectures {
            if arch.supports(config) {
                tracing::info!(arch = arch.name(), "selected VLM architecture");
                return arch.load(config, files, device);
            }
        }
        anyhow::bail!(
            "no VLM architecture supports model {}. Add an explicit backend/architecture hint.",
            config.name
        )
    }
}

pub fn build_logits_processor(
    seed: u64,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
) -> LogitsProcessor {
    let temp = temperature.unwrap_or(0.0);
    let sampling = if temp <= 0.0 {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature: temp },
            (Some(k), None) => Sampling::TopK { k, temperature: temp },
            (None, Some(p)) => Sampling::TopP { p, temperature: temp },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature: temp },
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}

pub fn sample_next(
    logits_processor: &mut LogitsProcessor,
    logits: &Tensor,
    tokens: &[u32],
    repeat_penalty: f32,
    repeat_last_n: usize,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = if repeat_penalty == 1.0 {
        logits
    } else {
        let start_at = tokens.len().saturating_sub(repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            repeat_penalty,
            &tokens[start_at..],
        )?
    };
    Ok(logits_processor.sample(&logits)?)
}

pub mod blip;
pub mod gemma4;
pub mod token_stream;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct DummyArch;
    impl VlmArchitecture for DummyArch {
        fn name(&self) -> &'static str { "dummy" }
        fn supports(&self, config: &crate::config::ModelConfig) -> bool {
            config.name == "dummy-model"
        }
        fn load(&self, _config: &crate::config::ModelConfig, _files: &ModelFiles, _device: &Device) -> Result<Box<dyn VlmModel>> {
            anyhow::bail!("unimplemented")
        }
    }

    #[test]
    fn registry_selects_matching_architecture() {
        let mut reg = VlmArchitectureRegistry::empty();
        reg.register(DummyArch);

        let config = crate::config::ModelConfig {
            name: "dummy-model".into(),
            ..Default::default()
        };
        let files = ModelFiles {
            config_path: PathBuf::new(),
            weights_paths: vec![],
            tokenizer_path: None,
            labels_path: None,
        };
        let res = reg.select(&config, &files, &Device::Cpu);
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(e.to_string().contains("unimplemented"));
        }
    }
}
