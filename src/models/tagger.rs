use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::vit::{Config, Model as VitModel};

use super::{loader, preprocess, CandleModel, ModelOutput, ModelOutputKind};

pub struct WdViTTagger {
    name: String,
    model: VitModel,
    labels: Vec<String>,
    device: Device,
    input_size: usize,
    threshold: f32,
}

impl WdViTTagger {
    pub fn load(
        name: &str,
        files: &loader::ModelFiles,
        device: Device,
        threshold: f32,
    ) -> Result<Self> {
        let config_text = std::fs::read_to_string(&files.config_path)
            .with_context(|| format!("failed to read config: {}", files.config_path.display()))?;
        let config: Config = serde_json::from_str(&config_text)
            .with_context(|| "failed to parse config.json as ViT Config")?;

        let labels_text = std::fs::read_to_string(&files.labels_path)
            .with_context(|| format!("failed to read labels: {}", files.labels_path.display()))?;
        let labels: Vec<String> = labels_text
            .lines()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .collect();

        // Memory-map the weights once and reuse the same mapping for metadata inspection and
        // model construction.
        let tensors = unsafe {
            // SAFETY: `MmapedSafetensors::new` memory-maps the weight file. The underlying file
            // must not be modified, truncated, or deleted for the lifetime of the returned
            // `MmapedSafetensors` (and any tensors/models derived from it), or the process may
            // encounter undefined behavior from the OS memory mapping.
            candle_core::safetensors::MmapedSafetensors::new(&files.weights_path)
        }
        .with_context(|| format!("failed to mmap weights: {}", files.weights_path.display()))?;

        // Inspect the classifier head output dimension directly from the weights so mismatches
        // with the labels file fail fast instead of loading garbage.
        let weight_num_labels = tensors
            .get("classifier.weight")
            .with_context(|| "classifier.weight not found in model weights")?
            .shape()
            .first()
            .copied()
            .with_context(|| "classifier.weight has no dimensions")?;
        if weight_num_labels != labels.len() {
            anyhow::bail!(
                "classifier output dimension mismatch: weights have {} classes, labels file has {}",
                weight_num_labels,
                labels.len()
            );
        }

        let vb = candle_nn::VarBuilder::from_backend(
            Box::new(tensors),
            candle_core::DType::F32,
            device.clone(),
        );

        let model = VitModel::new(&config, labels.len(), vb)
            .with_context(|| "failed to build ViT model")?;

        Ok(Self {
            name: name.to_string(),
            model,
            labels,
            device,
            input_size: 448,
            threshold,
        })
    }
}

#[async_trait::async_trait]
impl CandleModel for WdViTTagger {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> ModelOutputKind {
        ModelOutputKind::Tags
    }

    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        let tensor = preprocess::image_to_tensor(image_path, self.input_size, &self.device)?;
        let logits = self.model.forward(&tensor)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let mut tags = HashMap::new();
        for (i, &score) in probs_vec.iter().enumerate() {
            if score >= self.threshold && i < self.labels.len() {
                tags.insert(self.labels[i].clone(), score);
            }
        }

        Ok(ModelOutput::Tags(tags))
    }
}
