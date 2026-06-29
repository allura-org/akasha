use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::vit::{Config, Model as VitModel};

use super::{loader, preprocess, Model, ModelOutput};

/// Extract label names from a Hugging Face ViT-style `id2label` config field.
/// The field is an object mapping stringified indices to label strings, e.g.
/// `"0": "tench, Tinca tinca"`.
fn labels_from_config(config: &serde_json::Value) -> Result<Vec<String>> {
    let id2label = config
        .get("id2label")
        .context("no labels file and no id2label in config.json")?;
    let obj = id2label
        .as_object()
        .context("id2label in config.json is not an object")?;

    let mut pairs: Vec<(usize, String)> = obj
        .iter()
        .filter_map(|(k, v)| {
            let idx = k.parse::<usize>().ok()?;
            let label = v.as_str()?.to_string();
            Some((idx, label))
        })
        .collect();
    pairs.sort_by_key(|p| p.0);

    // Ensure contiguous indices starting at 0 so the vec position matches the
    // classifier output index.
    for (expected, (actual, _)) in pairs.iter().enumerate() {
        if expected != *actual {
            anyhow::bail!("id2label indices are not contiguous starting at 0");
        }
    }

    Ok(pairs.into_iter().map(|(_, label)| label).collect())
}

/// Standard Hugging Face ViT image-classifier tagger.
///
/// This uses the `candle_transformers` ViT implementation and applies a
/// sigmoid over the classifier logits so any label can be returned
/// independently. It is *not* compatible with timm-style checkpoints such as
/// `SmilingWolf/wd-vit-tagger-v3`.
pub struct ViTTagger {
    model: VitModel,
    labels: Vec<String>,
    device: Device,
    input_size: usize,
    threshold: f32,
    top_k: Option<usize>,
}

impl ViTTagger {
    pub fn load(
        _name: &str,
        files: &loader::ModelFiles,
        device: Device,
        options: &crate::config::ModelTagsOptions,
    ) -> Result<Self> {
        let config_text = std::fs::read_to_string(&files.config_path)
            .with_context(|| format!("failed to read config: {}", files.config_path.display()))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_text)
            .with_context(|| "failed to parse config.json")?;
        let config: Config = serde_json::from_value(config_value.clone())
            .with_context(|| "failed to parse config.json as ViT Config")?;

        let labels: Vec<String> = if let Some(labels_path) = &files.labels_path {
            let labels_text = std::fs::read_to_string(labels_path)
                .with_context(|| format!("failed to read labels: {}", labels_path.display()))?;
            labels_text
                .lines()
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            labels_from_config(&config_value)
                .with_context(|| "failed to read labels from config.json id2label")?
        };

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
            model,
            labels,
            device,
            input_size: config.image_size,
            threshold: options.threshold,
            top_k: options.top_k,
        })
    }
}

impl Model for ViTTagger {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        let tensor = preprocess::image_to_tensor(image_path, self.input_size, &self.device)?;
        let logits = self.model.forward(&tensor)?.squeeze(0)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let mut max_score = 0.0f32;
        let mut min_score = f32::INFINITY;
        let mut sum_score = 0.0f32;
        for &score in &probs_vec {
            if score > max_score {
                max_score = score;
            }
            if score < min_score {
                min_score = score;
            }
            sum_score += score;
        }
        let mean_score = if !probs_vec.is_empty() {
            sum_score / probs_vec.len() as f32
        } else {
            0.0
        };

        let mut tags = HashMap::new();
        for (i, &score) in probs_vec.iter().enumerate() {
            if score >= self.threshold && i < self.labels.len() {
                tags.insert(self.labels[i].clone(), score);
            }
        }

        if let Some(k) = self.top_k {
            if tags.len() > k {
                let mut sorted: Vec<_> = tags.into_iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                sorted.truncate(k);
                tags = sorted.into_iter().collect();
            }
        }

        tracing::info!(
            image = ?image_path,
            labels = self.labels.len(),
            threshold = self.threshold,
            max_score,
            min_score,
            mean_score,
            above_threshold = tags.len(),
            "WdViTTagger inference stats"
        );

        Ok(ModelOutput::Tags(tags))
    }
}

#[cfg(all(test, feature = "candle"))]
mod manual_tests {
    use super::*;
    use candle_core::Device;

    /// Manual smoke test: downloads a standard Hugging Face ViT image classification model and
    /// runs inference on a test image. Not part of the regular test suite.
    /// WD v3 uses a timm-style config and is not directly compatible with
    /// `candle_transformers::models::vit`, so this test uses `google/vit-base-patch16-224` to
    /// verify the candle pipeline end-to-end.
    #[test]
    #[ignore = "manual: downloads google/vit-base-patch16-224 weights from Hugging Face"]
    fn vit_base_tagger_smoke() -> Result<()> {
        let source = loader::resolve_source("google/vit-base-patch16-224")?;
        let files = loader::load_model_files(&source)?;
        let options = crate::config::ModelTagsOptions {
            threshold: 0.1,
            top_k: None,
        };
        let tagger = ViTTagger::load("vit-base-patch16-224", &files, Device::Cpu, &options)?;

        let output = tagger.infer(Path::new("test_imgs/dagnpats.png"))?;
        let ModelOutput::Tags(tags) = output else {
            anyhow::bail!("expected tag output");
        };

        let mut sorted: Vec<_> = tags.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("top tags:");
        for (tag, score) in sorted.iter().take(10) {
            println!("  {}: {:.4}", tag, score);
        }
        assert!(!sorted.is_empty(), "expected at least one tag above threshold");
        Ok(())
    }

    /// Verify the default threshold (0.35) still produces tags on the smoke image.
    /// Run with `cargo test --features candle -- --ignored --nocapture vit_base_tagger_default_threshold`.
    #[test]
    #[ignore = "manual: checks default threshold produces tags"]
    fn vit_base_tagger_default_threshold() -> Result<()> {
        let source = loader::resolve_source("google/vit-base-patch16-224")?;
        let files = loader::load_model_files(&source)?;
        let options = crate::config::ModelTagsOptions {
            threshold: 0.35,
            top_k: None,
        };
        let tagger = ViTTagger::load("vit-base-patch16-224", &files, Device::Cpu, &options)?;

        let output = tagger.infer(Path::new("test_imgs/dagnpats.png"))?;
        let ModelOutput::Tags(tags) = output else {
            anyhow::bail!("expected tag output");
        };

        println!("default threshold tag count: {}", tags.len());
        let mut sorted: Vec<_> = tags.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tag, score) in sorted.iter().take(5) {
            println!("  {}: {:.4}", tag, score);
        }
        assert!(!sorted.is_empty(), "expected at least one tag above default threshold");
        Ok(())
    }

    /// Manual performance baseline: run inference on 10 images and report total / per-image time.
    /// Run with `cargo test --features candle --release -- --ignored --nocapture vit_base_tagger_baseline`.
    #[test]
    #[ignore = "manual: benchmarks google/vit-base-patch16-224 CPU inference"]
    fn vit_base_tagger_baseline() -> Result<()> {
        let source = loader::resolve_source("google/vit-base-patch16-224")?;
        let files = loader::load_model_files(&source)?;
        let options = crate::config::ModelTagsOptions {
            threshold: 0.1,
            top_k: None,
        };
        let tagger = ViTTagger::load("vit-base-patch16-224", &files, Device::Cpu, &options)?;

        let image_paths: Vec<&str> = vec![
            "test_imgs/dagnpats.png",
            "test_imgs/dagnscritchies.png",
            "test_imgs/noise_0001_295694.png",
            "test_imgs/noise_0002_899443.png",
            "test_imgs/Th105Yuyuko.png",
            "test_imgs/tmpnc1whtei.png",
            "test_imgs/dagnpats.png",
            "test_imgs/dagnscritchies.png",
            "test_imgs/noise_0001_295694.png",
            "test_imgs/noise_0002_899443.png",
        ];

        let start = std::time::Instant::now();
        let mut tag_counts = Vec::new();
        for path in &image_paths {
            let output = tagger.infer(Path::new(path))?;
            let ModelOutput::Tags(tags) = output else {
                anyhow::bail!("expected tag output");
            };
            tag_counts.push(tags.len());
        }
        let elapsed = start.elapsed();

        println!(
            "baseline: {} images in {:.2}s ({:.2}s / image)",
            image_paths.len(),
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / image_paths.len() as f64
        );
        println!("tags per image: {:?}", tag_counts);
        Ok(())
    }
}
