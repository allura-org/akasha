//! Hydra-3.5 model implemented in Burn.

pub mod fused_ops;
pub mod image;
pub mod modules;
pub mod ops;
pub mod weights;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::DType;
use ndarray::Array4;

// F32 is used for the CPU spike regardless of backend; BF16 is slow on the
// Flex CPU path and the NdArray backend does not implement BF16 at all.
const MODEL_DTYPE: DType = DType::F32;

use crate::config::ModelConfig;
use crate::models::Model;

use self::fused_ops::{
    FastLinearBackend, FastRmsNormBackend, FusedAttentionBackend, FusedGluBackend,
    FusedHydraMidBlockBackend, FusedMlpBackend, FusedNaFlexAttnBackend,
};
use self::image::preprocess;
use self::modules::Hydra;
use self::weights::load_hydra;

const ONNX_MODELS_DIR: &str = "models/onnx";

pub struct HydraModel<B: Backend> {
    model: Hydra<B>,
    pos_embed: Array4<f32>,
    labels: Vec<String>,
    threshold: f32,
    top_k: Option<usize>,
    max_seq_len: usize,
    background: [u8; 3],
    _phantom: std::marker::PhantomData<B>,
}

impl<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedNaFlexAttnBackend,
> HydraModel<B>
{
    pub fn load(config: &ModelConfig, device: B::Device) -> Result<Self> {
        let path = config.path.as_deref().context("hydra model missing path")?;
        let file = resolve_model_file(path)?;
        if !file.is_file() {
            anyhow::bail!("hydra model file not found: {}", file.display());
        }

        tracing::info!(file = %file.display(), "Loading Hydra-3.5 Burn model");

        let model = load_hydra::<B>(&file, &device)
            .with_context(|| format!("failed to load Hydra weights from {}", file.display()))?;

        let pos_embed = load_pos_embed(&file)?;
        let labels = load_labels(&file)?;
        let background = load_background(&file)?;

        let threshold = config.tags.as_ref().map(|t| t.threshold).unwrap_or(0.35);
        let top_k = config.tags.as_ref().and_then(|t| t.top_k);

        Ok(Self {
            model,
            pos_embed,
            labels,
            threshold,
            top_k,
            max_seq_len: 1024,
            background,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedNaFlexAttnBackend
        + 'static,
> HydraModel<B>
{
    /// Run the model and return raw logits (one score per label).
    pub fn infer_logits(&self, image_path: &Path) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let device = <B::Device as Default>::default();
        let pre = preprocess(
            image_path,
            &self.pos_embed,
            self.max_seq_len,
            self.background,
        )
        .with_context(|| format!("failed to preprocess image: {}", image_path.display()))?;
        let t_pre = t0.elapsed();

        // Convert ndarray outputs to Burn tensors.
        let t1 = Instant::now();
        let patches = Tensor::<B, 1>::from_data(
            pre.patches.into_raw_vec_and_offset().0.as_slice(),
            (&device, MODEL_DTYPE),
        )
        .reshape([1, self.max_seq_len, 768]);

        let pos_embed = Tensor::<B, 1>::from_data(
            pre.pos_embed.into_raw_vec_and_offset().0.as_slice(),
            (&device, MODEL_DTYPE),
        )
        .reshape([1, self.max_seq_len, 1152]);

        // Skip building the bool mask when every patch is valid; this lets
        // Burn's attention take a faster no-mask path.
        let all_valid = pre.valid.iter().all(|&b| b);
        let mask: Option<Tensor<B, 2, Bool>> = if all_valid {
            None
        } else {
            // Candle does not support bool_from_data, so build the mask as an Int
            // tensor and compare to 1.
            let valid_raw: Vec<i32> = pre
                .valid
                .into_raw_vec_and_offset()
                .0
                .into_iter()
                .map(|b| if b { 1i32 } else { 0i32 })
                .collect();
            Some(
                Tensor::<B, 1, Int>::from_data(valid_raw.as_slice(), &device)
                    .reshape([1, self.max_seq_len])
                    .equal_elem(1),
            )
        };
        let t_tensors = t1.elapsed();

        let logits = self.model.forward(patches, pos_embed, mask);
        let t_forward = t1.elapsed();

        // Convert to f32 for post-processing.
        let t2 = Instant::now();
        let logits_f32 = logits.cast(DType::F32);
        let out = logits_f32.to_data().to_vec::<f32>().map_err(Into::into);
        let t_post = t2.elapsed();

        tracing::debug!(
            "infer_logits total={:.3}s preprocess={:.3}s tensor_create={:.3}s forward={:.3}s postprocess={:.3}s",
            t0.elapsed().as_secs_f64(),
            t_pre.as_secs_f64(),
            t_tensors.as_secs_f64(),
            t_forward.as_secs_f64(),
            t_post.as_secs_f64()
        );

        out
    }
}

impl<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedNaFlexAttnBackend
        + 'static,
> Model for HydraModel<B>
{
    fn infer(&self, image_path: &Path) -> Result<crate::models::ModelOutput> {
        let scores = self.infer_logits(image_path)?;

        let mut labels: HashMap<String, f32> = HashMap::new();
        for (idx, &score) in scores.iter().enumerate() {
            let prob = 1.0 / (1.0 + (-score).exp());
            let tag = self.labels.get(idx).map(|s| s.as_str()).unwrap_or("");
            if tag.is_empty() {
                continue;
            }
            if prob >= self.threshold {
                labels.insert(tag.to_string(), prob);
            }
        }

        // Simple top-k filtering.
        let mut tags: Vec<(String, f32)> = labels.into_iter().collect();
        if let Some(top_k) = self.top_k {
            tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tags.truncate(top_k);
        }

        Ok(crate::models::ModelOutput::Tags(tags.into_iter().collect()))
    }
}

fn resolve_model_file(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if p.is_dir() {
        let file = p.join("hydra-3.5.safetensors");
        if file.is_file() {
            return Ok(file);
        }
    }

    // Relative slug under ~/.local/share/akasha/models/onnx/<slug>/
    let slug = path.replace('/', "-");
    let data_dir = crate::config::Config::data_dir()?;
    let candidate = data_dir
        .join(ONNX_MODELS_DIR)
        .join(slug)
        .join("hydra-3.5.safetensors");
    if candidate.is_file() {
        return Ok(candidate);
    }

    anyhow::bail!("no Hydra-3.5 safetensors file found for path: {}", path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "manual: requires Hydra-3.5 safetensors"]
    fn hydra_burn_runs() {
        let _ = tracing_subscriber::fmt::try_init();
        let device = <crate::models::burn::BurnDevice as Default>::default();
        let cfg = ModelConfig {
            name: "hydra-3.5".into(),
            kind: crate::config::ModelKind::Local,
            backend: Some("burn".into()),
            path: Some(
                "/home/asriel/Projects/RedRocket--Hydra/models/hydra-3.5.safetensors".into(),
            ),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(crate::config::ModelTagsOptions {
                threshold: 0.35,
                top_k: Some(20),
            }),
            description: None,
            classification: None,
            remote: None,
            onnx: None,
            jtp3: None,
        };

        let model = HydraModel::<crate::models::burn::BurnBackendType>::load(&cfg, device)
            .expect("load model");
        let img_paths = [
            Path::new("/home/asriel/Projects/akasha/test_imgs/dagnpats.png"),
            Path::new("/home/asriel/Projects/akasha/test_imgs/portrait.png"),
            Path::new("/home/asriel/Projects/akasha/test_imgs/landscape.webp"),
        ];

        for img_path in &img_paths {
            eprintln!("\n--- Testing {img_path:?} ---");
            let logits = model.infer_logits(img_path).expect("infer");
            assert!(!logits.is_empty(), "expected non-empty logits");

            // Compare probabilities against a PyTorch reference for the canonical image.
            // The reference file contains the *sigmoid* output from Hydra's default
            // `load_model()` call (logit=False), so we apply sigmoid to our raw logits
            // before comparing.
            if img_path
                .file_name()
                .map(|n| n == "dagnpats.png")
                .unwrap_or(false)
            {
                let ref_path = Path::new("/tmp/hydra_ref_logits.f32");
                if ref_path.is_file() {
                    let ref_bytes = std::fs::read(ref_path).expect("read reference logits");
                    let ref_probs: Vec<f32> = ref_bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();

                    assert_eq!(
                        ref_probs.len(),
                        logits.len(),
                        "reference and Burn logits have different lengths"
                    );

                    let mut max_diff = 0.0f32;
                    let mut mean_diff = 0.0f32;
                    for (&r, &logit) in ref_probs.iter().zip(&logits) {
                        let prob = 1.0 / (1.0 + (-logit).exp());
                        let d = (r - prob).abs();
                        max_diff = max_diff.max(d);
                        mean_diff += d;
                    }
                    mean_diff /= logits.len() as f32;

                    eprintln!(
                        "Probs: {} values | max abs diff vs PyTorch BF16 ref: {:.6} | mean abs diff: {:.6}",
                        logits.len(),
                        max_diff,
                        mean_diff
                    );

                    // Reference is BF16 sigmoid output; allow tolerance for F32 CPU spike.
                    assert!(
                        max_diff < 0.1,
                        "max abs prob diff too large: {max_diff} (reference is BF16, spike is F32)"
                    );
                } else {
                    eprintln!("Reference logits not found at {ref_path:?}; skipping comparison");
                }
            }

            let output = model.infer(img_path).expect("infer tags");
            let tags = match output {
                crate::models::ModelOutput::Tags(tags) => {
                    assert!(
                        !tags.is_empty(),
                        "expected at least one tag above threshold"
                    );
                    tags
                }
                other => panic!("expected ModelOutput::Tags, got {:?}", other),
            };

            eprintln!(
                "Got {} tags, max score {}",
                tags.len(),
                tags.values().copied().fold(0.0f32, |a, b| a.max(b))
            );
        }
    }
}

fn load_background(path: &Path) -> Result<[u8; 3]> {
    let metadata = crate::models::burn::read_safetensors_metadata(path).with_context(|| {
        format!(
            "failed to read safetensors metadata from {}",
            path.display()
        )
    })?;

    match metadata.get("classifier.background").map(String::as_str) {
        Some("white") => Ok([255, 255, 255]),
        Some("grey") => Ok([127, 127, 127]),
        Some("black") | None => Ok([0, 0, 0]),
        Some(other) => {
            tracing::warn!("unknown classifier.background '{other}', defaulting to black");
            Ok([0, 0, 0])
        }
    }
}

fn load_labels(path: &Path) -> Result<Vec<String>> {
    let metadata = crate::models::burn::read_safetensors_metadata(path).with_context(|| {
        format!(
            "failed to read safetensors metadata from {}",
            path.display()
        )
    })?;

    let labels_str = metadata
        .get("classifier.labels")
        .cloned()
        .unwrap_or_default();

    let labels: Vec<String> = labels_str
        .lines()
        .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if labels.is_empty() {
        tracing::warn!("No labels found in model metadata; using empty label list");
    }

    Ok(labels)
}

fn load_pos_embed(path: &Path) -> Result<Array4<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read model file: {}", path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse safetensors: {}", path.display()))?;

    let tensor = tensors
        .tensor("embeds.pos_embed")
        .context("'embeds.pos_embed' tensor not found")?;

    let shape: Vec<usize> = tensor.shape().iter().map(|&d| d as usize).collect();
    anyhow::ensure!(
        shape == [1, 16, 16, 1152],
        "unexpected pos_embed shape {:?}",
        shape
    );

    let data: Vec<f32> = match tensor.dtype() {
        safetensors::Dtype::BF16 => tensor
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F32 => tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => anyhow::bail!("unsupported pos_embed dtype: {other:?}"),
    };

    Array4::from_shape_vec((1, 16, 16, 1152), data).context("failed to build pos_embed array")
}
