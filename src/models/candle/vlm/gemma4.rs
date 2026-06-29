use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use candle_transformers::models::gemma4::{
    self,
    config::{Gemma4Config, Gemma4VisionConfig},
};
use image::imageops::FilterType;
use serde_json::Value;

/// Number of image/vision tokens Gemma 4 uses for an image, taken from the
/// official processor config (`image_seq_length` / `max_soft_tokens`).
const GEMMA4_IMAGE_SEQ_LENGTH: usize = 280;
use tokenizers::Tokenizer;

use crate::config::{ModelConfig, ModelDescriptionOptions};
use crate::models::loader::ModelFiles;

use super::token_stream::{decode_tokens, TokenOutputStream};
use super::{build_logits_processor, sample_next, VlmArchitecture, VlmModel};

/// Custom `SimpleBackend` that transparently maps the unwrapped tensor names
/// used by `candle-transformers` to the `*.linear.weight` names produced by
/// Gemma 4's `Gemma4ClippableLinear` wrapper.
///
/// The clamp buffers (`input_max`, `output_max`, ...) are ignored; we load the
/// inner `linear.weight` and run inference without input/output clipping. This
/// matches the common "unwrap ClippableLinear" workaround used by PEFT and
/// vLLM for Gemma 4 checkpoints.
struct RemappedSafetensors(MmapedSafetensors);

impl RemappedSafetensors {
    fn new(mmap: MmapedSafetensors) -> Self {
        Self(mmap)
    }

    /// If `name` exists in the checkpoint use it directly. Otherwise, if it is
    /// a `.weight` tensor, try the equivalent `*.linear.weight` path used by
    /// Gemma 4's clippable linears.
    fn resolve(&self, name: &str) -> String {
        if self.0.get(name).is_ok() {
            return name.to_string();
        }
        if name.ends_with(".weight") {
            let prefix = &name[..name.len() - ".weight".len()];
            let linear_name = format!("{prefix}.linear.weight");
            if self.0.get(&linear_name).is_ok() {
                return linear_name;
            }
        }
        name.to_string()
    }
}

impl SimpleBackend for RemappedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let name = self.resolve(name);
        let tensor = self.0.load(&name, dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let name = self.resolve(name);
        self.0.load(&name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if self.0.get(name).is_ok() {
            return true;
        }
        if name.ends_with(".weight") {
            let prefix = &name[..name.len() - ".weight".len()];
            return self.0.get(&format!("{prefix}.linear.weight")).is_ok();
        }
        false
    }
}

pub struct Gemma4Architecture;

impl VlmArchitecture for Gemma4Architecture {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        // Selected by name/path heuristic: model id/path contains "gemma-4" or "gemma4".
        let haystack = format!(
            "{} {} {}",
            config.name,
            config.path.as_deref().unwrap_or(""),
            config.model_id.as_deref().unwrap_or("")
        )
        .to_lowercase();
        haystack.contains("gemma-4") || haystack.contains("gemma4")
    }

    fn load(
        &self,
        config: &ModelConfig,
        files: &ModelFiles,
        device: &Device,
    ) -> Result<Box<dyn VlmModel>> {
        let name = &config.name;
        let tokenizer_path = files
            .tokenizer_path
            .as_ref()
            .with_context(|| format!("tokenizer.json not found for {name}"))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {tokenizer_path:?}: {e}"))?;

        let config_text = std::fs::read_to_string(&files.config_path)
            .with_context(|| format!("failed to read config: {}", files.config_path.display()))?;
        let mut gemma_config: Gemma4Config = serde_json::from_str(&config_text)
            .with_context(|| "failed to parse Gemma4Config")?;
        let raw_config: Value = serde_json::from_str(&config_text)
            .with_context(|| "failed to parse raw config.json")?;
        let eos_token_ids = parse_eos_token_ids(&raw_config);

        // Akasha only uses the vision tower for image description, so skip the
        // audio tower entirely. This also avoids loading its ClippableLinear
        // wrapped weights, which candle-transformers 0.11.0 does not support.
        if gemma_config.audio_config.is_some() {
            tracing::debug!("gemma4: dropping audio_config; audio tower not needed");
            gemma_config.audio_config = None;
        }

        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let mmap = unsafe { MmapedSafetensors::multi(&files.weights_paths)? };
        let backend: Box<dyn SimpleBackend> = Box::new(RemappedSafetensors::new(mmap));
        let vb = VarBuilderArgs::new_with_args(backend, dtype, device);
        let model = gemma4::Model::new(&gemma_config, vb)?;

        let desc = config.description.clone().unwrap_or_default();

        Ok(Box::new(Gemma4Vlm {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            config: gemma_config,
            generation: desc,
            device: device.clone(),
            eos_token_ids,
        }))
    }
}

pub struct Gemma4Vlm {
    model: gemma4::Model,
    tokenizer: TokenOutputStream,
    config: Gemma4Config,
    generation: ModelDescriptionOptions,
    device: Device,
    /// EOS token IDs parsed from raw config.json, if present.
    eos_token_ids: Vec<u32>,
}

impl Gemma4Vlm {
    /// Preprocess the image and return the pixel tensor plus the number of vision
    /// tokens the Gemma 4 vision tower will produce for this resolution.
    fn preprocess_image(&self, path: &Path) -> Result<(Tensor, usize)> {
        preprocess_image_with_config(
            path,
            &self.config.vision_config,
            &self.generation,
            &self.device,
        )
    }

    fn build_prompt_tokens(&self, prompt: Option<&str>, vision_tokens: usize) -> Result<Vec<u32>> {
        build_prompt_tokens(
            self.tokenizer.tokenizer(),
            self.config.image_token_id,
            prompt,
            vision_tokens,
        )
    }

    fn eos_token(&self) -> Vec<u32> {
        // Stop on any explicit EOS IDs from config.json, plus tokenizer vocab lookups.
        // Default to ID 1 if nothing else is available.
        let mut ids = self.eos_token_ids.clone();
        if let Some(id) = self.tokenizer.get_token("</s>") {
            ids.push(id);
        }
        if let Some(id) = self.tokenizer.get_token("<eos>") {
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(1);
        }
        ids
    }
}

/// Default per-channel mean/std for Gemma 4 preprocessing, matching the
/// official processor config (`do_normalize: false`, `image_mean: [0,0,0]`,
/// `image_std: [1,1,1]`). Using these values makes normalization a no-op.
const DEFAULT_IMAGE_MEAN: [f32; 3] = [0.0, 0.0, 0.0];
const DEFAULT_IMAGE_STD: [f32; 3] = [1.0, 1.0, 1.0];
const DEFAULT_RESCALE_FACTOR: f32 = 1.0 / 255.0;

/// Preprocess the image and return the pixel tensor plus the number of vision
/// tokens the Gemma 4 vision tower will produce.
///
/// Defaults are taken from the official Gemma 4 E2B-it processor config:
/// - `do_convert_rgb: true` → input is converted to RGB.
/// - `do_rescale: true`, `rescale_factor: 1/255` → pixels are scaled by 1/255.
/// - `do_normalize: false`, `image_mean: [0,0,0]`, `image_std: [1,1,1]`
///   → normalization is a no-op unless `[models.description]` explicitly sets
///     `image_mean` and `image_std`, in which case `(pixel * rescale - mean) / std`
///     is applied per channel.
/// - `resample: 3` (PIL BICUBIC) → closest `image` crate filter is `CatmullRom`.
/// - `image_seq_length` / `max_soft_tokens: 280` → fixed 280 vision tokens.
fn preprocess_image_with_config(
    path: &Path,
    vision_config: &Gemma4VisionConfig,
    generation: &ModelDescriptionOptions,
    device: &Device,
) -> Result<(Tensor, usize)> {
    // Gemma 4 vision tower accepts any size divisible by patch_size.
    // patch_size (16) and pooling_kernel_size (3) come from config.vision_config
    // and match the processor config.
    let target_longest = 896u32;
    let patch_size = vision_config.patch_size as u32;

    let img = image::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let img = img.to_rgb8();
    let (w, h) = img.dimensions();

    let (new_w, new_h) = if w.max(h) > target_longest {
        let scale = target_longest as f32 / w.max(h) as f32;
        let nw = ((w as f32 * scale) as u32 / patch_size) * patch_size;
        let nh = ((h as f32 * scale) as u32 / patch_size) * patch_size;
        (nw.max(patch_size), nh.max(patch_size))
    } else {
        let nw = (w / patch_size) * patch_size;
        let nh = (h / patch_size) * patch_size;
        (nw.max(patch_size), nh.max(patch_size))
    };

    let img = image::imageops::resize(&img, new_w, new_h, FilterType::CatmullRom);

    let (mean, std) = match generation.image_mean.as_ref().zip(generation.image_std.as_ref()) {
        Some((m, s)) => {
            if m.len() != 3 || s.len() != 3 {
                anyhow::bail!("image_mean and image_std must each have exactly 3 channels");
            }
            ([m[0], m[1], m[2]], [s[0], s[1], s[2]])
        }
        None => (DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD),
    };

    let data: Vec<f32> = img
        .pixels()
        .flat_map(|p| {
            [
                (p[0] as f32 * DEFAULT_RESCALE_FACTOR - mean[0]) / std[0],
                (p[1] as f32 * DEFAULT_RESCALE_FACTOR - mean[1]) / std[1],
                (p[2] as f32 * DEFAULT_RESCALE_FACTOR - mean[2]) / std[2],
            ]
        })
        .collect();

    let tensor = Tensor::from_vec(data, (new_h as usize, new_w as usize, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)
        .with_context(|| "failed to build image tensor")?;

    Ok((tensor, GEMMA4_IMAGE_SEQ_LENGTH))
}

fn build_prompt_tokens(
    tokenizer: &Tokenizer,
    image_token_id: usize,
    prompt: Option<&str>,
    vision_tokens: usize,
) -> Result<Vec<u32>> {
    let prompt = prompt.unwrap_or("Describe this image.");

    // `tokenizers` 0.22 does not support `apply_chat_template`, so we manually
    // construct the special-token framing. Revisit this if a chat-template library
    // (e.g. minijinja) is added later.
    // Gemma 4 uses special tokens. Try the tokenizer vocab first, fall back to IDs from config.
    let vocab = tokenizer.get_vocab(true);
    let channel_id = vocab.get("<|channel|>").copied();
    let turn_id = vocab.get("<|turn|>").copied();
    let boi_id = vocab.get("<|image>").copied();
    let image_id = vocab
        .get("<|image|>")
        .copied()
        .unwrap_or(image_token_id as u32);

    let mut ids = Vec::new();
    if let Some(id) = channel_id {
        ids.push(id);
    }
    if let Some(id) = turn_id {
        ids.push(id);
    }
    ids.extend(
        tokenizer
            .encode("user\n", false)
            .map_err(|e| anyhow::anyhow!("encode user: {e}"))?
            .get_ids()
            .iter()
            .copied(),
    );
    if let Some(id) = boi_id {
        ids.push(id);
    }
    for _ in 0..vision_tokens {
        ids.push(image_id);
    }
    ids.extend(
        tokenizer
            .encode(format!("\n{}\n", prompt), false)
            .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?
            .get_ids()
            .iter()
            .copied(),
    );
    if let Some(id) = turn_id {
        ids.push(id);
    }
    ids.extend(
        tokenizer
            .encode("model\n", false)
            .map_err(|e| anyhow::anyhow!("encode model: {e}"))?
            .get_ids()
            .iter()
            .copied(),
    );

    Ok(ids)
}

fn parse_eos_token_ids(raw: &Value) -> Vec<u32> {
    let mut ids = Vec::new();
    if let Some(eos) = raw.get("eos_token_id") {
        if let Some(arr) = eos.as_array() {
            for v in arr {
                if let Some(n) = v.as_u64() {
                    ids.push(n as u32);
                }
            }
        } else if let Some(n) = eos.as_u64() {
            ids.push(n as u32);
        }
    }
    ids
}

impl VlmModel for Gemma4Vlm {
    fn generate(&mut self, image_path: &Path, prompt: Option<&str>) -> Result<String> {
        self.model.clear_kv_cache();

        let (pixel_values, vision_tokens) = self.preprocess_image(image_path)?;
        let mut tokens = self.build_prompt_tokens(prompt, vision_tokens)?;
        let prompt_len = tokens.len();

        let mut logits_processor = build_logits_processor(
            299792458,
            self.generation.temperature,
            self.generation.top_p,
            self.generation.top_k,
        );

        let eos_tokens = self.eos_token();
        let max_tokens = self.generation.max_tokens;
        let repeat_penalty = self.generation.repeat_penalty;
        let repeat_last_n = self.generation.repeat_last_n;
        let pixel_values_slice: Vec<Tensor> = vec![pixel_values];

        for index in 0..max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward_multimodal(
                &input_ids,
                Some(&pixel_values_slice),
                None,
                None,
                start_pos,
            )?;
            let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;

            let next_token = sample_next(
                &mut logits_processor,
                &logits,
                &tokens,
                repeat_penalty,
                repeat_last_n,
            )?;
            tokens.push(next_token);

            if eos_tokens.contains(&next_token) {
                break;
            }
        }

        let mut text = decode_tokens(self.tokenizer.tokenizer(), &tokens[prompt_len..])?;
        let eos_str = eos_tokens
            .first()
            .and_then(|id| self.tokenizer.tokenizer().id_to_token(*id))
            .unwrap_or_default();
        while !eos_str.is_empty() && text.ends_with(&eos_str) {
            text.truncate(text.len() - eos_str.len());
        }
        text = text.trim_end().to_string();
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn minimal_vision_config(patch_size: usize, pooling_kernel_size: usize) -> Gemma4VisionConfig {
        serde_json::from_value(serde_json::json!({
            "patch_size": patch_size,
            "pooling_kernel_size": pooling_kernel_size,
        }))
        .unwrap()
    }

    #[test]
    fn preprocess_shape_and_vision_tokens() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("solid.png");
        let img = image::RgbImage::from_pixel(144, 144, image::Rgb([128, 64, 32]));
        img.save(&path).unwrap();

        let vision_config = minimal_vision_config(16, 3);
        let generation = ModelDescriptionOptions::default();
        let device = Device::Cpu;

        let (tensor, vision_tokens) =
            preprocess_image_with_config(&path, &vision_config, &generation, &device).unwrap();

        assert_eq!(tensor.dims4().unwrap(), (1, 3, 144, 144));
        assert_eq!(vision_tokens, GEMMA4_IMAGE_SEQ_LENGTH);
    }

    #[test]
    fn preprocess_applies_mean_std_normalization() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("white.png");
        let img = image::RgbImage::from_pixel(16, 16, image::Rgb([255, 255, 255]));
        img.save(&path).unwrap();

        let vision_config = minimal_vision_config(16, 3);
        let generation = ModelDescriptionOptions {
            image_mean: Some(vec![0.5, 0.5, 0.5]),
            image_std: Some(vec![0.5, 0.5, 0.5]),
            ..Default::default()
        };
        let device = Device::Cpu;

        let (tensor, _) =
            preprocess_image_with_config(&path, &vision_config, &generation, &device).unwrap();
        let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in flat {
            assert!((v - 1.0).abs() < 1e-5, "expected normalized white pixel to be 1.0, got {v}");
        }
    }

    #[test]
    fn build_prompt_tokens_includes_image_token() {
        let tokenizer_json = r#"
        {
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<|image|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<|image|>": 0, "user": 1, "model": 2, "\n": 3, "Desc": 4, "ribe": 5},
                "unk_token": "<|image|>"
            }
        }
        "#;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).unwrap();
        let tokens = build_prompt_tokens(&tokenizer, 258880, None, 2).unwrap();

        assert!(!tokens.is_empty());
        assert!(tokens.contains(&0), "expected prompt tokens to contain image token id 0");
    }

    #[test]
    fn parse_eos_token_ids_from_array_and_scalar() {
        let raw = serde_json::json!({"eos_token_id": [1, 106]});
        assert_eq!(parse_eos_token_ids(&raw), vec![1, 106]);

        let raw = serde_json::json!({"eos_token_id": 42});
        assert_eq!(parse_eos_token_ids(&raw), vec![42]);
    }

    #[test]
    fn remapped_backend_loads_clippable_linear_weight() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("clippable.safetensors");
        let mut tensors = HashMap::new();
        tensors.insert(
            "vision.encoder.layers.0.self_attn.q_proj.linear.weight".to_string(),
            Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "vision.encoder.layers.0.self_attn.q_proj.input_max".to_string(),
            Tensor::new(1.0f32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "vision.encoder.layers.0.input_layernorm.weight".to_string(),
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path).unwrap();

        let mmap = unsafe { MmapedSafetensors::multi(&[path]).unwrap() };
        let backend: Box<dyn SimpleBackend> = Box::new(RemappedSafetensors::new(mmap));
        let vb = VarBuilderArgs::new_with_args(backend, DType::F32, &Device::Cpu);

        let weight = vb
            .get((4, 4), "vision.encoder.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(weight.dims(), &[4, 4]);

        let norm = vb
            .get(4, "vision.encoder.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(norm.dims(), &[4]);
    }
}
