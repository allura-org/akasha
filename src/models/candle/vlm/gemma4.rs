use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::gemma4::{self, config::Gemma4Config};
use image::imageops::FilterType;
use tokenizers::Tokenizer;

use crate::config::{ModelConfig, ModelDescriptionOptions};
use crate::models::loader::ModelFiles;

use super::token_stream::{decode_tokens, TokenOutputStream};
use super::{build_logits_processor, sample_next, VlmArchitecture, VlmModel};

pub struct Gemma4Architecture;

impl VlmArchitecture for Gemma4Architecture {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        // Explicit backend hint.
        if config.backend.as_deref() == Some("candle-gemma4") {
            return true;
        }
        // Heuristic: model id/path contains "gemma-4" or "gemma4".
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
        let tokenizer_path = files
            .tokenizer_path
            .as_ref()
            .context("Gemma4 requires a tokenizer.json file")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {tokenizer_path:?}: {e}"))?;

        let config_text = std::fs::read_to_string(&files.config_path)
            .with_context(|| format!("failed to read config: {}", files.config_path.display()))?;
        let gemma_config: Gemma4Config = serde_json::from_str(&config_text)
            .with_context(|| "failed to parse Gemma4Config")?;

        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&files.weights_paths, dtype, device)?
        };
        let model = gemma4::Model::new(&gemma_config, vb)?;

        let desc = config.description.clone().unwrap_or_default();

        Ok(Box::new(Gemma4Vlm {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            config: gemma_config,
            generation: desc,
            device: device.clone(),
        }))
    }
}

pub struct Gemma4Vlm {
    model: gemma4::Model,
    tokenizer: TokenOutputStream,
    config: Gemma4Config,
    generation: ModelDescriptionOptions,
    device: Device,
}

impl Gemma4Vlm {
    /// Preprocess the image and return the pixel tensor plus the number of vision
    /// tokens the Gemma 4 vision tower will produce for this resolution.
    fn preprocess_image(&self, path: &Path) -> Result<(Tensor, usize)> {
        // Gemma 4 vision tower accepts any size divisible by patch_size.
        // Resize longest side to a reasonable default; exact size should match the
        // processor config and available memory. 896 keeps CPU inference tractable.
        let target_longest = 896u32;
        let patch_size = self.config.vision_config.patch_size as u32;
        let pooling_kernel = self.config.vision_config.pooling_kernel_size as u32;

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

        let img = image::imageops::resize(&img, new_w, new_h, FilterType::Lanczos3);
        let data: Vec<f32> = img
            .pixels()
            .flat_map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
            .collect();

        let tensor = Tensor::from_vec(data, (new_h as usize, new_w as usize, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)
            .with_context(|| "failed to build image tensor")?;

        let ph = (new_h / patch_size) as usize;
        let pw = (new_w / patch_size) as usize;
        let num_patches = ph * pw;
        let vision_tokens = num_patches / (pooling_kernel as usize).pow(2);

        Ok((tensor, vision_tokens.max(1)))
    }

    fn build_prompt_tokens(&self, prompt: Option<&str>, vision_tokens: usize) -> Result<Vec<u32>> {
        let prompt = prompt.unwrap_or("Describe this image.");
        let t = self.tokenizer.tokenizer();

        // Gemma 4 uses special tokens. Try the tokenizer vocab first, fall back to IDs from config.
        let channel_id = self.tokenizer.get_token("<|channel|>");
        let turn_id = self.tokenizer.get_token("<|turn|>");
        let boi_id = self.tokenizer.get_token("<|image>");
        let image_id = self
            .tokenizer
            .get_token("<|image|>")
            .unwrap_or(self.config.image_token_id as u32);

        let mut ids = Vec::new();
        if let Some(id) = channel_id {
            ids.push(id);
        }
        if let Some(id) = turn_id {
            ids.push(id);
        }
        ids.extend(
            t.encode("user\n", false)
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
            t.encode(format!("\n{}\n", prompt), false)
                .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?
                .get_ids()
                .iter()
                .copied(),
        );
        if let Some(id) = turn_id {
            ids.push(id);
        }
        ids.extend(
            t.encode("model\n", false)
                .map_err(|e| anyhow::anyhow!("encode model: {e}"))?
                .get_ids()
                .iter()
                .copied(),
        );

        Ok(ids)
    }

    fn eos_token(&self) -> u32 {
        // The Gemma 4 config lists EOS as token 1; prefer tokenizer vocab lookup, then default.
        self.tokenizer
            .get_token("</s>")
            .or_else(|| self.tokenizer.get_token("<eos>"))
            .unwrap_or(1)
    }
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

        let eos_token = self.eos_token();
        let max_tokens = self.generation.max_tokens;
        let repeat_penalty = self.generation.repeat_penalty;
        let repeat_last_n = self.generation.repeat_last_n;

        for index in 0..max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward_multimodal(
                &input_ids,
                Some(&[pixel_values.clone()]),
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

            if next_token == eos_token {
                break;
            }
        }

        let mut text = decode_tokens(self.tokenizer.tokenizer(), &tokens[prompt_len..])?;
        let eos_str = self
            .tokenizer
            .tokenizer()
            .id_to_token(eos_token)
            .unwrap_or_default();
        while !eos_str.is_empty() && text.ends_with(&eos_str) {
            text.truncate(text.len() - eos_str.len());
        }
        text = text.trim_end().to_string();
        Ok(text)
    }
}
