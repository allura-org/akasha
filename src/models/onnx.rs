//! ONNX Runtime backend for local inference.
//!
//! Models are expected to live under `~/.local/share/akasha/models/onnx/<slug>/`,
//! where `<slug>` is the HuggingFace model id with `/` replaced by `-`.
//!
//! The backend tries to discover the ONNX model file, preprocessing config, and
//! tag/label files heuristically so users don't have to write per-model configs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;

use crate::config::ModelConfig;

use super::{Backend, Model, ModelOutput};

const ONNX_MODELS_DIR: &str = "models/onnx";

pub struct OrtBackend;

impl Backend for OrtBackend {
    fn id(&self) -> &'static str {
        "onnx"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        if config.kind != crate::config::ModelKind::Local {
            return false;
        }

        // Explicit backend selection takes precedence.
        match config.backend.as_deref() {
            Some("onnx") => return true,
            Some("candle") | Some(_) => return false,
            None => {}
        }

        // If no backend is specified, claim the model if it already resolves to
        // an ONNX model folder. We don't claim undownloaded HF slugs here so
        // Candle (registered earlier) can handle .safetensors models by default.
        if let Some(path) = &config.path {
            if let Ok(dir) = resolve_model_dir(path) {
                if find_model_file(&dir, None).is_ok() {
                    return true;
                }
            }
        }

        false
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let model = OrtModel::load(config)?;
        Ok(Arc::new(model))
    }
}

pub struct OrtModel {
    session: std::sync::Mutex<Session>,
    preprocessing: Preprocessing,
    tags: Vec<String>,
    threshold: f32,
    top_k: Option<usize>,
    input_name: String,
    output_name: String,
}

struct Preprocessing {
    width: u32,
    height: u32,
    mean: [f32; 3],
    std: [f32; 3],
    resize_to_fill: bool,
    channels_last: bool,
}

impl Default for Preprocessing {
    fn default() -> Self {
        Self {
            width: 224,
            height: 224,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            resize_to_fill: true,
            channels_last: false,
        }
    }
}

impl OrtModel {
    fn load(config: &ModelConfig) -> Result<Self> {
        let path = config.path.as_deref().context("onnx model missing path")?;
        let dir = ensure_model_dir(path)?;
        tracing::info!(dir = %dir.display(), "Loading ONNX model");

        let onnx_opts = config.onnx.clone().unwrap_or_default();
        let model_path = find_model_file(&dir, onnx_opts.model_file.as_deref())?;

        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
            .with_context(|| format!("failed to load ONNX model from {}", model_path.display()))?;

        let input = &session.inputs()[0];
        let input_name = input.name().to_string();
        let output_name = session.outputs()[0]
            .name()
            .to_string();

        let channels_last = infer_channels_last(input.dtype());

        tracing::info!(
            input = %input_name,
            output = %output_name,
            channels_last,
            "ONNX session ready"
        );

        let mut preprocessing = discover_preprocessing(&dir, onnx_opts.config_file.as_deref())?;
        preprocessing.channels_last = channels_last;
        let tags = discover_tags(&dir, onnx_opts.tags_file.as_deref())?;
        let threshold = config.tags.as_ref().map(|t| t.threshold).unwrap_or(0.35);
        let top_k = config.tags.as_ref().and_then(|t| t.top_k);

        Ok(Self {
            session: std::sync::Mutex::new(session),
            preprocessing,
            tags,
            threshold,
            top_k,
            input_name,
            output_name,
        })
    }
}

impl Model for OrtModel {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        let img = image::open(image_path)
            .with_context(|| format!("failed to open image: {}", image_path.display()))?;

        let resized = if self.preprocessing.resize_to_fill {
            img.resize_to_fill(
                self.preprocessing.width,
                self.preprocessing.height,
                FilterType::Lanczos3,
            )
        } else {
            img.resize_exact(
                self.preprocessing.width,
                self.preprocessing.height,
                FilterType::Lanczos3,
            )
        };

        let rgb = resized.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let input = if self.preprocessing.channels_last {
            let mut tensor_data = vec![0f32; 1 * h * w * 3];
            for (i, pixel) in rgb.pixels().enumerate() {
                let x = i % w;
                let y = i / w;
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                let base = y * w * 3 + x * 3;
                tensor_data[base + 0] = (r - self.preprocessing.mean[0]) / self.preprocessing.std[0];
                tensor_data[base + 1] = (g - self.preprocessing.mean[1]) / self.preprocessing.std[1];
                tensor_data[base + 2] = (b - self.preprocessing.mean[2]) / self.preprocessing.std[2];
            }
            let array = ndarray::Array::from_shape_vec((1, h, w, 3), tensor_data)
                .context("failed to build NHWC input tensor")?;
            Tensor::from_array(array).context("failed to create ONNX NHWC input tensor")?
        } else {
            let mut tensor_data = vec![0f32; 1 * 3 * h * w];
            for (i, pixel) in rgb.pixels().enumerate() {
                let x = i % w;
                let y = i / w;
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                tensor_data[y * w + x] = (r - self.preprocessing.mean[0]) / self.preprocessing.std[0];
                tensor_data[h * w + y * w + x] = (g - self.preprocessing.mean[1]) / self.preprocessing.std[1];
                tensor_data[2 * h * w + y * w + x] = (b - self.preprocessing.mean[2]) / self.preprocessing.std[2];
            }
            let array = ndarray::Array::from_shape_vec((1, 3, h, w), tensor_data)
                .context("failed to build NCHW input tensor")?;
            Tensor::from_array(array).context("failed to create ONNX NCHW input tensor")?
        };

        let mut session = self
            .session
            .lock()
            .expect("OrtModel session mutex poisoned");
        let outputs = session
            .run(ort::inputs![&self.input_name => input])
            .context("ONNX inference failed")?;

        let output = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()
            .context("failed to extract output tensor")?;

        // Output is expected to be 2D: (batch, num_labels).
        let scores: Vec<f32> = if output.ndim() == 2 {
            output.slice(ndarray::s![0, ..]).iter().copied().collect()
        } else if output.ndim() == 1 {
            output.iter().copied().collect()
        } else {
            anyhow::bail!("unexpected output rank: {}", output.ndim());
        };

        let mut tags = HashMap::new();
        let mut max_score = 0.0f32;
        let mut min_score = f32::INFINITY;
        let mut sum_score = 0.0f32;
        for (idx, &score) in scores.iter().enumerate() {
            let prob = 1.0 / (1.0 + (-score).exp());
            max_score = max_score.max(prob);
            min_score = min_score.min(prob);
            sum_score += prob;
            if prob >= self.threshold {
                let tag = self.tags.get(idx).map(|s| s.as_str()).unwrap_or("");
                if !tag.is_empty() {
                    tags.insert(tag.to_string(), prob);
                }
            }
        }

        tags = crate::models::tagger::apply_top_k(tags, self.top_k);

        let mean_score = if !scores.is_empty() { sum_score / scores.len() as f32 } else { 0.0 };

        tracing::info!(
            image = ?image_path,
            labels = self.tags.len(),
            threshold = self.threshold,
            max_score,
            min_score = if min_score.is_finite() { min_score } else { 0.0 },
            mean_score,
            above_threshold = tags.len(),
            "OrtModel inference stats"
        );

        Ok(ModelOutput::Tags(tags))
    }
}

/// Resolve a model path to an absolute directory.
///
/// If `path` is already absolute, use it directly. Otherwise, treat it as a
/// slug under `~/.local/share/akasha/models/onnx/`.
/// Ensure a model directory exists, downloading from HuggingFace if necessary.
///
/// If `path` is already absolute, return it directly. Otherwise, treat it as a
/// HuggingFace model slug, resolve the target directory under
/// `~/.local/share/akasha/models/onnx/<slug-with-dashes>/`, and download any
/// missing model files from the Hub.
fn ensure_model_dir(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }

    let slug = path.replace('/', "-");
    let data_dir = crate::config::Config::data_dir()?;
    let target_dir = data_dir.join(ONNX_MODELS_DIR).join(&slug);

    // If a model file already exists locally, don't re-download.
    if find_model_file(&target_dir, None).is_ok() {
        return Ok(target_dir);
    }

    // Without a slash, this isn't a HuggingFace slug; just return the empty dir
    // and let the caller fail cleanly if nothing is there.
    if !path.contains('/') {
        return Ok(target_dir);
    }

    tracing::info!(slug = path, dir = %target_dir.display(), "Downloading ONNX model from HuggingFace");

    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create model directory {}", target_dir.display()))?;

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(path.to_string());

    let info = repo.info().with_context(|| format!("failed to fetch repo info for {path}"))?;

    let wanted = [
        ("model.onnx", "model.onnx"),
        ("onnx/model.onnx", "model.onnx"),
        ("vision_model.onnx", "vision_model.onnx"),
        ("onnx/vision_model.onnx", "vision_model.onnx"),
        ("text_model.onnx", "text_model.onnx"),
        ("onnx/text_model.onnx", "text_model.onnx"),
        ("config.json", "config.json"),
        ("preprocess.json", "preprocess.json"),
        ("preprocessor_config.json", "preprocessor_config.json"),
        ("selected_tags.csv", "selected_tags.csv"),
        ("categories.json", "categories.json"),
        ("tags.json", "tags.json"),
        ("labels.txt", "labels.txt"),
    ];

    let available: std::collections::HashSet<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .collect();

    let mut downloaded_model = false;
    for (remote, local) in &wanted {
        if !available.contains(*remote) {
            continue;
        }
        match repo.get(remote) {
            Ok(src) => {
                let dst = target_dir.join(local);
                if let Err(e) = std::fs::copy(&src, &dst) {
                    tracing::warn!(src = %src.display(), dst = %dst.display(), error = %e, "Failed to copy downloaded file");
                } else {
                    tracing::info!(file = remote, "Downloaded");
                    if local.ends_with(".onnx") {
                        downloaded_model = true;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(file = remote, error = %e, "Failed to download file");
            }
        }
    }

    if !downloaded_model {
        anyhow::bail!("no ONNX model file could be downloaded from {path}");
    }

    Ok(target_dir)
}

/// Inspect the session's input type to decide whether the model expects NHWC (channels-last)
/// or NCHW (channels-first) image input. We assume a 4-D input where one of the dimensions is 3.
fn infer_channels_last(dtype: &ort::value::ValueType) -> bool {
    if let ort::value::ValueType::Tensor { shape, .. } = dtype {
        if shape.len() == 4 {
            // Typical image layouts:
            //   NCHW: [batch, 3, height, width]
            //   NHWC: [batch, height, width, 3]
            // If channel dim (3) is last, it's channels-last.
            return shape[1] != 3 && shape[3] == 3;
        }
    }
    false
}

fn resolve_model_dir(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }

    let slug = path.replace('/', "-");
    let data_dir = crate::config::Config::data_dir()?;
    Ok(data_dir.join(ONNX_MODELS_DIR).join(slug))
}

/// Find the ONNX model file in the directory.
fn find_model_file(dir: &Path, explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(name) = explicit {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
        anyhow::bail!("explicit ONNX model file not found: {}", path.display());
    }

    // Prefer a non-quantized, non-external-data `model.onnx` if present.
    let candidates = [
        "model.onnx",
        "vision_model.onnx",
        "model_optimized.onnx",
    ];
    for name in &candidates {
        for base in [dir, &dir.join("onnx")] {
            let path = base.join(name);
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    // Fall back to any `.onnx` file that isn't obviously a quantized/external variant.
    let mut fallback: Option<PathBuf> = None;
    for base in [dir.to_path_buf(), dir.join("onnx")] {
        if !base.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&base).context("failed to read model directory")? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if name.contains("quantized") || name.contains("external") || name.contains("_data") {
                    if fallback.is_none() {
                        fallback = Some(path);
                    }
                    continue;
                }
                return Ok(path);
            }
        }
    }

    fallback.context("no ONNX model file found in directory")
}

/// Discover preprocessing parameters by scanning config files in the model folder.
fn discover_preprocessing(dir: &Path, explicit: Option<&str>) -> Result<Preprocessing> {
    if let Some(name) = explicit {
        let path = dir.join(name);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read preprocessing config: {}", path.display()))?;
        return parse_preprocessing_json(&text);
    }

    let candidates = [
        "preprocess.json",
        "preprocessor_config.json",
        "config.json",
    ];

    for name in &candidates {
        let path = dir.join(name);
        if let Ok(text) = std::fs::read_to_string(&path) {
            match parse_preprocessing_json(&text) {
                Ok(cfg) => {
                    tracing::info!(file = name, "Discovered preprocessing config");
                    return Ok(cfg);
                }
                Err(e) => {
                    tracing::debug!(file = name, error = %e, "Preprocessing config file did not match expected schema");
                }
            }
        }
    }

    tracing::warn!("No preprocessing config found; using ONNX defaults");
    Ok(Preprocessing::default())
}

fn parse_preprocessing_json(text: &str) -> Result<Preprocessing> {
    let value: serde_json::Value = serde_json::from_str(text).context("invalid JSON")?;

    // PixAI / transformers.js style: {"stages": [{"type": "resize", "size": [H, W]}, {"type": "normalize", "mean": [...], "std": [...]}]}
    if let Some(stages) = value.get("stages").and_then(|v| v.as_array()) {
        let mut cfg = Preprocessing::default();
        for stage in stages {
            let stage_type = stage.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match stage_type {
                "resize" => {
                    if let Some(size) = stage.get("size") {
                        if let Some(arr) = size.as_array() {
                            if arr.len() == 2 {
                                cfg.height = arr[0].as_u64().unwrap_or(224) as u32;
                                cfg.width = arr[1].as_u64().unwrap_or(224) as u32;
                            } else if arr.len() == 1 {
                                let s = arr[0].as_u64().unwrap_or(224) as u32;
                                cfg.width = s;
                                cfg.height = s;
                            }
                        } else if let Some(s) = size.as_u64() {
                            cfg.width = s as u32;
                            cfg.height = s as u32;
                        }
                    }
                    cfg.resize_to_fill = stage.get("max_size").is_none();
                }
                "normalize" => {
                    if let Some(mean) = stage.get("mean").and_then(|v| v.as_array()) {
                        cfg.mean = read_f32_triplet(mean);
                    }
                    if let Some(std) = stage.get("std").and_then(|v| v.as_array()) {
                        cfg.std = read_f32_triplet(std);
                    }
                }
                _ => {}
            }
        }
        return Ok(cfg);
    }

    // timm / WD style: {"pretrained_cfg": {"input_size": [3, H, W], "mean": [...], "std": [...]}}
    if let Some(pretrained) = value.get("pretrained_cfg") {
        let mut cfg = Preprocessing::default();
        if let Some(input_size) = pretrained.get("input_size").and_then(|v| v.as_array()) {
            if input_size.len() == 3 {
                cfg.height = input_size[1].as_u64().unwrap_or(224) as u32;
                cfg.width = input_size[2].as_u64().unwrap_or(224) as u32;
            }
        }
        if let Some(mean) = pretrained.get("mean").and_then(|v| v.as_array()) {
            cfg.mean = read_f32_triplet(mean);
        }
        if let Some(std) = pretrained.get("std").and_then(|v| v.as_array()) {
            cfg.std = read_f32_triplet(std);
        }
        if let Some(crop_pct) = pretrained.get("crop_pct").and_then(|v| v.as_f64()) {
            // If crop_pct < 1.0 the original training pipeline resized then cropped.
            // We approximate with resize_to_fill for inference; close enough for tagging.
            cfg.resize_to_fill = crop_pct >= 1.0;
        }
        return Ok(cfg);
    }

    // Hugging Face preprocessor_config.json style: {"size": {"shortest_edge": 224}, "image_mean": [...], "image_std": [...]}
    if value.get("image_mean").is_some() || value.get("do_resize").is_some() {
        let mut cfg = Preprocessing::default();
        if let Some(size) = value.get("size") {
            if let Some(s) = size.as_u64() {
                cfg.width = s as u32;
                cfg.height = s as u32;
            } else if let Some(obj) = size.as_object() {
                if let Some(h) = obj.get("height").and_then(|v| v.as_u64()) {
                    cfg.height = h as u32;
                }
                if let Some(w) = obj.get("width").and_then(|v| v.as_u64()) {
                    cfg.width = w as u32;
                }
                if let Some(s) = obj.get("shortest_edge").and_then(|v| v.as_u64()) {
                    cfg.width = s as u32;
                    cfg.height = s as u32;
                }
            }
        }
        if let Some(mean) = value.get("image_mean").and_then(|v| v.as_array()) {
            cfg.mean = read_f32_triplet(mean);
        }
        if let Some(std) = value.get("image_std").and_then(|v| v.as_array()) {
            cfg.std = read_f32_triplet(std);
        }
        return Ok(cfg);
    }

    anyhow::bail!("no recognized preprocessing schema found")
}

fn read_f32_triplet(arr: &[serde_json::Value]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (i, v) in arr.iter().take(3).enumerate() {
        out[i] = v.as_f64().unwrap_or(0.0) as f32;
    }
    out
}

/// Discover tag/label names by scanning files in the model folder.
fn discover_tags(dir: &Path, explicit: Option<&str>) -> Result<Vec<String>> {
    if let Some(name) = explicit {
        let path = dir.join(name);
        return load_tags(&path);
    }

    let candidates = [
        "selected_tags.csv",
        "tags.json",
        "categories.json",
        "labels.txt",
    ];

    for name in &candidates {
        let path = dir.join(name);
        if path.is_file() {
            match load_tags(&path) {
                Ok(tags) => {
                    tracing::info!(file = name, count = tags.len(), "Discovered tag list");
                    return Ok(tags);
                }
                Err(e) => {
                    tracing::debug!(file = name, error = %e, "Tag file did not parse");
                }
            }
        }
    }

    anyhow::bail!("no tag/label file found in model directory")
}

fn load_tags(path: &Path) -> Result<Vec<String>> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "csv" => load_tags_csv(path),
        "json" => load_tags_json(path),
        "txt" => load_tags_txt(path),
        _ => anyhow::bail!("unsupported tag file format: {}", ext),
    }
}

fn load_tags_csv(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).context("failed to read tag CSV")?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(text.as_bytes());

    let headers = reader.headers().context("missing CSV header")?.clone();
    let name_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("name"))
        .context("tag CSV missing 'name' column")?;

    let mut tags = Vec::new();
    for result in reader.records() {
        let record = result?;
        if let Some(name) = record.get(name_idx) {
            tags.push(name.to_string());
        }
    }

    Ok(tags)
}

fn load_tags_json(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).context("failed to read tag JSON")?;
    let value: serde_json::Value = serde_json::from_str(&text).context("invalid tag JSON")?;

    // Categories JSON: [{"name": "general", "category": 0}, ...]
    if let Some(arr) = value.as_array() {
        let mut tags = Vec::new();
        for item in arr {
            if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
                tags.push(name.to_string());
            }
        }
        if !tags.is_empty() {
            return Ok(tags);
        }
    }

    // Flat string array: ["tag1", "tag2", ...]
    if let Some(arr) = value.as_array() {
        let tags: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        if !tags.is_empty() {
            return Ok(tags);
        }
    }

    anyhow::bail!("unrecognized tag JSON shape")
}

fn load_tags_txt(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).context("failed to read tag txt")?;
    Ok(text.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind};

    #[test]
    fn parse_pixai_preprocess_json() {
        let text = r#"{"stages":[{"type":"resize","size":[448,448]},{"type":"to_tensor"},{"type":"normalize","mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}]}"#;
        let cfg = parse_preprocessing_json(text).unwrap();
        assert_eq!(cfg.width, 448);
        assert_eq!(cfg.height, 448);
        assert_eq!(cfg.mean, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn parse_wd_config_json() {
        let text = r#"{"pretrained_cfg":{"input_size":[3,448,448],"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}}"#;
        let cfg = parse_preprocessing_json(text).unwrap();
        assert_eq!(cfg.width, 448);
        assert_eq!(cfg.height, 448);
    }

    #[test]
    fn ort_backend_supports_explicit_backend_id() {
        let backend = OrtBackend;
        let cfg = ModelConfig {
            name: "x".into(),
            kind: ModelKind::Local,
            backend: Some("onnx".into()),
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
        assert!(backend.supports(&cfg));
    }

    #[test]
    #[ignore = "manual: requires SmilingWolf/wd-vit-tagger-v3 ONNX model in ~/.local/share/akasha/models/onnx"]
    fn ort_backend_runs_wd_vit_tagger() {
        let model_dir = crate::config::Config::data_dir()
            .unwrap()
            .join("models/onnx/SmilingWolf-wd-vit-tagger-v3");
        if !model_dir.join("model.onnx").exists() {
            eprintln!("Skipping: model not found at {}", model_dir.display());
            return;
        }

        let cfg = ModelConfig {
            name: "wd-vit-tagger-v3".into(),
            kind: ModelKind::Local,
            backend: Some("onnx".into()),
            path: Some("SmilingWolf-wd-vit-tagger-v3".into()),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(crate::config::ModelTagsOptions { threshold: 0.35, top_k: None }),
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        };

        let model = OrtBackend.load(&cfg).expect("load model");
        let output = model.infer(Path::new("test_imgs/dagnpats.png")).expect("infer");

        match output {
            ModelOutput::Tags(tags) => {
                assert!(!tags.is_empty(), "expected at least one tag above threshold");
                eprintln!("Got {} tags", tags.len());
            }
            other => panic!("expected ModelOutput::Tags, got {:?}", other),
        }
    }
}
