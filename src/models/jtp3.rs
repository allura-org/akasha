//! JTP-3 ONNX backend.
//!
//! JTP-3 is a custom NaFlex (native-flexible) Vision Transformer tagger built
//! on SigLIP2 SO400M patch16 with a HydraPool attention head.  The original
//! PyTorch model is exported to ONNX with the position-embedding interpolation
//! removed from the graph; it is applied in preprocessing instead.
//!
//! A model directory must contain:
//!
//!   - `model.onnx` + `model.onnx.data`   the exported ONNX model (FP32 or FP16)
//!   - `pos_embed.safetensors`            learned 16x16 position embedding
//!   - `tags.txt`                         one tag per line

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{ImageDecoder, RgbImage};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView3, Axis, s, stack};
use ort::session::Session;
use ort::value::Tensor;

use crate::config::ModelConfig;

use super::{Backend, Model, ModelOutput};

const PATCH_SIZE: usize = 16;
const DEFAULT_MAX_SEQ_LEN: usize = 1024;
const POS_EMBED_H: usize = 16;
const POS_EMBED_W: usize = 16;
const EMBED_DIM: usize = 1152;
const ONNX_MODELS_DIR: &str = "models/onnx";

pub struct Jtp3Backend;

impl Backend for Jtp3Backend {
    fn id(&self) -> &'static str {
        "jtp3"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        if config.kind != crate::config::ModelKind::Local {
            return false;
        }

        // Explicit backend selection is the primary way to claim a JTP-3 model.
        if config.backend.as_deref() == Some("jtp3") {
            return true;
        }

        // Auto-detect a JTP-3 layout in the configured path.  Relative paths are
        // resolved under the normal ONNX models directory so JTP-3 models can live
        // alongside other ONNX models.
        if let Some(path) = &config.path {
            if let Ok(dir) = resolve_model_dir(path) {
                if dir.is_dir()
                    && find_model_file(&dir).is_ok()
                    && dir.join("pos_embed.safetensors").is_file()
                    && dir.join("tags.txt").is_file()
                {
                    return true;
                }
            }
        }

        false
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let model = Jtp3Model::load(config)?;
        Ok(Arc::new(model))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImplicationMode {
    Inherit,
    Constrain,
    Remove,
    ConstrainRemove,
    Off,
}

impl ImplicationMode {
    fn parse(s: &str) -> Self {
        match s {
            "inherit" => Self::Inherit,
            "constrain" => Self::Constrain,
            "remove" => Self::Remove,
            "constrain-remove" => Self::ConstrainRemove,
            _ => Self::Off,
        }
    }
}

pub struct Jtp3Model {
    session: std::sync::Mutex<Session>,
    tags: Vec<String>,
    pos_embed: Array4<f32>,
    threshold: f32,
    top_k: Option<usize>,
    max_seq_len: usize,
    /// Per-tag calibration thresholds loaded from `calibration.csv`.
    thresholds: HashMap<String, f32>,
    /// Tag implication graph loaded from tag metadata.
    implications: HashMap<String, Vec<String>>,
    /// Tag categories loaded from tag metadata.
    categories: HashMap<String, i32>,
    implication_mode: ImplicationMode,
    pos_embed_dtype: ort::value::TensorElementType,
}

impl Jtp3Model {
    fn load(config: &ModelConfig) -> Result<Self> {
        let path = config.path.as_deref().context("jtp3 model missing path")?;
        let dir = resolve_model_dir(path)?;
        if !dir.is_dir() {
            anyhow::bail!("jtp3 model directory not found: {}", dir.display());
        }

        let model_path = find_model_file(&dir)?;
        tracing::info!(dir = %dir.display(), model = %model_path.display(), "Loading JTP-3 ONNX model");

        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
            .with_context(|| format!("failed to load JTP-3 ONNX model from {}", model_path.display()))?;

        let pos_embed_dtype = session
            .inputs()
            .iter()
            .find(|i| i.name() == "pos_embed_padded")
            .and_then(|i| match i.dtype() {
                ort::value::ValueType::Tensor { ty, .. } => Some(*ty),
                _ => None,
            })
            .unwrap_or(ort::value::TensorElementType::Float32);

        let pos_embed_path = dir.join("pos_embed.safetensors");
        let pos_embed = load_pos_embed(&pos_embed_path)
            .with_context(|| format!("failed to load pos_embed from {}", pos_embed_path.display()))?;

        let tags_path = dir.join("tags.txt");
        let tags = load_tags(&tags_path)
            .with_context(|| format!("failed to load tags from {}", tags_path.display()))?;

        let threshold = config.tags.as_ref().map(|t| t.threshold).unwrap_or(0.35);
        let top_k = config.tags.as_ref().and_then(|t| t.top_k);

        let jtp3_opts = config.jtp3.clone().unwrap_or_default();

        // Optional per-tag calibration thresholds.
        let calibration_path = jtp3_opts
            .calibration_file
            .map(|n| dir.join(n))
            .or_else(|| Some(dir.join("calibration.csv")))
            .filter(|p| p.is_file())
            .unwrap_or_else(|| dir.join("calibration.csv"));
        let thresholds = if calibration_path.is_file() {
            load_calibration_csv(&calibration_path)
                .with_context(|| format!("failed to load calibration from {}", calibration_path.display()))?
        } else {
            HashMap::new()
        };

        // Optional tag metadata (categories + implications).
        let metadata_path = jtp3_opts
            .metadata_file
            .map(|n| dir.join(n))
            .or_else(|| find_metadata_file(&dir));
        let (categories, implications) = if let Some(path) = metadata_path {
            load_metadata_csv(&path)
                .with_context(|| format!("failed to load tag metadata from {}", path.display()))?
        } else {
            (HashMap::new(), HashMap::new())
        };

        let implication_mode = if jtp3_opts.implications.is_empty() {
            if implications.is_empty() {
                ImplicationMode::Off
            } else {
                ImplicationMode::Inherit
            }
        } else {
            ImplicationMode::parse(&jtp3_opts.implications)
        };

        tracing::info!(
            tags = tags.len(),
            threshold,
            top_k = ?top_k,
            calibrated = thresholds.len(),
            metadata = implications.len(),
            implication_mode = ?implication_mode,
            "JTP-3 model ready"
        );

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tags,
            pos_embed,
            threshold,
            top_k,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            thresholds,
            implications,
            categories,
            implication_mode,
            pos_embed_dtype,
        })
    }

    fn preprocess(&self, image_path: &Path) -> Result<(Array2<u8>, Array1<bool>, Array2<f32>)> {
        let icc_profile = image::ImageReader::open(image_path)
            .ok()
            .and_then(|r| r.into_decoder().ok())
            .and_then(|mut d| d.icc_profile().ok())
            .flatten();

        let img = image::open(image_path)
            .with_context(|| format!("failed to open image: {}", image_path.display()))?;
        let mut rgb = img.to_rgb8();
        rgb = apply_exif_orientation(image_path, rgb)
            .with_context(|| format!("failed to apply EXIF orientation: {}", image_path.display()))?;

        if let Some(icc) = icc_profile {
            rgb = apply_icc_profile(&icc, rgb)
                .with_context(|| format!("failed to apply ICC profile: {}", image_path.display()))?;
        }

        let (orig_w, orig_h) = (rgb.width() as usize, rgb.height() as usize);

        let (resize_h, resize_w) = compute_resize_for_seq(orig_h, orig_w, PATCH_SIZE, self.max_seq_len);
        let resized = image::imageops::resize(&rgb, resize_w as u32, resize_h as u32, FilterType::Lanczos3);

        let grid_h = resize_h / PATCH_SIZE;
        let grid_w = resize_w / PATCH_SIZE;
        let n_valid = grid_h * grid_w;

        // Patchify: (grid_h, patch_size, grid_w, patch_size, 3) -> (n_valid, patch_dim)
        let raw = resized.into_raw();
        let arr = Array3::from_shape_vec((resize_h, resize_w, 3), raw)
            .context("failed to build resized image array")?;
        let patches_view = arr
            .into_shape_with_order((grid_h, PATCH_SIZE, grid_w, PATCH_SIZE, 3))
            .context("failed to reshape into patches")?
            .permuted_axes([0, 2, 1, 3, 4]);
        let patches = patches_view
            .as_standard_layout()
            .into_shape_with_order((n_valid, PATCH_SIZE * PATCH_SIZE * 3))
            .context("failed to flatten patches")?;

        let mut patches_padded = Array2::<u8>::zeros((self.max_seq_len, PATCH_SIZE * PATCH_SIZE * 3));
        patches_padded.slice_mut(s![..n_valid, ..]).assign(&patches);

        let mut valid = Array1::<bool>::from_elem(self.max_seq_len, false);
        valid.slice_mut(s![..n_valid]).fill(true);

        let pos_embed_padded = interpolate_pos_embed(&self.pos_embed, grid_h, grid_w, self.max_seq_len)
            .context("failed to interpolate position embedding")?;

        Ok((patches_padded, valid, pos_embed_padded))
    }
}

impl Model for Jtp3Model {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        self.infer_batch(&[image_path])?
            .into_iter()
            .next()
            .context("JTP-3 batch inference returned no outputs")
    }

    fn infer_batch(&self, image_paths: &[&Path]) -> Result<Vec<ModelOutput>> {
        if image_paths.is_empty() {
            return Ok(Vec::new());
        }

        // Preprocess every image independently, then stack into one batch.
        let mut patches_list = Vec::with_capacity(image_paths.len());
        let mut valid_list = Vec::with_capacity(image_paths.len());
        let mut pos_embed_list = Vec::with_capacity(image_paths.len());

        for path in image_paths {
            let (patches, valid, pos_embed) = self
                .preprocess(path)
                .with_context(|| format!("failed to preprocess {}", path.display()))?;
            patches_list.push(patches);
            valid_list.push(valid);
            pos_embed_list.push(pos_embed);
        }

        let patches_batch = stack(Axis(0), &patches_list.iter().map(|a| a.view()).collect::<Vec<_>>())
            .context("failed to stack patch tensors")?;
        let valid_batch = stack(Axis(0), &valid_list.iter().map(|a| a.view()).collect::<Vec<_>>())
            .context("failed to stack patch_valid tensors")?;
        let pos_embed_batch = stack(
            Axis(0),
            &pos_embed_list.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .context("failed to stack pos_embed tensors")?;

        let patches_tensor = Tensor::from_array(patches_batch)
            .context("failed to create patches tensor")?;
        let valid_tensor = Tensor::from_array(valid_batch)
            .context("failed to create patch_valid tensor")?;
        let pos_embed_tensor: ort::value::DynValue = match self.pos_embed_dtype {
            ort::value::TensorElementType::Float16 => {
                let shape: Vec<i64> = pos_embed_batch.shape().iter().map(|&d| d as i64).collect();
                let data: Vec<half::f16> = pos_embed_batch
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
                Tensor::<half::f16>::from_array((shape, data))
                    .context("failed to create fp16 pos_embed tensor")?
                    .upcast()
                    .into()
            }
            _ => Tensor::from_array(pos_embed_batch)
                .context("failed to create pos_embed tensor")?
                .upcast()
                .into(),
        };

        let mut session = self
            .session
            .lock()
            .expect("Jtp3Model session mutex poisoned");
        let outputs = session
            .run(ort::inputs![
                "patches" => patches_tensor,
                "patch_valid" => valid_tensor,
                "pos_embed_padded" => pos_embed_tensor
            ])
            .context("JTP-3 ONNX batch inference failed")?;

        let output = outputs["logits"]
            .try_extract_array::<f32>()
            .context("failed to extract logits tensor")?;

        let mut results = Vec::with_capacity(image_paths.len());
        for (batch_idx, path) in image_paths.iter().enumerate() {
            let scores: Vec<f32> = output
                .slice(s![batch_idx, ..])
                .iter()
                .copied()
                .collect();

            let mut labels: HashMap<String, f32> = HashMap::new();
            let mut max_score = 0.0f32;
            let mut min_score = f32::INFINITY;
            let mut sum_score = 0.0f32;

            for (idx, &score) in scores.iter().enumerate() {
                let prob = 1.0 / (1.0 + (-score).exp());
                max_score = max_score.max(prob);
                min_score = min_score.min(prob);
                sum_score += prob;

                let tag = self.tags.get(idx).map(|s| s.as_str()).unwrap_or("");
                if tag.is_empty() {
                    continue;
                }

                let threshold = self.thresholds.get(tag).copied().unwrap_or(self.threshold);
                if prob >= threshold {
                    labels.insert(tag.to_string(), prob);
                }
            }

            apply_implications(&mut labels, &self.implications, self.implication_mode);

            let tags = crate::models::tagger::apply_top_k(labels, self.top_k);

            let mean_score = if !scores.is_empty() {
                sum_score / scores.len() as f32
            } else {
                0.0
            };

            tracing::info!(
                image = ?path,
                labels = self.tags.len(),
                threshold = self.threshold,
                max_score,
                min_score = if min_score.is_finite() { min_score } else { 0.0 },
                mean_score,
                above_threshold = tags.len(),
                "JTP-3 inference stats"
            );

            results.push(ModelOutput::Tags(tags));
        }

        Ok(results)
    }
}

/// Read the EXIF Orientation tag from `image_path` and apply the corresponding
/// transform to `rgb`.  If no orientation tag is present, `rgb` is returned
/// unchanged.  This matches PIL's `ImageOps.exif_transpose` behavior.
fn apply_exif_orientation(image_path: &Path, rgb: RgbImage) -> Result<RgbImage> {
    let file = std::fs::File::open(image_path)
        .with_context(|| format!("failed to open image for EXIF: {}", image_path.display()))?;
    let mut bufreader = std::io::BufReader::new(file);
    let exif_reader = exif::Reader::new();
    let exif = match exif_reader.read_from_container(&mut bufreader) {
        Ok(e) => e,
        Err(_) => return Ok(rgb),
    };

    let orientation_value = exif
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .unwrap_or(1);

    let oriented = match orientation_value {
        1 => rgb,
        2 => image::imageops::flip_horizontal(&rgb),
        3 => image::imageops::rotate180(&rgb),
        4 => image::imageops::flip_vertical(&rgb),
        5 => transpose_image(&rgb),
        6 => image::imageops::rotate90(&rgb),
        7 => image::imageops::rotate180(&transpose_image(&rgb)),
        8 => image::imageops::rotate270(&rgb),
        _ => rgb,
    };

    Ok(oriented)
}

/// Convert `rgb` from its embedded ICC profile to sRGB using qcms.
fn apply_icc_profile(icc: &[u8], rgb: RgbImage) -> Result<RgbImage> {
    let src_profile = qcms::Profile::new_from_slice(icc, false)
        .context("failed to parse embedded ICC profile")?;

    // Already sRGB; nothing to do.
    if src_profile.is_sRGB() {
        return Ok(rgb);
    }

    let dst_profile = qcms::Profile::new_sRGB();
    let transform = qcms::Transform::new(
        &src_profile,
        &dst_profile,
        qcms::DataType::RGB8,
        qcms::Intent::RelativeColorimetric,
    )
    .context("failed to create ICC transform")?;

    let (w, h) = (rgb.width(), rgb.height());
    let mut data = rgb.into_raw();
    transform.apply(&mut data);

    RgbImage::from_raw(w, h, data)
        .context("failed to rebuild RGB image after ICC conversion")
}

/// Transpose rows and columns (mirror across the top-left to bottom-right
/// diagonal).  Equivalent to PIL's `Image.Transpose.TRANSPOSE`.
fn transpose_image(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(h, w);
    for (x, y, pixel) in img.enumerate_pixels() {
        out.put_pixel(y, x, *pixel);
    }
    out
}

/// Binary-search for the largest resize that keeps the patch count within
/// `max_seq_len`, matching `model.get_image_size_for_seq`.
fn compute_resize_for_seq(
    orig_h: usize,
    orig_w: usize,
    patch_size: usize,
    max_seq_len: usize,
) -> (usize, usize) {
    let max_ratio: f64 = 1.0;
    let eps: f64 = 1e-5;

    let max_py = ((orig_h as f64 * max_ratio) / patch_size as f64).max(1.0) as usize;
    let max_px = ((orig_w as f64 * max_ratio) / patch_size as f64).max(1.0) as usize;

    if max_py * max_px <= max_seq_len {
        return (max_py * patch_size, max_px * patch_size);
    }

    let patchify = |ratio: f64| -> (usize, usize) {
        let py = ((orig_h as f64 * ratio) / patch_size as f64)
            .ceil()
            .min(max_py as f64) as usize;
        let px = ((orig_w as f64 * ratio) / patch_size as f64)
            .ceil()
            .min(max_px as f64) as usize;
        (py.max(1), px.max(1))
    };

    let (mut py, mut px) = patchify(eps);
    if py * px > max_seq_len {
        // Image is too large even at the minimum ratio.
        return (patch_size, patch_size);
    }

    let mut ratio = eps;
    let mut max_ratio = max_ratio;
    while (max_ratio - ratio) >= eps {
        let mid = (ratio + max_ratio) / 2.0;
        let (mpy, mpx) = patchify(mid);
        let seq_len = mpy * mpx;

        if seq_len > max_seq_len {
            max_ratio = mid;
            continue;
        }

        ratio = mid;
        py = mpy;
        px = mpx;

        if seq_len == max_seq_len {
            break;
        }
    }

    (py * patch_size, px * patch_size)
}

/// Bilinearly interpolate the learned 16x16 position embedding to the target
/// grid size, then pad/truncate to `max_seq_len`.  Matches PyTorch's
/// `F.interpolate(..., mode="bilinear", align_corners=False)`.
fn interpolate_pos_embed(
    pos_embed: &Array4<f32>,
    grid_h: usize,
    grid_w: usize,
    max_seq_len: usize,
) -> Result<Array2<f32>> {
    let view = pos_embed.index_axis(Axis(0), 0); // (16, 16, 1152)
    let mut out = Array2::<f32>::zeros((max_seq_len, EMBED_DIM));

    for y_out in 0..grid_h {
        let y_src = (y_out as f64 + 0.5) * (POS_EMBED_H as f64 / grid_h as f64) - 0.5;
        let y0 = y_src.floor() as isize;
        let dy = (y_src - y0 as f64) as f32;

        for x_out in 0..grid_w {
            let x_src = (x_out as f64 + 0.5) * (POS_EMBED_W as f64 / grid_w as f64) - 0.5;
            let x0 = x_src.floor() as isize;
            let dx = (x_src - x0 as f64) as f32;

            let row = y_out * grid_w + x_out;
            for c in 0..EMBED_DIM {
                let v00 = sample_pos_embed(&view, y0, x0, c);
                let v01 = sample_pos_embed(&view, y0, x0 + 1, c);
                let v10 = sample_pos_embed(&view, y0 + 1, x0, c);
                let v11 = sample_pos_embed(&view, y0 + 1, x0 + 1, c);

                let v0 = v00 * (1.0 - dx) + v01 * dx;
                let v1 = v10 * (1.0 - dx) + v11 * dx;
                out[[row, c]] = v0 * (1.0 - dy) + v1 * dy;
            }
        }
    }

    Ok(out)
}

#[inline]
fn sample_pos_embed(view: &ArrayView3<f32>, y: isize, x: isize, c: usize) -> f32 {
    let h = view.shape()[0] as isize;
    let w = view.shape()[1] as isize;
    let y = y.clamp(0, h - 1) as usize;
    let x = x.clamp(0, w - 1) as usize;
    view[[y, x, c]]
}

fn find_model_file(dir: &Path) -> Result<PathBuf> {
    let candidates = ["model.onnx", "jtp-3-hydra-fp16.onnx", "jtp-3-hydra-fp32.onnx"];
    for name in &candidates {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    anyhow::bail!("no ONNX model file found in {}", dir.display())
}

/// Resolve a model path the same way the regular OrtBackend does.
///
/// Absolute paths are used as-is. Relative paths are treated as a slug under
/// `~/.local/share/akasha/models/onnx/`.
fn resolve_model_dir(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }

    let slug = path.replace('/', "-");
    let data_dir = crate::config::Config::data_dir()?;
    Ok(data_dir.join(ONNX_MODELS_DIR).join(slug))
}

fn load_pos_embed(path: &Path) -> Result<Array4<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read pos_embed file: {}", path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse pos_embed safetensors: {}", path.display()))?;

    let tensor = tensors
        .tensor("pos_embed")
        .with_context(|| format!("'pos_embed' tensor not found in {}", path.display()))?;

    let shape: Vec<usize> = tensor.shape().iter().map(|&d| d as usize).collect();
    anyhow::ensure!(
        shape == [1, POS_EMBED_H, POS_EMBED_W, EMBED_DIM],
        "unexpected pos_embed shape {:?}, expected [1, {}, {}, {}]",
        shape,
        POS_EMBED_H,
        POS_EMBED_W,
        EMBED_DIM
    );

    let data: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Array4::from_shape_vec((1, POS_EMBED_H, POS_EMBED_W, EMBED_DIM), data)
        .context("failed to build pos_embed array")
}

fn load_tags(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read tags file: {}", path.display()))?;
    Ok(text
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn load_calibration_csv(path: &Path) -> Result<HashMap<String, f32>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read calibration CSV: {}", path.display()))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(text.as_bytes());

    let headers = reader.headers()?.clone();
    let tag_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("tag"))
        .context("calibration CSV missing 'tag' column")?;
    let threshold_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("threshold"))
        .context("calibration CSV missing 'threshold' column")?;

    let mut thresholds = HashMap::new();
    for result in reader.records() {
        let record = result?;
        let tag = record.get(tag_idx).unwrap_or("").trim().to_string();
        let value = record.get(threshold_idx).unwrap_or("").trim();
        if tag.is_empty() || value.is_empty() {
            continue;
        }
        let threshold: f32 = value.parse().with_context(|| format!("invalid threshold for tag {tag}"))?;
        if !(0.0..=1.0).contains(&threshold) {
            anyhow::bail!("threshold for tag {tag} must be between 0.0 and 1.0");
        }
        thresholds.insert(tag, threshold);
    }

    Ok(thresholds)
}

fn find_metadata_file(dir: &Path) -> Option<PathBuf> {
    for name in &["tags.csv", "jtp-3-hydra-tags.csv"] {
        let path = dir.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn load_metadata_csv(path: &Path) -> Result<(HashMap<String, i32>, HashMap<String, Vec<String>>)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read tag metadata CSV: {}", path.display()))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(text.as_bytes());

    let headers = reader.headers()?.clone();
    let tag_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("tag"))
        .context("metadata CSV missing 'tag' column")?;
    let category_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("category"))
        .context("metadata CSV missing 'category' column")?;
    let implications_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("implications"))
        .context("metadata CSV missing 'implications' column")?;

    let mut categories = HashMap::new();
    let mut implications = HashMap::new();
    for result in reader.records() {
        let record = result?;
        let tag = record.get(tag_idx).unwrap_or("").trim().to_string();
        if tag.is_empty() {
            continue;
        }

        if let Ok(category) = record.get(category_idx).unwrap_or("").trim().parse::<i32>() {
            categories.insert(tag.clone(), category);
        }

        let implied: Vec<String> = record
            .get(implications_idx)
            .unwrap_or("")
            .split_whitespace()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !implied.is_empty() {
            implications.insert(tag, implied);
        }
    }

    Ok((categories, implications))
}

fn apply_implications(
    labels: &mut HashMap<String, f32>,
    implications: &HashMap<String, Vec<String>>,
    mode: ImplicationMode,
) {
    if mode == ImplicationMode::Off || implications.is_empty() {
        return;
    }

    let tags: Vec<String> = labels.keys().cloned().collect();

    match mode {
        ImplicationMode::Inherit => {
            for tag in &tags {
                inherit_implications(labels, tag, implications);
            }
        }
        ImplicationMode::Constrain | ImplicationMode::ConstrainRemove => {
            for tag in &tags {
                constrain_implications(labels, tag, implications, None);
            }
        }
        ImplicationMode::Remove => {}
        ImplicationMode::Off => {}
    }

    if mode == ImplicationMode::Remove || mode == ImplicationMode::ConstrainRemove {
        for tag in &tags {
            if labels.contains_key(tag) {
                remove_implications(labels, tag, implications);
            }
        }
    }
}

fn inherit_implications(
    labels: &mut HashMap<String, f32>,
    antecedent: &str,
    implications: &HashMap<String, Vec<String>>,
) {
    let p = match labels.get(antecedent) {
        Some(&p) => p,
        None => return,
    };

    for consequent in implications.get(antecedent).into_iter().flatten() {
        if let Some(&q) = labels.get(consequent) {
            if q < p {
                labels.insert(consequent.clone(), p);
            }
        }
        inherit_implications(labels, consequent, implications);
    }
}

fn constrain_implications(
    labels: &mut HashMap<String, f32>,
    antecedent: &str,
    implications: &HashMap<String, Vec<String>>,
    target: Option<&str>,
) {
    let target = target.unwrap_or(antecedent);
    let target_p = match labels.get(target) {
        Some(&p) => p,
        None => return,
    };

    for consequent in implications.get(antecedent).into_iter().flatten() {
        if let Some(&q) = labels.get(consequent) {
            if target_p > q {
                labels.insert(target.to_string(), q);
            }
        }
        constrain_implications(labels, consequent, implications, Some(target));
    }
}

fn remove_implications(
    labels: &mut HashMap<String, f32>,
    antecedent: &str,
    implications: &HashMap<String, Vec<String>>,
) {
    for consequent in implications.get(antecedent).into_iter().flatten() {
        labels.remove(consequent);
        remove_implications(labels, consequent, implications);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_resize_keeps_small_image_native() {
        // 256x256 image fits in 1024 patches at 1:1.
        let (h, w) = compute_resize_for_seq(256, 256, 16, 1024);
        assert_eq!((h, w), (256, 256));
    }

    #[test]
    fn compute_resize_downscales_large_image() {
        // 2048x2048 would be 16384 patches; should downscale.
        let (h, w) = compute_resize_for_seq(2048, 2048, 16, 1024);
        assert!(h * w / (16 * 16) <= 1024);
        assert_eq!(h % 16, 0);
        assert_eq!(w % 16, 0);
    }

    #[test]
    fn interpolate_pos_embed_matches_expected_shape() {
        let pos = Array4::<f32>::zeros((1, 16, 16, 1152));
        let out = interpolate_pos_embed(&pos, 24, 24, 1024).unwrap();
        assert_eq!(out.shape(), &[1024, 1152]);
    }

    #[test]
    #[ignore = "manual: requires JTP-3 ONNX model in test_models/jtp3 or ~/.local/share/akasha/models/onnx/jtp-3-hydra"]
    fn jtp3_backend_runs() {
        // Prefer the normal ONNX models directory layout (relative slug).
        let model_dir = if resolve_model_dir("jtp-3-hydra")
            .map(|p| p.join("model.onnx").exists())
            .unwrap_or(false)
        {
            resolve_model_dir("jtp-3-hydra").unwrap()
        } else {
            PathBuf::from("test_models/jtp3").canonicalize().unwrap_or_else(|_| {
                PathBuf::from("test_models/jtp3")
            })
        };

        if !model_dir.join("model.onnx").exists() {
            eprintln!("Skipping: model not found at {}", model_dir.display());
            return;
        }

        let cfg = ModelConfig {
            name: "jtp-3-hydra".into(),
            kind: crate::config::ModelKind::Local,
            backend: Some("jtp3".into()),
            path: Some(model_dir.to_string_lossy().to_string()),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(crate::config::ModelTagsOptions { threshold: 0.35, top_k: Some(20) }),
            description: None,
            classification: None,
            remote: None,
            onnx: None,
            jtp3: None,
        };

        let model = Jtp3Backend.load(&cfg).expect("load model");
        let output = model.infer(Path::new("test_imgs/dagnpats.png")).expect("infer");

        match output {
            ModelOutput::Tags(tags) => {
                assert!(!tags.is_empty(), "expected at least one tag above threshold");
                eprintln!("Got {} tags", tags.len());
                for (tag, score) in tags.iter().take(10) {
                    eprintln!("  {}: {:.2}", tag, score);
                }
            }
            other => panic!("expected ModelOutput::Tags, got {:?}", other),
        }
    }
}
