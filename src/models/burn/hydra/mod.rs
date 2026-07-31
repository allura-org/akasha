//! Hydra-3.5 model implemented in Burn.
//!
//! The model graph in `modules.rs` is backend-agnostic. The `fused_ops` module
//! provides optional CPU fast paths behind backend traits; `HydraModel<B>`
//! requires those traits so it can call the fast paths when available. A new
//! backend (e.g. GPU) only needs to implement the same traits to opt into fast
//! inference, otherwise the generic Burn tensor ops are used.

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

// Compute dtype. CubeCL CUDA runs BF16: the checkpoint is natively BF16 (so
// the conversion is exact) and BF16 GEMMs hit the tensor-core path (~4x the
// FP32 rate on RTX-class GPUs) with half the memory traffic. CPU backends
// stay F32 — BF16 is slow on the Flex CPU path and the NdArray backend does
// not implement BF16 at all. wgpu stays F32 until BF16 is validated there.
#[cfg(feature = "burn-cuda")]
const MODEL_DTYPE: DType = DType::BF16;
#[cfg(not(feature = "burn-cuda"))]
const MODEL_DTYPE: DType = DType::F32;

use crate::config::ModelConfig;
use crate::models::Model;

use self::fused_ops::{
    FastLinearBackend, FastRmsNormBackend, FusedAttentionBackend, FusedGluBackend,
    FusedHydraMidBlockBackend, FusedHydraPoolBackend, FusedHydraPoolTailBackend, FusedMlpBackend,
    FusedNaFlexBlockBackend,
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
    max_batch_size: usize,
    background: [u8; 3],
    /// Persistent scratch buffers for the fused kernels. Kept across forwards
    /// so the large pool/FF buffers (~0.5 GB) are allocated and page-faulted
    /// once per model instead of once per image.
    workspace: std::sync::Mutex<fused_ops::BlockWorkspace>,
    /// Interpolated position embeddings keyed by patch grid `(h, w)`. Grid
    /// sizes are quantized by the resize search, so a whole collection hits a
    /// handful of buckets; caching skips a scalar ~1.2M-element bilinear
    /// interpolation per image.
    pos_embed_cache: std::sync::Mutex<HashMap<(usize, usize), ndarray::Array2<f32>>>,
    _phantom: std::marker::PhantomData<B>,
}

impl<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedHydraPoolTailBackend
        + FusedHydraPoolBackend
        + FusedNaFlexBlockBackend,
> HydraModel<B>
{
    pub fn load(config: &ModelConfig, device: B::Device) -> Result<Self> {
        // Configure the CubeCL runtime (single shared stream) before the first
        // GPU allocation; no-op on CPU backends and on later calls.
        super::init_cubecl_runtime();

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
        // Batching pays off on GPU (amortizes readback/fixed costs); the CPU
        // path already saturates all cores at batch 1 and measures slightly
        // slower when batched, so keep it at 1 there.
        #[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
        let max_batch_size = config
            .burn
            .as_ref()
            .map(|b| b.batch_size)
            .unwrap_or(crate::config::default_burn_batch_size())
            .max(1);
        #[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
        let max_batch_size = 1;

        Ok(Self {
            model,
            pos_embed,
            labels,
            threshold,
            top_k,
            max_seq_len: 1024,
            max_batch_size,
            background,
            workspace: std::sync::Mutex::new(fused_ops::BlockWorkspace::new()),
            pos_embed_cache: std::sync::Mutex::new(HashMap::new()),
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
        + FusedHydraPoolTailBackend
        + FusedHydraPoolBackend
        + FusedNaFlexBlockBackend
        + 'static,
> HydraModel<B>
{
    /// Run the model and return raw logits (one score per label).
    pub fn infer_logits(&self, image_path: &Path) -> Result<Vec<f32>> {
        Ok(self.infer_logits_batch(&[image_path])?.remove(0))
    }

    /// Run the model on a batch of images; returns one logits row per image.
    pub fn infer_logits_batch(&self, image_paths: &[&Path]) -> Result<Vec<Vec<f32>>> {
        let t0 = Instant::now();
        let device = <B::Device as Default>::default();
        let n = image_paths.len();
        anyhow::ensure!(n > 0, "infer_logits_batch: empty batch");

        // Preprocess (read/decode/resize/patchify) runs unlocked so pipelined
        // inference tasks overlap their CPU-side work; the pos-embed cache
        // locks itself per lookup, and only the forward below is serialized
        // (by the workspace mutex).
        let mut pres = Vec::with_capacity(n);
        for path in image_paths {
            pres.push(
                preprocess(
                    path,
                    &self.pos_embed,
                    self.max_seq_len,
                    self.background,
                    &self.pos_embed_cache,
                )
                .with_context(|| format!("failed to preprocess image: {}", path.display()))?,
            );
        }
        let t_pre = t0.elapsed();

        // Stack per-image preprocess outputs into batch tensors.
        let t1 = Instant::now();
        let seq = self.max_seq_len;
        let mut patches_flat = Vec::with_capacity(n * seq * 768);
        let mut pos_flat = Vec::with_capacity(n * seq * 1152);
        let mut valid_raw = Vec::with_capacity(n * seq);
        let mut n_valids = Vec::with_capacity(n);
        let mut all_valid = true;
        for pre in pres {
            patches_flat.extend(pre.patches.into_raw_vec_and_offset().0);
            pos_flat.extend(pre.pos_embed.into_raw_vec_and_offset().0);
            let valid = pre.valid.into_raw_vec_and_offset().0;
            let n_valid = valid.iter().filter(|&&b| b).count();
            if n_valid < seq {
                all_valid = false;
            }
            n_valids.push(n_valid);
            valid_raw.extend(valid.into_iter().map(|b| if b { 1i32 } else { 0i32 }));
        }

        let patches = Tensor::<B, 1>::from_data(
            patches_flat.as_slice(),
            (&device, MODEL_DTYPE),
        )
        .reshape([n, seq, 768]);

        let pos_embed =
            Tensor::<B, 1>::from_data(pos_flat.as_slice(), (&device, MODEL_DTYPE))
                .reshape([n, seq, 1152]);

        // Skip building the bool mask when every patch of every image is
        // valid; this lets attention take the faster no-mask path.
        let mask: Option<Tensor<B, 2, Bool>> = if all_valid {
            None
        } else {
            // Candle does not support bool_from_data, so build the mask as an
            // Int tensor and compare to 1.
            Some(
                Tensor::<B, 1, Int>::from_data(valid_raw.as_slice(), &device)
                    .reshape([n, seq])
                    .equal_elem(1),
            )
        };
        let t_tensors = t1.elapsed();

        let logits = {
            // Recover from poisoning: the workspace is pure scratch memory, so
            // a panic in a previous forward (e.g. GPU OOM) leaves nothing
            // inconsistent behind — only buffers to reuse.
            let mut workspace = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
            self.model
                .forward_with_workspace(patches, pos_embed, mask, Some(&n_valids), &mut workspace)
        };
        let t_forward = t1.elapsed();

        // Convert to f32 for post-processing.
        let t2 = Instant::now();
        let logits_f32 = logits.cast(DType::F32);
        let flat = logits_f32.to_data().to_vec::<f32>()?;
        let t_post = t2.elapsed();

        tracing::debug!(
            "infer_logits_batch total={:.3}s batch={} preprocess={:.3}s tensor_create={:.3}s forward={:.3}s postprocess={:.3}s",
            t0.elapsed().as_secs_f64(),
            n,
            t_pre.as_secs_f64(),
            t_tensors.as_secs_f64(),
            t_forward.as_secs_f64(),
            t_post.as_secs_f64()
        );

        let n_labels = flat.len() / n;
        Ok(flat.chunks_exact(n_labels).map(|c| c.to_vec()).collect())
    }

    /// Map raw logits to the thresholded, top-k filtered tag output.
    fn logits_to_output(&self, scores: Vec<f32>) -> crate::models::ModelOutput {
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

        crate::models::ModelOutput::Tags(tags.into_iter().collect())
    }
}

impl<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedHydraPoolTailBackend
        + FusedHydraPoolBackend
        + FusedNaFlexBlockBackend
        + 'static,
> Model for HydraModel<B>
{
    fn infer(&self, image_path: &Path) -> Result<crate::models::ModelOutput> {
        Ok(self.logits_to_output(self.infer_logits(image_path)?))
    }

    fn infer_batch(&self, image_paths: &[&Path]) -> Result<Vec<crate::models::ModelOutput>> {
        if image_paths.len() == 1 {
            return Ok(vec![self.infer(image_paths[0])?]);
        }
        let batch = self.infer_logits_batch(image_paths)?;
        Ok(batch
            .into_iter()
            .map(|scores| self.logits_to_output(scores))
            .collect())
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    fn release_memory(&self) {
        // Free the CubeCL memory pool's unused pages and sync so the deferred
        // frees execute and the driver's async mempool gets a release point.
        // No-ops on CPU backends.
        let device = <B::Device as Default>::default();
        B::memory_cleanup(&device);
        if let Err(e) = B::sync(&device) {
            tracing::warn!("hydra: backend sync after memory cleanup failed: {e}");
        }
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

    /// VRAM growth investigation: run `infer_logits` in a loop and print
    /// per-process GPU memory (via nvidia-smi) after each iteration.
    #[test]
    #[ignore = "manual: VRAM growth investigation"]
    fn hydra_burn_vram_loop() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
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
            burn: None,
        };

        let model = HydraModel::<crate::models::burn::BurnBackendType>::load(&cfg, device)
            .expect("load model");
        let img_paths = [
            Path::new("/home/asriel/Projects/akasha/test_imgs/dagnpats.png"),
            Path::new("/home/asriel/Projects/akasha/test_imgs/portrait.png"),
            Path::new("/home/asriel/Projects/akasha/test_imgs/landscape.webp"),
        ];

        let pid = std::process::id().to_string();
        let vram = move || -> String {
            match std::process::Command::new("nvidia-smi")
                .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
                .output()
            {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let mine: Vec<&str> = stdout
                        .lines()
                        .filter(|l| l.trim_start().starts_with(pid.as_str()))
                        .collect();
                    if mine.is_empty() {
                        format!("(no per-process entry; all: {})", stdout.trim())
                    } else {
                        mine.join(" | ")
                    }
                }
                Err(e) => format!("nvidia-smi error: {e}"),
            }
        };

        // Set AKASHA_HYDRA_VRAM_CLEANUP=1 to call `Backend::memory_cleanup`
        // after every inference, releasing the cubecl memory pool pages.
        let do_cleanup = std::env::var("AKASHA_HYDRA_VRAM_CLEANUP").is_ok();
        let device = <crate::models::burn::BurnDevice as Default>::default();

        eprintln!("after load: {}", vram());
        let iters: usize = std::env::var("AKASHA_HYDRA_VRAM_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        // AKASHA_HYDRA_VRAM_THREADS=N runs the loop on N std threads sharing
        // the same model, simulating tokio's blocking thread pool (each thread
        // gets its own CubeCL StreamId unless max_streams is capped).
        let threads: usize = std::env::var("AKASHA_HYDRA_VRAM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let model = std::sync::Arc::new(model);
        let run_loop = |model: &std::sync::Arc<HydraModel<crate::models::burn::BurnBackendType>>,
                        tid: usize| {
            for i in 0..iters {
                let img = img_paths[i % img_paths.len()];
                let t0 = std::time::Instant::now();
                let logits = model.infer_logits(img).expect("infer");
                let t_infer = t0.elapsed();
                std::hint::black_box(&logits);
                if do_cleanup {
                    use burn::tensor::backend::Backend as _;
                    let t1 = std::time::Instant::now();
                    crate::models::burn::BurnBackendType::memory_cleanup(&device);
                    crate::models::burn::BurnBackendType::sync(&device).expect("sync");
                    eprintln!(
                        "t{} iter {:02} {}: infer={:.3}s cleanup={:.3}s vram={}",
                        tid,
                        i,
                        img.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                        t_infer.as_secs_f64(),
                        t1.elapsed().as_secs_f64(),
                        vram()
                    );
                } else {
                    eprintln!(
                        "t{} iter {:02} {}: infer={:.3}s vram={}",
                        tid,
                        i,
                        img.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                        t_infer.as_secs_f64(),
                        vram()
                    );
                }
            }
        };

        if threads <= 1 {
            run_loop(&model, 0);
        } else {
            std::thread::scope(|s| {
                for tid in 0..threads {
                    let model = model.clone();
                    s.spawn(move || run_loop(&model, tid));
                }
            });
        }

        // Production-path check: the batch-end release (what the SearchWorker
        // calls after the queue drains) must return VRAM to baseline.
        crate::models::Model::release_memory(model.as_ref());
        eprintln!("after release_memory: {}", vram());
    }

    #[test]
    #[ignore = "manual: requires Hydra-3.5 safetensors"]
    fn hydra_burn_runs() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
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
            burn: None,
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

            // A/B helpers: set AKASHA_HYDRA_LOGITS_DUMP=<dir> to write raw
            // logits per image, and AKASHA_HYDRA_LOGITS_REF=<dir> to compare
            // this run against a previous dump (e.g. bf16 vs F32 GEMM paths).
            let img_name = img_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string());
            if let Ok(dir) = std::env::var("AKASHA_HYDRA_LOGITS_DUMP") {
                let mut bytes = Vec::with_capacity(logits.len() * 4);
                for v in &logits {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(Path::new(&dir).join(format!("{img_name}.f32")), &bytes)
                    .expect("dump logits");
            }
            if let Ok(dir) = std::env::var("AKASHA_HYDRA_LOGITS_REF") {
                let ref_file = Path::new(&dir).join(format!("{img_name}.f32"));
                if ref_file.is_file() {
                    let ref_bytes = std::fs::read(&ref_file).expect("read ref logits");
                    let ref_logits: Vec<f32> = ref_bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    assert_eq!(ref_logits.len(), logits.len(), "ref logits length mismatch");
                    let mut max_logit = 0.0f32;
                    let mut max_prob = 0.0f32;
                    for (&r, &l) in ref_logits.iter().zip(&logits) {
                        max_logit = max_logit.max((r - l).abs());
                        let pr = 1.0 / (1.0 + (-r).exp());
                        let pl = 1.0 / (1.0 + (-l).exp());
                        max_prob = max_prob.max((pr - pl).abs());
                    }
                    eprintln!(
                        "A/B vs {ref_file:?}: max abs logit diff {:.6}, max abs prob diff {:.6}",
                        max_logit, max_prob
                    );
                }
            }

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

    #[test]
    #[ignore = "manual: requires Hydra-3.5 safetensors"]
    fn hydra_burn_batch_matches_single() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
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
            burn: None,
        };

        let model = HydraModel::<crate::models::burn::BurnBackendType>::load(&cfg, device)
            .expect("load model");
        let img_paths = [
            Path::new("/home/asriel/Projects/akasha/test_imgs/dagnpats.png"),
            Path::new("/home/asriel/Projects/akasha/test_imgs/portrait.png"),
            Path::new("/home/asriel/Projects/akasha/test_imgs/landscape.webp"),
        ];

        // Single-image reference logits.
        let mut singles = Vec::new();
        let t_single0 = Instant::now();
        for img_path in &img_paths {
            singles.push(model.infer_logits(img_path).expect("infer"));
        }
        let t_singles = t_single0.elapsed();

        // One batched call. The three images have different aspect ratios, so
        // their valid patch counts differ and the ragged-mask path (pool
        // key-padding mask) is exercised.
        let refs: Vec<&Path> = img_paths.to_vec();
        let t_batch0 = Instant::now();
        let batch = model.infer_logits_batch(&refs).expect("batch infer");
        let t_batch = t_batch0.elapsed();

        assert_eq!(batch.len(), singles.len());
        // BF16 GPU runs tile GEMMs differently at different batch/seq shapes
        // (e.g. trimmed vs padded seq), so accumulation order — and the last
        // ~0.3% of probability — varies with batch composition. F32 runs use
        // identical kernels for identical rows and match far more tightly.
        let tol: f32 = if MODEL_DTYPE == DType::F32 { 1e-3 } else { 1e-2 };
        for (img, (s, b)) in img_paths.iter().zip(singles.iter().zip(batch.iter())) {
            assert_eq!(s.len(), b.len(), "logits length mismatch on {img:?}");
            let mut max_prob = 0.0f32;
            for (&x, &y) in s.iter().zip(b.iter()) {
                let px = 1.0 / (1.0 + (-x).exp());
                let py = 1.0 / (1.0 + (-y).exp());
                max_prob = max_prob.max((px - py).abs());
            }
            eprintln!("{img:?}: batch-vs-single max abs prob diff {max_prob:.6}");
            assert!(
                max_prob < tol,
                "batch vs single mismatch on {img:?}: {max_prob}"
            );
        }

        eprintln!(
            "batch of {} took {:.3}s; singles took {:.3}s total",
            img_paths.len(),
            t_batch.as_secs_f64(),
            t_singles.as_secs_f64()
        );

        // Steady-state: kernels for the batch shapes are now compiled.
        let t_batch2_0 = Instant::now();
        let batch2 = model.infer_logits_batch(&refs).expect("batch infer 2");
        let t_batch2 = t_batch2_0.elapsed();
        std::hint::black_box(&batch2);
        eprintln!(
            "steady-state batch of {} took {:.3}s ({:.3}s/image)",
            img_paths.len(),
            t_batch2.as_secs_f64(),
            t_batch2.as_secs_f64() / img_paths.len() as f64
        );

        // Ragged batch of 8: the full masked-attention scores tensor would be
        // ~7 GiB here, which exceeds CubeCL's max pool page; on GPU this
        // exercises the query-tiled masked attention.
        let ragged_idx = [0usize, 1, 2, 0, 1, 2, 1, 2];
        let ragged8: Vec<&Path> = ragged_idx.iter().map(|&i| img_paths[i]).collect();
        let ragged = model.infer_logits_batch(&ragged8).expect("ragged batch of 8");
        assert_eq!(ragged.len(), ragged_idx.len());
        for (i, logits) in ragged.iter().enumerate() {
            let expected = &singles[ragged_idx[i]];
            let mut max_prob = 0.0f32;
            for (&x, &y) in expected.iter().zip(logits.iter()) {
                let px = 1.0 / (1.0 + (-x).exp());
                let py = 1.0 / (1.0 + (-y).exp());
                max_prob = max_prob.max((px - py).abs());
            }
            assert!(
                max_prob < tol,
                "ragged batch item {i} mismatch: {max_prob}"
            );
        }
        eprintln!("ragged batch of 8 ok");

        // Diagnostics: AKASHA_HYDRA_BATCH_SAME[=N] batches the same (all-valid)
        // image N times (default 3) to isolate the masked/ragged path's cost
        // from batching itself.
        if let Ok(n_str) = std::env::var("AKASHA_HYDRA_BATCH_SAME") {
            let bn: usize = n_str.parse().unwrap_or(3);
            let same: Vec<&Path> = std::iter::repeat_n(img_paths[0], bn).collect();
            for _ in 0..2 {
                std::hint::black_box(model.infer_logits_batch(&same).expect("warm"));
            }
            let t0 = Instant::now();
            let reps = 5;
            for _ in 0..reps {
                std::hint::black_box(model.infer_logits_batch(&same).expect("same batch"));
            }
            eprintln!(
                "same-image batch of {bn}: {:.3}s/batch ({:.3}s/image) over {reps} reps",
                t0.elapsed().as_secs_f64() / reps as f64,
                t0.elapsed().as_secs_f64() / reps as f64 / bn as f64
            );
            let t1 = Instant::now();
            for _ in 0..reps {
                for p in &same {
                    std::hint::black_box(model.infer_logits(p).expect("single"));
                }
            }
            eprintln!(
                "steady singles: {:.3}s/image over {} reps",
                t1.elapsed().as_secs_f64() / (reps * bn) as f64,
                reps * bn
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

/// Throwaway phase-timing bench for the worker pipeline. Run with:
///   AKASHA_HYDRA_BENCH_LIST=<file with image paths, one per line> \
///   AKASHA_HYDRA_MODEL=<path to hydra-3.5.safetensors> \
///   cargo test --release --features burn-cuda hydra_pipeline_bench -- --ignored --nocapture
#[cfg(all(test, feature = "burn-cuda"))]
mod pipeline_bench {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelTagsOptions};
    use std::path::PathBuf;
    use std::sync::Arc;

    type Cuda = burn::backend::Cuda;

    fn report(name: &str, mut times: Vec<std::time::Duration>) {
        times.sort();
        let n = times.len();
        let sum: f64 = times.iter().map(|t| t.as_secs_f64()).sum();
        let pct = |p: usize| times[(n * p / 100).min(n - 1)].as_secs_f64() * 1000.0;
        println!(
            "{name}: n={n} mean={:.1}ms p50={:.1}ms p90={:.1}ms p99={:.1}ms => {:.2}/s",
            sum / n as f64 * 1000.0,
            pct(50),
            pct(90),
            pct(99),
            n as f64 / sum
        );
    }

    #[test]
    #[ignore]
    fn hydra_pipeline_bench() {
        let list = std::env::var("AKASHA_HYDRA_BENCH_LIST").expect("AKASHA_HYDRA_BENCH_LIST");
        let model_path = std::env::var("AKASHA_HYDRA_MODEL").expect("AKASHA_HYDRA_MODEL");
        let paths: Vec<PathBuf> = std::fs::read_to_string(list)
            .unwrap()
            .lines()
            .map(PathBuf::from)
            .collect();
        println!("benching {} images", paths.len());

        let config = ModelConfig {
            name: "hydra-3.5".into(),
            kind: ModelKind::Local,
            backend: Some("burn".into()),
            path: Some(model_path),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(ModelTagsOptions::default()),
            description: None,
            classification: None,
            remote: None,
            onnx: None,
            jtp3: None,
            burn: None,
        };
        let model = HydraModel::<Cuda>::load(&config, Default::default()).unwrap();

        // Phase 1: preprocess only, sequential.
        let mut pre_times = Vec::new();
        for p in &paths {
            let t = Instant::now();
            preprocess(
                p,
                &model.pos_embed,
                model.max_seq_len,
                model.background,
                &model.pos_embed_cache,
            )
            .unwrap();
            pre_times.push(t.elapsed());
        }
        report("preprocess(seq)", pre_times);

        // Warm up the forward path (kernel autotune-free, but first-run
        // module loads and workspace allocation still apply).
        for p in paths.iter().take(8) {
            model.infer_logits(p).unwrap();
        }

        // Phase 2: full infer (preprocess + H2D + forward + readback), sequential.
        let mut infer_times = Vec::new();
        for p in &paths {
            let t = Instant::now();
            model.infer_logits(p).unwrap();
            infer_times.push(t.elapsed());
        }
        report("infer(seq)", infer_times);

        // Phase 3: 4 threads sharing the model, mimicking PIPELINE_DEPTH=4.
        let model = Arc::new(model);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for chunk in paths.chunks((paths.len() + 3) / 4) {
                let m = model.clone();
                s.spawn(move || {
                    for p in chunk {
                        m.infer_logits(p).unwrap();
                    }
                });
            }
        });
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "infer(4 threads): {:.1}s wall for {} images => {:.2}/s",
            wall,
            paths.len(),
            paths.len() as f64 / wall
        );

        // Phase 4/5: tokio spawn_blocking pipeline mirroring the SearchWorker
        // loop (depth 4, one image per task), with and without real DB writes.
        async fn run_pipeline(
            model: Arc<HydraModel<Cuda>>,
            jobs: &[(i64, PathBuf)],
            pool: Option<&sqlx::SqlitePool>,
        ) -> f64 {
            const DEPTH: usize = 8;
            let mut pending: std::collections::VecDeque<(
                i64,
                tokio::task::JoinHandle<anyhow::Result<crate::models::ModelOutput>>,
            )> = std::collections::VecDeque::new();
            let mut next = 0usize;
            let t0 = Instant::now();
            while next < jobs.len() || !pending.is_empty() {
                while next < jobs.len() && pending.len() < DEPTH {
                    let (id, path) = &jobs[next];
                    let m = model.clone();
                    let p = path.clone();
                    pending.push_back((*id, tokio::task::spawn_blocking(move || m.infer(&p))));
                    next += 1;
                }
                let (id, handle) = pending.pop_front().unwrap();
                let out = handle.await.unwrap().unwrap();
                if let Some(pool) = pool {
                    if let crate::models::ModelOutput::Tags(tags) = out {
                        crate::db::searchable::update_tags_json(pool, id, "hydra-3.5", tags)
                            .await
                            .unwrap();
                    }
                }
            }
            t0.elapsed().as_secs_f64()
        }

        let jobs: Vec<(i64, PathBuf)> = std::fs::read_to_string(
            std::env::var("AKASHA_HYDRA_BENCH_JOBS").expect("AKASHA_HYDRA_BENCH_JOBS"),
        )
        .unwrap()
        .lines()
        .map(|l| {
            let (id, path) = l.split_once('|').unwrap();
            (id.parse().unwrap(), PathBuf::from(path))
        })
        .collect();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let (tokio_only_s, tokio_db_s) = rt.block_on(async {
            let t_no_db = run_pipeline(model.clone(), &jobs, None).await;

            let pool = sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(5)
                .connect_with(
                    sqlx::sqlite::SqliteConnectOptions::new()
                        .filename(
                            std::env::var("AKASHA_HYDRA_BENCH_DB").expect("AKASHA_HYDRA_BENCH_DB"),
                        )
                        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
                        .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
                        .busy_timeout(std::time::Duration::from_secs(5)),
                )
                .await
                .unwrap();
            let t_db = run_pipeline(model.clone(), &jobs, Some(&pool)).await;
            (t_no_db, t_db)
        });
        println!(
            "tokio pipeline (no DB): {:.1}s => {:.2}/s",
            tokio_only_s,
            jobs.len() as f64 / tokio_only_s
        );
        println!(
            "tokio pipeline (+ real tag writes): {:.1}s => {:.2}/s",
            tokio_db_s,
            jobs.len() as f64 / tokio_db_s
        );

        // Phase 6: A/B the SIMD resize path against the scalar reference at
        // the model level — logit deltas and above-threshold tag-set changes.
        // A same-path control (FIR twice) separates forward nondeterminism
        // from genuine resize-induced differences.
        let mut max_logit_diff_resize = 0f32;
        let mut max_logit_diff_control = 0f32;
        let mut tag_diff_resize = 0usize;
        let mut tag_diff_control = 0usize;
        let mut compared = 0usize;
        let above = |logits: &[f32], threshold: f32| -> std::collections::HashSet<usize> {
            logits
                .iter()
                .enumerate()
                .filter(|&(_, &v)| 1.0 / (1.0 + (-v).exp()) >= threshold)
                .map(|(i, _)| i)
                .collect()
        };
        let set_diff = |a: &std::collections::HashSet<usize>, b: &std::collections::HashSet<usize>| {
            a.symmetric_difference(b).count()
        };
        for p in paths.iter().take(16) {
            let fir1 = model.infer_logits(p).unwrap();
            let fir2 = model.infer_logits(p).unwrap();
            // SAFETY: single-threaded test; no concurrent env readers.
            unsafe { std::env::set_var("AKASHA_HYDRA_SCALAR_RESIZE", "1") };
            let scalar = model.infer_logits(p).unwrap();
            // SAFETY: single-threaded test; no concurrent env readers.
            unsafe { std::env::remove_var("AKASHA_HYDRA_SCALAR_RESIZE") };

            let d_control = fir1
                .iter()
                .zip(fir2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let d_resize = fir1
                .iter()
                .zip(scalar.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            max_logit_diff_control = max_logit_diff_control.max(d_control);
            max_logit_diff_resize = max_logit_diff_resize.max(d_resize);
            tag_diff_control += set_diff(&above(&fir1, model.threshold), &above(&fir2, model.threshold));
            tag_diff_resize += set_diff(&above(&fir1, model.threshold), &above(&scalar, model.threshold));
            compared += 1;
        }
        println!(
            "resize A/B ({} imgs): control max|dlogit|={:.4} ({} tags flipped total), \
             resize-vs-scalar max|dlogit|={:.4} ({} tags flipped total)",
            compared, max_logit_diff_control, tag_diff_control, max_logit_diff_resize, tag_diff_resize
        );
    }
}
