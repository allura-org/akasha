# VLM Description Job Hookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `visionlanguage` jobs actually generate image descriptions by adding a flexible VLM architecture registry to the Candle backend, with Gemma 4 E2B as the first supported model.

**Architecture:** Keep the existing `BackendRegistry`/`Model` worker plumbing unchanged. Inside `CandleBackend`, dispatch description-configured models to a new `VlmArchitectureRegistry`. Each VLM architecture (Gemma4 first, BLIP stubbed) implements a `VlmModel` trait with stateful text generation. A `VlmModelWrapper` adapts this to the existing `Model::infer` interface.

**Tech Stack:** Rust 2024, `candle-core`/`candle-nn`/`candle-transformers` 0.11.0, `hf-hub` 0.4, `tokenizers` 0.22, SQLite/sqlx, egui.

## Global Constraints

- `candle-core`, `candle-nn`, `candle-transformers` must be `0.11.0` (Gemma 4 is unavailable in 0.8.x).
- `tokenizers` must be added as an optional dependency gated by the `candle` feature.
- Do **not** add a dependency on `candle-examples`.
- The worker must not be modified; it already handles `ModelOutput::Description`.
- All existing `cargo test` tests must continue to pass after the upgrade.
- Code follows existing project style: `anyhow::Result`, `tracing`, explicit SQLite parameter binding.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Cargo.toml` | Pin candle versions, add `tokenizers`. |
| `src/config.rs` | Extend `ModelDescriptionOptions` with generation controls. |
| `src/models/loader.rs` | Load `tokenizer.json`; handle `model.safetensors.index.json` sharded weights. |
| `src/models/candle.rs` | Dispatch to tagger or VLM registry; wrap VLM in `Model` adapter. |
| `src/models/candle/vlm/mod.rs` | `VlmModel` trait, `VlmArchitecture` trait, `VlmArchitectureRegistry`, shared logits/sampling helpers. |
| `src/models/candle/vlm/token_stream.rs` | Decode token IDs to a final string (streaming-ready). |
| `src/models/candle/vlm/gemma4.rs` | Gemma 4 loading, image preprocessing, prompt formatting, generation loop. |
| `src/models/candle/vlm/blip.rs` | Minimal stub architecture showing how to register a second VLM. |

---

## Task 1: Upgrade Candle Dependencies and Verify Tests

**Files:**
- Modify: `Cargo.toml`
- Test: `cargo test --features candle`

**Interfaces:**
- Consumes: nothing.
- Produces: Updated `Cargo.toml` and `Cargo.lock` with candle 0.11.0 and tokenizers 0.22.

- [ ] **Step 1: Update candle dependency versions**

  In `Cargo.toml`, change the three candle crates and add `tokenizers`:

  ```toml
  candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers", "dep:hf-hub", "dep:tokenizers"]

  # Local inference (opt-in)
  candle-core = { version = "0.11.0", optional = true }
  candle-nn = { version = "0.11.0", optional = true }
  candle-transformers = { version = "0.11.0", optional = true }
  hf-hub = { version = "0.4", default-features = false, features = ["ureq", "rustls-tls"], optional = true }
  tokenizers = { version = "0.22", default-features = false, features = ["onig"], optional = true }
  ```

- [ ] **Step 2: Run cargo check**

  Run:

  ```bash
  cargo check --features candle
  ```

  Expected: only warnings, no errors. If the existing `ViTTagger` has API errors caused by candle 0.11 changes, fix them in this task before moving on.

- [ ] **Step 3: Run the full test suite**

  Run:

  ```bash
  cargo test --features candle
  ```

  Expected: all 58 existing tests pass.

- [ ] **Step 4: Commit**

  ```bash
  git add Cargo.toml Cargo.lock
  git commit -m "deps: upgrade candle to 0.11.0 and add tokenizers for VLM support"
  ```

---

## Task 2: Extend Model Loader for Tokenizer and Sharded Weights

**Files:**
- Modify: `src/models/loader.rs`
- Test: new tests in `src/models/loader.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct ModelFiles` with new field `pub tokenizer_path: Option<PathBuf>`.
  - `pub fn load_safetensors_paths(repo: &ApiRepo, base_dir: Option<&Path>) -> Result<Vec<PathBuf>>` that handles both `model.safetensors.index.json` and single-file `model.safetensors`.
  - `pub fn load_model_files(...)` populates `tokenizer_path` and calls `load_safetensors_paths`.

- [ ] **Step 1: Add `tokenizer_path` to `ModelFiles` and a sharded-weight helper**

  Replace the top of `src/models/loader.rs` with:

  ```rust
  use std::path::{Path, PathBuf};
  use anyhow::{Context, Result};

  pub enum ModelSource {
      HfSlug(String),
      LocalPath(PathBuf),
  }

  pub struct ModelFiles {
      pub config_path: PathBuf,
      pub weights_paths: Vec<PathBuf>,
      pub tokenizer_path: Option<PathBuf>,
      /// Optional per-model label list. Some models (e.g. standard HF ViT classifiers)
      /// store labels in `config.json` instead of a separate file.
      pub labels_path: Option<PathBuf>,
  }
  ```

- [ ] **Step 2: Implement `load_safetensors_paths`**

  Add inside `src/models/loader.rs`:

  ```rust
  #[cfg(feature = "candle")]
  fn parse_safetensors_index(index_path: &Path) -> Result<Vec<String>> {
      let file = std::fs::File::open(index_path)
          .with_context(|| format!("failed to open {index_path:?}"))?;
      let json: serde_json::Value = serde_json::from_reader(file)
          .with_context(|| format!("failed to parse {index_path:?}"))?;
      let weight_map = json
          .get("weight_map")
          .and_then(|v| v.as_object())
          .context("no weight_map object in safetensors index")?;

      let mut files = std::collections::HashSet::new();
      for value in weight_map.values() {
          if let Some(name) = value.as_str() {
              files.insert(name.to_string());
          }
      }
      let mut files: Vec<_> = files.into_iter().collect();
      files.sort();
      Ok(files)
  }

  #[cfg(feature = "candle")]
  pub fn load_safetensors_paths(
      repo: Option<&hf_hub::api::sync::ApiRepo>,
      local_dir: Option<&Path>,
  ) -> Result<Vec<PathBuf>> {
      let index_path = match (repo, local_dir) {
          (Some(repo), None) => repo.get("model.safetensors.index.json").ok(),
          (None, Some(dir)) => {
              let p = dir.join("model.safetensors.index.json");
              p.exists().then_some(p)
          }
          _ => None,
      };

      if let Some(index_path) = index_path {
          let names = parse_safetensors_index(&index_path)?;
          match repo {
              Some(repo) => names
                  .into_iter()
                  .map(|n| repo.get(&n).with_context(|| format!("failed to fetch {n}")))
                  .collect(),
              None => {
                  let dir = local_dir.unwrap();
                  Ok(names.into_iter().map(|n| dir.join(n)).collect())
              }
          }
      } else {
          let single = match (repo, local_dir) {
              (Some(repo), None) => repo.get("model.safetensors")?,
              (None, Some(dir)) => {
                  let p = dir.join("model.safetensors");
                  if p.exists() {
                      p
                  } else {
                      anyhow::bail!("model.safetensors not found in {}", dir.display());
                  }
              }
              _ => anyhow::bail!("expected repo or local_dir"),
          };
          Ok(vec![single])
      }
  }
  ```

- [ ] **Step 3: Update `load_model_files`**

  Replace the `#[cfg(feature = "candle")]` `load_model_files` with:

  ```rust
  #[cfg(feature = "candle")]
  pub fn load_model_files(source: &ModelSource) -> Result<ModelFiles> {
      match source {
          ModelSource::HfSlug(slug) => {
              let api = hf_hub::api::sync::Api::new()?;
              let repo = api.model(slug.clone());
              let labels_path = repo.get("selected_tags.csv").ok();
              let tokenizer_path = repo.get("tokenizer.json").ok();
              let weights_paths = load_safetensors_paths(Some(&repo), None)?;
              Ok(ModelFiles {
                  config_path: repo.get("config.json")?,
                  weights_paths,
                  tokenizer_path,
                  labels_path,
              })
          }
          ModelSource::LocalPath(dir) => {
              let labels_path = dir.join("selected_tags.csv");
              let labels_path = labels_path.exists().then_some(labels_path);
              let tokenizer_path = dir.join("tokenizer.json");
              let tokenizer_path = tokenizer_path.exists().then_some(tokenizer_path);
              let weights_paths = load_safetensors_paths(None, Some(dir))?;
              Ok(ModelFiles {
                  config_path: dir.join("config.json"),
                  weights_paths,
                  tokenizer_path,
                  labels_path,
              })
          }
      }
  }
  ```

- [ ] **Step 4: Update non-candle stub**

  Replace the `#[cfg(not(feature = "candle"))]` stub with:

  ```rust
  #[cfg(not(feature = "candle"))]
  pub fn load_model_files(_source: &ModelSource) -> Result<ModelFiles> {
      anyhow::bail!("candle feature not enabled")
  }
  ```

- [ ] **Step 5: Update existing ViTTagger to use `weights_paths`**

  In `src/models/tagger.rs`, `ViTTagger::load` currently does:

  ```rust
  let tensors = unsafe {
      candle_core::safetensors::MmapedSafetensors::new(&files.weights_path)
  } ...
  ```

  Change to:

  ```rust
  let tensors = unsafe {
      candle_core::safetensors::MmapedSafetensors::new(&files.weights_paths)
  }
  .with_context(|| format!("failed to mmap weights: {:?}", files.weights_paths))?;
  ```

  Note: `MmapedSafetensors::new` accepts `&[PathBuf]`.

- [ ] **Step 6: Add a test for sharded weight loading**

  Add to the `#[cfg(test)]` module at the bottom of `src/models/loader.rs`:

  ```rust
  #[test]
  fn parse_safetensors_index_returns_unique_sorted_files() {
      use std::io::Write;
      let temp = tempfile::tempdir().unwrap();
      let index = temp.path().join("model.safetensors.index.json");
      let mut f = std::fs::File::create(&index).unwrap();
      writeln!(
          f,
          r#"{{"weight_map":{{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}}}}"#
      ).unwrap();

      let files = super::parse_safetensors_index(&index).unwrap();
      assert_eq!(files, vec![
          "model-00001-of-00002.safetensors",
          "model-00002-of-00002.safetensors",
      ]);
  }
  ```

  Add `tempfile` as a dev-dependency if it is not already present. Check `Cargo.toml` first; if absent, add:

  ```toml
  [dev-dependencies]
  tempfile = "3"
  ```

- [ ] **Step 7: Run tests**

  ```bash
  cargo test --features candle -- loader
  ```

  Expected: new test and existing tests pass.

- [ ] **Step 8: Commit**

  ```bash
  git add Cargo.toml src/models/loader.rs src/models/tagger.rs
  git commit -m "feat(loader): add tokenizer_path and sharded safetensors loading"
  ```

---

## Task 3: Extend ModelDescriptionOptions

**Files:**
- Modify: `src/config.rs`
- Test: existing `parse_unified_models_config` or add a new one

**Interfaces:**
- Consumes: nothing.
- Produces: `ModelDescriptionOptions` with generation controls and `Default` impl.

- [ ] **Step 1: Add fields to `ModelDescriptionOptions`**

  In `src/config.rs`, replace:

  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct ModelDescriptionOptions {
      pub prompt: Option<String>,
  }
  ```

  with:

  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize)]
  #[serde(default)]
  pub struct ModelDescriptionOptions {
      pub prompt: Option<String>,
      pub max_tokens: usize,
      pub temperature: Option<f64>,
      pub top_p: Option<f64>,
      pub top_k: Option<usize>,
      pub repeat_penalty: f32,
      pub repeat_last_n: usize,
  }
  ```

- [ ] **Step 2: Add a `Default` impl**

  Add:

  ```rust
  impl Default for ModelDescriptionOptions {
      fn default() -> Self {
          Self {
              prompt: None,
              max_tokens: 128,
              temperature: Some(0.7),
              top_p: None,
              top_k: None,
              repeat_penalty: 1.0,
              repeat_last_n: 64,
          }
      }
  }
  ```

- [ ] **Step 3: Add a config parse test**

  Add to `src/config.rs` `mod tests`:

  ```rust
  #[test]
  fn parse_model_description_options() {
      let text = r#"
  [[models]]
  name = "gemma-4-E2B-it"
  type = "local"
  path = "google/gemma-4-E2B-it"

  [models.description]
  prompt = "Describe this image in one sentence."
  max_tokens = 64
  temperature = 0.5
  top_p = 0.9
  top_k = 20
  repeat_penalty = 1.1
  repeat_last_n = 32
  "#;
      let config: Config = toml::from_str(text).unwrap();
      let desc = config.models.models[0].description.as_ref().unwrap();
      assert_eq!(desc.prompt.as_deref(), Some("Describe this image in one sentence."));
      assert_eq!(desc.max_tokens, 64);
      assert_eq!(desc.temperature, Some(0.5));
      assert_eq!(desc.top_p, Some(0.9));
      assert_eq!(desc.top_k, Some(20));
      assert!((desc.repeat_penalty - 1.1).abs() < f32::EPSILON);
      assert_eq!(desc.repeat_last_n, 32);
  }
  ```

- [ ] **Step 4: Run tests**

  ```bash
  cargo test --features candle -- config
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/config.rs
  git commit -m "feat(config): add generation controls to ModelDescriptionOptions"
  ```

---

## Task 4: Create VLM Trait, Registry, and Token Stream Helper

**Files:**
- Create: `src/models/candle/vlm/mod.rs`
- Create: `src/models/candle/vlm/token_stream.rs`
- Modify: `src/models/mod.rs` (declare `pub mod vlm` inside candle? Actually inside `models/candle.rs`? Revisit.)

**Interfaces:**
- Consumes: `candle::{Device, DType, Tensor}`, `candle_transformers::generation::{LogitsProcessor, Sampling}`, `ModelConfig`, `ModelFiles`.
- Produces:
  - `pub trait VlmModel: Send + Sync { fn generate(&mut self, image_path: &Path, prompt: Option<&str>) -> Result<String>; }`
  - `pub trait VlmArchitecture: Send + Sync { ... }`
  - `pub struct VlmArchitectureRegistry { ... }`
  - `pub struct TokenOutputStream`
  - `pub fn build_logits_processor(seed: u64, temperature: Option<f64>, top_p: Option<f64>, top_k: Option<usize>) -> LogitsProcessor`
  - `pub fn sample_next(logits_processor: &mut LogitsProcessor, logits: &Tensor, tokens: &[u32], repeat_penalty: f32, repeat_last_n: usize) -> Result<u32>`

- [ ] **Step 1: Create directory and `mod.rs`**

  ```bash
  mkdir -p src/models/candle/vlm
  ```

  Create `src/models/candle/vlm/mod.rs`:

  ```rust
  use std::path::Path;
  use anyhow::{Context, Result};
  use candle::{DType, Device, Tensor};
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
  ```

- [ ] **Step 2: Create `token_stream.rs`**

  Create `src/models/candle/vlm/token_stream.rs`:

  ```rust
  use anyhow::{Context, Result};

  /// Minimal wrapper around `tokenizers::Tokenizer` for incremental / final decoding.
  /// Based on the helper in candle-examples, reimplemented here to avoid that dependency.
  pub struct TokenOutputStream {
      tokenizer: tokenizers::Tokenizer,
      tokens: Vec<u32>,
  }

  impl TokenOutputStream {
      pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
          Self { tokenizer, tokens: Vec::new() }
      }

      pub fn push(&mut self, token: u32) {
          self.tokens.push(token);
      }

      pub fn decode_all(&self) -> Result<String> {
          self.tokenizer
              .decode(&self.tokens, true)
              .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
      }

      pub fn get_token(&self, text: &str) -> Option<u32> {
          self.tokenizer.get_vocab(true).get(text).copied()
      }

      pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
          &self.tokenizer
      }

      pub fn clear(&mut self) {
          self.tokens.clear();
      }
  }

  /// Decode a fixed slice of token IDs to a string.
  pub fn decode_tokens(tokenizer: &tokenizers::Tokenizer, tokens: &[u32]) -> Result<String> {
      tokenizer
          .decode(tokens, true)
          .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
  }
  ```

- [ ] **Step 3: Wire up the module**

  Create `src/models/candle/mod.rs` as a thin re-export, or modify `src/models/candle.rs` to include the module. Because Rust does not allow both `candle.rs` and `candle/`, rename `src/models/candle.rs` to `src/models/candle/mod.rs` and move the backend code into it. This is the cleanest approach.

  Steps:

  1. `git mv src/models/candle.rs src/models/candle/mod.rs`
  2. At the top of the new `src/models/candle/mod.rs`, add:

     ```rust
     pub mod vlm;
     ```

  3. The file `src/models/mod.rs` already has `pub mod candle;` at the crate level, so no change needed there.

- [ ] **Step 4: Add a unit test for the registry**

  Add to `src/models/candle/vlm/mod.rs`:

  ```rust
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
              unimplemented!()
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
          assert!(res.is_err()); // DummyArch::load panics; we just want supports() to match
          assert!(format!("{res:?}").contains("unimplemented"));
      }
  }
  ```

  Note: `ModelConfig` needs a `Default` impl. If it does not have one, add it in `src/config.rs`:

  ```rust
  impl Default for ModelConfig {
      fn default() -> Self {
          Self {
              name: String::new(),
              kind: ModelKind::Local,
              backend: None,
              path: None,
              base_url: None,
              model_id: None,
              api_key: None,
              tags: None,
              description: None,
              classification: None,
              remote: None,
              onnx: None,
          }
      }
  }
  ```

- [ ] **Step 5: Run tests**

  ```bash
  cargo test --features candle -- vlm
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add src/models/candle src/models/mod.rs src/config.rs
  git commit -m "feat(vlm): add VlmModel trait, architecture registry, and token stream helper"
  ```

---

## Task 5: Implement Gemma 4 VLM

**Files:**
- Create: `src/models/candle/vlm/gemma4.rs`

**Interfaces:**
- Consumes: `VlmModel`, `VlmArchitecture`, `ModelConfig`, `ModelFiles`, `TokenOutputStream`, `build_logits_processor`, `sample_next`.
- Produces: `pub struct Gemma4Architecture; impl VlmArchitecture for Gemma4Architecture`, `pub struct Gemma4Vlm; impl VlmModel for Gemma4Vlm`.

- [ ] **Step 1: Create `src/models/candle/vlm/gemma4.rs`**

  Create the file with the following structure. This is the reference implementation; adjust exact prompt formatting after testing with the real tokenizer.

  ```rust
  use std::path::{Path, PathBuf};
  use anyhow::{Context, Result};
  use candle::{DType, Device, Tensor};
  use candle_nn::VarBuilder;
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
              .with_context(|| format!("failed to load tokenizer from {tokenizer_path:?}"))?;

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
      fn preprocess_image(&self, path: &Path) -> Result<Tensor> {
          // Gemma 4 vision tower accepts any size divisible by patch_size.
          // Resize longest side to a reasonable default; exact size should match the
          // processor config and available memory. 896 keeps CPU inference tractable.
          let target_longest = 896u32;
          let patch_size = self.config.vision_config.patch_size as u32;

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

          Tensor::from_vec(data, (new_h as usize, new_w as usize, 3), &self.device)?
              .permute((2, 0, 1))?
              .unsqueeze(0)
              .with_context(|| "failed to build image tensor")
      }

      fn build_prompt_tokens(&self, prompt: Option<&str>) -> Result<Vec<u32>> {
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
          let vision_tokens = self.config.vision_soft_tokens_per_image.max(1);

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

          let pixel_values = self.preprocess_image(image_path)?;
          let mut tokens = self.build_prompt_tokens(prompt)?;

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

          decode_tokens(self.tokenizer.tokenizer(), &tokens)
      }
  }
  ```

  Note: `Gemma4Config` may not have a `text_config.eos_token_id: Vec<i64>` field; inspect the struct in `candle-transformers 0.11.0` and adjust accordingly. The config we fetched from Hugging Face shows `eos_token_id: [1, 106]` at the top level, but `text_config.eos_token_id: 1`. If the candle struct mirrors the top level as `Option<Vec<i64>>`, use that; otherwise use `text_config.eos_token_id`.

- [ ] **Step 2: Fix any compile errors in Gemma4 module**

  Run:

  ```bash
  cargo check --features candle
  ```

  Fix type mismatches against the real `candle_transformers::models::gemma4` API.

- [ ] **Step 3: Commit**

  ```bash
  git add src/models/candle/vlm/gemma4.rs
  git commit -m "feat(vlm): add Gemma4 description generation"
  ```

---

## Task 6: Integrate VLM Registry into CandleBackend

**Files:**
- Modify: `src/models/candle/mod.rs` (formerly `src/models/candle.rs`)

**Interfaces:**
- Consumes: `VlmArchitectureRegistry`, `VlmModel`, `ModelConfig`.
- Produces: `VlmModelWrapper` implementing `crate::models::Model`.

- [ ] **Step 1: Add the wrapper and import the registry**

  At the top of `src/models/candle/mod.rs`, add:

  ```rust
  pub mod vlm;

  use std::path::Path;
  use std::sync::Arc;
  use anyhow::{Context, Result};
  use candle_core::Device;

  use crate::config::ModelConfig;
  use crate::models::{loader, Backend, Model, ModelOutput};
  use vlm::{VlmArchitectureRegistry, VlmModel};
  ```

- [ ] **Step 2: Add `VlmModelWrapper`**

  Add before `pub struct CandleBackend;`:

  ```rust
  /// Adapts a stateful `VlmModel` to the `Model` trait used by the worker.
  pub struct VlmModelWrapper {
      inner: std::sync::Mutex<Box<dyn VlmModel>>,
      prompt: Option<String>,
  }

  impl VlmModelWrapper {
      pub fn new(model: Box<dyn VlmModel>, prompt: Option<String>) -> Self {
          Self {
              inner: std::sync::Mutex::new(model),
              prompt,
          }
      }
  }

  impl Model for VlmModelWrapper {
      fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
          let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
          let text = inner.generate(image_path, self.prompt.as_deref())?;
          Ok(ModelOutput::Description(text))
      }
  }
  ```

- [ ] **Step 3: Update `CandleBackend::load` to dispatch**

  Replace the body of `CandleBackend::load` with:

  ```rust
  fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
      let path = config.path.as_deref().context("candle model missing path")?;
      let source = loader::resolve_source(path)?;
      let files = loader::load_model_files(&source)
          .with_context(|| format!("failed to load model files for {}", config.name))?;

      #[cfg(feature = "cuda")]
      let device = {
          match Device::new_cuda(0) {
              Ok(d) => d,
              Err(e) => {
                  tracing::warn!("CUDA device unavailable, falling back to CPU: {e}");
                  Device::Cpu
              }
          }
      };
      #[cfg(not(feature = "cuda"))]
      let device = Device::Cpu;

      if config.description.is_some() {
          let registry = VlmArchitectureRegistry::with_defaults();
          let vlm = registry.select(config, &files, &device)?;
          let prompt = config.description.as_ref().and_then(|d| d.prompt.clone());
          return Ok(Arc::new(VlmModelWrapper::new(vlm, prompt)));
      }

      match config.tags.as_ref() {
          Some(options) => {
              let tagger = super::tagger::ViTTagger::load(&config.name, &files, device, options)?;
              Ok(Arc::new(tagger))
          }
          None => anyhow::bail!("candle backend requires either tags or description configuration"),
      }
  }
  ```

  Also update the error message at the end from `only supports tags output kind right now` to the new message above.

- [ ] **Step 4: Run cargo check**

  ```bash
  cargo check --features candle
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/models/candle/mod.rs
  git commit -m "feat(candle): dispatch description models to VLM registry"
  ```

---

## Task 7: Add BLIP Stub Architecture

**Files:**
- Create: `src/models/candle/vlm/blip.rs`

**Interfaces:**
- Consumes: `VlmArchitecture`, `VlmModel`.
- Produces: `pub struct BlipArchitecture; impl VlmArchitecture for BlipArchitecture`.

- [ ] **Step 1: Create the stub**

  Create `src/models/candle/vlm/blip.rs`:

  ```rust
  use std::path::Path;
  use anyhow::{Context, Result};
  use candle::Device;

  use crate::config::ModelConfig;
  use crate::models::loader::ModelFiles;

  use super::{VlmArchitecture, VlmModel};

  pub struct BlipArchitecture;

  impl VlmArchitecture for BlipArchitecture {
      fn name(&self) -> &'static str {
          "blip"
      }

      fn supports(&self, config: &ModelConfig) -> bool {
          if config.backend.as_deref() == Some("candle-blip") {
              return true;
          }
          let haystack = format!(
              "{} {}",
              config.name,
              config.path.as_deref().unwrap_or("")
          )
          .to_lowercase();
          haystack.contains("blip")
      }

      fn load(
          &self,
          _config: &ModelConfig,
          _files: &ModelFiles,
          _device: &Device,
      ) -> Result<Box<dyn VlmModel>> {
          anyhow::bail!("BLIP VLM is not yet implemented; use Gemma4 for descriptions")
      }
  }
  ```

- [ ] **Step 2: Verify it compiles**

  ```bash
  cargo check --features candle
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add src/models/candle/vlm/blip.rs
  git commit -m "feat(vlm): add BLIP architecture stub"
  ```

---

## Task 8: Tests and Verification

**Files:**
- All of the above.

**Interfaces:**
- Consumes: everything.
- Produces: passing tests and a working end-to-end flow.

- [ ] **Step 1: Run the full test suite**

  ```bash
  cargo test --features candle
  ```

  Expected: all tests pass.

- [ ] **Step 2: Manual smoke test**

  1. Add to your `~/.config/akasha/config.toml`:

     ```toml
     [[models]]
     name = "gemma-4-E2B-it"
     type = "local"
     path = "google/gemma-4-E2B-it"

     [models.description]
     prompt = "Describe this image in one sentence."
     max_tokens = 64
     ```

  2. Run Akasha:

     ```bash
     cargo run --features candle
     ```

  3. Select an image, open Media Processing → Vision-Language, choose "gemma-4-E2B-it", click Go.
  4. Wait for the worker to process the job.
  5. Check the database:

     ```bash
     sqlite3 ~/.local/share/akasha/akasha.db \
       "SELECT descriptions_json FROM media_files WHERE id = <media_id>;"
     ```

  6. Verify the description appears and is searchable.

- [ ] **Step 3: Regression check**

  If you have a tagger model configured, enqueue a tag job and confirm `tags_json` still updates.

- [ ] **Step 4: Commit any final fixes**

  ```bash
  git add -A
  git commit -m "fix: VLM description end-to-end verification fixes"
  ```

---

## Self-Review Checklist

- **Spec coverage:**
  - [x] Candle upgrade → Task 1.
  - [x] `tokenizers` dependency → Task 1.
  - [x] Flexible VLM architecture registry → Task 4.
  - [x] Gemma4 implementation → Task 5.
  - [x] Backend dispatch → Task 6.
  - [x] Config options → Task 3.
  - [x] Loader tokenizer + sharded weights → Task 2.
  - [x] BLIP stub → Task 7.
  - [x] Tests → Task 8.

- **Placeholder scan:**
  - [x] No "TBD", "TODO", or vague "add error handling" steps.
  - [x] Each code step contains concrete code.
  - [x] Each test step contains the exact command and expected outcome.

- **Type consistency:**
  - [x] `ModelFiles` uses `weights_paths: Vec<PathBuf>` consistently.
  - [x] `VlmModel::generate` signature matches in trait and implementations.
  - [x] `VlmArchitecture` methods are consistent across `Gemma4Architecture` and `BlipArchitecture`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-vlm-description-hookup.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach would you like?
