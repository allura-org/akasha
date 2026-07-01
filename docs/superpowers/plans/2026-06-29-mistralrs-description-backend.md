# mistral.rs Description Backend

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `mistralrs`-based backend that runs `visionlanguage` description jobs using `google/gemma-4-E2B-it` (safetensors), bypassing the Candle MatFormer/PLE limitations that block the Gemma 4 E2B checkpoint.

**Architecture:** Keep the existing `BackendRegistry`/`Model`/`SearchWorker` plumbing unchanged. Add a new optional `mistralrs` feature and `src/models/mistralrs.rs`. The backend implements `Backend::load` by constructing a `mistralrs::MultimodalModelBuilder` (Gemma 4 auto-detected, ISQ Q4K by default), and `Model::infer` by loading the local image, sending a multimodal chat request, and returning `ModelOutput::Description`.

**Tech Stack:** Rust 2024, `mistralrs` 0.8.1, `image` 0.25, `tokio`, SQLite/sqlx.

## Global Constraints

- `mistralrs` must remain **off by default**; do not add it to the `default` feature set.
- The `cuda` feature must pass through to `mistralrs?/cuda`.
- Do not change the `Backend`, `Model`, or `SearchWorker` interfaces.
- `Model::infer` is synchronous and runs inside `tokio::task::spawn_blocking`. The `mistralrs` API is async, so bridge it with a dedicated tokio runtime or `Handle::current().block_on`.
- Use `config.model_id` (HF slug) first; fall back to `config.path` as a local directory.
- Default description prompt when `ModelDescriptionOptions::prompt` is `None`: `"Describe this image in detail."`.
- Respect `ModelDescriptionOptions::max_tokens`, `temperature`, `top_p`, `top_k` when building the request (map to the `RequestBuilder` / sampler knobs exposed by `mistralrs`).
- Keep image preprocessing minimal: open the file with `image::open`, pass the `DynamicImage` to `MultimodalMessages::add_image_message`.
- All existing `cargo test` matrices must continue to pass.
- Code follows existing project style: `anyhow::Result`, `tracing`, minimal scope.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Cargo.toml` | Add optional `mistralrs` dependency and `mistralrs`/`cuda` feature flags. |
| `src/models/mod.rs` | Conditionally declare `mod mistralrs` and register `MistralRsBackend` in `BackendRegistry::with_remote`. |
| `src/models/mistralrs.rs` | `MistralRsBackend` and `MistralRsModel` implementing `Backend` + `Model` for description jobs. |
| `config.example.toml` | Add a commented-out Gemma 4 description example using `backend = "mistralrs"`. |

---

## Task 1: Finalize Cargo.toml Feature Flags

**Files:**
- Modify: `Cargo.toml`
- Test: `cargo check --features mistralrs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Cargo.toml` with `mistralrs = ["dep:mistralrs"]` and `cuda = ["candle", "candle-core/cuda", "mistralrs?/cuda"]`.

- [ ] **Step 1: Ensure dependency is present**

  `Cargo.toml` should already contain:

  ```toml
  [features]
  mistralrs = ["dep:mistralrs"]
  cuda = ["candle", "candle-core/cuda", "mistralrs?/cuda"]

  [dependencies]
  mistralrs = { version = "0.8.1", optional = true }
  ```

  Verify the line is correct and the `default` feature set does **not** include `mistralrs`.

- [ ] **Step 2: Check that the feature resolves**

  Run:

  ```bash
  cargo check --features mistralrs
  ```

  Expected: dependency resolves and `mistralrs` types are importable. It is okay if no backend code uses them yet.

- [ ] **Step 3: Report**

  Note whether the dependency downloaded successfully and any build-time caveats (e.g., feature flags, C compiler requirements).

---

## Task 2: Implement the mistral.rs Backend

**Files:**
- Create: `src/models/mistralrs.rs`
- Modify: `src/models/mod.rs`
- Test: `cargo check --features mistralrs`, then `cargo test --features mistralrs`

**Interfaces:**
- Consumes: `ModelConfig`, `ModelDescriptionOptions`, `ModelOutput`, `Backend`, `Model`.
- Produces: `MistralRsBackend` (id `"mistralrs"`) and `MistralRsModel`.

- [ ] **Step 1: Add conditional module declaration**

  In `src/models/mod.rs`:

  ```rust
  #[cfg(feature = "mistralrs")]
  pub mod mistralrs;
  ```

- [ ] **Step 2: Implement `MistralRsBackend`**

  Create `src/models/mistralrs.rs`:

  ```rust
  use std::path::Path;
  use std::sync::Arc;
  use anyhow::{Context, Result};
  use image::DynamicImage;
  use mistralrs::{
      IsqType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
  };
  use tokio::runtime::Runtime;

  use crate::config::{ModelConfig, ModelDescriptionOptions};
  use crate::models::{Backend, Model, ModelOutput};

  pub struct MistralRsBackend;

  impl Backend for MistralRsBackend {
      fn id(&self) -> &'static str { "mistralrs" }
      fn is_available(&self) -> bool { true }
      fn supports(&self, config: &ModelConfig) -> bool {
          config.description.is_some()
      }
      fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
          let model_id = config
              .model_id
              .clone()
              .or_else(|| config.path.clone())
              .context("mistralrs backend requires a model_id or path")?;
          let opts = config.description.clone().unwrap_or_default();
          let runtime = Runtime::new().context("failed to create tokio runtime for mistralrs")?;
          let model = runtime.block_on(async {
              MultimodalModelBuilder::new(&model_id)
                  .with_isq(IsqType::Q4K)
                  .build()
                  .await
                  .context("failed to build mistralrs multimodal model")
          })?;
          Ok(Arc::new(MistralRsModel {
              model,
              opts,
              runtime,
          }))
      }
  }
  ```

  Adjust the exact builder method names (`with_isq`, `build`, etc.) and `IsqType` variant to match the actual `mistralrs` 0.8.1 API.

- [ ] **Step 3: Implement `MistralRsModel`**

  ```rust
  struct MistralRsModel {
      model: /* type returned by MultimodalModelBuilder, e.g., Arc<Mutex<Pipeline>> or a concrete Model */,
      opts: ModelDescriptionOptions,
      runtime: Runtime,
  }

  impl Model for MistralRsModel {
      fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
          let image = image::open(image_path)
              .with_context(|| format!("failed to open image {image_path:?}"))?;
          let prompt = self.opts.prompt.clone().unwrap_or_else(|| "Describe this image in detail.".into());

          self.runtime.block_on(async {
              let messages = MultimodalMessages::new()
                  .add_image_message(TextMessageRole::User, &prompt, vec![image]);
              let request = RequestBuilder::from(messages)
                  .set_max_tokens(self.opts.max_tokens);
              // apply temperature/top_p/top_k if present
              let response = self.model.send_chat_request(request).await
                  .context("mistralrs inference failed")?;
              let text = response.choices[0].message.content.clone()
                  .unwrap_or_default();
              Ok(ModelOutput::Description(text))
          })
      }
  }
  ```

  The exact request/response types and method names must be aligned with the real `mistralrs` crate. Use `cargo check` errors to guide corrections.

- [ ] **Step 4: Register backend**

  In `BackendRegistry::with_remote`:

  ```rust
  #[cfg(feature = "mistralrs")]
  reg.register(mistralrs::MistralRsBackend);
  ```

  Place it before `candle` so description jobs prefer it when no explicit `backend` is set.

- [ ] **Step 5: Compile and test**

  Run:

  ```bash
  cargo check --features mistralrs
  cargo test --features mistralrs
  ```

  Expected: no errors; existing tests pass. The backend will not be exercised by unit tests because it requires a multi-gigabyte model.

---

## Task 3: Update Config Example and Notes

**Files:**
- Modify: `config.example.toml`
- Modify: `.kimi/SESSION_NOTES.md` (append a note about the new backend)

- [ ] **Step 1: Add commented example**

  In `config.example.toml`, add under the existing model examples:

  ```toml
  # [[model]]
  # name = "gemma4-description"
  # type = "local"
  # backend = "mistralrs"
  # model_id = "google/gemma-4-E2B-it"
  # [model.description]
  # prompt = "Describe this image in detail."
  # max_tokens = 128
  # temperature = 0.7
  ```

- [ ] **Step 2: Document the pivot**

  Append to `.kimi/SESSION_NOTES.md` a brief note that the Candle Gemma 4 E2B path was abandoned due to MatFormer/PLE incompatibility in `candle-transformers` 0.11.0, and that a `mistralrs` backend was added as an optional feature for description jobs.

---

## Task 4: Verification

- [ ] **Step 1: Compile matrices**

  ```bash
  cargo test
  cargo test --features candle
  cargo test --features remote
  cargo test --features onnx
  cargo test --features mistralrs
  cargo test --no-default-features
  ```

  Expected: all pass (existing tests unchanged).

- [ ] **Step 2: Manual smoke test (optional but recommended)**

  If `google/gemma-4-E2B-it` is cached locally, configure a model with `backend = "mistralrs"` and enqueue a `visionlanguage` job via the Media Processing UI. Verify the job completes and writes a description to `searchable_values`.

  This is manual and should not block the automated verification above.

---

## Completion Criteria

- `cargo check --features mistralrs` succeeds.
- `cargo test` and all feature matrices pass.
- `BackendRegistry` can select a `"mistralrs"` backend for description configs.
- Config example documents the new backend.
- Code is gated behind the `mistralrs` feature and does not affect default builds.
