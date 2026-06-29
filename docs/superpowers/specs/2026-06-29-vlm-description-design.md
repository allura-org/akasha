# VLM Description Job Hookup — Design Spec

**Date:** 2026-06-29  
**Status:** Approved for implementation  
**Owner:** Kimi Code / agent session  

## Goal

Make the existing "Vision-Language" / description AI job kind actually run end-to-end in Akasha. Today the UI and worker already enqueue and store description jobs, but `CandleBackend` bails with:

```
candle backend only supports tags output kind right now
```

This spec adds a flexible, architecture-agnostic VLM path inside the Candle backend, with **Gemma 4 E2B** as the first supported model.

## Background

- `src/searchables/worker.rs` polls `job_queue`, loads a model through the `BackendRegistry`, runs inference, and writes the result.
- `src/models/mod.rs` defines the backend-agnostic `Model` / `Backend` traits and `ModelOutput` enum (which already includes `Description(String)`).
- `src/models/candle.rs` currently only knows how to load the `ViTTagger`.
- `src/ui/media_processing.rs` already exposes a "Vision-Language" tab that enqueues `visionlanguage` jobs when a model has `[models.description]` configured.
- `src/db/searchable.rs` already has `update_description_json` and a `description` searchable kind.

So the missing piece is **Candle-side model loading and inference for VLMs**.

## Decisions

1. **Upgrade `candle-*` from `0.8.x` to `0.11.0`.** Gemma 4 support does not exist in `0.8.4`; `0.11.0` ships `candle_transformers::models::gemma4`.
2. **Add a `tokenizers` dependency.** VLMs need prompt encoding / decoding. `candle-transformers 0.11.0` itself depends on `tokenizers 0.22`.
3. **Do not depend on `candle-examples`.** It is an examples crate. We will implement the small helpers we need (token streaming, sharded weight loading) ourselves.
4. **Use a flexible VLM architecture registry**, mirroring the existing `BackendRegistry` pattern, so adding BLIP / LLaVA / Moondream later does not require rewriting the backend.
5. **No worker changes.** The worker already understands `ModelOutput::Description`.

## Architecture

```text
BackendRegistry
    └── CandleBackend
            ├── Taggers (existing ViTTagger)
            │
            └── VlmArchitectureRegistry
                    ├── Gemma4Vlm  (first)
                    ├── BlipVlm    (stub / future)
                    └── LlavaVlm   (stub / future)
```

### New files

| File | Purpose |
|------|---------|
| `src/models/candle/vlm/mod.rs` | `VlmModel` trait, `VlmArchitectureRegistry`, shared text-generation utilities |
| `src/models/candle/vlm/gemma4.rs` | Gemma 4 multimodal loading, image preprocessing, prompt formatting, and generation |
| `src/models/candle/vlm/blip.rs` | Optional placeholder showing how another architecture registers itself |
| `src/models/candle/vlm/token_stream.rs` | Small `TokenOutputStream` helper for streaming / final decoding without pulling in `candle-examples` |

### Modified files

| File | Change |
|------|--------|
| `Cargo.toml` | Bump `candle-core`, `candle-nn`, `candle-transformers`, `hf-hub` to `0.11.0`; add `tokenizers = "0.22"`; fix any API fallout in existing tagger |
| `src/models/candle.rs` | In `load`, dispatch to VLM registry when `config.description.is_some()`, otherwise keep tagger path |
| `src/models/loader.rs` | Add `tokenizer_path` to `ModelFiles`; support `model.safetensors.index.json` sharded weights; support local tokenizer/config fallback |
| `src/config.rs` | Extend `ModelDescriptionOptions` with generation controls: `max_tokens`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `repeat_last_n` |

## Component Details

### `VlmModel` trait

```rust
pub trait VlmModel: Send + Sync {
    /// Generate a description for the image at `image_path`.
    /// `prompt` is the user-provided instruction (e.g. "Describe this image in one sentence.").
    fn generate(&mut self, image_path: &Path, prompt: Option<&str>) -> Result<String>;
}
```

`VlmModel` is intentionally stateful (`&mut self`) because generation needs to manage KV caches inside the loaded model.

### `VlmArchitectureRegistry`

```rust
pub trait VlmArchitecture: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, config: &ModelConfig) -> bool;
    fn load(&self, config: &ModelConfig, files: &ModelFiles, device: &Device) -> Result<Box<dyn VlmModel>>;
}

pub struct VlmArchitectureRegistry {
    architectures: Vec<Box<dyn VlmArchitecture>>,
}

impl VlmArchitectureRegistry {
    pub fn with_defaults() -> Self { ... }
    pub fn select(&self, config: &ModelConfig, files: &ModelFiles, device: &Device) -> Result<Box<dyn VlmModel>> { ... }
}
```

Selection rules (tentative):

- If `config.backend == "candle-gemma4"` or the model id/path contains `gemma-4`, use Gemma4.
- Otherwise default to Gemma4 when `description` is present.
- As more architectures land, selection heuristics will be added.

### Wrapper for the existing `Model` trait

The worker calls `Model::infer(&self, image_path: &Path) -> Result<ModelOutput>`. That trait takes `&self`, but VLMs need `&mut self` for KV-cache mutation. We will wrap a `Box<dyn VlmModel>` in a struct that:

1. Captures the prompt from `ModelDescriptionOptions` at load time.
2. Uses interior mutability (`Mutex` or `RefCell`) around the VLM so it can implement `Model::infer` with `&self`.

```rust
pub struct VlmModelWrapper {
    inner: std::sync::Mutex<Box<dyn VlmModel>>,
    prompt: Option<String>,
}

impl Model for VlmModelWrapper {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let text = inner.generate(image_path, self.prompt.as_deref())?;
        Ok(ModelOutput::Description(text))
    }
}
```

Note: generation happens inside the worker's `spawn_blocking` task. If a panic occurs, the mutex is poisoned; `unwrap_or_else(|e| e.into_inner())` ignores the poison and continues on the next job rather than crashing the worker.

### Gemma 4 implementation

**Model loading**

- Parse `config.json` as `candle_transformers::models::gemma4::config::Gemma4Config`.
- Load tokenizer from `tokenizer.json`.
- Load weights:
  - Try `model.safetensors.index.json` first (Gemma 4 E2B is sharded), fall back to single `model.safetensors`.
- Build `candle_transformers::models::gemma4::Model`.

**Image preprocessing**

Gemma 4 uses a SigLIP-style vision tower. The reference preprocessor (to be validated against the Gemma 4 processor config) is:

- Resize image to the vision config's expected input size with bicubic/Lanczos resampling.
- Convert to RGB.
- Normalize with ImageNet-style or model-specific mean/std.
- Produce a tensor of shape `(1, C, H, W)`.

Because `candle-transformers 0.11.0`'s Gemma4 example is text-only, we will implement the multimodal generation path ourselves using `Model::forward_multimodal`.

**Prompt formatting**

- If the tokenizer has a chat template, apply it with a single user message containing the image and the user's prompt.
- If not, use a simple default:
  - Place the model's `image_token_id` once.
  - Append the user prompt (or a default like "Describe this image.").

**Generation loop**

1. Encode the prompt to token IDs.
2. On the first forward pass, call `model.forward_multimodal(input_ids, Some(&[pixel_values]), None, None, 0)`.
3. Sample the next token with `LogitsProcessor` (respecting `temperature`, `top_p`, `top_k`).
4. For subsequent steps, call `forward_multimodal` with the updated token list and an increasing `seqlen_offset`.
5. Stop on EOS (`</s>` or tokenizer EOS id) or `max_tokens`.
6. Decode the generated tokens and return the string.

**KV cache**

- Call `model.clear_kv_cache()` before each new image generation.

## Config Changes

```toml
[[models]]
name = "gemma-4-E2B-it"
type = "local"
path = "google/gemma-4-E2B-it"
backend = "candle"

[models.description]
prompt = "Describe this image in one sentence."
max_tokens = 128
temperature = 0.7
top_p = 0.9
top_k = 40
repeat_penalty = 1.1
repeat_last_n = 64
```

All new fields have sensible defaults:

- `max_tokens`: 128
- `temperature`: 0.7
- `top_p`: None
- `top_k`: None
- `repeat_penalty`: 1.0 (no penalty)
- `repeat_last_n`: 64

`ModelDescriptionOptions::prompt` already exists; the others are added.

## Data Flow

1. User opens Media Processing → Vision-Language, selects the Gemma 4 model, clicks Go.
2. `app.rs::enqueue_media_processing_jobs` enqueues `visionlanguage` jobs with `searchable_config_id` pointing at the model's `description` config.
3. `SearchWorker::tick` claims the job.
4. `SearchWorker::process_one` reconstructs `ModelConfig` from the searchable config and calls `BackendRegistry::select_with_error`.
5. `CandleBackend::load` sees `description.is_some()`, asks `VlmArchitectureRegistry` for an implementation.
6. `Gemma4Vlm` loads tokenizer, config, and weights.
7. `Model::infer` runs the generation loop and returns `ModelOutput::Description(text)`.
8. Worker calls `update_description_json`, then `complete_job`.

## Error Handling

- **Model load failure:** Fail the job with the underlying error. Do not crash the worker.
- **Tokenizer missing:** Clear error: `"tokenizer.json not found for <model>"`.
- **Image preprocess failure:** Fail the job; preserve metadata.
- **Empty generation:** Store an empty description in `descriptions_json` and FTS. The job is still marked complete.
- **Generation panic:** The existing `spawn_blocking` + `?` in the worker catches panics and converts them to job failures.

## Testing Plan

1. **Candle upgrade compatibility:** Run `cargo test` after the version bump and fix any `ViTTagger` compilation errors caused by `candle-transformers` API changes.
2. **Unit tests for new code:**
   - `VlmArchitectureRegistry` selects Gemma4 for a Gemma-4-looking config.
   - `TokenOutputStream` correctly decodes a fixed token sequence.
   - Sharded weight loader parses `model.safetensors.index.json` and returns unique file paths.
3. **Manual smoke test:**
   - Configure `google/gemma-4-E2B-it`.
   - Enqueue a description job on a test image.
   - Verify `descriptions_json` and `searchable_text_fts` are populated.
   - Verify the description is searchable from the browser.
4. **Regression:** Existing tagger jobs still produce tags after the upgrade.

## Files to Create / Modify

- `Cargo.toml` — dependency versions
- `src/config.rs` — extend `ModelDescriptionOptions`
- `src/models/loader.rs` — tokenizer path + sharded weights
- `src/models/candle.rs` — dispatch to VLM registry
- `src/models/candle/vlm/mod.rs` — new
- `src/models/candle/vlm/token_stream.rs` — new
- `src/models/candle/vlm/gemma4.rs` — new
- `src/models/candle/vlm/blip.rs` — new (stub demonstrating second architecture)

## Open Questions

1. Does `candle-transformers 0.11.0`'s `Gemma4Config` expose the exact vision input size / normalization constants, or do we need to read them from the processor config?
2. Should we expose `system_prompt` and a chat-template override in `ModelDescriptionOptions` now, or defer until more VLMs land?
3. Should the VLM wrapper support batch inference in the future? (Out of scope for this milestone; keep batch_size = 1 per job.)
