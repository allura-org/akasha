# Backend-Agnostic Model Plugin Interface

## Status

Design for the `candle-integration` feature branch.

## Goal

Decouple Akasha's inference worker from candle-specific details so local and remote models with different architectures can be added by implementing a small trait and registering it. The first built-in backends remain **candle** (local ViT-style models) and **remote** (OpenAI-compatible HTTP endpoints). The design leaves room for future backends such as ONNX Runtime or external executable wrappers without changing the core worker.

## Non-Goals

- Dynamic plugin loading (`.so` plugins) is not implemented now, but the trait boundary must not prevent it later.
- Python-based backends are intentionally out of scope. If a user's model is unsupported, Akasha should advise running an inference server and connecting via the remote backend.
- The worker does not try to share preprocessing code between backends. Each backend owns its full pipeline.

## Core Abstractions

```rust
/// A backend knows how to decide whether it can run a model and how to load it.
pub trait Backend: Send + Sync {
    /// Stable identifier used in config, e.g. "candle", "remote".
    fn id(&self) -> &'static str;

    /// Whether this backend is available at runtime (feature flags, runtime checks).
    fn is_available(&self) -> bool;

    /// Whether this backend can run the given model config.
    fn supports(&self, config: &ModelConfig) -> bool;

    /// Load a model instance. The returned `Model` is cached by the worker.
    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>>;
}

/// A loaded model that can run inference on a single image.
pub trait Model: Send + Sync {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput>;
}
```

`ModelOutput` (already defined) remains the shared result type:

```rust
pub enum ModelOutput {
    Tags(HashMap<String, f32>),
    Description(String),
    Classification { label: String, score: f32 },
    Vector(Vec<f32>),
}
```

Each backend is responsible for decoding the image, preprocessing, running inference, and converting raw outputs into `ModelOutput`.

## Backend Registry

```rust
pub struct BackendRegistry {
    backends: Vec<Arc<dyn Backend>>,
}

impl BackendRegistry {
    pub fn empty() -> Self { ... }

    /// Register a backend. Order matters for automatic selection.
    pub fn register<B: Backend + 'static>(&mut self, backend: B) { ... }

    /// Built-in registry: candle (if feature enabled) + remote.
    pub fn default() -> Self { ... }

    /// Select a backend for a config.
    /// - If `config.backend` is set, match by exact `id`.
    /// - Otherwise, pick the first available backend whose `supports()` returns true.
    pub fn select(&self, config: &ModelConfig) -> Option<Arc<dyn Backend>>;
}
```

Selection rules:

1. If `config.backend` is `Some("candle")`, use the candle backend if available.
2. If `config.backend` is `Some("remote")`, use the remote backend.
3. If `config.backend` is `None`:
   - If `base_url` is set, use remote.
   - Otherwise, if candle is available and `path` resolves, use candle.
   - Otherwise, no backend supports the config.

If a requested backend is unavailable, the worker fails the job with a clear message.

## Config Changes

Add an optional `backend` field to `ModelConfig`:

```rust
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ModelKind,        // Local | Remote (kept for semantics/UI)
    pub backend: Option<String>, // "candle", "remote", future values
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub model_id: Option<String>,
    pub api_key: Option<String>,
    pub tags: Option<ModelTagsOptions>,
    pub description: Option<ModelDescriptionOptions>,
    pub classification: Option<ModelClassificationOptions>,
}
```

Example TOML:

```toml
[[models]]
name = "vit-base-patch16-224"
type = "local"
backend = "candle"
path = "google/vit-base-patch16-224"

[models.tags]
threshold = 0.35

[[models]]
name = "gpt-4o-mini"
type = "remote"
backend = "remote"
base_url = "https://api.openai.com/v1"
model_id = "gpt-4o-mini"
api_key = "..."

[models.description]
prompt = "Describe this image in one sentence."
```

`backend` is optional. Existing configs continue to work because `select()` falls back to `base_url` / `path` heuristics.

## Worker Refactor

`SearchWorker` stops owning a `CandleWorker` and instead owns:

```rust
pub struct SearchWorker {
    pool: Arc<SqlitePool>,
    batch_size: i64,
    registry: BackendRegistry,
    resident: Option<ResidentModel>,
}

struct ResidentModel {
    config_id: i64,
    backend_id: String,
    model: Arc<dyn Model>,
}
```

Per-tick flow:

1. Claim pending AI jobs (`tagger`, `classifier`, `visionlanguage`).
2. Partition by `searchable_config_id` and cluster so the resident model stays loaded.
3. For each job:
   - Reconstruct `ModelConfig` from `searchable_configs.options`.
   - Select a backend from the registry.
   - If the resident model does not match `(config_id, backend_id)`, load a new one via `backend.load()`.
   - Run `model.infer(image_path)` on `tokio::task::spawn_blocking`.
   - Write the result to the DB using existing `update_tags_json` / `update_description_json` helpers.
   - Mark the job complete or failed.

Reconstructing `ModelConfig` from the DB:

`sync_model_configs` already stores the base model fields (`path`, `base_url`, `model_id`) and per-output options (`threshold`, `prompt`) in `searchable_configs.options`. A helper turns this JSON back into a `ModelConfig`:

```rust
fn model_config_from_searchable_config(
    cfg: &SearchableConfig,
) -> Result<ModelConfig> { ... }
```

This avoids a schema migration; existing rows remain valid as long as the helper understands their options shape.

## Built-In Backends

### Candle Backend

- Feature-gated by `candle` (enabled by default alongside `remote`).
- Supports local and Hugging Face slug `path` values.
- Currently implements the `tags` output kind via `ViTTagger`.
- Description/classification/vector kinds return a "not yet implemented" error for now.

### Remote Backend

- Always compiled in (default feature).
- Requires `base_url` and `model_id`.
- Uses `reqwest` for HTTP calls.
- Per output kind:
  - **Description**: OpenAI-compatible chat completions with an image message. The prompt comes from `ModelDescriptionOptions::prompt` or a default.
  - **Tags / Classification**: POST the image to a configurable endpoint. The response format is documented and must be a JSON object mapping labels to scores (for tags) or `{label, score}` (for classification).
- API key is sent in the `Authorization` header.
- Endpoint paths are configurable per model and fall back to global defaults:
  - `ModelRemoteOptions::chat_endpoint` for description (default `/chat/completions`).
  - `ModelRemoteOptions::tag_endpoint` for tags (default `/tags`).
  - `ModelRemoteOptions::classify_endpoint` for classification (default `/classify`).

A new `reqwest` dependency is added, default-enabled as part of the `remote` feature.

## Error Handling

- **No supporting backend**: job fails with "No backend available for model '{name}'. Add `backend = \"...\"` or check feature flags."
- **Requested backend unavailable**: job fails with "Backend '{id}' is not compiled in. Rebuild with --features ..."
- **Backend load failure**: job fails; error stored in `job_queue.error`.
- **Inference failure**: job fails.
- Unknown `job_kind` continues to be rejected by `SearchWorker`.

## Testing

1. **Registry selection tests**: `BackendRegistry::select` with various configs.
2. **Mock backend/model**: implement a test-only backend that returns fixed `ModelOutput`. Use it in a worker integration test to verify the job-to-database flow without downloading real models.
3. **Candle tests**: keep existing `ViTTagger` smoke/baseline tests.
4. **Remote tests**: add a test with a `wiremock` or `tokio::net` stub server verifying request shape and response parsing.

## Migration / Compatibility

- No new SQL migrations are required.
- Existing `searchable_configs` rows have enough fields in `options` to reconstruct `ModelConfig`.
- Adding `backend` to `ModelConfig` is backward compatible for config loading (`Option<String>` defaults to `None`).

## Handling Custom Model Architectures

The `Backend`/`Model` boundary is at the *inference engine* level, not the *model architecture* level. A single backend can host many specialized model implementations and pick the right one at load time.

For example, the candle backend could internally dispatch like this:

```rust
fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
    if is_jtp_model(config) {
        Ok(Arc::new(JtpTagger::load(config)?))
    } else if is_wd_vit_v3(config) {
        Ok(Arc::new(WdViT3Tagger::load(config)?))
    } else {
        Ok(Arc::new(ViTTagger::load(config)?))
    }
}
```

Each concrete type implements `Model::infer` and returns `ModelOutput`. Akasha never sees the architecture-specific details.

For a model that cannot run in any compiled backend (e.g. a custom PyTorch checkpoint), the supported path is to run it behind an inference server and connect via the remote backend.

## Resident Model Caching

The worker keeps **one resident model** at a time. When the next job needs a different `(searchable_config_id, backend_id)`, the current model is dropped and the new one is loaded.

This matches the existing behavior and avoids premature optimization. If real-world usage shows users frequently alternating between two large models, we can replace the single resident with a small LRU cache later without changing the `Backend`/`Model` interface.

## Future Backends (Not Implemented Now)

- **ONNX Runtime (`ort`)**: implement `Backend` for `.onnx` files. Pre/postprocessing lives in the backend.
- **External executable**: implement `Backend` that spawns a subprocess and speaks JSON-lines or gRPC. The `Model::infer` call sends the image path and receives a `ModelOutput`-shaped response.

## Decisions

- Remote tags/classification endpoint paths are configurable per model with global defaults.
- Worker caches one resident model at a time.
- Custom architectures (JTP, WD v3, etc.) are handled inside the relevant backend by adding specialized `Model` implementations.
