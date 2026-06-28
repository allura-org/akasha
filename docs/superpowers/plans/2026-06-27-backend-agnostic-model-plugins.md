# Backend-Agnostic Model Plugin Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the candle-specific inference worker into a backend registry where `candle` and `remote` are interchangeable plugins, and any future backend can be added by implementing two traits.

**Architecture:** Move the generic `Model`/`Backend` traits and registry into `src/models/mod.rs`. Convert the existing candle code into a `CandleBackend` module. Add a `RemoteBackend` using `reqwest::blocking`. Refactor `SearchWorker` to select backends from the registry, keep one resident model, and run inference on `spawn_blocking`.

**Tech Stack:** Rust 2024, `tokio`, `sqlx`, `candle` (feature-gated), `reqwest` (default), `base64`.

## Global Constraints

- Default build must remain pure Rust; `candle` is a default feature but can be disabled with `--no-default-features`.
- `remote` is a default feature and must compile without `candle`.
- All inference runs on `tokio::task::spawn_blocking` so the async runtime is not blocked.
- No new SQL migrations.
- `ModelConfig` changes must be backward compatible (`backend` and `remote` are optional).

---

## File Structure

- `src/models/mod.rs` — Generic `ModelOutput`, `Model`, `Backend`, and `BackendRegistry`.
- `src/models/loader.rs` — Existing model file resolution (candle-local/HF slug).
- `src/models/preprocess.rs` — Image preprocessing helpers (candle).
- `src/models/tagger.rs` — `ViTTagger` (candle).
- `src/models/stub.rs` — Test stub tagger (candle).
- `src/models/candle.rs` — `CandleBackend` and `CandleModel` adapter.
- `src/models/remote.rs` — `RemoteBackend`, `RemoteModel`, request/response handling.
- `src/config.rs` — `ModelConfig` gains `backend` and `remote` fields.
- `src/db/searchable.rs` — `sync_model_configs` stores the new fields; helper reconstructs `ModelConfig`.
- `src/searchables/worker.rs` — Uses `BackendRegistry`, resident model cache, `spawn_blocking` inference.
- `Cargo.toml` — Adds `reqwest` (with `blocking`) and `base64` under default features.

---

### Task 1: Generalize `CandleModel` into `Model` and add `Backend`/`BackendRegistry`

**Files:**
- Modify: `src/models/mod.rs`, `src/config.rs`
- Test: inline `#[cfg(test)]` in `src/models/mod.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `pub trait Model: Send + Sync { fn infer(&self, image_path: &Path) -> Result<ModelOutput>; }`
  - `pub trait Backend: Send + Sync { fn id(&self) -> &'static str; fn is_available(&self) -> bool; fn supports(&self, config: &ModelConfig) -> bool; fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>>; }`
  - `pub struct BackendRegistry { backends: Vec<Arc<dyn Backend>> }` with `register`, `default`, `select`.
  - Re-exports/keeps `ModelOutput`, `ModelOutputKind`.

- [ ] **Step 1: Write the failing registry test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;
    use crate::config::{ModelConfig, ModelKind};

    struct AlwaysBackend;
    impl Backend for AlwaysBackend {
        fn id(&self) -> &'static str { "always" }
        fn is_available(&self) -> bool { true }
        fn supports(&self, _config: &ModelConfig) -> bool { true }
        fn load(&self, _config: &ModelConfig) -> Result<Arc<dyn Model>> {
            unimplemented!()
        }
    }

    #[test]
    fn registry_selects_by_backend_id() {
        let mut reg = BackendRegistry::empty();
        reg.register(AlwaysBackend);
        let config = ModelConfig {
            name: "x".into(),
            kind: ModelKind::Local,
            backend: Some("always".into()),
            path: None,
            base_url: None,
            model_id: None,
            api_key: None,
            tags: None,
            description: None,
            classification: None,
        };
        assert!(reg.select(&config).is_some());
    }

    #[test]
    fn registry_returns_none_for_missing_backend() {
        let reg = BackendRegistry::empty();
        let config = ModelConfig {
            name: "x".into(),
            kind: ModelKind::Local,
            backend: Some("nope".into()),
            path: None,
            base_url: None,
            model_id: None,
            api_key: None,
            tags: None,
            description: None,
            classification: None,
        };
        assert!(reg.select(&config).is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib models::mod::tests::registry_selects_by_backend_id`
Expected: FAIL (BackendRegistry not defined).

- [ ] **Step 3: Add `backend` field to `ModelConfig`**

In `src/config.rs`, add `backend: Option<String>` to `ModelConfig` so the registry can select an explicit backend. The field must be optional and backward compatible.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ModelKind,
    pub backend: Option<String>,
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub model_id: Option<String>,
    pub api_key: Option<String>,
    pub tags: Option<ModelTagsOptions>,
    pub description: Option<ModelDescriptionOptions>,
    pub classification: Option<ModelClassificationOptions>,
}
```

Do **not** add `remote` or `RemoteConfig` yet — those are Task 3.

- [ ] **Step 4: Implement traits and registry**

```rust
// src/models/mod.rs
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use crate::config::ModelConfig;

pub mod loader;
#[cfg(feature = "candle")]
pub mod preprocess;
#[cfg(feature = "candle")]
pub mod tagger;
#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "candle")]
pub mod stub;
#[cfg(feature = "remote")]
pub mod remote;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOutputKind {
    Tags,
    Description,
    Classification,
    Vector,
}

#[derive(Debug, Clone)]
pub enum ModelOutput {
    Tags(HashMap<String, f32>),
    Description(String),
    Classification { label: String, score: f32 },
    Vector(Vec<f32>),
}

pub trait Model: Send + Sync {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput>;
}

pub trait Backend: Send + Sync {
    fn id(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn supports(&self, config: &ModelConfig) -> bool;
    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>>;
}

pub struct BackendRegistry {
    backends: Vec<Arc<dyn Backend>>,
}

impl BackendRegistry {
    pub fn empty() -> Self {
        Self { backends: Vec::new() }
    }

    pub fn register<B: Backend + 'static>(&mut self, backend: B) {
        self.backends.push(Arc::new(backend));
    }

    pub fn default() -> Self {
        let mut reg = Self::empty();
        #[cfg(feature = "candle")]
        reg.register(candle::CandleBackend);
        #[cfg(feature = "remote")]
        reg.register(remote::RemoteBackend);
        reg
    }

    pub fn select(&self, config: &ModelConfig) -> Option<Arc<dyn Backend>> {
        if let Some(id) = &config.backend {
            self.backends
                .iter()
                .find(|b| b.id() == id && b.is_available())
                .cloned()
        } else {
            self.backends
                .iter()
                .find(|b| b.is_available() && b.supports(config))
                .cloned()
        }
    }
}
```

Note: `HashMap` import is already present in this file.

- [ ] **Step 5: Run tests**

Run: `cargo test --lib models::mod::tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/mod.rs src/config.rs
git commit -m "feat: add generic Model/Backend traits and registry"
```

---

### Task 2: Convert candle code into `CandleBackend`

**Files:**
- Create: `src/models/candle.rs`
- Delete: `src/models/worker.rs`
- Modify: `src/models/tagger.rs`, `src/models/stub.rs` (trait name `CandleModel` -> `Model`)
- Test: inline `#[cfg(all(test, feature = "candle"))]` in `src/models/candle.rs`

**Interfaces:**
- Consumes: `Model`, `Backend`, `ModelConfig`, `ModelOutput`, existing `loader`, `preprocess`, `tagger`, `stub`.
- Produces:
  - `pub struct CandleBackend;`
  - `impl Backend for CandleBackend`
  - `impl Model for ViTTagger`
  - `impl Model for StubTagger`

- [ ] **Step 1: Update tagger and stub to implement `Model`**

In `src/models/tagger.rs` and `src/models/stub.rs`, replace `impl CandleModel for ...` with `impl Model for ...`. Remove the now-unused `CandleModel` trait reference.

- [ ] **Step 2: Write the candle backend test**

```rust
// at the bottom of src/models/candle.rs
#[cfg(all(test, feature = "candle"))]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelTagsOptions};
    use std::sync::Arc;

    #[test]
    fn candle_backend_supports_local_path_or_hf_slug() {
        let backend = CandleBackend;
        let cfg = ModelConfig {
            name: "vit-base".into(),
            kind: ModelKind::Local,
            backend: None,
            path: Some("google/vit-base-patch16-224".into()),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(ModelTagsOptions { threshold: 0.1 }),
            description: None,
            classification: None,
        };
        assert!(backend.supports(&cfg));
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --features candle models::candle::tests::candle_backend_supports_local_path_or_hf_slug`
Expected: FAIL (CandleBackend not defined).

- [ ] **Step 4: Create `src/models/candle.rs`**

```rust
//! Local candle-based backend.

use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use candle_core::Device;

use crate::config::ModelConfig;

use super::{loader, Backend, Model, ModelOutput};

pub struct CandleBackend;

impl Backend for CandleBackend {
    fn id(&self) -> &'static str {
        "candle"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        // Candle only handles local models with a path right now.
        config.kind == crate::config::ModelKind::Local && config.path.is_some()
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let path = config.path.as_deref().context("candle model missing path")?;
        let source = loader::resolve_source(path)?;
        let files = loader::load_model_files(&source)
            .with_context(|| format!("failed to load model files for {}", config.name))?;
        let device = Device::Cpu;

        match config.tags.as_ref() {
            Some(options) => {
                let tagger = super::tagger::ViTTagger::load(
                    &config.name,
                    &files,
                    device,
                    options.threshold,
                )?;
                Ok(Arc::new(tagger))
            }
            None => anyhow::bail!("candle backend only supports tags output kind right now"),
        }
    }
}
```

- [ ] **Step 5: Delete `src/models/worker.rs`**

```bash
git rm src/models/worker.rs
```

- [ ] **Step 6: Run candle tests**

Run: `cargo test --features candle`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/models/candle.rs src/models/tagger.rs src/models/stub.rs src/models/worker.rs src/models/mod.rs
git commit -m "feat: convert candle worker into a CandleBackend plugin"
```

---

### Task 3: Add `backend` and `remote` fields to config and DB sync

**Files:**
- Modify: `src/config.rs`
- Modify: `src/db/searchable.rs`
- Test: `src/config.rs` tests and `src/db/searchable.rs` tests

**Interfaces:**
- Consumes: existing `ModelConfig` (with `backend` field added in Task 1).
- Produces:
  - `ModelConfig::remote: Option<ModelRemoteOptions>`
  - `ModelRemoteOptions { chat_endpoint, tag_endpoint, classify_endpoint }`
  - Top-level `[remote]` config section (`RemoteConfig`)
  - `model_config_from_searchable_config(cfg: &SearchableConfig) -> Result<ModelConfig>`

- [ ] **Step 1: Write the parsing test**

```rust
// in src/config.rs tests
#[test]
fn parse_model_with_backend_and_remote_options() {
    let text = r#"
[remote]
chat_endpoint = "/v1/chat"
tag_endpoint = "/v1/tag"

[[models]]
name = "remote-model"
type = "remote"
backend = "remote"
base_url = "https://example.com"
model_id = "m1"
api_key = "secret"

[models.remote]
classify_endpoint = "/v1/classify"
"#;

    let config: Config = toml::from_str(text).unwrap();
    let model = &config.models.models[0];
    assert_eq!(model.backend.as_deref(), Some("remote"));
    assert_eq!(model.remote.as_ref().unwrap().classify_endpoint.as_deref(), Some("/v1/classify"));
    assert_eq!(config.remote.chat_endpoint, "/v1/chat");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test config::tests::parse_model_with_backend_and_remote_options`
Expected: FAIL.

- [ ] **Step 3: Add types and fields**

```rust
// src/config.rs

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub ui: UiConfig,
    pub thumbnails: ThumbnailsConfig,
    pub debug: DebugConfig,
    pub models: ModelsConfig,
    pub remote: RemoteConfig,
    #[serde(alias = "folders")]
    pub imports: Vec<ImportConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RemoteConfig {
    pub chat_endpoint: String,
    pub tag_endpoint: String,
    pub classify_endpoint: String,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            chat_endpoint: "/chat/completions".into(),
            tag_endpoint: "/tags".into(),
            classify_endpoint: "/classify".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ModelKind,
    pub backend: Option<String>,
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub model_id: Option<String>,
    pub api_key: Option<String>,
    pub tags: Option<ModelTagsOptions>,
    pub description: Option<ModelDescriptionOptions>,
    pub classification: Option<ModelClassificationOptions>,
    pub remote: Option<ModelRemoteOptions>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRemoteOptions {
    #[serde(default = "default_chat_endpoint")]
    pub chat_endpoint: String,
    #[serde(default = "default_tag_endpoint")]
    pub tag_endpoint: String,
    #[serde(default = "default_classify_endpoint")]
    pub classify_endpoint: String,
}

fn default_chat_endpoint() -> String { "/chat/completions".into() }
fn default_tag_endpoint() -> String { "/tags".into() }
fn default_classify_endpoint() -> String { "/classify".into() }
```

Update `Config::default()` to include `remote: RemoteConfig::default()`.

- [ ] **Step 4: Update DB sync and add reconstruction helper**

In `src/db/searchable.rs`, modify `sync_model_configs` so each output-kind options object includes the base model fields and the `remote` options:

```rust
let base_options = serde_json::json!({
    "path": model.path,
    "base_url": model.base_url,
    "model_id": model.model_id,
    "api_key": model.api_key,
    "backend": model.backend,
    "remote": model.remote,
    "kind": model.kind,
});
```

Add helper:

```rust
pub fn model_config_from_searchable_config(cfg: &SearchableConfig) -> Result<crate::config::ModelConfig> {
    let opts = &cfg.options;

    let kind = opts
        .get("kind")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(crate::config::ModelKind::Local);

    let remote: Option<crate::config::ModelRemoteOptions> = opts
        .get("remote")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    let tags: Option<crate::config::ModelTagsOptions> = opts
        .get("threshold")
        .and_then(|t| serde_json::from_value(serde_json::json!({ "threshold": t })).ok());

    let description: Option<crate::config::ModelDescriptionOptions> = opts
        .get("prompt")
        .and_then(|p| serde_json::from_value(serde_json::json!({ "prompt": p })).ok());

    let classification: Option<crate::config::ModelClassificationOptions> =
        if cfg.kind == "classification" {
            Some(crate::config::ModelClassificationOptions {})
        } else {
            None
        };

    Ok(crate::config::ModelConfig {
        name: cfg.name.clone(),
        kind,
        backend: opts.get("backend").and_then(|v| v.as_str()).map(|s| s.to_string()),
        path: opts.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()),
        base_url: opts.get("base_url").and_then(|v| v.as_str()).map(|s| s.to_string()),
        model_id: opts.get("model_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
        api_key: opts.get("api_key").and_then(|v| v.as_str()).map(|s| s.to_string()),
        remote,
        tags,
        description,
        classification,
    })
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test --features candle`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/config.rs src/db/searchable.rs
git commit -m "feat: add backend/remote config fields and DB reconstruction helper"
```

---

### Task 4: Implement `RemoteBackend`

**Files:**
- Create: `src/models/remote.rs`
- Modify: `Cargo.toml`
- Test: inline tests with a local HTTP server stub

**Interfaces:**
- Consumes: `Backend`, `Model`, `ModelConfig`, `ModelOutput`, `RemoteConfig`.
- Produces: `pub struct RemoteBackend; impl Backend for RemoteBackend`, `struct RemoteModel`, `impl Model for RemoteModel`.

- [ ] **Step 1: Add dependencies**

```toml
# Cargo.toml
[features]
default = ["simd-thumbnails", "remote", "candle"]
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers", "dep:hf-hub"]
remote = ["dep:reqwest", "dep:base64"]

[dependencies]
reqwest = { version = "0.12", features = ["blocking", "json"], optional = true }
base64 = { version = "0.22", optional = true }
```

- [ ] **Step 2: Write the remote backend test**

```rust
// src/models/remote.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelRemoteOptions, ModelTagsOptions};

    #[test]
    fn remote_backend_supports_base_url() {
        let backend = RemoteBackend;
        let cfg = ModelConfig {
            name: "remote".into(),
            kind: ModelKind::Remote,
            backend: None,
            path: None,
            base_url: Some("https://example.com".into()),
            model_id: Some("m1".into()),
            api_key: None,
            tags: Some(ModelTagsOptions { threshold: 0.35 }),
            description: None,
            classification: None,
            remote: Some(ModelRemoteOptions::default()),
        };
        assert!(backend.supports(&cfg));
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --features remote models::remote::tests::remote_backend_supports_base_url`
Expected: FAIL.

- [ ] **Step 4: Implement `src/models/remote.rs`**

```rust
//! Remote HTTP backend for OpenAI-compatible and custom inference endpoints.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};

use crate::config::{ModelConfig, ModelKind, RemoteConfig};
use super::{Backend, Model, ModelOutput};

pub struct RemoteBackend;

impl Backend for RemoteBackend {
    fn id(&self) -> &'static str { "remote" }
    fn is_available(&self) -> bool { true }

    fn supports(&self, config: &ModelConfig) -> bool {
        config.kind == ModelKind::Remote || config.base_url.is_some()
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let base_url = config.base_url.as_deref().context("remote model missing base_url")?;
        let model_id = config.model_id.as_deref().context("remote model missing model_id")?;
        let client = reqwest::blocking::Client::new();
        Ok(Arc::new(RemoteModel {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model_id: model_id.to_string(),
            api_key: config.api_key.clone(),
            endpoints: config.remote.clone().unwrap_or_default(),
        }))
    }
}

struct RemoteModel {
    client: reqwest::blocking::Client,
    base_url: String,
    model_id: String,
    api_key: Option<String>,
    endpoints: crate::config::ModelRemoteOptions,
}

impl Model for RemoteModel {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        // Default to tags for this milestone.
        self.tag_image(image_path)
    }
}

impl RemoteModel {
    fn tag_image(&self, image_path: &Path) -> Result<ModelOutput> {
        let image_bytes = std::fs::read(image_path)
            .with_context(|| format!("failed to read image: {}", image_path.display()))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

        let url = format!("{}{}", self.base_url, self.endpoints.tag_endpoint);
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": self.model_id,
            "image": b64,
        }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response: serde_json::Value = req.send()
            .context("remote tag request failed")?
            .json()
            .context("failed to parse remote tag response")?;

        let mut tags = HashMap::new();
        if let Some(obj) = response.as_object() {
            for (k, v) in obj {
                if let Some(score) = v.as_f64() {
                    tags.insert(k.clone(), score as f32);
                }
            }
        }
        Ok(ModelOutput::Tags(tags))
    }
}
```

For `ModelRemoteOptions::default()` to work, derive `Default` on it.

- [ ] **Step 5: Run tests**

Run: `cargo test --features remote models::remote::tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/remote.rs Cargo.toml Cargo.lock
git commit -m "feat: add RemoteBackend with tag support"
```

---

### Task 5: Refactor `SearchWorker` to use `BackendRegistry`

**Files:**
- Modify: `src/searchables/worker.rs`
- Test: inline integration test with a mock backend

**Interfaces:**
- Consumes: `BackendRegistry`, `Backend`, `Model`, `model_config_from_searchable_config`, `update_tags_json`, `update_description_json`, `complete_job`, `fail_job`.
- Produces: `SearchWorker::new` takes a registry; `process_jobs` runs inference generically.

- [ ] **Step 1: Write the integration test**

```rust
// in src/searchables/worker.rs tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelTagsOptions};
    use crate::models::{Backend, Model, ModelOutput};
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    struct MockModel;
    impl Model for MockModel {
        fn infer(&self, _path: &Path) -> anyhow::Result<ModelOutput> {
            let mut tags = HashMap::new();
            tags.insert("mock_tag".to_string(), 0.99);
            Ok(ModelOutput::Tags(tags))
        }
    }

    struct MockBackend;
    impl Backend for MockBackend {
        fn id(&self) -> &'static str { "mock" }
        fn is_available(&self) -> bool { true }
        fn supports(&self, config: &ModelConfig) -> bool {
            config.backend.as_deref() == Some("mock")
        }
        fn load(&self, _config: &ModelConfig) -> anyhow::Result<Arc<dyn Model>> {
            Ok(Arc::new(MockModel))
        }
    }

    #[tokio::test]
    async fn search_worker_runs_mock_backend_job() {
        use crate::db;
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();

        let fid = db::folder::insert(&pool, None, "/tmp", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let mid = db::media::upsert(&pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None)
            .await
            .unwrap();

        let cfg_id = db::searchable::upsert_config(
            &pool,
            "mock",
            "tags",
            true,
            serde_json::json!({"backend": "mock", "kind": "local", "threshold": 0.0}),
        )
        .await
        .unwrap();
        db::searchable::enqueue_job(&pool, mid, "tagger", "{}", Some(cfg_id)).await.unwrap();

        let mut reg = BackendRegistry::empty();
        reg.register(MockBackend);
        let mut worker = SearchWorker::with_registry(Arc::new(pool.clone()), reg);
        worker.tick().await.unwrap();

        let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
            .bind(mid)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(row.0.contains("mock_tag"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features candle searchables::worker::tests::search_worker_runs_mock_backend_job`
Expected: FAIL (worker not refactored yet).

- [ ] **Step 3: Refactor worker**

```rust
// src/searchables/worker.rs
use std::sync::Arc;
use std::time::Duration;
use sqlx::SqlitePool;
use tokio::time::interval;

use crate::models::{BackendRegistry, Model};

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

impl SearchWorker {
    pub fn new(pool: Arc<SqlitePool>) -> Self {
        Self {
            pool,
            batch_size: 4,
            registry: BackendRegistry::default(),
            resident: None,
        }
    }

    #[cfg(test)]
    pub fn with_registry(pool: Arc<SqlitePool>, registry: BackendRegistry) -> Self {
        Self { pool, batch_size: 4, registry, resident: None }
    }

    pub async fn run(mut self) { ... }

    async fn tick(&mut self) -> anyhow::Result<usize> { ... }
}
```

Implement `tick` and `process_one`:

```rust
async fn tick(&mut self) -> anyhow::Result<usize> {
    let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
    if jobs.is_empty() {
        return Ok(0);
    }

    let ai_kinds = ["tagger", "classifier", "visionlanguage"];
    let (mut to_process, ignored): (Vec<_>, Vec<_>) = jobs
        .into_iter()
        .partition(|j| ai_kinds.contains(&j.job_kind.as_str()));

    for job in ignored {
        let _ = crate::db::searchable::fail_job(
            &self.pool,
            job.id,
            &format!("unknown job kind: {}", job.job_kind),
        ).await;
    }

    let count = to_process.len();
    if count == 0 {
        return Ok(0);
    }

    cluster_jobs(&mut to_process, self.resident.as_ref().map(|r| r.config_id));

    for job in &to_process {
        if let Err(e) = self.process_one(job).await {
            tracing::warn!(job_id = job.id, error = %e, "SearchWorker: job failed");
            let _ = crate::db::searchable::fail_job(&self.pool, job.id, &e.to_string()).await;
        }
    }

    Ok(count)
}

async fn process_one(&mut self, job: &crate::db::searchable::JobRow) -> anyhow::Result<()> {
    use std::path::Path;

    let cfg = crate::db::searchable::get_config_by_id(
        &self.pool,
        job.searchable_config_id.unwrap_or(0),
    )
    .await?
    .context("missing searchable_config for job")?;

    let model_config = crate::db::searchable::model_config_from_searchable_config(&cfg)?;
    let backend = self.registry.select(&model_config)
        .with_context(|| format!("no backend available for model {}", model_config.name))?;

    let backend_id = backend.id().to_string();
    let needs_load = self.resident.as_ref()
        .map(|r| r.config_id != cfg.id || r.backend_id != backend_id)
        .unwrap_or(true);

    if needs_load {
        tracing::info!(model = model_config.name, backend = backend_id, "SearchWorker: loading model");
        let model = tokio::task::spawn_blocking({
            let backend = backend.clone();
            let model_config = model_config.clone();
            move || backend.load(&model_config)
        })
        .await
        .map_err(|e| anyhow::anyhow!("model loading task panicked: {e}"))??;
        self.resident = Some(ResidentModel { config_id: cfg.id, backend_id, model });
        tracing::info!(model = model_config.name, "SearchWorker: model loaded");
    }

    let model = self.resident.as_ref().unwrap().model.clone();
    let media = crate::db::media::get_by_id(&self.pool, job.media_file_id)
        .await?
        .context("missing media file")?;
    let image_path = media.absolute_path.clone();

    let output = tokio::task::spawn_blocking(move || model.infer(Path::new(&image_path)))
        .await
        .map_err(|e| anyhow::anyhow!("inference task panicked: {e}"))??;

    match output {
        crate::models::ModelOutput::Tags(tags) => {
            crate::db::searchable::update_tags_json(&self.pool, job.media_file_id, &cfg.name, tags).await?;
        }
        crate::models::ModelOutput::Description(text) => {
            crate::db::searchable::update_description_json(&self.pool, job.media_file_id, &cfg.name, &text).await?;
        }
        _ => {}
    }

    crate::db::searchable::complete_job(&self.pool, job.id).await?;
    Ok(())
}
```

Keep the existing `cluster_jobs` helper.

- [ ] **Step 4: Run tests**

Run: `cargo test --features candle`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/searchables/worker.rs
git commit -m "feat: refactor SearchWorker to use BackendRegistry"
```

---

### Task 6: Wire up default registry and verify end-to-end

**Files:**
- Modify: `src/main.rs` (if needed to pass registry)
- Test: run full test suite

- [ ] **Step 1: Ensure `main.rs` still compiles**

`src/main.rs` calls `SearchWorker::new(self.pool.clone()).run()`. The new constructor has the same signature, so no change is required.

- [ ] **Step 2: Run full test suite**

Run:
```bash
cargo test
cargo test --features candle
cargo test --no-default-features
```

Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: verify backend-agnostic worker end-to-end"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Backend trait, registry, remote backend, config fields, worker refactor, custom architectures, resident caching, testing — all have tasks.
- [ ] **Placeholder scan:** No `TODO`/`TBD`/vague steps remain.
- [ ] **Type consistency:** `ModelConfig` fields, `Backend` trait signatures, and `SearchWorker` resident struct names match across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-27-backend-agnostic-model-plugins.md`.

**Execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task with review gates.
2. **Inline Execution** — Execute tasks in this session using `executing-plans` with checkpoints.

Which approach?
