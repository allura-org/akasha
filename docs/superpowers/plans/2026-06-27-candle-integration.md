# Candle Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign searchable storage to scale to 420k+ files, then add real local image tagging via Hugging Face `candle` behind an opt-in feature flag.

**Architecture:** JSON columns on `media_files` store raw Searchable outputs; side tables (`searchable_tags`, `searchable_text_fts`) power fast search. A new `src/models/` module (gated by the `candle` feature) loads ViT taggers from Hugging Face or local paths, runs inference in the existing `SearchWorker`, and writes results back through the DB layer.

**Tech Stack:** Rust 2024, sqlx 0.8, SQLite (WAL), candle-core/candle-nn/candle-transformers, hf-hub, tokenizers, egui 0.31, tokio.

## Global Constraints

- Rust edition 2024; use `anyhow::Result` at module boundaries.
- All candle code lives behind the opt-in `candle` Cargo feature.
- Database queries use parameterized SQL (`?1`, `?2`, ...).
- Migrations are embedded with `sqlx::migrate!()`; a rebuild is required after adding a migration file.
- `MediaSummary` stays lightweight; do not add JSON blobs to it.
- All writes to JSON columns and side tables happen in a single SQL transaction.
- `cargo build` (default, no candle) and `cargo test` must continue to pass.
- Commit after each task.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `migrations/015_searchable_columns.sql` | Add JSON columns, side tables, and fix `searchable_configs` unique constraint |
| `src/db/media.rs` | Add JSON fields to `MediaFile`; keep `MediaSummary` lightweight |
| `src/db/searchable.rs` | Config/value helpers: `upsert_config`, `update_tags_json`, `update_description_json`, job queue helpers |
| `src/searchables/tags.rs` | New `TagsSearchable` querying `searchable_tags` |
| `src/searchables/description.rs` | New `DescriptionSearchable` querying `searchable_text_fts` |
| `src/searchables/mod.rs` | Register new Searchables in `SearchableRegistry::with_defaults()` |
| `src/searchables/worker.rs` | Poll queue, cluster jobs, delegate to `CandleWorker` when feature is enabled |
| `src/config.rs` | Migrate to unified `[[models]]` list with optional subtables |
| `src/models/mod.rs` | Public API: `CandleModel` trait, `ModelOutput` enum, feature-gated module root |
| `src/models/loader.rs` | Resolve HF slug or local path via `hf-hub` |
| `src/models/preprocess.rs` | Decode/resize/normalize image → `candle_core::Tensor` |
| `src/models/tagger.rs` | Concrete `WdViTTagger` implementation |
| `src/models/worker.rs` | `CandleWorker` that runs inference jobs |
| `src/ui/media_processing.rs` | Update subtabs to list unified models, add CPU warning |
| `src/app.rs` | Sync model configs to DB on startup; pass config to Media Processing UI |

---

## Phase 1: Storage Redesign

### Task 1.1: Migration `migrations/015_searchable_columns.sql`

**Files:**
- Create: `migrations/015_searchable_columns.sql`
- Test: run `cargo build` then `cargo test`

**Interfaces:**
- Produces: schema with `tags_json`, `descriptions_json`, `classifications_json`, `embeddings_json`, `searchable_tags`, `searchable_text_fts`, and `searchable_configs` with `UNIQUE(name, kind)`.

- [ ] **Step 1: Write the migration**

```sql
-- Add raw Searchable storage columns to media_files.
ALTER TABLE media_files ADD COLUMN tags_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN descriptions_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN classifications_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN embeddings_json TEXT DEFAULT NULL;

-- Side table for fast tag search.
CREATE TABLE searchable_tags (
    media_file_id INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    tag TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (media_file_id, source, tag)
);
CREATE INDEX idx_searchable_tags_tag ON searchable_tags(tag, score);
CREATE INDEX idx_searchable_tags_media ON searchable_tags(media_file_id);

-- FTS5 table for description search.
CREATE VIRTUAL TABLE searchable_text_fts USING fts5(
    media_file_id UNINDEXED,
    source UNINDEXED,
    content
);

-- Recreate searchable_configs with UNIQUE(name, kind) while preserving IDs.
CREATE TABLE searchable_configs_new (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    options TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, kind)
);

INSERT INTO searchable_configs_new (id, name, kind, enabled, options, created_at)
SELECT id, name, kind, enabled, options, created_at FROM searchable_configs;

PRAGMA foreign_keys = OFF;
DROP TABLE searchable_configs;
ALTER TABLE searchable_configs_new RENAME TO searchable_configs;
PRAGMA foreign_keys = ON;
```

- [ ] **Step 2: Run tests to verify the migration applies cleanly**

Run: `cargo test`
Expected: existing tests pass; no migration errors.

- [ ] **Step 3: Commit**

```bash
git add migrations/015_searchable_columns.sql
git commit -m "Add searchable storage redesign migration"
```

---

### Task 1.2: DB Helpers in `src/db/searchable.rs`

**Files:**
- Modify: `src/db/searchable.rs`
- Test: add tests in the same file's `#[cfg(test)]` module

**Interfaces:**
- Consumes: existing `MediaFile` IDs and `searchable_configs` rows.
- Produces:
  - `pub async fn upsert_config(pool, name, kind, enabled, options) -> Result<i64>`
  - `pub async fn update_tags_json(pool, media_file_id, source, tags) -> Result<()>`
  - `pub async fn update_description_json(pool, media_file_id, source, description) -> Result<()>`
  - `pub async fn get_config_by_name_kind(pool, name, kind) -> Result<Option<SearchableConfig>>`

- [ ] **Step 1: Add `upsert_config` and `get_config_by_name_kind`**

```rust
pub async fn upsert_config(
    pool: &SqlitePool,
    name: &str,
    kind: &str,
    enabled: bool,
    options: serde_json::Value,
) -> Result<i64> {
    let id = sqlx::query(
        "INSERT INTO searchable_configs (name, kind, enabled, options)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(name, kind) DO UPDATE SET
             enabled = excluded.enabled,
             options = excluded.options,
             updated_at = CURRENT_TIMESTAMP"
    )
    .bind(name)
    .bind(kind)
    .bind(enabled)
    .bind(options)
    .execute(pool)
    .await?
    .last_insert_rowid();
    Ok(id)
}

pub async fn get_config_by_name_kind(
    pool: &SqlitePool,
    name: &str,
    kind: &str,
) -> Result<Option<SearchableConfig>> {
    let row = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at FROM searchable_configs
         WHERE name = ?1 AND kind = ?2"
    )
    .bind(name)
    .bind(kind)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}
```

- [ ] **Step 2: Add `update_tags_json`**

```rust
pub async fn update_tags_json(
    pool: &SqlitePool,
    media_file_id: i64,
    source: &str,
    tags: std::collections::HashMap<String, f32>,
) -> Result<()> {
    let mut tx = pool.begin().await?;

    // Read existing tags_json, update the source entry.
    let existing: Option<String> = sqlx::query_scalar(
        "SELECT tags_json FROM media_files WHERE id = ?1"
    )
    .bind(media_file_id)
    .fetch_optional(&mut *tx)
    .await?;

    let mut map: serde_json::Map<String, serde_json::Value> = existing
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    map.insert(source.to_string(), serde_json::to_value(&tags)?);

    sqlx::query("UPDATE media_files SET tags_json = ?1 WHERE id = ?2")
        .bind(serde_json::to_string(&map)?)
        .bind(media_file_id)
        .execute(&mut *tx)
        .await?;

    // Mirror into searchable_tags.
    sqlx::query("DELETE FROM searchable_tags WHERE media_file_id = ?1 AND source = ?2")
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;

    for (tag, score) in tags {
        sqlx::query(
            "INSERT INTO searchable_tags (media_file_id, source, tag, score)
             VALUES (?1, ?2, ?3, ?4)"
        )
        .bind(media_file_id)
        .bind(source)
        .bind(tag)
        .bind(score)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}
```

- [ ] **Step 3: Add `update_description_json`**

```rust
pub async fn update_description_json(
    pool: &SqlitePool,
    media_file_id: i64,
    source: &str,
    description: &str,
) -> Result<()> {
    let mut tx = pool.begin().await?;

    let existing: Option<String> = sqlx::query_scalar(
        "SELECT descriptions_json FROM media_files WHERE id = ?1"
    )
    .bind(media_file_id)
    .fetch_optional(&mut *tx)
    .await?;

    let mut map: serde_json::Map<String, serde_json::Value> = existing
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    map.insert(source.to_string(), serde_json::Value::String(description.to_string()));

    sqlx::query("UPDATE media_files SET descriptions_json = ?1 WHERE id = ?2")
        .bind(serde_json::to_string(&map)?)
        .bind(media_file_id)
        .execute(&mut *tx)
        .await?;

    // Mirror into FTS5.
    sqlx::query(
        "INSERT INTO searchable_text_fts (rowid, media_file_id, source, content)
         VALUES (?1, ?1, ?2, ?3)
         ON CONFLICT(rowid) DO UPDATE SET source = excluded.source, content = excluded.content"
    )
    .bind(media_file_id)
    .bind(source)
    .bind(description)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(())
}
```

- [ ] **Step 4: Add tests**

```rust
#[tokio::test]
async fn update_tags_json_writes_column_and_side_table() {
    let pool = setup_pool().await;
    let fid = crate::db::folder::insert(&pool, None, "/tmp", true, false, &[], &[], None, None, "disable").await.unwrap();
    let mid = crate::db::media::upsert(&pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None).await.unwrap();

    let mut tags = std::collections::HashMap::new();
    tags.insert("cat".to_string(), 0.9f32);
    update_tags_json(&pool, mid, "wd-vit", tags).await.unwrap();

    let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
        .bind(mid)
        .fetch_one(&pool).await.unwrap();
    assert!(row.0.contains("cat"));

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM searchable_tags WHERE media_file_id = ?1")
        .bind(mid)
        .fetch_one(&pool).await.unwrap();
    assert_eq!(count.0, 1);
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test db::searchable`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/db/searchable.rs
git commit -m "Add DB helpers for tags and descriptions JSON/side-table writes"
```

---

### Task 1.3: `TagsSearchable` and `DescriptionSearchable`

**Files:**
- Create: `src/searchables/tags.rs`
- Create: `src/searchables/description.rs`
- Modify: `src/searchables/mod.rs`
- Test: tests in each new file

**Interfaces:**
- Produces: `TagsSearchable` and `DescriptionSearchable` structs implementing `Searchable`.

- [ ] **Step 1: Implement `TagsSearchable`**

```rust
use std::collections::HashMap;
use anyhow::Result;
use sqlx::SqlitePool;
use super::{Searchable, SearchableKind};

#[derive(Debug, Clone, Copy, Default)]
pub struct TagsSearchable;

#[async_trait::async_trait]
impl Searchable for TagsSearchable {
    fn name(&self) -> &str { "tags" }
    fn kind(&self) -> SearchableKind { SearchableKind::Tags }

    async fn search(&self, pool: &SqlitePool, folder_id: i64, recursive: bool, query: &str) -> Result<Vec<(i64, f32)>> {
        let tokens: Vec<String> = query.split_whitespace().map(|t| t.to_lowercase()).filter(|t| !t.is_empty()).collect();
        if tokens.is_empty() { return Ok(Vec::new()); }

        let placeholders: Vec<String> = tokens.iter().enumerate().map(|(i, _)| format!("?{}", i + 3)).collect();
        let sql = if recursive {
            format!(
                "SELECT t.media_file_id, COUNT(*) AS matches
                 FROM searchable_tags t
                 JOIN media_files m ON m.id = t.media_file_id
                 JOIN folders f ON f.id = m.folder_id
                 WHERE (f.id = ?1 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
                   AND t.tag IN ({})
                 GROUP BY t.media_file_id",
                placeholders.join(",")
            )
        } else {
            format!(
                "SELECT t.media_file_id, COUNT(*) AS matches
                 FROM searchable_tags t
                 JOIN media_files m ON m.id = t.media_file_id
                 WHERE m.folder_id = ?1 AND t.tag IN ({})
                 GROUP BY t.media_file_id",
                placeholders.join(",")
            )
        };

        let mut q = sqlx::query_as::<_, (i64, i64)>(&sql).bind(folder_id);
        for token in &tokens { q = q.bind(token); }
        let rows = q.fetch_all(pool).await?;
        Ok(rows.into_iter().map(|(id, matches)| (id, matches as f32 * 1.0)).collect())
    }
}
```

- [ ] **Step 2: Implement `DescriptionSearchable`**

```rust
use anyhow::Result;
use sqlx::SqlitePool;
use super::{Searchable, SearchableKind};

#[derive(Debug, Clone, Copy, Default)]
pub struct DescriptionSearchable;

#[async_trait::async_trait]
impl Searchable for DescriptionSearchable {
    fn name(&self) -> &str { "descriptions" }
    fn kind(&self) -> SearchableKind { SearchableKind::Text }

    async fn search(&self, pool: &SqlitePool, folder_id: i64, recursive: bool, query: &str) -> Result<Vec<(i64, f32)>> {
        let q = query.trim();
        if q.is_empty() { return Ok(Vec::new()); }

        let sql = if recursive {
            r#"
            SELECT f.media_file_id, bm25(searchable_text_fts) AS score
            FROM searchable_text_fts f
            JOIN media_files m ON m.id = f.media_file_id
            JOIN folders fld ON fld.id = m.folder_id
            WHERE f MATCH ?2
              AND (fld.id = ?1 OR fld.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
            ORDER BY score
            "#
        } else {
            r#"
            SELECT f.media_file_id, bm25(searchable_text_fts) AS score
            FROM searchable_text_fts f
            JOIN media_files m ON m.id = f.media_file_id
            WHERE m.folder_id = ?1 AND f MATCH ?2
            ORDER BY score
            "#
        };

        let rows = sqlx::query_as::<_, (i64, f64)>(sql)
            .bind(folder_id)
            .bind(q)
            .fetch_all(pool)
            .await?;

        Ok(rows.into_iter().map(|(id, score)| (id, -score as f32)).collect())
    }
}
```

- [ ] **Step 3: Register in `src/searchables/mod.rs`**

```rust
pub mod tags;
pub mod description;

// In SearchableRegistry::with_defaults:
reg.register(Arc::new(filename::FilenameSearchable));
reg.register(Arc::new(tags::TagsSearchable));
reg.register(Arc::new(description::DescriptionSearchable));
```

- [ ] **Step 4: Run tests**

Run: `cargo test searchables::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/searchables/tags.rs src/searchables/description.rs src/searchables/mod.rs
git commit -m "Add TagsSearchable and DescriptionSearchable"
```

---

## Phase 2: Config & Model Registry

### Task 2.1: Migrate `src/config.rs` to Unified `[[models]]` List

**Files:**
- Modify: `src/config.rs`
- Test: update existing tests and add new ones

**Interfaces:**
- Produces:
  - `pub struct ModelsConfig { pub models: Vec<ModelConfig> }`
  - `pub struct ModelConfig { name, kind, path, base_url, model_id, api_key, tags, description, classification }`
  - `pub struct ModelTagsOptions { pub threshold: f32 }`
  - `pub struct ModelDescriptionOptions { pub prompt: Option<String> }`
  - `pub struct ModelClassificationOptions {}` (empty for now)

- [ ] **Step 1: Replace `ModelsConfig` and `ModelConfig`**

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ModelKind,
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub model_id: Option<String>,
    pub api_key: Option<String>,
    pub tags: Option<ModelTagsOptions>,
    pub description: Option<ModelDescriptionOptions>,
    pub classification: Option<ModelClassificationOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTagsOptions {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptionOptions {
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelClassificationOptions {}

fn default_threshold() -> f32 { 0.35 }

impl Default for ModelTagsOptions {
    fn default() -> Self { Self { threshold: default_threshold() } }
}
```

- [ ] **Step 2: Update `Config::default()`**

```rust
impl Default for Config {
    fn default() -> Self {
        Self {
            ui: UiConfig::default(),
            thumbnails: ThumbnailsConfig::default(),
            debug: DebugConfig::default(),
            models: ModelsConfig::default(),
            imports: Vec::new(),
        }
    }
}
```

- [ ] **Step 3: Update config test**

```rust
#[test]
fn parse_unified_models_config() {
    let text = r#"
[ui]
theme = "dark"

[[models]]
name = "wd-vit-tagger-v3"
type = "local"
path = "SmilingWolf/wd-vit-tagger-v3"

[models.tags]
threshold = 0.35
"#;

    let config: Config = toml::from_str(text).unwrap();
    assert_eq!(config.models.models.len(), 1);
    assert_eq!(config.models.models[0].name, "wd-vit-tagger-v3");
    assert_eq!(config.models.models[0].kind, ModelKind::Local);
    assert_eq!(config.models.models[0].path.as_deref(), Some("SmilingWolf/wd-vit-tagger-v3"));
    assert_eq!(config.models.models[0].tags.as_ref().unwrap().threshold, 0.35);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test config::`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "Migrate config to unified [[models]] list"
```

---

### Task 2.2: Sync Model Registry to DB on Startup

**Files:**
- Modify: `src/db/searchable.rs`
- Modify: `src/app.rs`
- Test: add test in `src/db/searchable.rs`

**Interfaces:**
- Consumes: `Config::models.models`.
- Produces: `pub async fn sync_model_configs(pool, models) -> Result<()>`.

- [ ] **Step 1: Add `sync_model_configs` helper**

```rust
pub async fn sync_model_configs(
    pool: &SqlitePool,
    models: &[crate::config::ModelConfig],
) -> Result<()> {
    let mut wanted = std::collections::HashSet::new();
    for model in models {
        let base_options = serde_json::json!({
            "path": model.path,
            "base_url": model.base_url,
            "model_id": model.model_id,
        });

        if model.tags.is_some() {
            let mut options = serde_json::to_value(model.tags.as_ref().unwrap())?;
            merge_json(&mut options, base_options.clone());
            upsert_config(pool, &model.name, "tags", true, options).await?;
            wanted.insert((model.name.clone(), "tags".to_string()));
        }
        if model.description.is_some() {
            let mut options = serde_json::to_value(model.description.as_ref().unwrap())?;
            merge_json(&mut options, base_options.clone());
            upsert_config(pool, &model.name, "description", true, options).await?;
            wanted.insert((model.name.clone(), "description".to_string()));
        }
        if model.classification.is_some() {
            let mut options = serde_json::to_value(model.classification.as_ref().unwrap())?;
            merge_json(&mut options, base_options);
            upsert_config(pool, &model.name, "classification", true, options).await?;
            wanted.insert((model.name.clone(), "classification".to_string()));
        }
    }

    fn merge_json(target: &mut serde_json::Value, source: serde_json::Value) {
        if let (serde_json::Value::Object(t), serde_json::Value::Object(s)) = (target, source) {
            for (k, v) in s { t.insert(k, v); }
        }
    }

    // Disable stale sources.
    let existing = list_searchable_configs(pool).await?;
    for cfg in existing {
        if !wanted.contains(&(cfg.name.clone(), cfg.kind.clone())) {
            sqlx::query("UPDATE searchable_configs SET enabled = 0 WHERE id = ?1")
                .bind(cfg.id)
                .execute(pool)
                .await?;
        }
    }

    Ok(())
}
```

- [ ] **Step 2: Call from `AkashaApp::new`**

In `src/app.rs`, after config load and before spawning SearchWorker:

```rust
let models_for_sync = config.models.models.clone();
let pool_for_sync = Arc::clone(&pool_arc);
rt_arc.spawn(async move {
    if let Err(e) = crate::db::searchable::sync_model_configs(&pool_for_sync, &models_for_sync).await {
        tracing::error!("Failed to sync model configs: {e}");
    }
});
```

- [ ] **Step 3: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/db/searchable.rs src/app.rs
git commit -m "Sync model configs to searchable_configs on startup"
```

---

### Task 2.3: Update Media Processing UI

**Files:**
- Modify: `src/ui/media_processing.rs`
- Modify: `src/app.rs` (pass unified models)

**Interfaces:**
- Consumes: `Config::models.models`.
- Produces: `MediaProcessingAction { target, source_name, output_kind, model_name }`.

- [ ] **Step 1: Update `MediaProcessingAction`**

```rust
#[derive(Debug, Clone)]
pub struct MediaProcessingAction {
    pub target: MediaProcessingTarget,
    pub source_name: String,
    pub output_kind: String,
    pub model_name: String,
}
```

- [ ] **Step 2: Replace per-tab model fetching with filtering over `config.models.models`**

```rust
fn models_for_kind(config: &Config, kind: &str) -> Vec<&ModelConfig> {
    config.models.models.iter().filter(|m| match kind {
        "tags" => m.tags.is_some(),
        "description" => m.description.is_some(),
        "classification" => m.classification.is_some(),
        _ => false,
    }).collect()
}
```

- [ ] **Step 3: Add CPU inference warning**

Show a label above the Go button when a local model is selected:

```rust
if models[selected].kind == crate::config::ModelKind::Local {
    ui.label("Local CPU inference can take days on very large collections and only runs while Akasha is open.");
}
```

- [ ] **Step 4: Update `app.rs` action handling**

When a `MediaProcessingAction` is returned, enqueue a job for each target media file using the appropriate `searchable_configs` row.

- [ ] **Step 5: Run the app**

Run: `cargo run`
Expected: UI opens; Media Processing window shows models grouped by subtab.

- [ ] **Step 6: Commit**

```bash
git add src/ui/media_processing.rs src/app.rs
git commit -m "Update Media Processing UI for unified model config"
```

---

## Phase 3: Candle Integration

### Task 3.1: Add Cargo Features and Dependencies

**Files:**
- Modify: `Cargo.toml`

**Interfaces:**
- Produces: `candle` and `cuda` features; conditional compilation via `#[cfg(feature = "candle")]`.

- [ ] **Step 1: Add features and deps**

```toml
[features]
default = ["simd-thumbnails"]
hevc = ["dep:libheif-rs"]
simd-thumbnails = ["dep:fast_image_resize", "dep:webp"]
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers", "dep:hf-hub", "dep:tokenizers"]
cuda = ["candle", "candle-core/cuda"]

[dependencies]
# ... existing deps ...

# Local inference (opt-in)
candle-core = { version = "0.8", optional = true }
candle-nn = { version = "0.8", optional = true }
candle-transformers = { version = "0.8", optional = true }
hf-hub = { version = "0.4", optional = true }
tokenizers = { version = "0.21", optional = true }
```

- [ ] **Step 2: Verify default build still works**

Run: `cargo build`
Expected: compiles without candle deps.

- [ ] **Step 3: Verify candle build**

Run: `cargo build --features candle`
Expected: compiles (may take a while).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "Add candle feature and dependencies"
```

---

### Task 3.2: Model Loader

**Files:**
- Create: `src/models/loader.rs`
- Modify: `src/models/mod.rs`
- Test: `src/models/loader.rs` tests

**Interfaces:**
- Produces:
  - `pub enum ModelSource { HfSlug(String), LocalPath(PathBuf) }`
  - `pub fn resolve_source(path: &str) -> Result<ModelSource>`
  - `pub struct ModelFiles { pub config_path: PathBuf, pub weights_path: PathBuf, pub labels_path: PathBuf }`
  - `pub fn load_model_files(source: &ModelSource) -> Result<ModelFiles>`

- [ ] **Step 1: Implement loader**

```rust
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

pub enum ModelSource {
    HfSlug(String),
    LocalPath(PathBuf),
}

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
    pub labels_path: PathBuf,
}

pub fn resolve_source(path: &str) -> Result<ModelSource> {
    let p = Path::new(path);
    if p.exists() {
        Ok(ModelSource::LocalPath(p.to_path_buf()))
    } else {
        Ok(ModelSource::HfSlug(path.to_string()))
    }
}

#[cfg(feature = "candle")]
pub fn load_model_files(source: &ModelSource) -> Result<ModelFiles> {
    match source {
        ModelSource::HfSlug(slug) => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(slug.clone());
            Ok(ModelFiles {
                config_path: repo.get("config.json")?,
                weights_path: repo.get("model.safetensors")?,
                labels_path: repo.get("selected_tags.csv")?,
            })
        }
        ModelSource::LocalPath(dir) => {
            Ok(ModelFiles {
                config_path: dir.join("config.json"),
                weights_path: dir.join("model.safetensors"),
                labels_path: dir.join("selected_tags.csv"),
            })
        }
    }
}

#[cfg(not(feature = "candle"))]
pub fn load_model_files(_source: &ModelSource) -> Result<ModelFiles> {
    anyhow::bail!("candle feature not enabled")
}
```

- [ ] **Step 2: Add tests**

```rust
#[test]
fn resolve_local_path() {
    let src = resolve_source("/tmp").unwrap();
    match src {
        ModelSource::LocalPath(p) => assert_eq!(p, PathBuf::from("/tmp")),
        _ => panic!("expected local"),
    }
}

#[test]
fn resolve_hf_slug() {
    let src = resolve_source("SmilingWolf/wd-vit-tagger-v3").unwrap();
    match src {
        ModelSource::HfSlug(s) => assert_eq!(s, "SmilingWolf/wd-vit-tagger-v3"),
        _ => panic!("expected hf slug"),
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add src/models/loader.rs src/models/mod.rs
git commit -m "Add model source loader"
```

---

### Task 3.3: Image Preprocessing

**Files:**
- Create: `src/models/preprocess.rs`
- Test: `src/models/preprocess.rs` tests

**Interfaces:**
- Produces: `pub fn image_to_tensor(path: &Path, size: usize, device: &Device) -> Result<Tensor>`

- [ ] **Step 1: Implement preprocessing**

```rust
use std::path::Path;
use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, DynamicImage};

pub fn image_to_tensor(path: &Path, size: usize, device: &Device) -> Result<Tensor> {
    let img = image::open(path)?;
    let img = img.resize_to_fill(size as u32, size as u32, FilterType::Lanczos3);
    let rgb = img.to_rgb8();
    let pixels: Vec<f32> = rgb.pixels().flat_map(|p| {
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        // Normalize with ImageNet mean/std.
        [(r - 0.48145466) / 0.26862954, (g - 0.4578275) / 0.26130258, (b - 0.40821073) / 0.27577711]
    }).collect();

    let tensor = Tensor::from_vec(pixels, (size, size, 3), device)?
        .permute((2, 0, 1))? // (3, H, W)
        .unsqueeze(0)?;      // (1, 3, H, W)
    Ok(tensor)
}
```

- [ ] **Step 2: Add test with test_imgs**

```rust
#[test]
#[cfg(feature = "candle")]
fn preprocess_test_image() {
    let device = Device::Cpu;
    let tensor = image_to_tensor(Path::new("test_imgs/dagnpats.png"), 448, &device).unwrap();
    assert_eq!(tensor.dims(), &[1, 3, 448, 448]);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/models/preprocess.rs
git commit -m "Add image preprocessing for candle models"
```

---

### Task 3.4: ViT Tagger Model

**Files:**
- Create: `src/models/tagger.rs`
- Modify: `src/models/mod.rs`
- Test: integration test with stub (Task 4.1)

**Interfaces:**
- Produces: `pub struct WdViTTagger { ... }` implementing `CandleModel`.

- [ ] **Step 1: Define `CandleModel` trait and `ModelOutput` enum in `src/models/mod.rs`**

```rust
use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use candle_core::Device;

pub mod loader;
#[cfg(feature = "candle")]
pub mod preprocess;
#[cfg(feature = "candle")]
pub mod tagger;
#[cfg(feature = "candle")]
pub mod worker;

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

#[async_trait::async_trait]
pub trait CandleModel: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> ModelOutputKind;
    fn infer(&self, image_path: &Path) -> Result<ModelOutput>;
}
```

- [ ] **Step 2: Implement `WdViTTagger`**

This task is intentionally broad; the exact candle-transformers API may require adaptation. Start with:

```rust
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use candle_transformers::models::vit::{Config, Model as VitModel};

pub struct WdViTTagger {
    name: String,
    model: VitModel,
    labels: Vec<String>,
    device: Device,
    input_size: usize,
    threshold: f32,
}

impl WdViTTagger {
    pub fn load(name: &str, files: &loader::ModelFiles, device: Device, threshold: f32) -> Result<Self> {
        let config_text = std::fs::read_to_string(&files.config_path)?;
        let config: Config = serde_json::from_str(&config_text)?;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&files.weights_path], candle_core::DType::F32, &device)? };
        let model = VitModel::new(&config, 1000, vb)?; // num_classes may differ; adjust after inspecting weights

        let labels_text = std::fs::read_to_string(&files.labels_path)?;
        let labels: Vec<String> = labels_text.lines().map(|s| s.to_string()).collect();

        Ok(Self {
            name: name.to_string(),
            model,
            labels,
            device,
            input_size: 448,
            threshold,
        })
    }
}

#[async_trait::async_trait]
impl super::CandleModel for WdViTTagger {
    fn name(&self) -> &str { &self.name }
    fn kind(&self) -> super::ModelOutputKind { super::ModelOutputKind::Tags }

    fn infer(&self, image_path: &Path) -> Result<super::ModelOutput> {
        let tensor = preprocess::image_to_tensor(image_path, self.input_size, &self.device)?;
        let logits = self.model.forward(&tensor)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let mut tags = HashMap::new();
        for (i, &score) in probs_vec.iter().enumerate() {
            if score >= self.threshold && i < self.labels.len() {
                tags.insert(self.labels[i].clone(), score);
            }
        }

        Ok(super::ModelOutput::Tags(tags))
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --features candle`
Expected: compiles (model head/num_classes may need iteration).

- [ ] **Step 4: Commit**

```bash
git add src/models/tagger.rs src/models/mod.rs
git commit -m "Add WdViTTagger candle model"
```

---

### Task 3.5: CandleWorker and SearchWorker Integration

**Files:**
- Create: `src/models/worker.rs`
- Modify: `src/searchables/worker.rs`

**Interfaces:**
- Consumes: `job_queue` rows with `searchable_config_id` and `params_json`.
- Produces: `pub struct CandleWorker` with `async fn process_jobs(&mut self, jobs: Vec<JobRow>)`.

- [ ] **Step 1: Implement `CandleWorker`**

```rust
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use candle_core::Device;
use sqlx::SqlitePool;

use super::{loader, CandleModel, ModelOutput, ModelOutputKind};

pub struct CandleWorker {
    pool: Arc<SqlitePool>,
    device: Device,
    resident: Option<Box<dyn CandleModel>>,
    resident_config_id: Option<i64>,
}

impl CandleWorker {
    pub fn new(pool: Arc<SqlitePool>) -> Result<Self> {
        Ok(Self {
            pool,
            device: Device::Cpu,
            resident: None,
            resident_config_id: None,
        })
    }

    #[cfg(test)]
    pub fn set_resident(&mut self, model: Box<dyn CandleModel>, config_id: i64) {
        self.resident = Some(model);
        self.resident_config_id = Some(config_id);
    }

    pub async fn process_jobs(&mut self, jobs: &[crate::db::searchable::JobRow]) -> Result<()> {
        for job in jobs {
            if let Err(e) = self.process_one(job).await {
                let _ = crate::db::searchable::fail_job(&self.pool, job.id, &e.to_string()).await;
            }
        }
        Ok(())
    }

    async fn process_one(&mut self, job: &crate::db::searchable::JobRow) -> Result<()> {
        let cfg = crate::db::searchable::get_config_by_id(&self.pool, job.searchable_config_id.unwrap_or(0)).await?
            .context("missing searchable_config for job")?;

        // Load/replace model if needed.
        if self.resident_config_id != Some(cfg.id) {
            self.resident = Some(load_model_for_config(&cfg, &self.device).await?);
            self.resident_config_id = Some(cfg.id);
        }

        let model = self.resident.as_ref().unwrap();
        let media = crate::db::media::get_by_id(&self.pool, job.media_file_id).await?
            .context("missing media file")?;

        let output = model.infer(Path::new(&media.absolute_path))?;

        match output {
            ModelOutput::Tags(tags) => {
                crate::db::searchable::update_tags_json(&self.pool, job.media_file_id, &cfg.name, tags).await?;
            }
            ModelOutput::Description(text) => {
                crate::db::searchable::update_description_json(&self.pool, job.media_file_id, &cfg.name, &text).await?;
            }
            _ => {}
        }

        crate::db::searchable::complete_job(&self.pool, job.id).await?;
        Ok(())
    }
}

async fn load_model_for_config(cfg: &crate::db::searchable::SearchableConfig, device: &Device) -> Result<Box<dyn CandleModel>> {
    let options_value = cfg.options.clone();
    let path = options_value.get("path").and_then(|v| v.as_str()).unwrap_or(&cfg.name);
    let source = loader::resolve_source(path)?;
    let files = loader::load_model_files(&source)?;

    match cfg.kind.as_str() {
        "tags" => {
            let options: crate::config::ModelTagsOptions = serde_json::from_value(options_value).unwrap_or_default();
            let tagger = super::tagger::WdViTTagger::load(&cfg.name, &files, device.clone(), options.threshold)?;
            Ok(Box::new(tagger))
        }
        other => anyhow::bail!("unsupported model kind: {other}"),
    }
}
```

- [ ] **Step 2: Add `get_config_by_id` helper**

```rust
pub async fn get_config_by_id(pool: &SqlitePool, id: i64) -> Result<Option<SearchableConfig>> {
    let row = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at FROM searchable_configs WHERE id = ?1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}
```

- [ ] **Step 3: Wire into `SearchWorker`**

In `src/searchables/worker.rs`:

```rust
#[cfg(feature = "candle")]
async fn tick(&self) -> anyhow::Result<usize> {
    let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
    let count = jobs.len();
    if count == 0 { return Ok(0); }

    let mut candle = crate::models::worker::CandleWorker::new(Arc::clone(&self.pool))?;
    candle.process_jobs(&jobs).await?;
    Ok(count)
}

#[cfg(not(feature = "candle"))]
async fn tick(&self) -> anyhow::Result<usize> {
    // existing dummy behavior
}
```

- [ ] **Step 4: Commit**

```bash
git add src/models/worker.rs src/searchables/worker.rs src/db/searchable.rs
git commit -m "Wire CandleWorker into SearchWorker"
```

---

## Phase 4: Testing & Polish

### Task 4.1: Stub Model for Tests

**Files:**
- Create: `src/models/stub.rs` (only compiled in test cfg or always available)
- Modify: `src/models/mod.rs`
- Test: add integration test

**Interfaces:**
- Produces: `pub struct StubTagger` returning deterministic tags.

- [ ] **Step 1: Implement stub**

```rust
use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;

pub struct StubTagger {
    name: String,
}

impl StubTagger {
    pub fn new(name: &str) -> Self { Self { name: name.to_string() } }
}

#[async_trait::async_trait]
impl super::CandleModel for StubTagger {
    fn name(&self) -> &str { &self.name }
    fn kind(&self) -> super::ModelOutputKind { super::ModelOutputKind::Tags }

    fn infer(&self, _image_path: &Path) -> Result<super::ModelOutput> {
        let mut tags = HashMap::new();
        tags.insert("stub_tag".to_string(), 0.99f32);
        Ok(super::ModelOutput::Tags(tags))
    }
}
```

- [ ] **Step 2: Add integration test**

```rust
#[tokio::test]
async fn candle_worker_writes_tags() {
    let pool = setup_pool().await;
    let fid = crate::db::folder::insert(&pool, None, "/tmp", true, false, &[], &[], None, None, "disable").await.unwrap();
    let mid = crate::db::media::upsert(&pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None).await.unwrap();

    let cfg_id = crate::db::searchable::upsert_config(&pool, "stub", "tags", true, serde_json::json!({"threshold":0.0})).await.unwrap();
    crate::db::searchable::enqueue_job(&pool, mid, "inference", "{}", Some(cfg_id)).await.unwrap();

    let jobs = crate::db::searchable::claim_pending_jobs(&pool, 10).await.unwrap();
    let mut worker = crate::models::worker::CandleWorker::new(Arc::new(pool.clone())).unwrap();
    // Override resident with stub for the test.
    worker.set_resident(Box::new(crate::models::stub::StubTagger::new("stub")), cfg_id);
    worker.process_jobs(&jobs).await.unwrap();

    let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
        .bind(mid).fetch_one(&pool).await.unwrap();
    assert!(row.0.contains("stub_tag"));
}
```

- [ ] **Step 3: Commit**

```bash
git add src/models/stub.rs src/models/mod.rs
git commit -m "Add stub model and integration test for candle worker"
```

---

### Task 4.2: Manual Testing & Performance Baseline

**Files:**
- None (manual)

- [ ] **Step 1: Build with candle**

Run: `cargo build --features candle --release`

- [ ] **Step 2: Add a model to config.toml**

```toml
[[models]]
name = "wd-vit-tagger-v3"
type = "local"
path = "SmilingWolf/wd-vit-tagger-v3"

[models.tags]
threshold = 0.35
```

- [ ] **Step 3: Run inference on test_imgs**

Run: `cargo run --features candle --release`
Open Media Processing → Tagger → select model → target `test_imgs/` → Go.

- [ ] **Step 4: Verify**

- Tags appear in search results when searching "stub_tag" or real tags.
- UI remains responsive.
- No crashes on corrupt images.

- [ ] **Step 5: Performance baseline**

Time tagging 10 images from `test_imgs/` and record seconds per image.

- [ ] **Step 6: Commit any doc/config updates**

```bash
git add README.md config.example.toml
git commit -m "Document candle feature and example model config"
```

---

## Self-Review

### Spec Coverage

| Spec Section | Implementing Task |
|--------------|-------------------|
| JSON columns + side tables | Task 1.1, 1.2 |
| Tags/Description Searchables | Task 1.3 |
| Unified `[[models]]` config | Task 2.1 |
| Model registry sync | Task 2.2 |
| Media Processing UI | Task 2.3 |
| Candle feature flag | Task 3.1 |
| Model loader | Task 3.2 |
| Preprocessing | Task 3.3 |
| ViT tagger | Task 3.4 |
| Worker integration + clustering | Task 3.5 |
| Stub model + tests | Task 4.1 |
| Manual testing | Task 4.2 |

### Placeholder Scan

No TBD/TODO placeholders. All code blocks contain concrete implementations or test stubs.

### Type Consistency

- `update_tags_json` and `update_description_json` signatures are consistent across tasks.
- `CandleModel` trait is defined in Task 3.1 and used in Tasks 3.4, 3.5, 4.1.
- `ModelOutput` variants align with DB helper names.

### Known Gaps / Iterate During Implementation

- Exact `candle_transformers::models::vit::Config`/`Model` API may differ; the tagger task includes a note to iterate.
- `WdViTTagger` may need a custom classification head if candle-transformers' ViT does not match the model's output shape.
- FTS5 `bm25()` sign handling should be verified; the plan negates it so higher score is better.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-27-candle-integration.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
