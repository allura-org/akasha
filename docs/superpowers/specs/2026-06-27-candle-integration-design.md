# Candle Integration Design Spec

**Date:** 2026-06-27  
**Topic:** Local inference with Hugging Face `candle` + Searchable storage redesign  
**Status:** Approved for implementation planning  

## Section 1 — Overview & Scope

### Goal

Replace the dummy `SearchWorker` with real local inference using Hugging Face `candle`, starting with an image tagger. Before inference lands, redesign searchable storage so it scales to 420k+ media files without exploding the row count.

### In Scope

- Implement the JSON-column + side-table storage redesign from `searchable-storage-plan.md`.
- Refactor the `Searchable` trait and registry to operate on the new storage model.
- Add `candle` dependencies behind an opt-in Cargo feature (`candle`).
- Implement one concrete candle tagger as the first real model:
  - Primary target: `SmilingWolf/wd-vit-tagger-v3` (standard ViT + classification head; candle has ViT primitives).
  - Fallback/alternative: a SigLIP2-based zero-shot tagger if SigLIP2 support in candle-transformers is ready.
- Keep inference manual-only via the Media Processing UI for this milestone.
- CPU inference first; CUDA support added later as an additional feature flag.
- Preserve existing remote OpenAI-compatible model support (`ModelKind::Remote`).

### Explicitly Deferred

- Auto-enqueueing inference jobs during scan/import.
- Batched inference (process one image per forward pass for now).
- Vector search backend (`sqlite-vec`, HNSW, etc.).
- JTP-3's custom SigLIP2 head (port after the generic tagger pipeline is proven).
- GPU backends beyond a future CUDA feature gate.

### Non-Goals

- Removing Python/ONNX entirely from the user's environment (they may still use either for other tools).
- Supporting every candle example model out of the box.
- Real-time inference progress bars or cancellation UI in this milestone.

---

## Section 2 — Storage Redesign (Schema & DB Layer)

### Schema Changes

Add four JSON columns to `media_files` for the well-known Searchable output types:

```sql
ALTER TABLE media_files ADD COLUMN tags_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN descriptions_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN classifications_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN embeddings_json TEXT DEFAULT NULL;
```

Shape:

- `tags_json`: `{ "model-name": { "tag": confidence, ... } }`
- `descriptions_json`: `{ "model-name": "caption text" }`
- `classifications_json`: `{ "model-name": { label details } }` (shape TBD when first classifier is added)
- `embeddings_json`: `{ "model-name": [float, ...] }` (storage/serialization only; search deferred)

Add search-optimized side tables:

```sql
CREATE TABLE searchable_tags (
    media_file_id INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    tag TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (media_file_id, source, tag)
);
CREATE INDEX idx_searchable_tags_tag ON searchable_tags(tag, score);
CREATE INDEX idx_searchable_tags_media ON searchable_tags(media_file_id);

CREATE VIRTUAL TABLE searchable_text_fts USING fts5(
    media_file_id UNINDEXED,
    source UNINDEXED,
    content,
    content_rowid=media_file_id
);
```

Classifications and vectors are intentionally deferred:
- Classifications stay in `classifications_json` until a classifier model is added.
- Vectors stay in `embeddings_json` until a vector-search backend is chosen.

### Migration Strategy

- New migration `migrations/015_searchable_columns.sql` adds the columns and side tables.
- Existing `searchable_values` and `searchable_configs` are left untouched for now (rollback path).
- Once the redesign is stable, a follow-up migration drops the old EAV table and any unused indexes.

### DB Layer Changes (`src/db/`)

- Add the new JSON fields to `MediaFile`. `MediaSummary` stays lightweight; it does not need the JSON blobs for the grid.
- Add helpers in `src/db/searchable.rs` (or a new module):
  - `update_tags_json(pool, media_id, source, tags)` — writes `tags_json` and mirrors the data into `searchable_tags` inside a transaction.
  - `update_description_json(pool, media_id, source, description)` — writes `descriptions_json` and inserts/updates `searchable_text_fts`.
  - Similar helpers for `classifications_json` and `embeddings_json` (write JSON only for now).
- Keep config helpers (`list_searchable_configs`, `insert_config`) but repurpose `searchable_configs` to represent **sources** (model names, image-board sources, `user`, etc.) rather than abstract Searchable types.

### Searchable Trait Update

- `FilenameSearchable` stays as-is (queries `media_files.relative_path`).
- Add `TagsSearchable` that queries `searchable_tags` and returns `(media_file_id, score_contribution)`.
- Add `DescriptionSearchable` that queries `searchable_text_fts`.
- `ClassificationSearchable` and `VectorSearchable` remain placeholders until their data backends exist.
- The `SearchEngine` aggregation logic stays the same; only the data sources change.
- `TagsSearchable` and `DescriptionSearchable` are always registered in `SearchableRegistry::with_defaults()` alongside `FilenameSearchable`; they become no-ops when no data exists.

---

## Section 3 — Candle Integration Architecture

### Module Layout

Add a new module `src/inference/` (or `src/models/`) that is independent of the UI and the `Searchable` registry:

```
src/inference/
  mod.rs          — public API: `InferenceEngine`, `ModelHandle`
  loader.rs       — resolve model source (HF slug or local path), download/cache
  preprocess.rs   — image decoding, resize, normalize, tensorize
  tagger.rs       — concrete ViT-based tagger pipeline
  worker.rs       — `CandleWorker` that consumes `job_queue` rows and runs inference
```

`src/searchables/worker.rs` keeps polling the queue but delegates AI jobs to `src/inference/worker.rs` when the `candle` feature is enabled.

### Cargo Feature

Add an opt-in feature in `Cargo.toml`:

```toml
[features]
default = ["simd-thumbnails"]
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers", "dep:hf-hub", "dep:tokenizers"]
cuda = ["candle", "candle-core/cuda"]
```

When `candle` is disabled, AI jobs remain no-ops (log + mark done) as they are today.

### Model Abstraction

Introduce a small trait that each concrete model implements:

```rust
pub trait CandleModel: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> ModelOutputKind; // Tags | Description | Classification | Vector
    fn load(source: &ModelSource, device: &Device) -> Result<Self> where Self: Sized;
    fn infer(&self, image_path: &Path) -> Result<ModelOutput>;
}
```

`ModelOutput` is an enum mirroring the Searchable types:

```rust
pub enum ModelOutput {
    Tags(HashMap<String, f32>),
    Description(String),
    Classification { label: String, score: f32 },
    Vector(Vec<f32>),
}
```

A single physical model that can produce multiple Searchable kinds is represented by one `CandleModel` instance per output kind (or a single instance if it naturally returns multiple outputs at once). For this milestone we target taggers that return a single `Tags` output.

### First Concrete Model: ViT Tagger

Implement `WdViTTagger` (or a generic `ViTTagger`) that:

1. Loads a Hugging Face `config.json`, `model.safetensors`, and a labels file (e.g. `selected_tags.csv`).
2. Builds a `candle_transformers::models::vit::VitModel` (or custom config) on top of `candle-nn`.
3. Runs image preprocessing:
   - Decode with the existing `image` crate.
   - Resize to model input size (e.g. 448×448 for wd-vit-tagger-v3).
   - Normalize with dataset mean/std.
   - Convert to `candle_core::Tensor` of shape `(1, 3, H, W)`.
4. Forward pass → logits → sigmoid → filter by a configurable confidence threshold.
5. Returns `ModelOutput::Tags`.

If `wd-vit-tagger-v3` turns out to need a custom head not present in candle-transformers, we implement only the head in our crate and reuse candle's ViT backbone.

### Model Loading & Caching

`src/inference/loader.rs` resolves `ModelSource::HfSlug("SmilingWolf/wd-vit-tagger-v3")` or `ModelSource::LocalPath("/path/to/model")`:

- For HF slugs, use the `hf-hub` crate. It respects the `HF_HOME` environment variable and uses the same cache layout as the Python `huggingface_hub` library (default `~/.cache/huggingface`). No separate Akasha cache is required.
- For local paths, read files directly.
- Required files: `config.json`, `model.safetensors`, labels file. Optional: `preprocessor_config.json`.

### Worker Integration

`CandleWorker` is owned by `SearchWorker` and runs inside the same Tokio task:

- Maintain one resident `Box<dyn CandleModel>` (or an `Option<ModelKind, Box<dyn CandleModel>>` map if we later allow multiple resident models).
- On each claimed job:
  1. Parse `params_json` for `source` and `output_kind`.
  2. If the resident model does not match the job's source, load/replace it.
  3. Run `infer(absolute_path)`.
  4. Write the result via `db::searchable` helpers inside a transaction (JSON column + side table).
  5. Mark the job done.
- Jobs for missing files (`is_present = 0`) are skipped at the DB level as they are today.

### Error Handling

- Model load failures: mark the job failed with the error; do not crash the worker.
- Inference failures (e.g. corrupt image): mark failed and increment `attempts`; after N attempts the job stays failed until manually retried.
- Device selection failure (e.g. CUDA requested but unavailable): fall back to CPU and log a warning.

---

## Section 4 — Config & Model Registry

### Config Format

Use a single unified `[[models]]` list. The output kinds a model produces are implied by which parameter subtables it has (`[models.tags]`, `[models.description]`, `[models.classification]`):

```toml
[[models]]
name = "wd-vit-tagger-v3"
type = "local"
path = "SmilingWolf/wd-vit-tagger-v3"

[models.tags]
threshold = 0.35

[[models]]
name = "nsfw-classifier"
type = "remote"
base_url = "http://localhost:8000/v1"
model_id = "nsfw-classifier"

[models.classification]
# classifier-specific params
```

- `path` accepts either a Hugging Face model slug or an absolute/local directory path.
- `type = "local"` means run via candle (when the feature is enabled).
- `type = "remote"` means call an OpenAI-compatible endpoint using `base_url`, `model_id`, and optionally `api_key`.
- A model can have multiple subtables if it can perform multiple tasks (e.g. a CLIP model might one day have both `[models.classification]` and `[models.vector]`).

### Model Registry in the DB

Repurpose `searchable_configs` to represent **sources** rather than abstract Searchable types:

- `name`: model name from config (e.g. `wd-vit-tagger-v3`).
- `kind`: the Searchable kind this source produces (`tags`, `description`, `classification`, `vector`).
- `enabled`: whether the user has enabled this source.
- `options`: model-specific JSON (threshold, prompt, etc.).

### Rust Config Struct

Replace the existing `ModelsConfig` (with separate `tagger`, `classifier`, `visionlanguage` vectors) with a single `Vec<ModelConfig>`:

```rust
pub struct ModelsConfig {
    pub models: Vec<ModelConfig>,
}

pub struct ModelConfig {
    pub name: String,
    pub kind: ModelKind, // Local | Remote
    pub path: Option<String>, // HF slug or local directory for Local models
    pub base_url: Option<String>, // for Remote models
    pub model_id: Option<String>, // for Remote models
    pub api_key: Option<String>, // for Remote models
    pub tags: Option<ModelTagsOptions>,
    pub description: Option<ModelDescriptionOptions>,
    pub classification: Option<ModelClassificationOptions>,
}
```

The presence of `tags`, `description`, or `classification` determines which output kinds the model produces and which Media Processing subtabs it appears in.

### Model Registry in the DB

Repurpose `searchable_configs` to represent **sources** rather than abstract Searchable types:

- `name`: model name from config (e.g. `wd-vit-tagger-v3`).
- `kind`: the Searchable kind this source produces (`tags`, `description`, `classification`, `vector`).
- `enabled`: whether the user has enabled this source.
- `options`: model-specific JSON (threshold, prompt, etc.).

A source can have multiple rows if it produces multiple Searchable kinds.

On startup:
1. Parse `config.models`.
2. For each model, inspect its populated option fields (`tags`, `description`, `classification`) to derive output kinds.
3. For each output kind, upsert a row in `searchable_configs` keyed by `(name, kind)`.
4. Disable any DB rows whose `(name, kind)` no longer appears in config.

### Media Processing UI

Keep the existing `Tagger`/`Classifier`/`VisionLanguage` subtabs in `src/ui/media_processing.rs`; they are good separate surfaces for the three task types:

- Each subtab lists only the models that have the matching parameter subtable (`[models.tags]`, `[models.classification]`, `[models.description]`).
- A model with multiple subtables appears in every matching subtab.
- Show each model's name, `path`/`base_url`, and whether it is local or remote.
- When the user clicks **Go**, enqueue jobs for the selected model and target.
- Add a small "candle not compiled in" hint if the user configures local models but the `candle` feature is disabled.

### Job Queue Row Shape

Encode the model source and options in `params_json` (no schema change to `job_queue`):

```json
{
  "source": "wd-vit-tagger-v3",
  "output_kind": "tags",
  "threshold": 0.35
}
```

The worker parses `params_json` to determine which model to load. If it doesn't recognize the source, the job fails with a clear error.

For clustering, the claim query can order by `json_extract(params_json, '$.source')` so jobs for the same model are returned together.

---

## Section 5 — Data Flow & Lifecycle

### Startup Flow

1. `Config::load()` parses `config.toml`.
2. For every configured model and each of its output kinds, `app.rs` calls `db::searchable::upsert_config(source, kind, enabled, options)`.
3. `searchable_configs` rows whose source no longer exists in config are disabled (not deleted, to preserve history).
4. `SearchWorker` is spawned. If the `candle` feature is enabled, it creates a `CandleWorker` internally.
5. The worker polls `job_queue` every 5 seconds.

### Enqueueing a Job (Manual)

1. User opens Media Processing, picks a subtab (Tagger/Classifier/VisionLanguage), selects a model, and picks a target (single file or folder).
2. `app.rs` expands folder targets into a list of present media file IDs.
3. For each media file, call `db::searchable::enqueue_job(media_id, job_kind, params_json, config_id)`.
4. `params_json` contains `source`, `output_kind`, and model-specific options (threshold, prompt, etc.).
5. Duplicate pending jobs for the same `(media_id, source)` are skipped.

### Job Processing

1. `SearchWorker::tick()` claims a batch of pending jobs that reference present files.
2. The worker **clusters jobs by model** to minimize model load/unload:
   - If a model is already resident, claim pending jobs for that `source` first.
   - If no resident model exists or no pending jobs match it, claim jobs for the most-represented pending `source`.
   - The claim query orders pending jobs by `source` and creation time, then the worker reorders the claimed batch so same-source jobs are contiguous and the currently resident source is at the front.
3. For each job:
   - Parse `params_json`.
   - If the resident `CandleModel`'s name does not match `source`, drop the old model and load the new one from cache/disk.
   - Run `infer(absolute_path)`.
   - On success, write the result:
     - `ModelOutput::Tags` → `update_tags_json(media_id, source, tags)` (updates `tags_json` + `searchable_tags`).
     - `ModelOutput::Description` → `update_description_json(media_id, source, text)` (updates `descriptions_json` + `searchable_text_fts`).
     - Other kinds write only their JSON column for now.
   - All writes happen inside one SQL transaction so the JSON column and side table never drift.
   - Mark the job done.
4. On failure, increment `attempts`, set `status = 'failed'`, and store the error.

### Search Flow

1. User types a query and selects enabled Searchables.
2. `SearchEngine::execute()` loads enabled source configs from `searchable_configs`.
3. For each enabled Searchable:
   - `TagsSearchable` queries `searchable_tags` for tag matches and sums scores per media file.
   - `DescriptionSearchable` queries `searchable_text_fts` for token matches and sums scores.
   - `FilenameSearchable` queries `media_files.relative_path` as today.
4. Scores are aggregated per media file ID.
5. Matching IDs are hydrated into `MediaSummary` rows scoped to the current folder.
6. Results are sorted by score descending.

### Scan / Import / Watcher

- No automatic inference job enqueueing in this milestone.
- A scanned file is inserted/updated in `media_files` with empty JSON columns.
- The user must explicitly run a model via Media Processing to populate Searchable values.
- Auto-enqueue on scan is deferred to a later milestone.

---

## Section 6 — Error Handling & Resource Controls

### Model Load Failures

- If a model fails to load (missing files, unsupported architecture, unsupported dtype), the job is marked failed with the error message.
- The worker does **not** keep trying to reload the same broken model on every tick; it records that the source is unloadable for this tick and moves to other sources.
- If all pending jobs are for the same unloadable source, the worker sleeps and retries on the next tick (files/cache may have changed).

### Inference Failures

- Corrupt/unsupported images: mark the job failed, increment `attempts`, and store the error.
- After a configurable number of attempts (default 3), the job stays in `failed` status until the user manually retries or clears it.
- The UI shows failed job count and the latest error per source.

### Retry Policy

- Failed jobs are not automatically retried by the worker.
- A future UI action ("Retry failed jobs") can reset `status = 'pending'` for failed rows.
- Pending jobs are otherwise FIFO within a source cluster.

### Resource Controls

- Only **one** inference job runs at a time in this milestone (single-threaded worker, one resident model).
- The worker yields between jobs with a short sleep (e.g. 10–50 ms) so the Tokio runtime and UI remain responsive.
- CPU inference is expected to be slow; we do not attempt to cap CPU usage further.
- Memory: candle models are loaded on demand and unloaded when switching sources. We do not pre-load all configured models.

### Disk / Cache

- HF downloads use the `hf-hub` crate, which respects `HF_HOME` (default `~/.cache/huggingface`). This matches the Python Hugging Face tooling and avoids duplicating large model files.
- The worker does not clean the cache; users can delete it manually or set `HF_HUB_OFFLINE=1` to prevent new downloads.
- A future setting may cap cache size or prune unused models.

### Feature Flag Disabled

- If the user configures local models but builds without `--features candle`, local model jobs are marked failed with a message like "candle feature not enabled".
- Remote models (`ModelKind::Remote`) still work if their endpoints are reachable.

---

## Section 7 — Testing Plan

### Unit Tests

- `db::searchable` helpers:
  - `update_tags_json` writes both `tags_json` and `searchable_tags` correctly.
  - `update_description_json` writes both `descriptions_json` and `searchable_text_fts`.
  - Deleting a media file cascades to side tables.
  - Upserting a config disables stale sources.
- `searchables::TagsSearchable` and `DescriptionSearchable` return correct scores.
- `inference::loader` resolves HF slug vs local path and returns the expected file paths.
- `inference::preprocess` produces tensors of the expected shape and dtype for a given input image.

### Integration Tests

- Enqueue a tagger job in `job_queue`, run `SearchWorker`/`CandleWorker` end-to-end, and verify:
  - Job status becomes `done`.
  - `tags_json` is populated.
  - `searchable_tags` contains the expected rows.
  - Tag search returns the media file.

### Manual Tests

- Build with `--features candle` and run `cargo run`.
- Configure `SmilingWolf/wd-vit-tagger-v3` (or the chosen first model) in `config.toml`.
- Run inference on `test_imgs/` files via Media Processing UI.
- Verify tags appear in search results and the UI remains responsive.

### Build Matrix

- `cargo build` (default, no candle) must compile and run unchanged.
- `cargo build --features candle` must compile.
- `cargo test` and `cargo test --features candle` must pass.

### Performance Baseline

- Time tagging a representative set of `test_imgs/` images on CPU to establish a baseline before any optimization work.

---
