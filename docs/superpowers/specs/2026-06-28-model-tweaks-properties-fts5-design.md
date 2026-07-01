# Design: Model Tweaks, Properties Window, and FTS5 Tag Search

Date: 2026-06-28

## Overview

This spec covers four related UI/data improvements:

1. **`top_k` option** for tagger model configs, capping the number of returned tags.
2. **Overwrite checkbox** in the Media Processing window to clear a model's previous predictions before re-running inference.
3. **FTS5 trigram search** for partial tag matching.
4. **Properties window** for viewing media metadata, tags, descriptions, classifications, and (placeholder) embeddings.

Implementation will be split into two milestones:

- **Milestone 1:** `top_k`, Overwrite, and FTS5 trigram (data/AI focused).
- **Milestone 2:** Properties window + settings config (UI focused).

---

## 1. `top_k` model config option

### Goal

Prevent poorly-tuned taggers from returning hundreds of low-confidence tags by capping the output to the `k` highest-scoring tags that still pass the threshold.

### Config schema

Add `top_k` to `ModelTagsOptions` in `src/config.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTagsOptions {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_top_k")]
    pub top_k: Option<usize>,
}

fn default_top_k() -> Option<usize> {
    Some(100)
}
```

- Default: `Some(100)`.
- `None` means no cap (backward-compatible escape hatch).

### Behavior

For each backend that produces `ModelOutput::Tags`:

1. Run inference.
2. Apply sigmoid if needed.
3. Filter by `threshold`.
4. If `top_k` is `Some(k)` and the filtered tag count exceeds `k`, keep only the `k` highest-scoring tags.

### Where to apply

Apply the cap inside each tag-producing backend so the worker stays simple:

- `src/models/tagger.rs` — `ViTTagger::infer`
- `src/models/onnx.rs` — `OrtModel::infer`
- `src/models/remote.rs` — `RemoteBackend::infer` tag path

The worker continues to write whatever `ModelOutput::Tags` contains.

---

## 2. Media Processing "Overwrite" checkbox

### Goal

When re-running inference with tweaked settings, let the user replace a model's existing predictions instead of merging with them.

### Scope

Overwrite only affects predictions from the **selected model/source and output kind**. Other sources/models are untouched.

### UI change

In `src/ui/media_processing.rs`:

- Add `overwrite: bool` to `MediaProcessingAction`.
- Add a checkbox labeled **"Overwrite existing predictions"** above the **Go** button.

### Enqueue behavior

In `app.rs::enqueue_media_processing_jobs`:

- If `action.overwrite` is true, delete existing predictions for the target media + `source_name` + `output_kind` before enqueueing new jobs.
- For tags: delete matching rows from `searchable_tags` and remove the source entry from `media_files.tags_json`.
- For descriptions: delete matching rows from `searchable_text_fts` and remove the source entry from `media_files.descriptions_json`.
- For classifications and embeddings, apply the same pattern once those output kinds are wired.

Deletion must happen in the same transaction as the `tags_json`/`descriptions_json` update so the JSON column and side tables stay consistent.

---

## 3. FTS5 trigram tag search

### Goal

Allow partial tag matching so searching `hair` returns `black_hair`, `long_hair`, etc., while still supporting short queries like `ai` that trigrams cannot match.

### Schema change

New migration file:

```sql
CREATE VIRTUAL TABLE searchable_tags_fts USING fts5(
    tag,
    media_file_id UNINDEXED,
    source UNINDEXED,
    tokenize='trigram'
);
```

`searchable_tags` remains the source of truth for scores and source attribution.

### Sync strategy

Extend `db::searchable::update_tags_json` to maintain `searchable_tags_fts` inside the same transaction:

```rust
DELETE FROM searchable_tags_fts WHERE media_file_id = ?1 AND source = ?2;
-- one INSERT per tag
INSERT INTO searchable_tags_fts (tag, media_file_id, source) VALUES (?1, ?2, ?3);
```

This mirrors the existing `searchable_tags` sync pattern. Write amplification is acknowledged; optimization is deferred until profiling shows it matters.

### Query strategy

For each query token:

- If token length >= 3, use FTS5 trigram `MATCH` on `searchable_tags_fts`.
- If token length < 3, fall back to exact case-insensitive match on `searchable_tags`.

Tokens are processed independently; results are merged in Rust by summing `matches` per `media_file_id`.

#### FTS5 query

```sql
SELECT fts.media_file_id, COUNT(*) AS matches
FROM searchable_tags_fts fts
JOIN media_files m ON m.id = fts.media_file_id
JOIN folders f ON f.id = m.folder_id
WHERE fts.tag MATCH ?2
  AND (f.id = ?1 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
GROUP BY fts.media_file_id
```

The join back to `searchable_tags` is unnecessary because `media_file_id` is available directly in the FTS table.

#### Short-token fallback query

```sql
SELECT t.media_file_id, COUNT(*) AS matches
FROM searchable_tags t
JOIN media_files m ON m.id = t.media_file_id
JOIN folders f ON f.id = m.folder_id
WHERE LOWER(t.tag) = ?2
  AND (f.id = ?1 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
GROUP BY t.media_file_id
```

### MATCH escaping

User tokens must be treated as literal phrases. Wrap each token in double quotes and escape internal double quotes by doubling them:

- `ai` → `"ai"`
- `foo"bar` → `"foo""bar"`

This prevents FTS5 query-syntax injection (`OR`, `AND`, `NOT`, `*`, etc.).

### Exact-match scoring note

Exact matches are naturally included by trigram substring search. They do **not** receive an extra score boost. The short-token fallback uses exact matching only because trigrams cannot match tokens under 3 characters.

### Future idea (deferred)

A reviewer suggested running exact matches alongside substring matches and ranking exact matches higher. This is noted for future tuning but not implemented now.

---

## 4. Properties window

### Goal

Display metadata and per-source AI predictions for a single selected media item.

### New module

`src/ui/properties.rs` defines:

```rust
pub struct PropertiesState {
    pub open: bool,
    pub media_id: Option<i64>,
}

pub enum PropertiesAction {
    Open(i64),
}

pub struct PropertiesData {
    pub media: MediaFile,
    pub tags: HashMap<String, HashMap<String, f32>>,      // source -> tag -> score
    pub descriptions: HashMap<String, String>,            // source -> text
    pub classifications: HashMap<String, Vec<String>>,    // source -> classes
    pub embeddings: Vec<String>,                          // placeholder source names
}

pub fn show(
    ctx: &egui::Context,
    open: &mut bool,
    media_id: Option<i64>,
    data: Option<&PropertiesData>,
    advanced: bool,
) -> Vec<PropertiesAction>;
```

### Opening the window

- Grid context menu: add **"Properties"** item.
- Viewer action bar: add **"Properties"** button and `I` hotkey.
- If no single media is selected, the window shows a placeholder.

### Data fetching

`app.rs` fetches `PropertiesData` asynchronously when:

- The window opens.
- The target `media_id` changes.

Fetch all tabs in one query; cache the result in app state. Refresh on target change, not on tab switch.

### Tabs

#### General

Always shown:

- Filename
- Absolute path
- Folder path
- Dimensions
- Format
- File size
- Created / modified times
- Presence status
- BLAKE3 hash

If `show_advanced_media_properties` is enabled, also show raw DB columns (`id`, `folder_id`, etc.).

#### Tags

- Source switcher at the top.
- Scrollable list of `tag -> score` pairs, sorted by score descending.

#### Descriptions

- Source switcher at the top.
- Scrollable, selectable text.

#### Classifications

- Source switcher at the top.
- List of classes for the selected source.

#### Embeddings

Placeholder tab. Show a message like "Embeddings viewer not yet implemented" and, if useful, the list of sources that produced embeddings.

### Source switcher layout

Fixed-width dropdown with left/right arrows:

```text
|[<-] [v Source________________] [->]|
```

- Dropdown is a fixed width (e.g. 200 px), text left-aligned.
- Left/right arrows cycle to the previous/next source for the current tab.
- Only sources that have data for the current tab appear in the dropdown.

### Multi-select behavior

For now, Properties requires a single selection. With zero or multiple items selected, show a placeholder message. Multi-select support can be revisited once the browser supports it.

---

## 5. Settings config addition

Add to `UiConfig` in `src/config.rs`:

```rust
#[serde(default)]
pub show_advanced_media_properties: bool,
```

Default: `false` (via standard `Default` for `bool`).

In the Settings window, add a checkbox **"Show advanced media properties"**. When toggled:

- Save the config.
- The Properties window General tab immediately reflects the change.

---

## 6. Testing/verification

- Unit tests for `top_k` cap behavior in tag-producing backends.
- Unit test for overwrite deletion in `db::searchable`.
- Unit tests for `TagsSearchable` covering:
  - Substring match (`hair` → `black_hair`).
  - Short-token exact match (`ai` → `ai`, but not `hair`).
  - Multiple token scoring.
- Manual verification of Properties window opening from grid context menu and viewer hotkey.

---

## Open questions / deferred

- Should exact substring matches receive a score boost over partial matches? Deferred; current behavior treats all matches equally.
- Should Properties window support comparing two sources side-by-side? Deferred.
- Should the source switcher remember the last-selected source per tab? Deferred.
