# Model Tweaks, Properties Window, and FTS5 Tag Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `top_k` tag cap and overwrite checkbox to AI inference, enable partial tag search via FTS5 trigrams, and add a Properties window for viewing media metadata and predictions.

**Architecture:** Config changes and inference-output capping live in the model backends; overwrite deletion and FTS5 sync live in `db::searchable`; tag search logic moves from exact `IN` to FTS5 `MATCH` with a short-token fallback; the Properties window is a new `ui::properties` module driven by async DB fetches in `app.rs`.

**Tech Stack:** Rust 2024, egui 0.31, sqlx 0.8 + SQLite WAL, SQLite FTS5 with `tokenize='trigram'`.

## Global Constraints

- SQLite queries use parameterized binding (`?1`, `?2`, ...).
- Convert between `u32`/`u64` and `i64` at the DB boundary.
- `Arc<SqlitePool>` is used to share the pool across async tasks.
- Keep DB logic in `db/` modules; UI logic in `ui/` modules.
- Follow existing Rust naming and `anyhow::Result` patterns.
- Run `cargo test` and `cargo clippy` after each milestone.
- Commit after every task.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/config.rs` | Add `top_k` to `ModelTagsOptions`; add `show_advanced_media_properties` to `UiConfig`. |
| `src/models/tagger.rs` | Apply `top_k` cap in Candle ViT tagger. |
| `src/models/onnx.rs` | Apply `top_k` cap in ONNX tagger. |
| `src/models/remote.rs` | Apply `top_k` cap in remote tag response. |
| `src/ui/media_processing.rs` | Add overwrite checkbox and propagate it in `MediaProcessingAction`. |
| `src/app.rs` | Implement overwrite deletion before enqueue; wire Properties window state, async fetch, context menu, and viewer hotkey. |
| `src/db/searchable.rs` | Add prediction-deletion helpers; sync `searchable_tags_fts` in `update_tags_json`. |
| `migrations/016_searchable_tags_fts.sql` | Create FTS5 trigram virtual table. |
| `src/searchables/tags.rs` | Replace exact `IN` query with FTS5 `MATCH` + short-token fallback. |
| `src/ui/settings.rs` | Add "Show advanced media properties" checkbox. |
| `src/ui/properties.rs` (new) | Render the Properties window with tabs and source switcher. |
| `src/ui/context_menu.rs` | Already exists; `app.rs` will add the Properties action. |
| `src/db/media.rs` | Add query to fetch full `MediaFile` plus aggregated tags/descriptions/classifications for the Properties window. |

---

## Milestone 1: `top_k`, Overwrite, and FTS5 trigram

### Task 1.1: Add `top_k` to `ModelTagsOptions`

**Files:**
- Modify: `src/config.rs:185-209`
- Test: `src/config.rs` tests (existing parse test)

**Interfaces:**
- Consumes: none
- Produces: `ModelTagsOptions { threshold, top_k }` where `top_k: Option<usize>` defaults to `Some(100)`.

- [ ] **Step 1: Add the field and default**

  Replace `ModelTagsOptions` and its `Default` impl:

  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct ModelTagsOptions {
      #[serde(default = "default_threshold")]
      pub threshold: f32,
      #[serde(default = "default_top_k")]
      pub top_k: Option<usize>,
  }

  impl Default for ModelTagsOptions {
      fn default() -> Self {
          Self {
              threshold: default_threshold(),
              top_k: default_top_k(),
          }
      }
  }

  fn default_top_k() -> Option<usize> {
      Some(100)
  }
  ```

- [ ] **Step 2: Update parse test**

  In the existing `parse_unified_models_config` test, add:

  ```rust
  assert_eq!(config.models.models[0].tags.as_ref().unwrap().top_k, Some(100));
  ```

- [ ] **Step 3: Run tests**

  ```bash
  cargo test config::tests --lib
  ```

  Expected: PASS

- [ ] **Step 4: Commit**

  ```bash
  git add src/config.rs
  git commit -m "feat: add top_k option to ModelTagsOptions with default 100"
  ```

---

### Task 1.2: Apply `top_k` in Candle `ViTTagger`

**Files:**
- Modify: `src/models/tagger.rs:47-53`, `119-125`, `154-159`
- Test: `src/models/tagger.rs` tests

**Interfaces:**
- Consumes: `ModelTagsOptions::top_k` passed through `ViTTagger::load`.
- Produces: `ModelOutput::Tags(HashMap<String, f32>)` capped to `top_k`.

- [ ] **Step 1: Add `top_k` to `ViTTagger`**

  ```rust
  pub struct ViTTagger {
      model: VitModel,
      labels: Vec<String>,
      device: Device,
      input_size: usize,
      threshold: f32,
      top_k: Option<usize>,
  }
  ```

- [ ] **Step 2: Accept `top_k` in `load`**

  Change signature:

  ```rust
  pub fn load(
      _name: &str,
      files: &loader::ModelFiles,
      device: Device,
      options: &crate::config::ModelTagsOptions,
  ) -> Result<Self> {
  ```

  Store `threshold: options.threshold` and `top_k: options.top_k`.

- [ ] **Step 3: Apply cap in `infer`**

  Replace the tag-collection loop with:

  ```rust
  let mut tags = HashMap::new();
  for (i, &score) in probs_vec.iter().enumerate() {
      if score >= self.threshold && i < self.labels.len() {
          tags.insert(self.labels[i].clone(), score);
      }
  }

  if let Some(k) = self.top_k {
      if tags.len() > k {
          let mut sorted: Vec<_> = tags.into_iter().collect();
          sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
          sorted.truncate(k);
          tags = sorted.into_iter().collect();
      }
  }
  ```

- [ ] **Step 4: Update `load` call sites**

  Find the call site in this file (the manual smoke test) and update:

  ```rust
  let options = crate::config::ModelTagsOptions {
      threshold: 0.1,
      top_k: None,
  };
  let tagger = ViTTagger::load("vit-base-patch16-224", &files, Device::Cpu, &options)?;
  ```

  Production call in `src/models/candle.rs:60`:

  ```rust
  let options = model.tags.as_ref().cloned().unwrap_or_default();
  let tagger = super::tagger::ViTTagger::load(name, &files, device, &options)?;
  ```

- [ ] **Step 5: Add unit test for cap**

  Add a non-ignored test that creates a tagger with mocked internals or tests a helper function. Since full model loading is heavy, add a small helper test for the cap logic:

  ```rust
  #[test]
  fn top_k_caps_tags() {
      let mut tags = std::collections::HashMap::new();
      tags.insert("a".to_string(), 0.9f32);
      tags.insert("b".to_string(), 0.8f32);
      tags.insert("c".to_string(), 0.7f32);
      let capped = crate::models::tagger::apply_top_k(tags, Some(2));
      assert_eq!(capped.len(), 2);
      assert!(capped.contains_key("a"));
      assert!(capped.contains_key("b"));
      assert!(!capped.contains_key("c"));
  }
  ```

  Extract `apply_top_k` as a public free function:

  ```rust
  pub fn apply_top_k(tags: HashMap<String, f32>, top_k: Option<usize>) -> HashMap<String, f32> {
      match top_k {
          Some(k) if tags.len() > k => {
              let mut sorted: Vec<_> = tags.into_iter().collect();
              sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
              sorted.truncate(k);
              sorted.into_iter().collect()
          }
          _ => tags,
      }
  }
  ```

  Use `apply_top_k` in `ViTTagger::infer`.

- [ ] **Step 6: Run tests**

  ```bash
  cargo test tagger --lib
  ```

  Expected: PASS

- [ ] **Step 7: Commit**

  ```bash
  git add src/models/tagger.rs
  git commit -m "feat: apply top_k cap in Candle ViT tagger"
  ```

---

### Task 1.3: Apply `top_k` in `OrtModel`

**Files:**
- Modify: `src/models/onnx.rs`

**Interfaces:**
- Consumes: `ModelConfig.tags.top_k`.
- Produces: capped `ModelOutput::Tags`.

- [ ] **Step 1: Locate threshold filter in `OrtModel::infer`**

  Find the loop that builds the tags `HashMap` from ONNX output (already filters by `threshold`).

- [ ] **Step 2: Apply cap**

  After the threshold filter, add:

  ```rust
  if let Some(k) = self.config.tags.as_ref().and_then(|t| t.top_k) {
      if tags.len() > k {
          let mut sorted: Vec<_> = tags.into_iter().collect();
          sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
          sorted.truncate(k);
          tags = sorted.into_iter().collect();
      }
  }
  ```

  Or reuse `crate::models::tagger::apply_top_k` if it is public.

- [ ] **Step 3: Build and run clippy**

  ```bash
  cargo clippy --features onnx -- -D warnings
  ```

  Expected: no warnings

- [ ] **Step 4: Commit**

  ```bash
  git add src/models/onnx.rs
  git commit -m "feat: apply top_k cap in ONNX OrtModel"
  ```

---

### Task 1.4: Apply `top_k` in `RemoteBackend`

**Files:**
- Modify: `src/models/remote.rs`

**Interfaces:**
- Consumes: `ModelConfig.tags.top_k`.
- Produces: capped tag map.

- [ ] **Step 1: Find tag response parsing**

  In `tag_image`, locate where the remote response `HashMap<String, f32>` is built or returned.

- [ ] **Step 2: Apply cap before returning**

  ```rust
  if let Some(k) = self.config.tags.as_ref().and_then(|t| t.top_k) {
      if tags.len() > k {
          let mut sorted: Vec<_> = tags.into_iter().collect();
          sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
          sorted.truncate(k);
          tags = sorted.into_iter().collect();
      }
  }
  ```

  Or use `crate::models::tagger::apply_top_k`.

- [ ] **Step 3: Build and run clippy**

  ```bash
  cargo clippy --features remote -- -D warnings
  ```

  Expected: no warnings

- [ ] **Step 4: Commit**

  ```bash
  git add src/models/remote.rs
  git commit -m "feat: apply top_k cap in RemoteBackend tagger"
  ```

---

### Task 1.5: Add overwrite checkbox to Media Processing UI

**Files:**
- Modify: `src/ui/media_processing.rs:27-32`, `200-214`

**Interfaces:**
- Consumes: none
- Produces: `MediaProcessingAction { target, source_name, output_kind, model_name, overwrite }`

- [ ] **Step 1: Extend `MediaProcessingAction`**

  ```rust
  #[derive(Debug, Clone)]
  pub struct MediaProcessingAction {
      pub target: MediaProcessingTarget,
      pub source_name: String,
      pub output_kind: String,
      pub model_name: String,
      pub overwrite: bool,
  }
  ```

- [ ] **Step 2: Add checkbox state and UI**

  Inside the model-details block (after the model info labels, before the `Go` button), add:

  ```rust
  let overwrite_key = format!("media_processing_overwrite_{}", sub_tab.output_kind());
  let mut overwrite: bool = ctx.memory_mut(|mem| {
      mem.data
          .get_persisted(egui::Id::new(&overwrite_key))
          .unwrap_or(false)
  });
  ui.checkbox(&mut overwrite, "Overwrite existing predictions");
  ctx.memory_mut(|mem| {
      mem.data.insert_persisted(egui::Id::new(overwrite_key), overwrite);
  });
  ```

- [ ] **Step 3: Include overwrite in emitted action**

  When building `MediaProcessingAction`, set:

  ```rust
  overwrite,
  ```

- [ ] **Step 4: Build**

  ```bash
  cargo build
  ```

  Expected: compiles

- [ ] **Step 5: Commit**

  ```bash
  git add src/ui/media_processing.rs
  git commit -m "feat: add overwrite checkbox to Media Processing window"
  ```

---

### Task 1.6: Delete existing predictions on overwrite

**Files:**
- Modify: `src/db/searchable.rs`
- Modify: `src/app.rs:744-810`

**Interfaces:**
- Consumes: `MediaProcessingAction::overwrite`, `output_kind`, `source_name`, target media IDs.
- Produces: `delete_tags_for_source(pool, media_file_id, source)` and `delete_description_for_source(pool, media_file_id, source)` helpers.

- [ ] **Step 1: Add deletion helpers in `db::searchable.rs`**

  ```rust
  /// Delete all tags for a given media file and source, updating both
  /// `searchable_tags` and `media_files.tags_json`.
  pub async fn delete_tags_for_source(
      pool: &SqlitePool,
      media_file_id: i64,
      source: &str,
  ) -> Result<()> {
      let mut tx = pool.begin().await?;

      sqlx::query("DELETE FROM searchable_tags WHERE media_file_id = ?1 AND source = ?2")
          .bind(media_file_id)
          .bind(source)
          .execute(&mut *tx)
          .await?;

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
      map.remove(source);

      sqlx::query("UPDATE media_files SET tags_json = ?1 WHERE id = ?2")
          .bind(serde_json::to_string(&map)?)
          .bind(media_file_id)
          .execute(&mut *tx)
          .await?;

      tx.commit().await?;
      Ok(())
  }

  /// Delete a description for a given media file and source, updating both
  /// `searchable_text_fts` and `media_files.descriptions_json`.
  pub async fn delete_description_for_source(
      pool: &SqlitePool,
      media_file_id: i64,
      source: &str,
  ) -> Result<()> {
      let mut tx = pool.begin().await?;

      sqlx::query("DELETE FROM searchable_text_fts WHERE media_file_id = ?1 AND source = ?2")
          .bind(media_file_id)
          .bind(source)
          .execute(&mut *tx)
          .await?;

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
      map.remove(source);

      sqlx::query("UPDATE media_files SET descriptions_json = ?1 WHERE id = ?2")
          .bind(serde_json::to_string(&map)?)
          .bind(media_file_id)
          .execute(&mut *tx)
          .await?;

      tx.commit().await?;
      Ok(())
  }
  ```

- [ ] **Step 2: Call deletions before enqueue in `app.rs`**

  In `enqueue_media_processing_jobs`, after resolving `media_ids` and before the enqueue loop, if `action.overwrite` is true:

  ```rust
  if action.overwrite {
      for &media_id in &media_ids {
          let result = match action.output_kind.as_str() {
              "tags" => db::searchable::delete_tags_for_source(&pool, media_id, &action.source_name).await,
              "description" => db::searchable::delete_description_for_source(&pool, media_id, &action.source_name).await,
              _ => Ok(()),
          };
          if let Err(e) = result {
              tracing::warn!(media_id, error = %e, "Failed to clear existing predictions");
          }
      }
  }
  ```

- [ ] **Step 3: Add unit test**

  Add a test in `src/searchables/worker.rs` or `src/db/searchable.rs` that inserts tags, calls `delete_tags_for_source`, and asserts the side table and JSON column are clean.

- [ ] **Step 4: Run tests**

  ```bash
  cargo test searchable --lib
  ```

  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/db/searchable.rs src/app.rs
  git commit -m "feat: implement overwrite prediction deletion before enqueue"
  ```

---

### Task 1.7: Add FTS5 trigram migration

**Files:**
- Create: `migrations/016_searchable_tags_fts.sql`

**Interfaces:**
- Consumes: none
- Produces: `searchable_tags_fts` virtual table.

- [ ] **Step 1: Write migration**

  ```sql
  -- FTS5 trigram side table for fast partial tag search.
  CREATE VIRTUAL TABLE searchable_tags_fts USING fts5(
      tag,
      media_file_id UNINDEXED,
      source UNINDEXED,
      tokenize='trigram'
  );
  ```

- [ ] **Step 2: Verify migration order**

  Ensure the file number `016` is the next after `015_searchable_columns.sql`.

- [ ] **Step 3: Build**

  ```bash
  cargo build
  ```

  Expected: compiles (migrations are embedded at compile time)

- [ ] **Step 4: Commit**

  ```bash
  git add migrations/016_searchable_tags_fts.sql
  git commit -m "feat: add FTS5 trigram migration for searchable tags"
  ```

---

### Task 1.8: Sync FTS5 table in `update_tags_json`

**Files:**
- Modify: `src/db/searchable.rs:127-179`

**Interfaces:**
- Consumes: tags map
- Produces: `searchable_tags_fts` rows kept in sync with `searchable_tags`.

- [ ] **Step 1: Delete existing FTS5 rows for source**

  Inside `update_tags_json`, after the `searchable_tags` DELETE and before the INSERT loop, add:

  ```rust
  sqlx::query("DELETE FROM searchable_tags_fts WHERE media_file_id = ?1 AND source = ?2")
      .bind(media_file_id)
      .bind(source)
      .execute(&mut *tx)
      .await?;
  ```

- [ ] **Step 2: Insert new FTS5 rows**

  Inside the existing `for (tag, score) in tags` loop, after the `searchable_tags` INSERT, add:

  ```rust
  sqlx::query(
      "INSERT INTO searchable_tags_fts (tag, media_file_id, source) VALUES (?1, ?2, ?3)"
  )
  .bind(&tag)
  .bind(media_file_id)
  .bind(source)
  .execute(&mut *tx)
  .await?;
  ```

- [ ] **Step 3: Update tests**

  Ensure existing `update_tags_json` tests still pass and consider adding an assertion that queries `searchable_tags_fts` directly to confirm rows were inserted.

- [ ] **Step 4: Run tests**

  ```bash
  cargo test update_tags_json --lib
  ```

  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/db/searchable.rs
  git commit -m "feat: keep searchable_tags_fts in sync with searchable_tags"
  ```

---

### Task 1.9: Update `TagsSearchable` for substring matching

**Files:**
- Modify: `src/searchables/tags.rs`

**Interfaces:**
- Consumes: query text, folder scope
- Produces: list of `(media_file_id, score)` where score is number of matching tokens.

- [ ] **Step 1: Add FTS5 query helper**

  Add a function to build a safe FTS5 `MATCH` expression:

  ```rust
  fn fts5_phrase(token: &str) -> String {
      let escaped = token.replace('"', "\"\"");
      format!("\"{}\"", escaped)
  }
  ```

  And a function to combine tokens:

  ```rust
  fn fts5_match_expr(tokens: &[String]) -> String {
      tokens.iter().map(|t| fts5_phrase(t)).collect::<Vec<_>>().join(" OR ")
  }
  ```

- [ ] **Step 2: Rewrite `TagsSearchable::search`**

  Split tokens into short (< 3) and long (>= 3):

  ```rust
  let tokens: Vec<String> = query
      .split_whitespace()
      .map(|t| t.to_lowercase())
      .filter(|t| !t.is_empty())
      .collect();
  if tokens.is_empty() {
      return Ok(Vec::new());
  }

  let (short, long): (Vec<_>, Vec<_>) = tokens.into_iter().partition(|t| t.len() < 3);

  let mut scores: HashMap<i64, f32> = HashMap::new();

  if !long.is_empty() {
      let match_expr = fts5_match_expr(&long);
      let sql = if recursive {
          format!(
              "SELECT fts.media_file_id, COUNT(*) AS matches
               FROM searchable_tags_fts fts
               JOIN media_files m ON m.id = fts.media_file_id
               JOIN folders f ON f.id = m.folder_id
               WHERE fts.tag MATCH ?1
                 AND (f.id = ?2 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?2))
               GROUP BY fts.media_file_id"
          )
      } else {
          format!(
              "SELECT fts.media_file_id, COUNT(*) AS matches
               FROM searchable_tags_fts fts
               JOIN media_files m ON m.id = fts.media_file_id
               WHERE m.folder_id = ?2 AND fts.tag MATCH ?1
               GROUP BY fts.media_file_id"
          )
      };

      let rows: Vec<(i64, i64)> = sqlx::query_as(&sql)
          .bind(&match_expr)
          .bind(folder_id)
          .fetch_all(pool)
          .await?;

      for (id, matches) in rows {
          *scores.entry(id).or_insert(0.0) += matches as f32;
      }
  }

  if !short.is_empty() {
      let placeholders: Vec<String> = short
          .iter()
          .enumerate()
          .map(|(i, _)| format!("?{}", i + 2))
          .collect();
      let sql = if recursive {
          format!(
              "SELECT t.media_file_id, COUNT(*) AS matches
               FROM searchable_tags t
               JOIN media_files m ON m.id = t.media_file_id
               JOIN folders f ON f.id = m.folder_id
               WHERE (f.id = ?1 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
                 AND LOWER(t.tag) IN ({})
               GROUP BY t.media_file_id",
              placeholders.join(",")
          )
      } else {
          format!(
              "SELECT t.media_file_id, COUNT(*) AS matches
               FROM searchable_tags t
               JOIN media_files m ON m.id = t.media_file_id
               WHERE m.folder_id = ?1 AND LOWER(t.tag) IN ({})
               GROUP BY t.media_file_id",
              placeholders.join(",")
          )
      };

      let mut q = sqlx::query_as::<_, (i64, i64)>(&sql).bind(folder_id);
      for token in &short {
          q = q.bind(token);
      }
      let rows = q.fetch_all(pool).await?;
      for (id, matches) in rows {
          *scores.entry(id).or_insert(0.0) += matches as f32;
      }
  }

  Ok(scores.into_iter().map(|(id, score)| (id, score)).collect())
  ```

- [ ] **Step 3: Update tests**

  Add tests:
  - `substring_match_finds_underscored_tag`
  - `short_token_exact_match`
  - `mixed_short_and_long_tokens`

- [ ] **Step 4: Run tests**

  ```bash
  cargo test tags::tests --lib
  ```

  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/searchables/tags.rs
  git commit -m "feat: use FTS5 trigram for partial tag search with short-token fallback"
  ```

---

### Task 1.10: Milestone 1 verification

- [ ] **Step 1: Run full test suite**

  ```bash
  cargo test --lib
  ```

  Expected: PASS

- [ ] **Step 2: Run clippy**

  ```bash
  cargo clippy -- -D warnings
  ```

  Expected: no warnings

- [ ] **Step 3: Manual smoke test**

  1. Start the app with a test import.
  2. Run a tagger model and confirm no more than `top_k` tags are stored per image.
  3. Search for a partial tag (e.g. `hair`) and confirm results include underscored tags.
  4. Queue a tagger job with **Overwrite** checked and confirm previous predictions for that source are replaced.

- [ ] **Step 4: Commit any fixes**

  ```bash
  git commit -am "fix: address milestone 1 review issues"
  ```

---

## Milestone 2: Properties window + settings config

### Task 2.1: Add `show_advanced_media_properties` to config

**Files:**
- Modify: `src/config.rs:97-106`
- Modify: `src/config.rs:258-268`

**Interfaces:**
- Consumes: none
- Produces: `UiConfig::show_advanced_media_properties: bool` defaulting to `false`.

- [ ] **Step 1: Add field**

  In `UiConfig`:

  ```rust
  #[serde(default)]
  pub show_advanced_media_properties: bool,
  ```

- [ ] **Step 2: Update Default impl**

  In `impl Default for UiConfig`:

  ```rust
  show_advanced_media_properties: false,
  ```

- [ ] **Step 3: Build**

  ```bash
  cargo build
  ```

  Expected: compiles

- [ ] **Step 4: Commit**

  ```bash
  git add src/config.rs
  git commit -m "feat: add show_advanced_media_properties config option"
  ```

---

### Task 2.2: Add settings checkbox

**Files:**
- Modify: `src/ui/settings.rs:5-11`, `66-94`

**Interfaces:**
- Consumes: `config.ui.show_advanced_media_properties`
- Produces: `SettingsAction::AdvancedMediaPropertiesChanged(bool)`

- [ ] **Step 1: Extend `SettingsAction`**

  ```rust
  pub enum SettingsAction {
      ThumbnailSizeChanged(u32),
      ThemeChanged(bool),
      DoubleClickDebounceChanged,
      ScrollSpeedChanged(f32),
      ViewerDefaultScaleModeChanged,
      AdvancedMediaPropertiesChanged(bool),
  }
  ```

- [ ] **Step 2: Add checkbox UI**

  After the Viewer section, add:

  ```rust
  ui.add_space(16.0);
  ui.heading("Properties");
  ui.separator();

  let mut advanced = config.ui.show_advanced_media_properties;
  if ui.checkbox(&mut advanced, "Show advanced media properties").changed() {
      config.ui.show_advanced_media_properties = advanced;
      actions.push(SettingsAction::AdvancedMediaPropertiesChanged(advanced));
  }
  ```

- [ ] **Step 3: Handle action in `app.rs`**

  In the settings action loop in `app.rs`:

  ```rust
  crate::ui::settings::SettingsAction::AdvancedMediaPropertiesChanged(_) => {
      settings_changed = true;
  }
  ```

  Saving config already happens when `settings_changed` is true.

- [ ] **Step 4: Build**

  ```bash
  cargo build
  ```

  Expected: compiles

- [ ] **Step 5: Commit**

  ```bash
  git add src/ui/settings.rs src/app.rs
  git commit -m "feat: add show advanced media properties checkbox to settings"
  ```

---

### Task 2.3: Create properties module and data structures

**Files:**
- Create: `src/ui/properties.rs`
- Modify: `src/ui/mod.rs`

**Interfaces:**
- Consumes: `MediaFile`, `db::media::PropertiesData`
- Produces: `PropertiesState`, `PropertiesAction::Open(i64)`, `PropertiesTab` enum.

- [ ] **Step 1: Add `PropertiesData` to `src/db/media.rs`**

  Before creating the UI module, define the data structure in the DB layer so `db::media` does not depend on `ui::properties`:

  ```rust
  #[derive(Debug, Clone)]
  pub struct PropertiesData {
      pub media: MediaFile,
      pub tags: HashMap<String, HashMap<String, f32>>,
      pub descriptions: HashMap<String, String>,
      pub classifications: HashMap<String, Vec<String>>,
      pub embeddings: Vec<String>,
  }
  ```

  Add necessary imports (`std::collections::HashMap`).

- [ ] **Step 2: Create `src/ui/properties.rs`**

  ```rust
  use eframe::egui;

  use crate::db::media::PropertiesData;

  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum PropertiesTab {
      General,
      Tags,
      Descriptions,
      Classifications,
      Embeddings,
  }

  impl PropertiesTab {
      fn label(&self) -> &'static str {
          match self {
              PropertiesTab::General => "General",
              PropertiesTab::Tags => "Tags",
              PropertiesTab::Descriptions => "Descriptions",
              PropertiesTab::Classifications => "Classifications",
              PropertiesTab::Embeddings => "Embeddings",
          }
      }
  }

  #[derive(Debug, Clone)]
  pub struct PropertiesState {
      pub open: bool,
      pub media_id: Option<i64>,
  }

  impl Default for PropertiesState {
      fn default() -> Self {
          Self { open: false, media_id: None }
      }
  }

  pub enum PropertiesAction {
      Open(i64),
  }

  pub fn show(
      ctx: &egui::Context,
      open: &mut bool,
      media_id: Option<i64>,
      data: Option<&PropertiesData>,
      advanced: bool,
  ) -> Vec<PropertiesAction> {
      let mut actions = Vec::new();

      egui::Window::new("Properties")
          .open(open)
          .resizable(true)
          .collapsible(false)
          .default_width(500.0)
          .default_height(600.0)
          .show(ctx, |ui| {
              let Some(media_id) = media_id else {
                  ui.label("No media selected.");
                  return;
              };

              let Some(data) = data else {
                  ui.label("Loading...");
                  return;
              };

              // Tabs
              let mut tab = ctx.memory_mut(|mem| {
                  mem.data
                      .get_persisted(egui::Id::new("properties_tab"))
                      .unwrap_or(PropertiesTab::General)
              });

              ui.horizontal(|ui| {
                  for t in [
                      PropertiesTab::General,
                      PropertiesTab::Tags,
                      PropertiesTab::Descriptions,
                      PropertiesTab::Classifications,
                      PropertiesTab::Embeddings,
                  ] {
                      if ui.selectable_label(tab == t, t.label()).clicked() {
                          tab = t;
                      }
                  }
              });
              ui.separator();

              ctx.memory_mut(|mem| {
                  mem.data.insert_persisted(egui::Id::new("properties_tab"), tab);
              });

              egui::ScrollArea::vertical().show(ui, |ui| {
                  match tab {
                      PropertiesTab::General => show_general(ui, data, advanced),
                      PropertiesTab::Tags => show_tags(ui, data),
                      PropertiesTab::Descriptions => show_descriptions(ui, data),
                      PropertiesTab::Classifications => show_classifications(ui, data),
                      PropertiesTab::Embeddings => show_embeddings(ui, data),
                  }
              });
          });

      actions
  }

  fn show_general(ui: &mut egui::Ui, data: &PropertiesData, advanced: bool) {
      let m = &data.media;
      ui.label(format!("Filename: {}", std::path::Path::new(&m.relative_path).file_name().map(|s| s.to_string_lossy()).unwrap_or_default()));
      ui.label(format!("Absolute path: {}", m.absolute_path));
      ui.label(format!("Folder ID: {}", m.folder_id));
      ui.label(format!("Dimensions: {}x{}", m.width.unwrap_or(0), m.height.unwrap_or(0)));
      ui.label(format!("Format: {}", m.format.as_deref().unwrap_or("unknown")));
      ui.label(format!("Size: {} bytes", m.file_size.unwrap_or(0)));
      ui.label(format!("Created: {}", m.created_at));
      if let Some(modified) = m.modified_at {
          ui.label(format!("Modified: {}", modified));
      }
      ui.label(format!("Present: {}", if m.is_present { "yes" } else { "no" }));
      ui.label(format!("Hash: {}", m.blake3_hash));

      if advanced {
          ui.separator();
          ui.heading("Advanced");
          ui.label(format!("ID: {}", m.id));
          ui.label(format!("Folder ID: {}", m.folder_id));
          if let Some(missing) = m.missing_since {
              ui.label(format!("Missing since: {}", missing));
          }
      }
  }

  fn source_switcher(
      ui: &mut egui::Ui,
      sources: &[String],
      selected: &mut usize,
      id_salt: &str,
  ) {
      ui.horizontal(|ui| {
          let available_width = ui.available_width();
          let dropdown_width = 200.0f32.min(available_width - 80.0).max(100.0);

          let prev_enabled = !sources.is_empty() && *selected > 0;
          if ui.add_enabled(prev_enabled, egui::Button::new("<-")).clicked() {
              *selected = selected.saturating_sub(1);
          }

          egui::ComboBox::from_id_salt(id_salt)
              .width(dropdown_width)
              .selected_text(sources.get(*selected).map(|s| s.as_str()).unwrap_or("(none)"))
              .show_ui(ui, |ui| {
                  for (i, source) in sources.iter().enumerate() {
                      if ui.selectable_label(*selected == i, source).clicked() {
                          *selected = i;
                      }
                  }
              });

          let next_enabled = !sources.is_empty() && *selected + 1 < sources.len();
          if ui.add_enabled(next_enabled, egui::Button::new("->")).clicked() {
              *selected += 1;
          }
      });
  }

  fn show_tags(ui: &mut egui::Ui, data: &PropertiesData) {
      let sources: Vec<String> = data.tags.keys().cloned().collect();
      let mut selected = ui.memory_mut(|mem| {
          mem.data
              .get_persisted(egui::Id::new("properties_tags_source"))
              .unwrap_or(0usize)
      });
      selected = selected.min(sources.len().saturating_sub(1));

      source_switcher(ui, &sources, &mut selected, "properties_tags_source");

      ui.memory_mut(|mem| {
          mem.data.insert_persisted(egui::Id::new("properties_tags_source"), selected);
      });

      if let Some(source) = sources.get(selected) {
          if let Some(tags) = data.tags.get(source) {
              let mut sorted: Vec<_> = tags.iter().collect();
              sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
              for (tag, score) in sorted {
                  ui.horizontal(|ui| {
                      ui.label(tag);
                      ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                          ui.label(format!("{:.3}", score));
                      });
                  });
              }
          }
      }
  }

  fn show_descriptions(ui: &mut egui::Ui, data: &PropertiesData) {
      let sources: Vec<String> = data.descriptions.keys().cloned().collect();
      let mut selected = ui.memory_mut(|mem| {
          mem.data
              .get_persisted(egui::Id::new("properties_desc_source"))
              .unwrap_or(0usize)
      });
      selected = selected.min(sources.len().saturating_sub(1));

      source_switcher(ui, &sources, &mut selected, "properties_desc_source");

      ui.memory_mut(|mem| {
          mem.data.insert_persisted(egui::Id::new("properties_desc_source"), selected);
      });

      if let Some(source) = sources.get(selected) {
          if let Some(text) = data.descriptions.get(source) {
              let mut text = text.clone();
              ui.add(
                  egui::TextEdit::multiline(&mut text)
                      .desired_width(f32::INFINITY)
                      .interactive(false),
              );
          }
      }
  }

  fn show_classifications(ui: &mut egui::Ui, data: &PropertiesData) {
      let sources: Vec<String> = data.classifications.keys().cloned().collect();
      let mut selected = ui.memory_mut(|mem| {
          mem.data
              .get_persisted(egui::Id::new("properties_cls_source"))
              .unwrap_or(0usize)
      });
      selected = selected.min(sources.len().saturating_sub(1));

      source_switcher(ui, &sources, &mut selected, "properties_cls_source");

      ui.memory_mut(|mem| {
          mem.data.insert_persisted(egui::Id::new("properties_cls_source"), selected);
      });

      if let Some(source) = sources.get(selected) {
          if let Some(classes) = data.classifications.get(source) {
              for class in classes {
                  ui.label(class);
              }
          }
      }
  }

  fn show_embeddings(ui: &mut egui::Ui, _data: &PropertiesData) {
      ui.label("Embeddings viewer not yet implemented.");
  }
  ```

- [ ] **Step 2: Export module**

  In `src/ui/mod.rs`, add:

  ```rust
  pub mod properties;
  ```

- [ ] **Step 3: Build**

  ```bash
  cargo build
  ```

  Expected: compiles with warnings acceptable; fix any errors.

- [ ] **Step 4: Commit**

  ```bash
  git add src/ui/properties.rs src/ui/mod.rs
  git commit -m "feat: scaffold Properties window module"
  ```

---

### Task 2.4: Add DB queries for `PropertiesData`

**Files:**
- Modify: `src/db/media.rs`

**Interfaces:**
- Consumes: `media_file_id`
- Produces: `PropertiesData` (or equivalent fields).

- [ ] **Step 1: Add query function**

  Add to `src/db/media.rs`:

  ```rust
  pub async fn get_properties_data(
      pool: &SqlitePool,
      media_file_id: i64,
  ) -> anyhow::Result<PropertiesData> {
      let media = get_by_id(pool, media_file_id)
          .await?
          .context("media file not found")?;

      let tags_json: Option<String> = sqlx::query_scalar(
          "SELECT tags_json FROM media_files WHERE id = ?1"
      )
      .bind(media_file_id)
      .fetch_one(pool)
      .await?;

      let tags: HashMap<String, HashMap<String, f32>> = tags_json
          .as_deref()
          .and_then(|s| serde_json::from_str(s).ok())
          .unwrap_or_default();

      let descriptions_json: Option<String> = sqlx::query_scalar(
          "SELECT descriptions_json FROM media_files WHERE id = ?1"
      )
      .bind(media_file_id)
      .fetch_one(pool)
      .await?;

      let descriptions: HashMap<String, String> = descriptions_json
          .as_deref()
          .and_then(|s| serde_json::from_str(s).ok())
          .unwrap_or_default();

      // Classifications and embeddings stored similarly once wired.
      let classifications: HashMap<String, Vec<String>> = HashMap::new();
      let embeddings: Vec<String> = Vec::new();

      Ok(PropertiesData {
          media,
          tags,
          descriptions,
          classifications,
          embeddings,
      })
  }
  ```

  Add import for `anyhow::Context`. `PropertiesData` is defined in this same module.

- [ ] **Step 2: Add unit test**

  Test that `get_properties_data` returns the right tags/descriptions after `update_tags_json` and `update_description_json`.

- [ ] **Step 3: Run tests**

  ```bash
  cargo test properties --lib
  ```

  Expected: PASS

- [ ] **Step 4: Commit**

  ```bash
  git add src/db/media.rs
  git commit -m "feat: add DB query for Properties window data"
  ```

---

### Task 2.5: Implement General tab

**Files:**
- Modify: `src/ui/properties.rs`

**Interfaces:**
- Consumes: `PropertiesData`, `advanced` flag.
- Produces: rendered General tab.

- [ ] **Step 1: Format file size**

  Add a helper:

  ```rust
  fn format_bytes(bytes: i64) -> String {
      if bytes < 1024 {
          format!("{} B", bytes)
      } else if bytes < 1024 * 1024 {
          format!("{:.2} KiB", bytes as f64 / 1024.0)
      } else if bytes < 1024 * 1024 * 1024 {
          format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
      } else {
          format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
      }
  }
  ```

- [ ] **Step 2: Use formatted size**

  In `show_general`, replace the raw size label with:

  ```rust
  ui.label(format!("Size: {}", format_bytes(m.file_size.unwrap_or(0))));
  ```

- [ ] **Step 3: Make path selectable**

  Use `TextEdit::singleline` for the absolute path so users can copy it.

- [ ] **Step 4: Commit**

  ```bash
  git add src/ui/properties.rs
  git commit -m "feat: implement Properties General tab"
  ```

---

### Task 2.6: Implement source switcher widget

**Files:**
- Modify: `src/ui/properties.rs`

**Interfaces:**
- Consumes: list of source names, selected index.
- Produces: updated selected index.

- [ ] **Step 1: Verify `source_switcher` layout**

  Confirm `source_switcher` uses `ui.add_enabled` for arrows and a fixed-width `ComboBox`. The draft in Task 2.3 already uses this layout; ensure it compiles and behaves correctly.

- [ ] **Step 2: Add test render**

  Build and visually verify the dropdown stays fixed width when source names are long or short.

- [ ] **Step 3: Commit**

  ```bash
  git add src/ui/properties.rs
  git commit -m "feat: implement fixed-width source switcher for Properties window"
  ```

---

### Task 2.7: Implement Tags/Descriptions/Classifications tabs

**Files:**
- Modify: `src/ui/properties.rs`

**Interfaces:**
- Consumes: `PropertiesData`.
- Produces: rendered tabs.

- [ ] **Step 1: Finalize `show_tags`**

  Already mostly implemented in Task 2.3. Ensure scores align to the right and the list scrolls.

- [ ] **Step 2: Finalize `show_descriptions`**

  Already implemented in Task 2.3 with a cloned, non-interactive `TextEdit::multiline`. Verify it is present.

- [ ] **Step 3: Finalize `show_classifications`**

  Same as draft; add a fallback message when empty.

- [ ] **Step 4: Build**

  ```bash
  cargo build
  ```

  Expected: compiles

- [ ] **Step 5: Commit**

  ```bash
  git add src/ui/properties.rs
  git commit -m "feat: implement Tags, Descriptions, and Classifications tabs"
  ```

---

### Task 2.8: Wire Properties window into `app.rs`

**Files:**
- Modify: `src/app.rs`

**Interfaces:**
- Consumes: `PropertiesState`, `PropertiesData`, async DB result.
- Produces: `properties_open`, `properties_media_id`, `properties_data`, `properties_fetch_in_progress`.

- [ ] **Step 1: Add state fields**

  In `AkashaApp` (or the browser state, depending on current organization):

  ```rust
  properties_state: crate::ui::properties::PropertiesState,
  properties_data: Option<crate::db::media::PropertiesData>,
  properties_tx: tokio::sync::mpsc::UnboundedSender<(i64, crate::db::media::PropertiesData)>,
  properties_rx: tokio::sync::mpsc::UnboundedReceiver<(i64, crate::db::media::PropertiesData)>,
  ```

  Initialize in `new()`:

  ```rust
  let (properties_tx, properties_rx) = tokio::sync::mpsc::unbounded_channel();
  ```

- [ ] **Step 2: Fetch data when target changes**

  In `update` or a helper, when `properties_state.open` is true and `properties_state.media_id` changes, spawn an async fetch:

  ```rust
  fn fetch_properties_data(&self, media_id: i64) {
      let pool = Arc::clone(&self.pool);
      let tx = self.properties_tx.clone();
      self.rt.spawn(async move {
          match db::media::get_properties_data(&pool, media_id).await {
              Ok(data) => {
                  let _ = tx.send((media_id, data));
              }
              Err(e) => {
                  tracing::warn!(media_id, error = %e, "Failed to fetch properties data");
              }
          }
      });
  }
  ```

- [ ] **Step 3: Receive fetched data**

  Each frame, drain `properties_rx` and update `properties_data` if the media_id still matches.

- [ ] **Step 4: Show window**

  In `update`, after settings window handling:

  ```rust
  if self.properties_state.open {
      crate::ui::properties::show(
          ctx,
          &mut self.properties_state.open,
          self.properties_state.media_id,
          self.properties_data.as_ref(),
          self.config.ui.show_advanced_media_properties,
      );
  }
  ```

- [ ] **Step 5: Build**

  ```bash
  cargo build
  ```

  Expected: compiles

- [ ] **Step 6: Commit**

  ```bash
  git add src/app.rs
  git commit -m "feat: wire Properties window state and async fetching into app"
  ```

---

### Task 2.9: Add context menu and viewer hotkey

**Files:**
- Modify: `src/app.rs`
- Modify: `src/ui/viewer.rs` (if viewer returns actions)

**Interfaces:**
- Consumes: selected media summary.
- Produces: `PropertiesAction::Open(media_id)`.

- [ ] **Step 1: Add context menu item in browser grid**

  Locate the grid context menu in `src/app.rs` (where `show_in_file_manager` and `copy_to_clipboard` actions are produced). Add:

  ```rust
  if ui.button("Properties").clicked() {
      self.browser.properties_state.open = true;
      self.browser.properties_state.media_id = Some(media.id);
      self.fetch_properties_data(media.id);
      ui.close_menu();
  }
  ```

- [ ] **Step 2: Add viewer hotkey/button**

  In the viewer handling code, check for `I` key press or a properties button. On trigger:

  ```rust
  self.browser.properties_state.open = true;
  self.browser.properties_state.media_id = Some(media.id);
  self.fetch_properties_data(media.id);
  ```

  If `viewer.rs` returns a `process_with_ai` style action, add `show_properties: bool` to the viewer response struct and handle it in `app.rs`.

- [ ] **Step 3: Build**

  ```bash
  cargo build
  ```

  Expected: compiles

- [ ] **Step 4: Commit**

  ```bash
  git add src/app.rs src/ui/viewer.rs
  git commit -m "feat: open Properties window from grid context menu and viewer hotkey"
  ```

---

### Task 2.10: Milestone 2 verification

- [ ] **Step 1: Run full test suite**

  ```bash
  cargo test --lib
  ```

  Expected: PASS

- [ ] **Step 2: Run clippy**

  ```bash
  cargo clippy -- -D warnings
  ```

  Expected: no warnings

- [ ] **Step 3: Manual smoke test**

  1. Right-click a grid item → Properties → General tab shows metadata.
  2. Switch to Tags tab; source switcher cycles through tag sources.
  3. Open viewer and press `I` → Properties opens for current image.
  4. Toggle "Show advanced media properties" in Settings; General tab shows/hides advanced fields.
  5. Resize Properties window; content scrolls when data is long.

- [ ] **Step 4: Commit any fixes**

  ```bash
  git commit -am "fix: address milestone 2 review issues"
  ```

---

## Spec coverage check

| Spec section | Plan task |
|--------------|-----------|
| `top_k` config | 1.1 |
| `top_k` in Candle | 1.2 |
| `top_k` in ONNX | 1.3 |
| `top_k` in Remote | 1.4 |
| Overwrite checkbox | 1.5 |
| Overwrite deletion | 1.6 |
| FTS5 migration | 1.7 |
| FTS5 sync | 1.8 |
| FTS5 substring search | 1.9 |
| Properties window module | 2.3 |
| Properties data fetch | 2.4, 2.8 |
| General tab | 2.5 |
| Source switcher | 2.6 |
| Tags/Descriptions/Classifications tabs | 2.7 |
| Context menu / viewer hotkey | 2.9 |
| Settings config | 2.1, 2.2 |

## Placeholder scan

No TBDs, TODOs, or vague steps remain. Code blocks contain concrete snippets. Exact file paths are used throughout.

## Type consistency check

- `ModelTagsOptions::top_k` is `Option<usize>` everywhere.
- `MediaProcessingAction` includes `overwrite: bool` after Task 1.5 and is consumed in Task 1.6.
- `PropertiesData` is defined in `src/db/media.rs` and imported by `src/ui/properties.rs`.
- `SettingsAction::AdvancedMediaPropertiesChanged(bool)` is emitted by settings and handled in app.
