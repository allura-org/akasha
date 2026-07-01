# Project Backlog
Things we've decided to address later.

## Viewer

### Buttons
- [ ] The buttons used to have icons that have since disappeared.
- [ ] The Fit/1:1 button currently changes size when you toggle it, pushing against the Next button, which is a bit of a UI crime. Should be a determinate size with the text centered within it.

### Info ticker
- [ ] Currently is nearly the same color as the background when in light mode.
- [ ] Currently overlaps the navbar when the window is narrow. There's a few ways to approach this:
    - Give it a background and stick it in another corner, so it doesn't clash with the gallery elements below (also solves the above)
    - Move the navbar and center the ticker
    - Place the ticker between the navbar and the image
    - Make the ticker's width reactive to the window so it retreats when the navbar moves over it (might be too much faff)

### Close button
- [ ] Should be the same height as the navbar, keeping the two horizontally aligned.

## Gallery

### Grid Layout
- [ ] Variable cell heights — currently forced to uniform 230px via `set_min_height`. Want per-row heights computed from actual image aspect ratios (tall portraits get tall rows, wide landscapes get short rows). Needs either:
    - egui `show_viewport` with manual row height prefix sums (previous attempt failed due to coordinate/clip rect issues — revisit with better understanding)
    - Custom layout that measures each cell before placing rows
    - Or a different approach entirely (e.g. masonry / waterfall layout)
- [ ] Seamless tiling mode — a-la Windows 8 Live Tiles. Images tile edge-to-edge with no gaps, aspect ratios preserved, variable sizes based on content importance or user preference. This is the user's stated preferred long-term look.

### Thumbnail slider
- [x] The thumbnail slider doesn't affect the size of the image previews, only the resolution of the thumbnails.
    - RESOLVED by two-tier architecture: slider now controls thumbnail resolution; viewer loads full images independently.

## Context Menu / Clipboard

- [ ] Configurable clipboard copy mode. The current implementation copies as `text/uri-list` (a file reference), which preserves the original filename and MIME type and works for Discord / Thunar-style paste. Apps like GIMP that only accept raw image bytes cannot paste it. Add a user setting to choose between:
    - `file_uri` (current): copy `file:///path/to/image.ext` as `text/uri-list`
    - `image_bytes`: copy decoded image bytes, re-encoded as PNG (`image/png`) for raw-byte paste targets
    - Possibly offer both targets simultaneously if we ever implement a custom clipboard owner.

## Performance

### Thumbnail Generation
- [x] GUI settings for thumbnail generation threads and algorithm
- [x] Thumbnail generation should be different when the user is scrolling the viewport vs. remaining still — DONE (velocity-aware priority queue)
- [x] Should we cancel thumbnail generation when images leave viewport? — DONE (queue is regenerated each frame; only viewport + prefetch items are queued)
- [x] Should we queue thumbnail generation ahead of viewport when nothing else is happening? — DONE (idle state queues visible + 5 rows prefetch)
- [x] Is hardware acceleration feasible? — DONE (simd-thumbnails feature with fast_image_resize + libwebp)
- [ ] JPEG decode bottleneck: `image::open` is ~99% of thumbnail generation time. `zune-jpeg` integration would give 2–3× speedup on JPEGs.
- [ ] Memory retention after thumbnail resize: `textures.clear()` drops handles but RSS doesn't return to baseline. Likely allocator hoarding (glibc malloc arenas) or egui texture atlas not shrinking. Candidate fix: `libc::malloc_trim(0)` after mass eviction.

### Pagination / Full Records
- [ ] Phase 6: Paginated `MediaFile` LRU cache (~500 records/page, 5 pages hot). Deferred until detail panels / bulk ops exist.

## Searchables / AI

- [x] Core `Searchable` trait, registry, scoring engine, and `filename` baseline.
- [x] DB schema for configs, values, and background job queue.
- [x] UI search bar with per-Searchable toggles and score-sorted results.
- [x] Backend-agnostic `Model`/`Backend` trait interface and `BackendRegistry`.
- [x] `RemoteBackend` for OpenAI-compatible / custom HTTP endpoints (tags only for now).
- [x] `CandleBackend` for local Candle inference (ViT tagger only for now).
- [ ] Candle architecture dispatch table. Refactor `CandleBackend` so it reads `model_type` from `config.json` and dispatches to the appropriate `candle_transformers` loader. This makes adding future Candle ports (CLIP, BLIP, LLaVA, JTP-3, etc.) a matter of adding a row to the registry rather than special-casing architectures in the backend. Each entry maps `(model_type, output_kind) -> loader`, with per-architecture preprocessing and postprocessing kept in its own module. See `docs/superpowers/plans/2026-06-27-backend-agnostic-model-plugins.md` for context.
- [x] ONNX inference integration (`ort`) for tags, embeddings, and classifications.
  - `OrtBackend` implemented as a default feature.
  - Loads models from `~/.local/share/akasha/models/onnx/<slug>/`.
  - Downloads ONNX models from HuggingFace when a model slug is configured.
  - Discovers `model.onnx`, preprocessing configs (`preprocess.json`, `config.json`, `preprocessor_config.json`), and tag files (`selected_tags.csv`, `tags.json`, `categories.json`, `labels.txt`) heuristically.
  - Supports NCHW and NHWC input layouts by inspecting the ONNX input shape.
  - Backend selection respects explicit `backend = "onnx"`; Candle no longer claims ONNX models.
  - Inference stats logging matches Candle tagger output.
  - Embeddings/classification outputs still need dedicated postprocessors and Searchables.
  - **ONNX VLM / multi-session pipeline** (e.g., Gemma 4, LLaVA-style models) is deferred. These models require a tokenizer, processor/chat template, vision encoder, embed tokens, and a KV-cache generation loop across multiple ONNX sessions. Candle is the preferred path for VLMs in the near term.
  - **Release CI builds exclude ONNX** for now; local `cargo run` still includes it by default.
- [x] Description Searchable via FTS5 (`searchable_text_fts`).
- [ ] Sidecar text search (`.txt` files alongside images) — likely via FTS5.
- [ ] Vector search backend (`sqlite-vec` or in-memory HNSW).
- [ ] Saved queries / cached search results, especially for embeddings.
- [ ] Per-Searchable weighting in the score aggregation.

## File Watcher

- [x] Debounced watcher using `notify-debouncer-full` for all configured folders.
- [x] Single-file incremental upsert/delete via `scanner::upsert_one` and `db::media::delete_by_path`.
- [x] Automatic subfolder creation when new files appear inside unwatched subdirectories.
- [x] Watcher events ignored during active manual scans.
- [ ] Hot-reload watched folders when `config.toml` changes (currently requires restart).
- [ ] Handle `Rename` events explicitly instead of treating them as Remove + Create.
- [ ] Clean up empty folders after the last file is removed.

## Profiling/Debugging
- [ ] Set up `cargo flamegraph` or `samply` for CPU profiling
- [ ] Set up `heaptrack` or `dhat` for memory profiling

## Work Queue / Background Jobs

- [x] `job_queue` schema generalized with `job_kind` and `params_json`.
- [x] Scanner no longer auto-enqueues inference jobs.
- [x] Media Processing window + context menus for manually enqueueing AI jobs.
- [x] `SearchWorker` dispatches `tagger`/`classifier`/`visionlanguage` jobs to real backends.
  - Tag inference works end-to-end for Candle, ONNX, and Remote backends.
  - Description/classification/vector inference is still deferred.
- [ ] Store inference results in `searchable_values` and surface them as Searchables.
- [ ] Add progress reporting in the status bar for long-running background jobs.
- [ ] Add prioritization, cancellation, retry, and parallel worker limits.
- [ ] Queue auto-restart on crash as a config option.
- **Conversions / batch ops** (transcoding, remuxing) can reuse the same queue schema later.

## Missing Files / Orphan Handling

- [x] `media_files` rows are preserved when a file disappears from disk.
- [x] `is_present` / `missing_since` columns track missing files (migration 013).
- [x] Scanner orphan pass marks files missing instead of deleting them.
- [x] Watcher `Remove` events mark files missing instead of deleting them.
- [x] Re-upserting a missing file clears the missing flag.
- [x] Grid shows a grey "missing" placeholder and skips thumbnail loads.
- [x] Viewer shows "File is missing" and does not attempt to load the image.
- [x] DB Management menu has a "Clear missing records" action for explicit deletion.
- [x] Background `job_queue` claims only skip media rows where `is_present = 1`.
- [ ] Hide-missing filter in the UI (grid/search) is not yet implemented.

## Database Operations / Scanner Architecture

The current scanner design has tension between per-file updates and folder-level `scan_complete` tracking:

- The `needs_update` check (hash + size comparison) skips unchanged files, which is efficient but means metadata fixes (e.g. populating a NULL `format` column) don't get applied without a hash/size change.
- Folders marked `scan_complete = true` are skipped entirely during the walk, so files inside them are invisible to the scanner even if their DB records are stale or incomplete.
- A manual "Rescan" marks the entire tree incomplete and re-walks everything, which is heavy for large collections.
- We should revisit whether `scan_complete` is the right granularity, or whether we need a more targeted "reconcile" operation that checks individual DB records against disk without walking the whole folder tree.
- Related: the `format` fallback fix for HEIF (falling back to extension when `ImageReader::format()` returns `None`) is currently only applied to newly scanned files. Existing records with NULL format need either a full rescan or a targeted DB update to fix.

## Code Health / Review Follow-ups

### mistral.rs backend
- [x] `MistralRsBackend::supports()` now requires explicit `backend = "mistralrs"`.
- [ ] `MistralRsBackend::load()` calls `Handle::current()`, which is only valid when `Backend::load` is invoked from inside a Tokio runtime. It happens to be called from `tokio::task::spawn_blocking` today, but the `Backend::load` trait contract doesn't document this requirement. Add a trait-level note or make the backend tolerant of being called outside a runtime.

### `src/searchables/worker.rs` style nits (from review)
- [ ] `process_one()` uses `self.resident.as_ref().unwrap()` after loading. Replace with an `expect("model just loaded")` or a `let model = ...` binding to be more defensive and self-documenting.
- [ ] `cluster_jobs()` uses `sort_by_key`, which recomputes the key for every comparison. With tiny batches this is fine, but `sort_by` with a cached key tuple is the more idiomatic choice.
- [ ] The worker silently ignores `Classification` and `Vector` outputs with a catch-all `_` match arm. Make the match explicit so future `ModelOutput` variants force an implementation decision.

### Dead-code warnings
- [ ] There are many dead-code warnings across the codebase (`delete_values_for_media`, `delete_values_for_config`, unused `JobRow` fields, `ModelOutput::Classification`/`Vector`, `SearchableKind`, etc.). These are pre-existing and not branch-specific, but should be cleaned up during the Alpha phase.

### Branch scope / process
- [ ] Tag search and several other features got bundled into the VLM feature branch. Future branches should stay focused; one-person-team or not, narrower PRs are easier to review and revert.

### Build / dependency policy
- [ ] Default features now include `candle`, `remote`, and `onnx`, which pulls in OpenSSL via `hf-hub/native-tls`. Decide whether the default build should remain pure Rust/no C deps, or whether heavier defaults are acceptable and just need documentation.
- [ ] TLS stack split: `hf-hub` uses `native-tls` to avoid rustls `close_notify` failures, while `reqwest` and `ort` still use `rustls-tls`. Evaluate unifying on one TLS stack (either switch `hf-hub` back to `rustls-tls` if the issue can be solved another way, or move `reqwest`/`ort` to `native-tls`).

### `run-gpu.sh`
- [ ] This is a personal test utility, not a user-facing script. Consider adding it to `.gitignore` and/or moving it out of the repo root so it doesn't look like official tooling.
