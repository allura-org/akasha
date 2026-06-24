# Akasha — Agent Guide

Akasha is a Linux-native, database-backed image gallery desktop application written in Rust. It is in early MVP stage and uses an immediate-mode GUI (egui via eframe). The intended audience is data hoarders who want fast local browsing, AI classification, and search over large image/video collections without moving files.

---

## Project Overview

- **Name:** akasha
- **Version:** 0.1.0
- **License:** GNU General Public License v3.0 (or later) — see `LICENSE`
- **Language:** Rust (edition 2024)
- **Platform:** Linux desktop (optimized for desktops, not homelabs or servers)
- **Architecture:** Single-threaded GUI (eframe/egui) with a multi-threaded Tokio runtime for async I/O and database work
- **Database:** SQLite (WAL mode, 5s busy timeout), managed via `sqlx` with embedded migrations
- **Config format:** TOML, human-readable, stored in XDG directories

### Key Goals (from `concept.md`)
- Keep media in-place; use hashes to avoid duplicates within the app
- Browse by folder tree (recursive or flat per-folder)
- Add folders to watch (recursive or not) with blacklist glob patterns
- Modular "Searchables" abstraction for AI classification (classifiers, embeddings, VLMs, etc.)
- Extensible, backwards-compatible schema

---

## Technology Stack

| Concern | Crate(s) |
|---------|----------|
| GUI framework | `eframe` 0.31, `egui` 0.31, `egui_extras` 0.31 |
| Async runtime | `tokio` (rt-multi-thread, macros, sync) |
| Async traits | `async-trait` 0.1 (used by the Searchables abstraction) |
| Database | `sqlx` 0.8 (sqlite, runtime-tokio, chrono, json) |
| Migrations | `sqlx::migrate!()` (embedded in binary) |
| Serialization | `serde`, `serde_json`, `toml` |
| Image processing | `image` 0.25 (png, jpeg, webp, gif, bmp, tiff); `libheif-rs` 2.7 (optional HEVC → HEIF/HEIC); `fast_image_resize` 6 + `webp` 0.3 (optional SIMD thumbnails) |
| File hashing | `blake3` |
| Directory traversal | `walkdir` |
| Glob blacklists | `globset` |
| Logging | `tracing`, `tracing-subscriber` (env-filter) |
| Error handling | `anyhow` |
| XDG directories | `directories` 6 |
| Date/time | `chrono` (with serde support) |
| File system watching (planned) | `notify` 8, `notify-debouncer-full` 0.5 |

---

## Build and Run

Standard Cargo workflow:

```bash
# Debug build (pure Rust, no C dependencies)
cargo build

# Run
cargo run

# Release build (pure Rust)
cargo build --release

# With SIMD thumbnail acceleration (requires libwebp-dev / libwebp-devel)
cargo build --release --features simd-thumbnails

# Tests (none exist yet, but this is the command)
cargo test
```

### Feature flags

- `hevc` — Enables HEVC-coded media. Currently covers HEIF/HEIC images; will extend to HEVC video in MP4 when video support lands. Requires system libraries:
  - `libheif-dev` >= 1.17.0 (and its dependency `libde265-dev` for HEVC decoding)
  - Install on Debian/Ubuntu: `sudo apt install libheif-dev libde265-dev`
  - Build with the feature: `cargo build --features hevc`
  - If the libraries are missing, disable the feature: `cargo build --no-default-features` (or remove `hevc` from `default` in `Cargo.toml`)
- `simd-thumbnails` — Enables SIMD-optimized thumbnail generation via `fast_image_resize` (AVX2/NEON) and `libwebp`. Opt-in; enable with `cargo build --features simd-thumbnails`. Requires `libwebp-dev` (Debian/Ubuntu) or `libwebp-devel` (Fedora).

**Important:** `sqlx::migrate!()` embeds migrations at compile time. After adding a new migration file, you **must** rebuild (`cargo build` / `cargo run`) before the migration will be applied.

### Runtime Startup Flow
1. Initialize `tracing` subscriber at `INFO` level.
2. Load `Config` from TOML (or create defaults and persist them).
3. Create a Tokio runtime.
4. Initialize the SQLite pool and run migrations via `sqlx::migrate!()`.
5. Launch the native eframe window (`1280x800`, titled "Akasha").
6. On startup, check each configured folder:
   - If missing or `scan_complete = false` → scan it
   - If `scan_complete = true` → skip
7. During a scan, the scanner uses an **mtime + size fast path** to skip unchanged files without re-hashing them. Only files whose `modified_at` or `file_size` differ from the DB record (or whose `format` is missing) are hashed and upserted.
8. Changed files are written to the DB in **batches of 500** wrapped in explicit transactions, rather than one implicit transaction per file.
9. Send `ScanEvent::Complete("Existing data loaded", 0)` if nothing needs scanning.
10. Start the filesystem watcher (`watcher::spawn`) for every configured folder. Watcher events are ignored while a manual scan is in flight to avoid races.
11. Start the background Searchables worker (`SearchWorker`) polling `job_queue`.

---

## Code Organization

```
src/
  main.rs        — Entry point, tracing setup, config + DB + runtime bootstrap, eframe launch
  app.rs         — `AkashaApp` implements `eframe::App`; main UI orchestrator (~1000 lines). Uses a two-tier media list: `media_summaries` (lightweight, all items) for the grid + thumbnail queue, and `media_items` (paginated full records, reserved for future detail panels).
  config.rs      — TOML config with XDG paths; `UiConfig`, `ThumbnailConfig`, `FolderConfig`
  scanner.rs     — Directory scanning: walkdir traversal, hashing, dimensions, per-subfolder completion tracking
  searchables/   — Searchables abstraction: trait, registry, engine, built-in `filename` Searchable, background worker stub
  thumbnailer.rs — Thumbnail generation, resize, WebP encoding, cache path resolution (global/per-folder/custom, sharded 2-level hash prefix). SIMD pipeline via `fast_image_resize` + `libwebp` when `simd-thumbnails` feature is enabled.
  watcher.rs     — Filesystem watcher using `notify-debouncer-full`; emits batched Create/Modify/Remove events to `app.rs`
  theme.rs       — Custom flat egui theme
  db/
    mod.rs       — `init_pool()` creates SQLite pool (WAL mode) and runs migrations
    folder.rs    — Folder CRUD: `list_all`, `list_roots`, `list_children`, `get_by_path`, `get_or_create`, `insert`, `update_scan_complete`, `update_scan_complete_recursive`, `update_show_recursive`
    media.rs     — Media file CRUD: `MediaFile` (full record), `MediaSummary` (lightweight grid record), `list_by_folder`, `list_by_folder_recursive`, `list_summaries_by_folder` (streaming), `count_by_folder`, `get_by_id`, `list_page_by_folder`, `upsert`, `delete_orphans`, `search_summaries`
    searchable.rs — Searchable config/value CRUD and `job_queue` helpers
  ui/
    mod.rs       — Re-exports `browser`, `viewer`, `widgets`
    browser.rs   — `BrowserPanel` placeholder (unused; browser UI is inline in `app.rs`)
    viewer.rs    — Full-screen viewer overlay: zoom fit/1:1, prev/next, info ticker, keyboard shortcuts (Escape, ArrowLeft, ArrowRight)
    widgets.rs   — Shared UI helpers (currently a single placeholder fn)
```

### Module Relationships
- `main.rs` depends on all top-level modules.
- `app.rs` is the orchestrator: holds config, DB pool, Tokio runtime, thumbnailer, Searchables engine, and all UI state.
- `db::folder`, `db::media`, and `db::searchable` are the DB access layers.
- `scanner` and `thumbnailer` are called from async tasks spawned by `app.rs`.
- `searchables::SearchWorker` runs as a background tokio task started from `app.rs`.
- `watcher::spawn` is called from `app.rs` on startup; watcher events are polled each frame.
- `ui::viewer` is a pure function called from `app.rs` viewer state; `ui::browser` now owns the folder tree and search bar.

---

## Database Schema

Migrations live in `migrations/` and are embedded at compile time.

### `folders`
- `id`, `parent_id` (FK, self-referencing, cascade delete)
- `path` (unique, absolute)
- `recursive` (bool), `show_recursive` (bool)
- `scan_complete` (bool, DEFAULT 0) — per-subfolder completion tracking
- `blacklist` (JSON array string)
- `thumbnail_cache_mode` (optional string: 'disabled', 'global', 'per_folder', 'custom')
- `created_at`
- Index: `idx_folder_parent` on `parent_id`

### `media_files`
- `id`, `folder_id` (FK, cascade delete)
- `relative_path`, `absolute_path`
- `blake3_hash`, `width`, `height`, `format`, `file_size`, `modified_at`
- `created_at`
- Unique on `(folder_id, relative_path)`
- Indexes: `idx_media_hash` (blake3_hash), `idx_media_folder` (folder_id), `idx_media_modified_at` (modified_at), `idx_media_summary` (covering index for lightweight grid queries)

### `searchable_configs`
- `id`, `name` (unique), `kind` (`text` | `tags` | `vector` | `classification`)
- `enabled` (bool), `options` (JSON)
- Index: `idx_searchable_config_name`

### `searchable_values`
- `id`, `media_file_id` (FK → `media_files`, cascade delete), `searchable_config_id` (FK → `searchable_configs`, cascade delete)
- `value_json` — stores strings, string arrays, or float arrays depending on `kind`
- `created_at`, `updated_at`
- Unique on `(media_file_id, searchable_config_id)`
- Indexes: `idx_searchable_values_media`, `idx_searchable_values_config`

### `job_queue`
- `id`, `media_file_id` (FK), `searchable_config_id` (FK)
- `status` (`pending` | `running` | `done` | `failed`), `attempts`, `error`
- `created_at`, `updated_at`
- Index: `idx_job_queue_pending`

### Notes
- `blacklist` is stored as a JSON string and deserialized via `serde_json`.
- `media_files` uses `UPSERT` (`ON CONFLICT ... DO UPDATE SET`) in `db::media::upsert` and `scanner::flush_batch`.
- Orphan cleanup uses `json_each()` for batch path comparison.
- Recursive CTEs are used for tree queries (e.g., `list_by_folder_recursive`, `update_scan_complete_recursive`).
- Summary queries (`list_summaries_by_folder*`) stream rows incrementally via `sqlx::query_as().fetch()` rather than `.fetch_all()`, avoiding a massive allocation spike for large folders.
- Search results are hydrated with `search_summaries()`, which uses `json_each()` to match a batch of media IDs.
- The thumbnail cache uses a 2-level hash prefix (`aa/bb/{hash}_{size}.webp`) to avoid ext4/xfs metadata stress with hundreds of thousands of files.
- The bare-minimum Searchable is `filename` (kind `text`), seeded by migration `008_seed_filename_searchable.sql`.

---

## Configuration

Config path: `~/.config/akasha/config.toml`
Database path: `~/.local/share/akasha/akasha.db`
Cache path: `~/.cache/akasha/`

### Default Config
```toml
[ui]
theme = "dark"
thumbnail_size = 256

[thumbnails]
cache_mode = "global"   # "disabled" | "global" | "per_folder" | "custom"
custom_path = ""

# folders = []
```

Per-folder config can override `thumbnail_cache_mode`. Blacklists are glob patterns stored per-folder.

---

## Code Style Guidelines

- Follow standard Rust naming (`PascalCase` types, `snake_case` functions/variables, `SCREAMING_SNAKE_CASE` constants).
- Use `anyhow::Result` for fallible operations at the application/module boundary.
- Use `tracing::info!` (and appropriate levels) for operational logging.
- Use `sqlx::query` / `sqlx::query_as` with explicit parameter binding (`?1`, `?2`, ...).
- Convert between `u32`/`u64` and `i64` at the DB boundary (schema stores integers as SQLite `INTEGER`, which maps to `i64`).
- `Arc<SqlitePool>` is used to share the pool across async tasks.
- Keep DB logic in `db/` modules. Business logic (scanning, thumbnailing) stays at the crate root.

---

## Git Workflow

- **Commit often** — commits are useful for rolling back bad edits.
- **Do not push unless explicitly instructed** — pushes should be deferred until a feature, fix, or rework is actually done.

---

## Testing Instructions

- Use `cargo test` to run unit and integration tests.
- Existing tests live in `src/searchables/` and use an in-memory SQLite database (`sqlite::memory:`) with embedded migrations.
- For DB-dependent tests, run `sqlx::migrate!("./migrations").run(&pool)` in test setup.
- The project uses `sqlx`, so `SQLX_OFFLINE` may be relevant if query macros are used in the future (currently raw SQL strings are used).

---

## Security Considerations

- The app is a local desktop application with no network server. All data stays on the local filesystem.
- File identity is verified with `blake3` hashes.
- SQLite queries use parameterized binding to prevent injection.
- `absolute_path` and `relative_path` are stored as plain text; ensure path canonicalization is applied before display or file access if untrusted input is ever introduced.
- The project depends on `notify` for future file-watching; ensure watcher paths are validated against the configured folder list to avoid unintended traversal.

---

## Development Phases (from original plan)

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Scaffold Cargo project, config loading, SQLite + migrations | ✅ Complete |
| 2 | Folder scanner (walk, hash, insert) | ✅ Complete |
| 3 | Thumbnailer + cache | ✅ Complete |
| 4 | egui browser UI (folder tree + thumbnail grid) | ✅ Complete |
| 5 | Image viewer + keyboard nav | ✅ Complete |
| 6 | Polish: theme, error toasts, settings UI | ✅ Complete |
| — | **MVP Complete** — usable gallery | ✅ Complete |
| 7 | `notify` file watcher, incremental updates | ✅ Complete (debounced watcher, single-file upsert/delete, subfolder creation) |
| 8 | Searchables trait + filename baseline | ✅ Complete (trait, registry, and `filename` Searchable implemented; ONNX deferred) |
| 9 | Vector search (HNSW or sqlite-vss) + text search (FTS5) | ❌ Not started |
| 10 | Unified search UI | 🔄 In progress (search bar + scoring implemented; advanced blending/tuning deferred) |

The full original plan (database evaluation, Searchables trait definition, extensibility hooks, open questions) lives in `SESSION_NOTES.md` under "Full Architectural Roadmap".

## Known Gaps / TODOs

- `ui/widgets.rs` — only contains a placeholder label helper.
- ONNX inference for tags/embeddings/classifications is not yet implemented; `job_queue` and `SearchWorker` are stubs that mark jobs done.
- Vector search backend (`sqlite-vec` / HNSW) is not yet chosen or implemented.
- Text Searchables currently use `LIKE` queries; FTS5 can be added later for descriptions/sidecars.
- Watcher config is loaded once at startup; editing `config.toml` requires a restart to update watched folders.
- Cross-root file moves appear as a Remove + Create pair; no move deduplication.
- `delete_orphans_for_root` in `db/media.rs` is no longer called (replaced by per-folder cleanup) and has a bug (only matches direct children, not all descendants).
- **Paginated full records (Phase 6):** `media_items` in `app.rs` is currently empty. An LRU cache of `MediaFile` pages (~500 records/page, 5 pages hot) is planned for detail panels / bulk ops, but deferred until those features exist.
- **Thumbnail queue velocity tuning:** the scroll-velocity thresholds (60/240 rows/sec) are initial guesses and may need adjustment based on real-world feel.

---

## Useful Files for Agents

| File | Purpose |
|------|---------|
| `Cargo.toml` | Dependencies and package metadata |
| `migrations/*.sql` | Database schema evolution (source of truth) |
| `src/config.rs` | Config serialization, defaults, and XDG paths |
| `src/db/media.rs` | Media file queries and `MediaFile` struct |
| `src/db/folder.rs` | Folder queries and `Folder` struct |
| `src/app.rs` | Central app state and `eframe::App` implementation |
| `src/scanner.rs` | Directory scanning with per-subfolder resume |
| `src/searchables/mod.rs` | `Searchable` trait, kinds, and registry |
| `src/searchables/engine.rs` | Search orchestration and score aggregation |
| `src/searchables/filename.rs` | Built-in filename Searchable |
| `src/searchables/worker.rs` | Background `job_queue` worker stub |
| `src/db/searchable.rs` | Searchable config/value and job queue queries |
| `src/thumbnailer.rs` | Thumbnail generation and cache path resolution |
| `src/ui/browser.rs` | Folder tree, thumbnail grid, and search bar |
| `src/ui/viewer.rs` | Full-screen image viewer overlay |
| `src/watcher.rs` | Filesystem watcher and event classification |
| `generate_test_noise.py` | Helper script to create random noise PNGs for watcher testing |

### Scratchpad Folder (`.kimi/`)

Session notes, backlogs, and architectural documents live in `.kimi/` to keep the project root clean. Treat it as a working scratchpad:

| File | Purpose |
|------|---------|
| `.kimi/concept.md` | High-level product vision and planned features |
| `.kimi/SESSION_NOTES.md` | Session-by-session progress and next-steps |
| `.kimi/BACKLOG.md` | Deferred work and known issues |
| `.kimi/plan_pagination_thumbnails.md` | Original implementation plan for two-tier media list + priority thumbnails |
| `.kimi/review_pagination_thumbnails.md` | Backend reviewer's feedback on the plan |
| `.kimi/viewer_and_gallery_tweaks.md` | (Deprecated — contents merged into `BACKLOG.md`) |
