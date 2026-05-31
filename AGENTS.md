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
| Database | `sqlx` 0.8 (sqlite, runtime-tokio, chrono, json) |
| Migrations | `sqlx::migrate!()` (embedded in binary) |
| Serialization | `serde`, `serde_json`, `toml` |
| Image processing | `image` 0.25 (png, jpeg, webp, gif, bmp, tiff) |
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
# Debug build
cargo build

# Run
cargo run

# Release build
cargo build --release

# Tests (none exist yet, but this is the command)
cargo test
```

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
7. Send `ScanEvent::Complete("Existing data loaded", 0)` if nothing needs scanning.

---

## Code Organization

```
src/
  main.rs        — Entry point, tracing setup, config + DB + runtime bootstrap, eframe launch
  app.rs         — `AkashaApp` implements `eframe::App`; main UI orchestrator (~950 lines)
  config.rs      — TOML config with XDG paths; `UiConfig`, `ThumbnailConfig`, `FolderConfig`
  scanner.rs     — Directory scanning: walkdir traversal, hashing, dimensions, per-subfolder completion tracking
  thumbnailer.rs — Thumbnail generation, resize, WebP encoding, cache path resolution (global/per-folder/custom)
  theme.rs       — Custom flat egui theme
  db/
    mod.rs       — `init_pool()` creates SQLite pool (WAL mode) and runs migrations
    folder.rs    — Folder CRUD: `list_all`, `list_roots`, `list_children`, `get_by_path`, `get_or_create`, `insert`, `update_scan_complete`, `update_scan_complete_recursive`, `update_show_recursive`
    media.rs     — Media file CRUD: `list_by_folder`, `list_by_folder_recursive`, `upsert`, `delete_orphans`, `delete_orphans_for_root`
  ui/
    mod.rs       — Re-exports `browser`, `viewer`, `widgets`
    browser.rs   — `BrowserPanel` placeholder (unused; browser UI is inline in `app.rs`)
    viewer.rs    — Full-screen viewer overlay: zoom fit/1:1, prev/next, info ticker, keyboard shortcuts (Escape, ArrowLeft, ArrowRight)
    widgets.rs   — Shared UI helpers (currently a single placeholder fn)
```

### Module Relationships
- `main.rs` depends on all top-level modules.
- `app.rs` is the orchestrator: holds config, DB pool, Tokio runtime, thumbnailer, and all UI state.
- `db::folder` and `db::media` are the only DB access layers.
- `scanner` and `thumbnailer` are called from async tasks spawned by `app.rs`.
- `ui::viewer` is a pure function called from `app.rs` viewer state; `ui::browser` is unused.

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
- Indexes: `idx_media_hash` (blake3_hash), `idx_media_folder` (folder_id)

### Notes
- `blacklist` is stored as a JSON string and deserialized via `serde_json`.
- `media_files` uses `UPSERT` (`ON CONFLICT ... DO UPDATE SET`) in `db::media::upsert`.
- Orphan cleanup uses `json_each()` for batch path comparison.
- Recursive CTEs are used for tree queries (e.g., `list_by_folder_recursive`, `update_scan_complete_recursive`).

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

## Testing Instructions

There are currently no tests in the repository. When adding tests:

- Use `cargo test` to run unit and integration tests.
- For DB-dependent tests, consider using an in-memory SQLite database (`:memory:`) or a temporary file, and running migrations in test setup.
- The project uses `sqlx`, so `SQLX_OFFLINE` may be relevant if query macros are used in the future (currently raw SQL strings are used).

---

## Security Considerations

- The app is a local desktop application with no network server. All data stays on the local filesystem.
- File identity is verified with `blake3` hashes.
- SQLite queries use parameterized binding to prevent injection.
- `absolute_path` and `relative_path` are stored as plain text; ensure path canonicalization is applied before display or file access if untrusted input is ever introduced.
- The project depends on `notify` for future file-watching; ensure watcher paths are validated against the configured folder list to avoid unintended traversal.

---

## Known Gaps / TODOs

- `ui/browser.rs` — `BrowserPanel` is a placeholder; the actual browser UI (folder tree + grid) is inline in `app.rs`.
- `ui/widgets.rs` — only contains a placeholder label helper.
- File system watching (`notify`) is listed as a dependency but not yet integrated.
- AI/ONNX "Searchables" abstraction is described in `concept.md` but not yet present in code.
- `delete_orphans_for_root` in `db/media.rs` is no longer called (replaced by per-folder cleanup) and has a bug (only matches direct children, not all descendants).
- No tests exist yet.

---

## Useful Files for Agents

| File | Purpose |
|------|---------|
| `Cargo.toml` | Dependencies and package metadata |
| `migrations/*.sql` | Database schema evolution (source of truth) |
| `concept.md` | High-level product vision and planned features |
| `SESSION_NOTES.md` | Session-by-session progress and next-steps |
| `viewer_and_gallery_tweaks.md` | UI polish backlog |
| `src/config.rs` | Config serialization, defaults, and XDG paths |
| `src/db/media.rs` | Media file queries and `MediaFile` struct |
| `src/db/folder.rs` | Folder queries and `Folder` struct |
| `src/app.rs` | Central app state and `eframe::App` implementation |
| `src/scanner.rs` | Directory scanning with per-subfolder resume |
| `src/thumbnailer.rs` | Thumbnail generation and cache path resolution |
| `src/ui/viewer.rs` | Full-screen image viewer overlay |
