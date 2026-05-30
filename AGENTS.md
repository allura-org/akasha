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
- **Database:** SQLite, managed via `sqlx` with embedded migrations
- **Config format:** TOML, human-readable, stored in XDG directories

### Key Goals (from `concept.md`)
- Keep media in-place; use hashes to avoid duplicates within the app
- Browse by folder tree
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

### Runtime Startup Flow
1. Initialize `tracing` subscriber at `INFO` level.
2. Load `Config` from TOML (or create defaults and persist them).
3. Create a Tokio runtime.
4. Initialize the SQLite pool and run migrations via `sqlx::migrate!()`.
5. Launch the native eframe window (`1280x800`, titled "Akasha").

---

## Code Organization

```
src/
  main.rs        — Entry point, tracing setup, config + DB + runtime bootstrap, eframe launch
  app.rs         — `AkashaApp` implements `eframe::App`; owns `Config`, `Arc<Mutex<SqlitePool>>`, `Arc<Runtime>`
  config.rs      — TOML config with XDG paths; `UiConfig`, `ThumbnailConfig`, `FolderConfig`
  scanner.rs     — Directory scanning logic (hash, dimensions, format). Mostly TODO.
  thumbnailer.rs — Thumbnail generation and cache path resolution. Mostly TODO.
  db/
    mod.rs       — `init_pool()` creates SQLite pool and runs migrations
    folder.rs    — Folder CRUD: `list_all`, `insert`
    media.rs     — Media file CRUD: `list_by_folder`, `upsert`, `delete_orphans`
  ui/
    mod.rs       — Re-exports `browser`, `viewer`, `widgets`
    browser.rs   — `BrowserPanel` (folder tree / grid browser). Placeholder.
    viewer.rs    — `ViewerPanel` (single media view). Placeholder.
    widgets.rs   — Shared UI helpers (currently a single placeholder fn).
```

### Module Relationships
- `main.rs` depends on all top-level modules.
- `app.rs` is the orchestrator: it holds references to config, DB pool, and Tokio runtime so UI panels can spawn async DB work.
- `db::folder` and `db::media` are the only DB access layers.
- `scanner` and `thumbnailer` are intended to be called from async tasks (e.g., triggered by UI or file watchers).

---

## Database Schema

Migrations live in `migrations/001_initial.sql` and are embedded at compile time.

### `folders`
- `id`, `path` (unique), `recursive`, `blacklist` (JSON array string), `thumbnail_cache_mode`, `created_at`

### `media_files`
- `id`, `folder_id` (FK, cascade delete), `relative_path`, `absolute_path`, `blake3_hash`, `width`, `height`, `format`, `file_size`, `modified_at`, `created_at`
- Unique on `(folder_id, relative_path)`
- Indexes: `idx_media_hash` (blake3_hash), `idx_media_folder` (folder_id)

### Notes
- `blacklist` is stored as a JSON string and deserialized via `serde_json`.
- `media_files` uses `UPSERT` (`ON CONFLICT ... DO UPDATE SET`) in `db::media::upsert`.
- Orphan cleanup is done with `json_each()` for batch path comparison.

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
- `Arc<Mutex<SqlitePool>>` is used to share the pool across UI panels; the runtime is similarly `Arc`-wrapped.
- Keep DB logic in `db/` modules. Keep UI logic in `ui/` modules. Business logic (scanning, thumbnailing) stays at the crate root.

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

Many modules are still scaffolding:
- `scanner.rs` — directory walking, hashing, and metadata extraction are unimplemented.
- `thumbnailer.rs` — image resize and WebP encoding are unimplemented; `PerFolder` cache path resolution is stubbed.
- `ui/browser.rs` and `ui/viewer.rs` — UI panels are placeholders.
- `ui/widgets.rs` — only contains a placeholder label helper.
- File system watching (`notify`) is listed as a dependency but not yet integrated.
- AI/ONNX "Searchables" abstraction is described in `concept.md` but not yet present in code.

---

## Useful Files for Agents

| File | Purpose |
|------|---------|
| `Cargo.toml` | Dependencies and package metadata |
| `migrations/001_initial.sql` | Database schema (source of truth) |
| `concept.md` | High-level product vision and planned features |
| `src/config.rs` | Config serialization, defaults, and XDG paths |
| `src/db/media.rs` | Media file queries and `MediaFile` struct |
| `src/db/folder.rs` | Folder queries and `Folder` struct |
| `src/app.rs` | Central app state and `eframe::App` implementation |
