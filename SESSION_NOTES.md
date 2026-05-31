# Session Notes — 2026-05-30

## What We Built Today

Went from a basic scaffold to a functional image gallery with resume-capable scanning.

### Committed
- `6f7e1a1` — Add per-subfolder scan completion tracking and auto-resume

### Scanner / Resume
- **Per-subfolder completion tracking**: `scan_complete` boolean on `folders` table (migration 003)
- **Walker-level skip**: Already-complete subfolders are skipped entirely via `filter_entry`; no filesystem stat for files inside them
- **Directory stack**: `Vec<(PathBuf, folder_id)>` tracks directories being walked; marked complete when walker leaves them
- **Per-folder orphan cleanup**: `delete_orphans(pool, folder_id, paths)` per visited folder instead of root-wide — fixes deep-nested orphans and avoids deleting files in skipped complete subtrees
- **Startup resume**: Checks each configured folder individually; scans only incomplete/new ones; sends "Existing data loaded" if all complete
- **Manual rescan**: Resets entire tree (`update_scan_complete_recursive`) before scanning so nothing is skipped

### Previous sessions (already in main)
- Flat egui theme, toasts, settings panel
- Thumbnail loading capped at 8 concurrent, decode off UI thread, visibility-based loading
- Grid virtualization via `ScrollArea::show_rows()` (only visible rows rendered)
- Folder trees with `parent_id`, `show_recursive`, recursive collapsible sidebar
- SQLite WAL mode, 5s busy timeout, batch commits every 1k files with `yield_now()`
- Async refresh via channels (no `block_on` in UI thread)
- O(1) thumbnail queue via `HashSet<usize>` storing `(idx, hash)`
- `hard_reset` parameter on `refresh_media_async` to distinguish folder switch vs background refresh
- Viewer with zoom fit/1:1, info ticker, prev/next navigation

---

## Current State

| Feature | Status |
|---------|--------|
| Folder scanner (recursive, blacklist, hash, dimensions) | ✅ Done |
| Per-subfolder scan resume | ✅ Done |
| Thumbnail generation + cache | ✅ Done |
| Folder tree sidebar (collapsible, recursive toggle) | ✅ Done |
| Grid browser (virtualized, thumbnail slider) | ✅ Done |
| Single-image viewer (fit/1:1, prev/next, info ticker) | ✅ Done |
| Settings panel | ✅ Done |
| Toast notifications | ✅ Done |
| SQLite WAL + batch commits | ✅ Done |
| Config (TOML, XDG paths) | ✅ Done |
| Right-click context menu | ❌ Not started |
| Hotkeys / keyboard shortcuts | ❌ Not started |
| AI / Searchables (ONNX classifiers, embeddings, etc.) | ❌ Not started |
| File system watching (`notify`) | ❌ Not started |

---

## Backlog Files

- `viewer_and_gallery_tweaks.md` — UI polish items (button sizing, info ticker color, close button alignment, thumbnail slider behavior)
- `concept.md` — Long-term vision: AI classification with "Searchables" abstraction (ONNX models, classifiers, embeddings, VLMs)

---

## Next Session Plans (from user)

1. **Right-click context menu** on thumbnails and in folder tree
2. **Hotkeys / keyboard shortcuts** (navigation, zoom, etc.)
3. **Make Akasha stand out from the crowd** — whatever that means; probably unique/differentiating features beyond "yet another image gallery"

---

## Full Architectural Roadmap (from original plan)

### Where We Are
| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Scaffold, config, SQLite + migrations | ✅ Done |
| 2 | Folder scanner (walk, hash, insert) | ✅ Done |
| 3 | Thumbnailer + cache | ✅ Done |
| 4 | egui browser UI (folder tree + grid) | ✅ Done |
| 5 | Image viewer + keyboard nav | ✅ Done |
| 6 | Polish: theme, toasts, settings UI | ✅ Done |
| — | **MVP Complete** — usable gallery | ✅ Done |

### Post-MVP Phases
| Phase | Deliverable | Time Estimate |
|-------|-------------|---------------|
| 7 | `notify` file watcher, incremental updates | ~1 session |
| 8 | Searchables trait + ONNX scaffolding | ~1–2 sessions |
| 9 | Vector search (HNSW or sqlite-vss) + text search (FTS5) | ~1–2 sessions |
| 10 | Unified search UI | ~1 session |

### Phase 2: File Watcher & Live Updates
- Use `notify` crate (already in `Cargo.toml`) to watch configured folders
- Debounce events (~500ms)
- On `Create`/`Modify`/`Remove`, update DB incrementally instead of full re-scans
- Show "Watching" / "Scanning..." in the status bar

### Phase 3: The "Searchables" Abstraction (Core AI/Classification)
This is the main differentiator. Any model that takes image/video in and spits out something searchable is a Searchable.

**Trait (conceptual):**
```rust
pub trait Searchable: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SearchableKind; // Vector | Text | Tags | Classification
    fn process(&self, media: &MediaInput) -> anyhow::Result<SearchableValue>;
}

pub enum SearchableKind {
    Vector(usize),      // e.g., 512-dim CLIP embedding
    Text,               // free-form description
    Tags(Vec<String>),  // predefined vocabulary
    Classification { label: String, confidence: f32 },
}
```

**DB additions:**
- `searchable_configs` table: model path, kind, thresholds, enabled flag
- `searchables` table: `(media_file_id, searchable_config_id, value_json)`

**ONNX Integration:**
- Crate: `ort` (ONNX Runtime Rust bindings)
- Each Searchable loads its own `.onnx` model
- Inference runs in a bounded tokio task pool
- Background job queue: process new media through all enabled Searchables

**Search & Retrieval:**
- **Vector**: in-memory HNSW index (`hnsw` crate) rebuilt on startup, or `sqlite-vec` extension
- **Text/Tags/Classifications**: SQLite FTS5 virtual table
- **Unified search UI**: query hits FTS5 + vector similarity simultaneously, results blended
- Per-Searchable on/off toggles

### Phase 4: Extensibility Hooks
| Future Feature | Hook Point |
|----------------|------------|
| gallery-dl downloader | `Downloader` trait, background job queue |
| Remuxer/transcoder | `MediaProcessor` trait |
| Interaction API (REST/JSON-RPC) | `axum` or `jsonrpsee` server binary in same workspace |
| Plugin system (user-provided Searchables) | `libloading` or WASM sandbox |

### Open Questions (from original plan)
1. **egui vs iced**: We started with egui. If theming hits a wall, pivoting is localized to `src/ui/`.
2. **Vector search backend**: Evaluate `sqlite-vec` (loadable extension, single-file) vs. in-memory Rust HNSW crate. Contingent on sticking with SQLite.
3. **Video support**: MVP is image-only. Video thumbnails/frame extraction will need `ffmpeg` bindings; defer until Phase 3+.

### Backwards Compatibility Strategy
- All schema changes are additive migrations (`sqlx migrate add`)
- Never rename columns; add new ones with defaults
- Config TOML uses `serde` defaults for missing fields
- If breaking change is unavoidable, version the config file and provide a migration script

---

## Known Issues / Notes for Next Time

- The `delete_orphans_for_root` function still exists in `src/db/media.rs` but is no longer called by the scanner (replaced by per-folder cleanup). It also has a bug: it only matches direct children (`parent_id = ?1`), not all descendants via CTE. Could be removed or fixed.
- `AGENTS.md` was rewritten but should be kept in sync with future changes.
- `README.md` is empty.
- No tests exist yet.
- The user's collection is ~424k items across `gallery-dl` with many subfolders. Test with `test_imgs/` inside the project; do NOT browse user's home directory.
- Build reminder: `cargo build` (or `cargo run`) is required after adding migrations — `sqlx::migrate!()` embeds them at compile time.
