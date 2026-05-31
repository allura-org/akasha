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

## Known Issues / Notes for Next Time

- The `delete_orphans_for_root` function still exists in `src/db/media.rs` but is no longer called by the scanner (replaced by per-folder cleanup). It also has a bug: it only matches direct children (`parent_id = ?1`), not all descendants via CTE. Could be removed or fixed.
- `AGENTS.md` is severely out of date — describes scanner/thumbnailer/browser/viewer as "Mostly TODO" and "placeholders". Needs a rewrite.
- `README.md` is empty.
- No tests exist yet.
- The user's collection is ~424k items across `gallery-dl` with many subfolders. Test with `test_imgs/` inside the project; do NOT browse user's home directory.
- Build reminder: `cargo build` (or `cargo run`) is required after adding migrations — `sqlx::migrate!()` embeds them at compile time.
