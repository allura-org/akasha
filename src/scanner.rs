use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

const SUPPORTED_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "avif",
];

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_lowercase();
            if SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
                return true;
            }
            #[cfg(feature = "hevc")]
            if matches!(ext.as_str(), "heif" | "heic") {
                return true;
            }
            false
        })
        .unwrap_or(false)
}

/// Returns true if `path` should be excluded based on the user's exclude list.
/// An entry that resolves to an absolute path performs exact-path matching;
/// otherwise the entry is treated as a substring matched against the full path.
fn is_excluded(path: &Path, patterns: &[String]) -> bool {
    for pattern in patterns {
        let candidate = std::path::Path::new(pattern);
        if candidate.is_absolute() {
            if path == candidate {
                return true;
            }
        } else {
            let path_str = path.to_string_lossy();
            if path_str.contains(pattern) {
                return true;
            }
        }
    }
    false
}

/// Returns true if `path` is allowed by the user's include list.
/// If the include list is empty, everything is allowed.
/// An entry that resolves to an absolute path performs exact-path matching;
/// otherwise the entry is treated as a substring matched against the full path.
fn is_included(path: &Path, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    for pattern in patterns {
        let candidate = std::path::Path::new(pattern);
        if candidate.is_absolute() {
            if path == candidate {
                return true;
            }
        } else {
            let path_str = path.to_string_lossy();
            if path_str.contains(pattern) {
                return true;
            }
        }
    }
    false
}

pub async fn scan_folder(
    pool: &sqlx::SqlitePool,
    root_folder_id: i64,
    folder_path: &Path,
    recursive: bool,
    exclude: &[String],
    include: &[String],
    progress_tx: Option<&std::sync::mpsc::Sender<crate::app::ScanEvent>>,
) -> anyhow::Result<usize> {
    info!(
        "Scanning folder: {} (recursive={}, exclude={:?}, include={:?})",
        folder_path.display(),
        recursive,
        exclude,
        include
    );

    let walker = if recursive {
        walkdir::WalkDir::new(folder_path)
    } else {
        walkdir::WalkDir::new(folder_path).max_depth(1)
    };

    // Map absolute path -> folder_id for directories we've seen
    let mut folder_ids: HashMap<std::path::PathBuf, i64> = HashMap::new();
    folder_ids.insert(folder_path.to_path_buf(), root_folder_id);

    // Query complete subfolders to skip during walk
    let complete_subfolders: std::collections::HashSet<std::path::PathBuf> =
        sqlx::query_scalar::<_, String>(
            "WITH RECURSIVE subtree(id) AS (
                SELECT ?1
                UNION ALL
                SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
            )
            SELECT path FROM folders WHERE scan_complete = 1 AND id IN (SELECT id FROM subtree) AND id != ?1"
        )
        .bind(root_folder_id)
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(|p| std::path::PathBuf::from(p))
        .collect();
    info!("Found {} complete subfolders to skip", complete_subfolders.len());

    // Bulk-load existing files for the ENTIRE tree under this root (recursive CTE)
    let existing: HashMap<String, (String, i64, Option<chrono::NaiveDateTime>, Option<String>)> = sqlx::query_as::<_, (String, String, i64, Option<chrono::NaiveDateTime>, Option<String>)>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
        )
        SELECT relative_path, blake3_hash, file_size, modified_at, format FROM media_files WHERE folder_id IN (SELECT id FROM subtree)"
    )
    .bind(root_folder_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|(path, hash, size, mtime, format)| (path, (hash, size, mtime, format)))
    .collect();

    let mut folder_paths: HashMap<i64, Vec<String>> = HashMap::new();
    let mut scanned_count = 0usize;
    let mut pending: Vec<PendingUpsert> = Vec::new();
    const UPSERT_BATCH_SIZE: usize = 500;

    // Track directories currently being walked; mark them complete when we leave them
    let mut dir_stack: Vec<(std::path::PathBuf, i64)> = Vec::new();
    let mut completed_dirs: Vec<i64> = Vec::new();

    for entry in walker.into_iter().filter_entry(|e| {
        if e.depth() == 0 {
            return true;
        }
        let path = e.path();
        if is_excluded(path, exclude) {
            return false;
        }
        if !is_included(path, include) {
            return false;
        }
        if e.file_type().is_dir() && complete_subfolders.contains(e.path()) {
            info!("Skipping complete subfolder: {}", e.path().display());
            return false;
        }
        true
    }) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Walkdir error: {}", e);
                continue;
            }
        };

        let path = entry.path();

        // Pop directories we've left
        while let Some((top_path, _)) = dir_stack.last() {
            if path.starts_with(top_path) {
                break;
            }
            let (_, id) = dir_stack.pop().unwrap();
            completed_dirs.push(id);
        }

        // Flush completions periodically
        if completed_dirs.len() >= 10 {
            let count = completed_dirs.len();
            for id in completed_dirs.drain(..) {
                let _ = crate::db::folder::update_scan_complete(pool, id, true).await;
            }
            info!("Marked {} directories complete during scan", count);
        }

        if path.is_dir() {
            if recursive && !folder_ids.contains_key(path) {
                let parent_path = path.parent().unwrap_or(folder_path);
                let parent_id = folder_ids.get(parent_path).copied();
                match crate::db::folder::get_or_create(
                    pool,
                    parent_id,
                    &path.to_string_lossy(),
                    false,       // subfolders are non-recursive by default
                    false,       // not known to be complete until walker leaves it
                    exclude,
                    include,
                    None,        // inherit cache mode
                    None,        // inherit cache folder
                    "disable",   // inherit fallback
                )
                .await
                {
                    Ok(id) => {
                        folder_ids.insert(path.to_path_buf(), id);
                    }
                    Err(e) => {
                        warn!("Failed to create folder entry for {}: {}", path.display(), e);
                        continue;
                    }
                }
            }
            if let Some(&id) = folder_ids.get(path) {
                if dir_stack.last().map(|(p, _)| p != path).unwrap_or(true) {
                    dir_stack.push((path.to_path_buf(), id));
                }
            }
            continue;
        }

        if !path.is_file() || !is_supported(path) {
            continue;
        }

        let relative_path = match path.strip_prefix(folder_path) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => {
                warn!("Could not compute relative path for {}", path.display());
                continue;
            }
        };

        // Determine which folder this file belongs to
        let parent_dir = path.parent().unwrap_or(folder_path);
        let folder_id = *folder_ids.get(parent_dir).unwrap_or(&root_folder_id);

        folder_paths.entry(folder_id).or_default().push(relative_path.clone());

        let absolute_path = path.to_string_lossy().to_string();
        let metadata = match tokio::fs::metadata(path).await {
            Ok(m) => m,
            Err(e) => {
                warn!("Failed to read metadata for {}: {}", path.display(), e);
                continue;
            }
        };

        let file_size = metadata.len();
        let modified_at = metadata
            .modified()
            .ok()
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
            })
            .flatten()
            .map(|dt| dt.naive_utc());

        // Check if file changed since last scan (in-memory, O(1))
        let needs_update = match existing.get(&relative_path) {
            Some((_old_hash, old_size, old_modified_at, old_format)) => {
                let size_matches = *old_size as u64 == file_size;
                let mtime_matches = *old_modified_at == modified_at;
                !(size_matches && mtime_matches && old_format.is_some())
            }
            None => true,
        };

        if !needs_update {
            debug!("Skipping unchanged file: {}", relative_path);
            continue;
        }

        let path_buf = path.to_path_buf();
        let (hash, width, height, format) = match tokio::task::spawn_blocking(move || {
            process_file(&path_buf)
        })
        .await
        {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                warn!("Failed to process {}: {}", path.display(), e);
                continue;
            }
            Err(e) => {
                warn!("Task panicked for {}: {}", path.display(), e);
                continue;
            }
        };

        pending.push(PendingUpsert {
            folder_id,
            relative_path,
            absolute_path,
            hash,
            width,
            height,
            format,
            file_size,
            modified_at,
        });

        scanned_count += 1;

        // Send progress every 5000 files
        if scanned_count % 5000 == 0 {
            if let Some(tx) = progress_tx {
                let _ = tx.send(crate::app::ScanEvent::Progress(
                    folder_path.to_string_lossy().to_string(),
                    scanned_count,
                ));
            }
        }

        // Flush batch when full
        if pending.len() >= UPSERT_BATCH_SIZE {
            flush_batch(pool, &mut pending).await?;
            tokio::task::yield_now().await;
        }
    }

    // Pop remaining directories
    let mut final_completions = 0usize;
    while let Some((_, id)) = dir_stack.pop() {
        completed_dirs.push(id);
        final_completions += 1;
    }
    for id in completed_dirs.drain(..) {
        let _ = crate::db::folder::update_scan_complete(pool, id, true).await;
    }
    info!("Marked {} directories complete at end of scan", final_completions);

    // Flush any remaining pending upserts
    if !pending.is_empty() {
        flush_batch(pool, &mut pending).await?;
    }

    // Mark missing files per visited folder (avoids touching files in skipped complete subtrees)
    for (folder_id, paths) in &folder_paths {
        let marked = crate::db::media::mark_missing(pool, *folder_id, paths).await?;
        if marked > 0 {
            info!("Marked {} missing records in folder {}", marked, folder_id);
        }
    }

    info!(
        "Scan complete: {} files processed",
        scanned_count
    );
    Ok(scanned_count)
}

#[derive(Debug)]
struct PendingUpsert {
    folder_id: i64,
    relative_path: String,
    absolute_path: String,
    hash: String,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
    file_size: u64,
    modified_at: Option<chrono::NaiveDateTime>,
}

async fn flush_batch(
    pool: &sqlx::SqlitePool,
    batch: &mut Vec<PendingUpsert>,
) -> anyhow::Result<Vec<i64>> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }
    let mut tx = pool.begin().await?;
    let mut media_ids = Vec::with_capacity(batch.len());
    for item in batch.iter() {
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO media_files
             (folder_id, relative_path, absolute_path, blake3_hash, width, height, format, file_size, modified_at, is_present, missing_since)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, NULL)
             ON CONFLICT(folder_id, relative_path) DO UPDATE SET
                 absolute_path = excluded.absolute_path,
                 blake3_hash = excluded.blake3_hash,
                 width = excluded.width,
                 height = excluded.height,
                 format = excluded.format,
                 file_size = excluded.file_size,
                 modified_at = excluded.modified_at,
                 is_present = 1,
                 missing_since = NULL
             RETURNING id"
        )
        .bind(item.folder_id)
        .bind(&item.relative_path)
        .bind(&item.absolute_path)
        .bind(&item.hash)
        .bind(item.width.map(|v| v as i64))
        .bind(item.height.map(|v| v as i64))
        .bind(item.format.as_deref())
        .bind(item.file_size as i64)
        .bind(item.modified_at)
        .fetch_one(&mut *tx)
        .await?;
        media_ids.push(id);
    }
    tx.commit().await?;
    batch.clear();
    Ok(media_ids)
}

/// Process and upsert a single file incrementally. Used by the file watcher.
pub async fn upsert_one(
    pool: &sqlx::SqlitePool,
    folder_id: i64,
    relative_path: &str,
    absolute_path: &str,
) -> anyhow::Result<i64> {
    let path = std::path::PathBuf::from(absolute_path);

    if !path.is_file() || !is_supported(&path) {
        anyhow::bail!("unsupported or missing file: {}", absolute_path);
    }

    let metadata = tokio::fs::metadata(&path).await?;
    let file_size = metadata.len();
    let modified_at = metadata
        .modified()
        .ok()
        .and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
        })
        .flatten()
        .map(|dt| dt.naive_utc());

    let path_buf = path.clone();
    let (hash, width, height, format) = tokio::task::spawn_blocking(move || process_file(&path_buf))
        .await
        .map_err(|e| anyhow::anyhow!("task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("process_file failed: {e}"))?;

    let id = crate::db::media::upsert(
        pool,
        folder_id,
        relative_path,
        absolute_path,
        &hash,
        width,
        height,
        format.as_deref(),
        Some(file_size),
        modified_at,
    )
    .await?;

    Ok(id)
}

fn process_file(path: &Path) -> anyhow::Result<(String, Option<u32>, Option<u32>, Option<String>)> {
    // Hash
    let mut hasher = blake3::Hasher::new();
    let mut file = std::fs::File::open(path)?;
    std::io::copy(&mut file, &mut hasher)?;
    let hash = hasher.finalize().to_hex().to_string();

    // Dimensions & format
    let reader = image::ImageReader::open(path)?;
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase());
    let format = reader
        .format()
        .map(|f| format!("{:?}", f).to_lowercase())
        .or(ext);
    let (width, height) = reader.into_dimensions().ok().unzip();

    Ok((hash, width, height, format))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclude_matches_exact_absolute_path() {
        let patterns = vec!["/tmp/skip".to_string()];
        assert!(is_excluded(Path::new("/tmp/skip"), &patterns));
        assert!(!is_excluded(Path::new("/tmp/keep"), &patterns));
    }

    #[test]
    fn exclude_matches_substring() {
        let patterns = vec!["node_modules".to_string()];
        assert!(is_excluded(Path::new("/home/user/node_modules/foo.png"), &patterns));
        assert!(!is_excluded(Path::new("/home/user/photos/foo.png"), &patterns));
    }

    #[test]
    fn include_requires_match_when_non_empty() {
        let patterns = vec!["photos".to_string()];
        assert!(is_included(Path::new("/home/user/photos/foo.png"), &patterns));
        assert!(!is_included(Path::new("/home/user/videos/foo.mp4"), &patterns));
    }

    #[test]
    fn empty_include_allows_all() {
        let patterns: Vec<String> = Vec::new();
        assert!(is_included(Path::new("/anything"), &patterns));
    }

    #[test]
    fn exclude_wins_over_include() {
        let exclude = vec!["private".to_string()];
        let include = vec!["photos".to_string()];
        let path = Path::new("/home/user/photos/private/foo.png");
        assert!(is_excluded(path, &exclude));
        assert!(is_included(path, &include));
        // When applying both, exclude takes precedence: the path is rejected.
        assert!(
            is_excluded(path, &exclude) || !is_included(path, &include),
            "exclude should win over include"
        );
    }
}
