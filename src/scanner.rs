use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

const SUPPORTED_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "avif",
];

fn is_supported(path: &Path) -> bool {
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

fn build_blacklist_set(blacklist: &[String]) -> anyhow::Result<globset::GlobSet> {
    let mut builder = globset::GlobSetBuilder::new();
    for pattern in blacklist {
        let glob = globset::Glob::new(pattern)?;
        builder.add(glob);
    }
    Ok(builder.build()?)
}

fn check_blacklist(entry: &walkdir::DirEntry, blacklist: &globset::GlobSet) -> bool {
    let name = entry.file_name().to_string_lossy();
    if blacklist.is_match(&*name) {
        return true;
    }
    if let Some(relative) = entry.path().file_name() {
        let rel = relative.to_string_lossy();
        if blacklist.is_match(&*rel) {
            return true;
        }
    }
    false
}

pub async fn scan_folder(
    pool: &sqlx::SqlitePool,
    root_folder_id: i64,
    folder_path: &Path,
    recursive: bool,
    show_recursive: bool,
    blacklist: &[String],
    progress_tx: Option<&std::sync::mpsc::Sender<crate::app::ScanEvent>>,
) -> anyhow::Result<usize> {
    info!(
        "Scanning folder: {} (recursive={}, show_recursive={}, blacklist={:?})",
        folder_path.display(),
        recursive,
        show_recursive,
        blacklist
    );

    let blacklist_set = build_blacklist_set(blacklist)?;
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
    let existing: HashMap<String, (String, i64, Option<String>)> = sqlx::query_as::<_, (String, String, i64, Option<String>)>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
        )
        SELECT relative_path, blake3_hash, file_size, format FROM media_files WHERE folder_id IN (SELECT id FROM subtree)"
    )
    .bind(root_folder_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|(path, hash, size, format)| (path, (hash, size, format)))
    .collect();

    let mut folder_paths: HashMap<i64, Vec<String>> = HashMap::new();
    let mut scanned_count = 0usize;
    let mut batch_count = 0usize;
    const BATCH_SIZE: usize = 1000;

    // Track directories currently being walked; mark them complete when we leave them
    let mut dir_stack: Vec<(std::path::PathBuf, i64)> = Vec::new();
    let mut completed_dirs: Vec<i64> = Vec::new();

    for entry in walker.into_iter().filter_entry(|e| {
        if e.depth() == 0 {
            return true;
        }
        if check_blacklist(e, &blacklist_set) {
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
                    show_recursive,
                    false,       // not known to be complete until walker leaves it
                    blacklist,
                    None,        // inherit cache mode
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
            Some((old_hash, old_size, old_format)) => {
                old_hash.is_empty() || *old_size as u64 != file_size || old_format.is_none()
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

        // Upsert via the pool (WAL mode allows concurrent reads)
        sqlx::query(
            "INSERT INTO media_files
             (folder_id, relative_path, absolute_path, blake3_hash, width, height, format, file_size, modified_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(folder_id, relative_path) DO UPDATE SET
                 absolute_path = excluded.absolute_path,
                 blake3_hash = excluded.blake3_hash,
                 width = excluded.width,
                 height = excluded.height,
                 format = excluded.format,
                 file_size = excluded.file_size,
                 modified_at = excluded.modified_at"
        )
        .bind(folder_id)
        .bind(&relative_path)
        .bind(&absolute_path)
        .bind(&hash)
        .bind(width.map(|v| v as i64))
        .bind(height.map(|v| v as i64))
        .bind(format.as_deref())
        .bind(file_size as i64)
        .bind(modified_at)
        .execute(pool)
        .await?;

        scanned_count += 1;
        batch_count += 1;

        // Send progress every 5000 files
        if scanned_count % 5000 == 0 {
            if let Some(tx) = progress_tx {
                let _ = tx.send(crate::app::ScanEvent::Progress(
                    folder_path.to_string_lossy().to_string(),
                    scanned_count,
                ));
            }
        }

        // Yield every batch to let other DB operations through
        if batch_count >= BATCH_SIZE {
            batch_count = 0;
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

    // Delete orphans per visited folder (avoids deleting files in skipped complete subtrees)
    for (folder_id, paths) in &folder_paths {
        let deleted = crate::db::media::delete_orphans(pool, *folder_id, paths).await?;
        if deleted > 0 {
            info!("Deleted {} orphan records from folder {}", deleted, folder_id);
        }
    }

    info!(
        "Scan complete: {} files processed",
        scanned_count
    );
    Ok(scanned_count)
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
