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
            SUPPORTED_EXTENSIONS.contains(&ext.as_str())
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
    folder_id: i64,
    folder_path: &Path,
    recursive: bool,
    blacklist: &[String],
) -> anyhow::Result<usize> {
    info!(
        "Scanning folder: {} (recursive={}, blacklist={:?})",
        folder_path.display(),
        recursive,
        blacklist
    );

    let blacklist_set = build_blacklist_set(blacklist)?;
    let walker = if recursive {
        walkdir::WalkDir::new(folder_path)
    } else {
        walkdir::WalkDir::new(folder_path).max_depth(1)
    };

    let mut existing_paths: Vec<String> = Vec::new();
    let mut scanned_count = 0usize;

    for entry in walker.into_iter().filter_entry(|e| {
        if e.depth() == 0 {
            return true;
        }
        !check_blacklist(e, &blacklist_set)
    }) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Walkdir error: {}", e);
                continue;
            }
        };

        let path = entry.path();
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

        existing_paths.push(relative_path.clone());

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

        // Check if file changed since last scan
        let needs_update = {
            let row = sqlx::query_as::<_, (Option<i64>, Option<String>, Option<i64>)>(
                "SELECT id, blake3_hash, file_size FROM media_files WHERE folder_id = ?1 AND relative_path = ?2"
            )
            .bind(folder_id)
            .bind(&relative_path)
            .fetch_optional(pool)
            .await?;

            match row {
                Some((_, Some(old_hash), Some(old_size))) => {
                    old_hash.is_empty() || old_size as u64 != file_size
                }
                _ => true,
            }
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

        crate::db::media::upsert(
            pool,
            folder_id,
            &relative_path,
            &absolute_path,
            &hash,
            width,
            height,
            format.as_deref(),
            Some(file_size),
            modified_at,
        )
        .await?;

        scanned_count += 1;
    }

    let deleted = crate::db::media::delete_orphans(pool, folder_id, &existing_paths).await?;
    if deleted > 0 {
        info!("Deleted {} orphan records from folder {}", deleted, folder_id);
    }

    info!(
        "Scan complete: {} files processed, {} orphans removed",
        scanned_count, deleted
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
    let format = reader.format().map(|f| format!("{:?}", f).to_lowercase());
    let (width, height) = reader.into_dimensions().ok().unzip();

    Ok((hash, width, height, format))
}
