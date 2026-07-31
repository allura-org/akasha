use std::collections::HashMap;

use anyhow::Context;
use futures_util::stream::TryStreamExt;
use sqlx::SqlitePool;

#[derive(Debug, Clone)]
pub struct MediaFile {
    pub id: i64,
    pub folder_id: i64,
    pub relative_path: String,
    pub absolute_path: String,
    pub blake3_hash: String,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub format: Option<String>,
    pub file_size: Option<i64>,
    pub is_present: bool,
    pub missing_since: Option<chrono::NaiveDateTime>,
    pub created_at: chrono::NaiveDateTime,
    pub modified_at: Option<chrono::NaiveDateTime>,
}

#[derive(Debug, Clone)]
pub struct PropertiesData {
    pub media: MediaFile,
    pub folder_path: String,
    pub tags: HashMap<String, HashMap<String, f32>>,
    pub descriptions: HashMap<String, String>,
    pub classifications: HashMap<String, Vec<String>>,
    pub embeddings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MediaSummary {
    pub id: i64,
    pub folder_id: i64,
    pub relative_path: String,
    pub absolute_path: String,
    pub blake3_hash: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub format: Option<String>,
    pub file_size: Option<i64>,
    pub created_at: chrono::NaiveDateTime,
    pub modified_at: Option<chrono::NaiveDateTime>,
    pub is_present: bool,
    pub missing_since: Option<chrono::NaiveDateTime>,
    /// Populated when this summary is the result of a search query.
    pub search_score: Option<f32>,
}

pub async fn count_by_folder(pool: &SqlitePool, folder_id: i64) -> anyhow::Result<i64> {
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM media_files WHERE folder_id = ?1")
        .bind(folder_id)
        .fetch_one(pool)
        .await?;
    Ok(row.0)
}

pub async fn count_by_folder_recursive(pool: &SqlitePool, folder_id: i64) -> anyhow::Result<i64> {
    let row: (i64,) = sqlx::query_as(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         SELECT COUNT(*) FROM media_files WHERE folder_id IN (SELECT id FROM subtree)"
    )
    .bind(folder_id)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

pub async fn list_summaries_by_folder(
    pool: &SqlitePool,
    folder_id: i64,
) -> anyhow::Result<Vec<MediaSummary>> {
    let mut summaries = Vec::new();
    // INDEXED BY: forces the idx_media_summary covering-index scan. The
    // planner otherwise probes the fat media_files rows (tags_json makes
    // them several KB each) per row — at collection scale that is the
    // difference between ~1s and ~40s per folder load.
    let mut stream = sqlx::query_as::<_, MediaSummaryRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash, width, height, format, file_size, created_at, modified_at, is_present, missing_since
         FROM media_files INDEXED BY idx_media_summary
         WHERE folder_id = ?1
         ORDER BY id"
    )
    .bind(folder_id)
    .fetch(pool);

    while let Some(row) = stream.try_next().await? {
        summaries.push(into_summary(row));
    }

    Ok(summaries)
}

pub async fn list_summaries_by_folder_recursive(
    pool: &SqlitePool,
    folder_id: i64,
) -> anyhow::Result<Vec<MediaSummary>> {
    let mut summaries = Vec::new();
    // See the INDEXED BY note on list_summaries_by_folder.
    let mut stream = sqlx::query_as::<_, MediaSummaryRow>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         SELECT m.id, m.folder_id, m.relative_path, m.absolute_path, m.blake3_hash, m.width, m.height, m.format, m.file_size, m.created_at, m.modified_at, m.is_present, m.missing_since
         FROM media_files m INDEXED BY idx_media_summary
         JOIN subtree s ON m.folder_id = s.id
         ORDER BY m.id"
    )
    .bind(folder_id)
    .fetch(pool);

    while let Some(row) = stream.try_next().await? {
        summaries.push(into_summary(row));
    }

    Ok(summaries)
}

/// Load `MediaSummary` rows for a set of media file IDs, scoped to a folder.
pub async fn search_summaries(
    pool: &SqlitePool,
    folder_id: i64,
    recursive: bool,
    ids: &[i64],
) -> anyhow::Result<Vec<MediaSummary>> {
    // sqlx's bundled SQLite (3.46) plans the recursive id-filter join as a
    // subtree_folders × ids nested loop of index probes — billions of probes
    // at collection scale, i.e. the query never finishes. Sidestep the
    // planner entirely for large id sets: scan the subtree through the
    // covering index (already ~1-2s, see list_summaries_by_folder_recursive)
    // and filter in Rust. Small id sets keep the cheap rowid-probe SQL path.
    const RUST_FILTER_THRESHOLD: usize = 1000;
    if recursive && ids.len() >= RUST_FILTER_THRESHOLD {
        let t_start = std::time::Instant::now();
        tracing::info!(ids = ids.len(), "search_summaries: subtree scan + rust filter");
        let id_set: std::collections::HashSet<i64> = ids.iter().copied().collect();
        let mut summaries = list_summaries_by_folder_recursive(pool, folder_id).await?;
        summaries.retain(|s| id_set.contains(&s.id));
        tracing::info!(
            rows = summaries.len(),
            elapsed_ms = t_start.elapsed().as_millis(),
            "search_summaries: done"
        );
        return Ok(summaries);
    }

    let mut summaries = Vec::new();
    let ids_json = serde_json::to_string(ids)?;
    let t_start = std::time::Instant::now();
    tracing::info!(ids = ids.len(), "search_summaries: start");

    // Recursive + few ids: rowid probes with a subtree membership check.
    // Non-recursive: single-folder probe, safe at any id count; INDEXED BY
    // keeps it on the covering index.
    let sql = if recursive {
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
        )
        SELECT m.id, m.folder_id, m.relative_path, m.absolute_path, m.blake3_hash,
               m.width, m.height, m.format, m.file_size, m.created_at, m.modified_at, m.is_present, m.missing_since
        FROM media_files m
        JOIN subtree s ON m.folder_id = s.id
        WHERE m.id IN (SELECT CAST(value AS INTEGER) FROM json_each(?2))
        ORDER BY m.id"
        .to_string()
    } else {
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size, created_at, modified_at, is_present, missing_since
         FROM media_files INDEXED BY idx_media_summary
         WHERE folder_id = ?1 AND id IN (SELECT CAST(value AS INTEGER) FROM json_each(?2))
         ORDER BY id"
        .to_string()
    };

    let mut stream = sqlx::query_as::<_, MediaSummaryRow>(&sql)
        .bind(folder_id)
        .bind(ids_json)
        .fetch(pool);

    while let Some(row) = stream.try_next().await? {
        summaries.push(into_summary(row));
    }

    tracing::info!(
        rows = summaries.len(),
        elapsed_ms = t_start.elapsed().as_millis(),
        "search_summaries: done"
    );

    Ok(summaries)
}

pub async fn get_by_id(pool: &SqlitePool, id: i64) -> anyhow::Result<Option<MediaFile>> {
    let row = sqlx::query_as::<_, MediaFileRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size, is_present, missing_since, created_at, modified_at
         FROM media_files WHERE id = ?1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(into_media))
}

pub async fn get_properties_data(
    pool: &SqlitePool,
    media_file_id: i64,
) -> anyhow::Result<PropertiesData> {
    let media = get_by_id(pool, media_file_id)
        .await?
        .context("media file not found")?;

    let folder_path: String = sqlx::query_scalar(
        "SELECT path FROM folders WHERE id = ?1"
    )
    .bind(media.folder_id)
    .fetch_one(pool)
    .await
    .context("folder not found for media file")?;

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
        folder_path,
        tags,
        descriptions,
        classifications,
        embeddings,
    })
}

pub async fn get_by_path(
    pool: &SqlitePool,
    folder_id: i64,
    relative_path: &str,
) -> anyhow::Result<Option<MediaFile>> {
    let row = sqlx::query_as::<_, MediaFileRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size, is_present, missing_since, created_at, modified_at
         FROM media_files WHERE folder_id = ?1 AND relative_path = ?2"
    )
    .bind(folder_id)
    .bind(relative_path)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(into_media))
}

pub async fn delete_by_path(
    pool: &SqlitePool,
    folder_id: i64,
    relative_path: &str,
) -> anyhow::Result<u64> {
    let mut tx = pool.begin().await?;

    // Virtual FTS5 tables cannot have foreign keys, so clean them up explicitly
    // before deleting the parent media_files row.
    let media_id: Option<i64> = sqlx::query_scalar(
        "SELECT id FROM media_files WHERE folder_id = ?1 AND relative_path = ?2"
    )
    .bind(folder_id)
    .bind(relative_path)
    .fetch_optional(&mut *tx)
    .await?;

    if let Some(id) = media_id {
        sqlx::query("DELETE FROM searchable_text_fts WHERE rowid = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await?;
    }

    let rows = sqlx::query(
        "DELETE FROM media_files WHERE folder_id = ?1 AND relative_path = ?2"
    )
    .bind(folder_id)
    .bind(relative_path)
    .execute(&mut *tx)
    .await?
    .rows_affected();

    tx.commit().await?;
    Ok(rows)
}

pub async fn list_page_by_folder(
    pool: &SqlitePool,
    folder_id: i64,
    after_id: i64,
    limit: i64,
) -> anyhow::Result<Vec<MediaFile>> {
    let rows = sqlx::query_as::<_, MediaFileRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size, is_present, missing_since, created_at, modified_at
         FROM media_files
         WHERE folder_id = ?1 AND id > ?2
         ORDER BY id
         LIMIT ?3"
    )
    .bind(folder_id)
    .bind(after_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_media).collect())
}

pub async fn list_page_by_folder_recursive(
    pool: &SqlitePool,
    folder_id: i64,
    after_id: i64,
    limit: i64,
) -> anyhow::Result<Vec<MediaFile>> {
    let rows = sqlx::query_as::<_, MediaFileRow>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         SELECT m.id, m.folder_id, m.relative_path, m.absolute_path, m.blake3_hash,
                m.width, m.height, m.format, m.file_size, m.is_present, m.missing_since,
                m.created_at, m.modified_at
         FROM media_files m
         JOIN subtree s ON m.folder_id = s.id
         WHERE m.id > ?2
         ORDER BY m.id
         LIMIT ?3"
    )
    .bind(folder_id)
    .bind(after_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_media).collect())
}

// Legacy full-record queries (still used during scans and for detail panels)
pub async fn list_by_folder(pool: &SqlitePool, folder_id: i64) -> anyhow::Result<Vec<MediaFile>> {
    let rows = sqlx::query_as::<_, MediaFileRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size, is_present, missing_since, created_at, modified_at
         FROM media_files WHERE folder_id = ?1"
    )
    .bind(folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_media).collect())
}

pub async fn list_by_folder_recursive(pool: &SqlitePool, folder_id: i64) -> anyhow::Result<Vec<MediaFile>> {
    let rows = sqlx::query_as::<_, MediaFileRow>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size, is_present, missing_since, created_at, modified_at
         FROM media_files WHERE folder_id IN (SELECT id FROM subtree)"
    )
    .bind(folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_media).collect())
}

pub async fn upsert(
    pool: &SqlitePool,
    folder_id: i64,
    relative_path: &str,
    absolute_path: &str,
    hash: &str,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<&str>,
    file_size: Option<u64>,
    modified_at: Option<chrono::NaiveDateTime>,
) -> anyhow::Result<i64> {
    let id = sqlx::query(
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
             missing_since = NULL"
    )
    .bind(folder_id)
    .bind(relative_path)
    .bind(absolute_path)
    .bind(hash)
    .bind(width.map(|v| v as i64))
    .bind(height.map(|v| v as i64))
    .bind(format)
    .bind(file_size.map(|v| v as i64))
    .bind(modified_at)
    .execute(pool)
    .await?
    .last_insert_rowid();

    Ok(id)
}

/// Mark every file in `folder_id` that is not in `existing_paths` as missing.
/// Existing metadata is preserved so it can be restored if the file reappears.
pub async fn mark_missing(pool: &SqlitePool, folder_id: i64, existing_paths: &[String]) -> anyhow::Result<u64> {
    let paths_json = serde_json::to_string(existing_paths)?;
    let result = sqlx::query(
        "UPDATE media_files
         SET is_present = 0, missing_since = CURRENT_TIMESTAMP
         WHERE folder_id = ?1
           AND (is_present = 1 OR is_present IS NULL)
           AND relative_path NOT IN (SELECT value FROM json_each(?2))"
    )
    .bind(folder_id)
    .bind(paths_json)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Mark a single file as missing. Used by the file watcher when a path is removed.
pub async fn mark_missing_by_path(
    pool: &SqlitePool,
    folder_id: i64,
    relative_path: &str,
) -> anyhow::Result<u64> {
    let result = sqlx::query(
        "UPDATE media_files
         SET is_present = 0, missing_since = CURRENT_TIMESTAMP
         WHERE folder_id = ?1 AND relative_path = ?2"
    )
    .bind(folder_id)
    .bind(relative_path)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Mark a single file as present again. The normal upsert path also clears the
/// missing flag, but this helper is useful for explicit reconcile operations.
pub async fn mark_present_by_path(
    pool: &SqlitePool,
    folder_id: i64,
    relative_path: &str,
) -> anyhow::Result<u64> {
    let result = sqlx::query(
        "UPDATE media_files
         SET is_present = 1, missing_since = NULL
         WHERE folder_id = ?1 AND relative_path = ?2"
    )
    .bind(folder_id)
    .bind(relative_path)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Permanently delete all media rows that are currently marked missing, plus
/// all folder rows marked missing. Deleting a missing folder cascade-deletes
/// its subfolders and their media (present or not — everything under a missing
/// folder is unreachable by definition).
/// This is an explicit, user-initiated action from the DB Management menu.
/// Returns `(media rows deleted, folder rows deleted)`.
pub async fn delete_missing(pool: &SqlitePool) -> anyhow::Result<(u64, u64)> {
    let mut tx = pool.begin().await?;

    // Virtual FTS5 tables cannot declare foreign keys, so clean up orphans
    // explicitly before deleting the parent media_files rows. This covers both
    // media marked missing directly and media under a missing folder (which is
    // removed by the folder cascade below).
    const MISSING_MEDIA: &str =
        "SELECT id FROM media_files WHERE is_present = 0 OR folder_id IN (SELECT id FROM folders WHERE is_present = 0)";
    sqlx::query(&format!(
        "DELETE FROM searchable_text_fts WHERE rowid IN ({MISSING_MEDIA})"
    ))
    .execute(&mut *tx)
    .await?;

    let media = sqlx::query("DELETE FROM media_files WHERE is_present = 0")
        .execute(&mut *tx)
        .await?;

    let folders = sqlx::query("DELETE FROM folders WHERE is_present = 0")
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok((media.rows_affected(), folders.rows_affected()))
}

#[derive(sqlx::FromRow)]
struct MediaFileRow {
    id: i64,
    folder_id: i64,
    relative_path: String,
    absolute_path: String,
    blake3_hash: String,
    width: Option<i64>,
    height: Option<i64>,
    format: Option<String>,
    file_size: Option<i64>,
    is_present: i64,
    missing_since: Option<chrono::NaiveDateTime>,
    created_at: chrono::NaiveDateTime,
    modified_at: Option<chrono::NaiveDateTime>,
}

#[derive(sqlx::FromRow)]
struct MediaSummaryRow {
    id: i64,
    folder_id: i64,
    relative_path: String,
    absolute_path: String,
    blake3_hash: String,
    width: Option<i64>,
    height: Option<i64>,
    format: Option<String>,
    file_size: Option<i64>,
    created_at: chrono::NaiveDateTime,
    modified_at: Option<chrono::NaiveDateTime>,
    is_present: i64,
    missing_since: Option<chrono::NaiveDateTime>,
}

fn into_media(row: MediaFileRow) -> MediaFile {
    MediaFile {
        id: row.id,
        folder_id: row.folder_id,
        relative_path: row.relative_path,
        absolute_path: row.absolute_path,
        blake3_hash: row.blake3_hash,
        width: row.width,
        height: row.height,
        format: row.format,
        file_size: row.file_size,
        is_present: row.is_present != 0,
        missing_since: row.missing_since,
        created_at: row.created_at,
        modified_at: row.modified_at,
    }
}

fn into_summary(row: MediaSummaryRow) -> MediaSummary {
    MediaSummary {
        id: row.id,
        folder_id: row.folder_id,
        relative_path: row.relative_path,
        absolute_path: row.absolute_path,
        blake3_hash: row.blake3_hash,
        width: row.width.map(|v| v as u32),
        height: row.height.map(|v| v as u32),
        format: row.format,
        file_size: row.file_size,
        created_at: row.created_at,
        modified_at: row.modified_at,
        is_present: row.is_present != 0,
        missing_since: row.missing_since,
        search_score: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::folder;
    use crate::db::searchable;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn get_by_path_and_delete_by_path_round_trip() {
        let pool = setup_pool().await;
        let fid = folder::insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();

        let id = upsert(
            &pool,
            fid,
            "foo.jpg",
            "/tmp/root/foo.jpg",
            "hash",
            Some(100),
            Some(200),
            Some("jpeg"),
            Some(1024),
            Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();

        let found = get_by_path(&pool, fid, "foo.jpg").await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);

        let deleted = delete_by_path(&pool, fid, "foo.jpg").await.unwrap();
        assert_eq!(deleted, 1);

        let found = get_by_path(&pool, fid, "foo.jpg").await.unwrap();
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn mark_missing_preserves_record_and_upsert_clears_it() {
        let pool = setup_pool().await;
        let fid = folder::insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();

        upsert(
            &pool,
            fid,
            "foo.jpg",
            "/tmp/root/foo.jpg",
            "hash",
            Some(100),
            Some(200),
            Some("jpeg"),
            Some(1024),
            Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();

        let marked = mark_missing_by_path(&pool, fid, "foo.jpg").await.unwrap();
        assert_eq!(marked, 1);

        let found = get_by_path(&pool, fid, "foo.jpg").await.unwrap().expect("row gone");
        assert!(!found.is_present);
        assert!(found.missing_since.is_some());

        // Re-upserting the file clears the missing flag.
        upsert(
            &pool,
            fid,
            "foo.jpg",
            "/tmp/root/foo.jpg",
            "hash2",
            Some(100),
            Some(200),
            Some("jpeg"),
            Some(2048),
            Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();

        let found = get_by_path(&pool, fid, "foo.jpg").await.unwrap().expect("row gone");
        assert!(found.is_present);
        assert!(found.missing_since.is_none());
        assert_eq!(found.blake3_hash, "hash2");
    }

    #[tokio::test]
    async fn mark_missing_and_delete_missing_works() {
        let pool = setup_pool().await;
        let fid = folder::insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();

        upsert(
            &pool,
            fid,
            "present.jpg",
            "/tmp/root/present.jpg",
            "hash1",
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        upsert(
            &pool,
            fid,
            "gone.jpg",
            "/tmp/root/gone.jpg",
            "hash2",
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let marked = mark_missing(&pool, fid, &["present.jpg".to_string()]).await.unwrap();
        assert_eq!(marked, 1);

        let all = list_by_folder(&pool, fid).await.unwrap();
        assert_eq!(all.len(), 2);
        let gone = all.iter().find(|m| m.relative_path == "gone.jpg").unwrap();
        assert!(!gone.is_present);

        let (media_deleted, folders_deleted) = delete_missing(&pool).await.unwrap();
        assert_eq!(media_deleted, 1);
        assert_eq!(folders_deleted, 0);

        let all = list_by_folder(&pool, fid).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].relative_path, "present.jpg");
    }

    #[tokio::test]
    async fn delete_missing_purges_missing_folders_and_their_media() {
        let pool = setup_pool().await;
        let root = folder::insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let sub = folder::insert(
            &pool, Some(root), "/tmp/root/gone", true, false, &[], &[], None, None, "disable",
        )
        .await
        .unwrap();

        let mid = upsert(
            &pool,
            sub,
            "a.jpg",
            "/tmp/root/gone/a.jpg",
            "hash1",
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        // Present media under the soon-to-be-missing folder: it should be
        // cascade-deleted with the folder, and its FTS rows cleaned up.
        crate::db::searchable::update_description_json(&pool, mid, "blip", "a cat")
            .await
            .unwrap();
        let keep = upsert(
            &pool,
            root,
            "keep.jpg",
            "/tmp/root/keep.jpg",
            "hash2",
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let marked = folder::mark_missing(&pool, sub).await.unwrap();
        assert_eq!(marked, 1);

        // The folder's media is marked missing with it.
        let media = get_by_id(&pool, mid).await.unwrap().unwrap();
        assert!(!media.is_present);

        let (media_deleted, folders_deleted) = delete_missing(&pool).await.unwrap();
        assert_eq!(folders_deleted, 1);
        assert!(media_deleted >= 1);

        // Folder, its media, and its FTS rows are gone; unrelated rows survive.
        assert!(folder::get_by_path(&pool, "/tmp/root/gone").await.unwrap().is_none());
        assert!(get_by_id(&pool, mid).await.unwrap().is_none());
        assert!(get_by_id(&pool, keep).await.unwrap().is_some());
        let fts_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM searchable_text_fts WHERE media_file_id = ?1",
        )
        .bind(mid)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(fts_count.0, 0);
    }

    #[tokio::test]
    async fn get_properties_data_returns_tags_and_descriptions() {
        let pool = setup_pool().await;
        let fid = folder::insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();

        let id = upsert(
            &pool,
            fid,
            "foo.jpg",
            "/tmp/root/foo.jpg",
            "hash",
            Some(100),
            Some(200),
            Some("jpeg"),
            Some(1024),
            Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();

        let mut tags = HashMap::new();
        tags.insert("cat".to_string(), 0.95f32);
        tags.insert("dog".to_string(), 0.23f32);
        searchable::update_tags_json(&pool, id, "wd-vit", tags.clone())
            .await
            .unwrap();

        searchable::update_description_json(&pool, id, "blip", "a cat on a mat")
            .await
            .unwrap();

        let props = get_properties_data(&pool, id).await.unwrap();
        assert_eq!(props.media.id, id);
        assert_eq!(props.media.relative_path, "foo.jpg");
        assert_eq!(props.folder_path, "/tmp/root");

        let source_tags = props.tags.get("wd-vit").expect("wd-vit tags missing");
        assert_eq!(source_tags.len(), 2);
        assert!((source_tags.get("cat").copied().unwrap() - 0.95f32).abs() < f32::EPSILON);
        assert!((source_tags.get("dog").copied().unwrap() - 0.23f32).abs() < f32::EPSILON);

        assert_eq!(
            props.descriptions.get("blip"),
            Some(&"a cat on a mat".to_string())
        );
        assert!(props.classifications.is_empty());
        assert!(props.embeddings.is_empty());
    }
}

#[cfg(test)]
mod live_hydration_bench {
    /// Reproduce the app's exact hydration path through sqlx (bundled SQLite)
    /// against the real database. Run with:
    ///   AKASHA_LIVE_DB=~/.local/share/akasha/akasha.db \
    ///   cargo test --release live_hydration -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn live_hydration_explain_and_time() {
        let db = std::env::var("AKASHA_LIVE_DB").expect("AKASHA_LIVE_DB");
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(2)
            .connect_with(
                sqlx::sqlite::SqliteConnectOptions::new()
                    .filename(db)
                    .read_only(true)
                    .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal),
            )
            .await
            .unwrap();

        let version: (String,) = sqlx::query_as("SELECT sqlite_version()")
            .fetch_one(&pool)
            .await
            .unwrap();
        println!("sqlx bundled sqlite version: {}", version.0);

        let ids: Vec<i64> = sqlx::query_scalar(
            "SELECT DISTINCT media_file_id FROM searchable_tags WHERE tag IN ('solo','solo_focus')",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        println!("ids: {}", ids.len());

        let ids_json = serde_json::to_string(&ids).unwrap();
        let sql = "WITH RECURSIVE subtree(id) AS (
                SELECT ?1 UNION ALL
                SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
            )
            SELECT m.id FROM media_files m INDEXED BY idx_media_summary
            JOIN subtree s ON m.folder_id = s.id
            WHERE m.id IN (SELECT CAST(value AS INTEGER) FROM json_each(?2))
            ORDER BY m.id";
        let plan: Vec<(i64, i64, i64, String)> =
            sqlx::query_as(&format!("EXPLAIN QUERY PLAN {sql}"))
                .bind(2i64)
                .bind(&ids_json)
                .fetch_all(&pool)
                .await
                .unwrap();
        for row in &plan {
            println!("plan: {} {} {} {}", row.0, row.1, row.2, row.3);
        }

        let t = std::time::Instant::now();
        let summaries = super::search_summaries(&pool, 2, true, &ids).await.unwrap();
        println!(
            "search_summaries: {} rows in {:.2}s",
            summaries.len(),
            t.elapsed().as_secs_f64()
        );
    }
}
