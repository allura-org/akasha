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
    let mut stream = sqlx::query_as::<_, MediaSummaryRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash, width, height, format, file_size, created_at, modified_at, is_present, missing_since
         FROM media_files
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
    let mut stream = sqlx::query_as::<_, MediaSummaryRow>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         SELECT m.id, m.folder_id, m.relative_path, m.absolute_path, m.blake3_hash, m.width, m.height, m.format, m.file_size, m.created_at, m.modified_at, m.is_present, m.missing_since
         FROM media_files m
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
///
/// `ids_json` should be a JSON array of integers, e.g. `[1, 2, 3]`.
pub async fn search_summaries(
    pool: &SqlitePool,
    folder_id: i64,
    recursive: bool,
    ids_json: &str,
) -> anyhow::Result<Vec<MediaSummary>> {
    let mut summaries = Vec::new();

    let sql = if recursive {
        r#"
        WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
        )
        SELECT m.id, m.folder_id, m.relative_path, m.absolute_path, m.blake3_hash,
               m.width, m.height, m.format, m.file_size, m.created_at, m.modified_at, m.is_present, m.missing_since
        FROM media_files m
        JOIN subtree s ON m.folder_id = s.id
        WHERE m.id IN (SELECT CAST(value AS INTEGER) FROM json_each(?2))
        ORDER BY m.id
        "#
    } else {
        r#"
        SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
               width, height, format, file_size, created_at, modified_at, is_present, missing_since
        FROM media_files
        WHERE folder_id = ?1 AND id IN (SELECT CAST(value AS INTEGER) FROM json_each(?2))
        ORDER BY id
        "#
    };

    let mut stream = sqlx::query_as::<_, MediaSummaryRow>(sql)
        .bind(folder_id)
        .bind(ids_json)
        .fetch(pool);

    while let Some(row) = stream.try_next().await? {
        summaries.push(into_summary(row));
    }

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
        sqlx::query("DELETE FROM searchable_tags_fts WHERE media_file_id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM searchable_text_fts WHERE media_file_id = ?1")
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

/// Permanently delete all rows that are currently marked missing.
/// This is an explicit, user-initiated action from the DB Management menu.
pub async fn delete_missing(pool: &SqlitePool) -> anyhow::Result<u64> {
    let mut tx = pool.begin().await?;

    // Virtual FTS5 tables cannot declare foreign keys, so clean up orphans
    // explicitly before deleting the parent media_files rows.
    sqlx::query(
        "DELETE FROM searchable_tags_fts WHERE media_file_id IN (SELECT id FROM media_files WHERE is_present = 0)"
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "DELETE FROM searchable_text_fts WHERE media_file_id IN (SELECT id FROM media_files WHERE is_present = 0)"
    )
    .execute(&mut *tx)
    .await?;

    let result = sqlx::query("DELETE FROM media_files WHERE is_present = 0")
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(result.rows_affected())
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

        let deleted = delete_missing(&pool).await.unwrap();
        assert_eq!(deleted, 1);

        let all = list_by_folder(&pool, fid).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].relative_path, "present.jpg");
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
