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
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash, width, height, format, file_size, created_at, modified_at
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
         SELECT m.id, m.folder_id, m.relative_path, m.absolute_path, m.blake3_hash, m.width, m.height, m.format, m.file_size, m.created_at, m.modified_at
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
               m.width, m.height, m.format, m.file_size, m.created_at, m.modified_at
        FROM media_files m
        JOIN subtree s ON m.folder_id = s.id
        WHERE m.id IN (SELECT CAST(value AS INTEGER) FROM json_each(?2))
        ORDER BY m.id
        "#
    } else {
        r#"
        SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
               width, height, format, file_size, created_at, modified_at
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
                width, height, format, file_size
         FROM media_files WHERE id = ?1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(into_media))
}

pub async fn list_page_by_folder(
    pool: &SqlitePool,
    folder_id: i64,
    after_id: i64,
    limit: i64,
) -> anyhow::Result<Vec<MediaFile>> {
    let rows = sqlx::query_as::<_, MediaFileRow>(
        "SELECT id, folder_id, relative_path, absolute_path, blake3_hash,
                width, height, format, file_size
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
                m.width, m.height, m.format, m.file_size
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
                width, height, format, file_size
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
                width, height, format, file_size
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

pub async fn delete_orphans(pool: &SqlitePool, folder_id: i64, existing_paths: &[String]) -> anyhow::Result<u64> {
    let paths_json = serde_json::to_string(existing_paths)?;
    let result = sqlx::query(
        "DELETE FROM media_files
         WHERE folder_id = ?1
           AND relative_path NOT IN (SELECT value FROM json_each(?2))"
    )
    .bind(folder_id)
    .bind(paths_json)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

pub async fn delete_orphans_for_root(pool: &SqlitePool, root_folder_id: i64, existing_paths: &[String]) -> anyhow::Result<u64> {
    let paths_json = serde_json::to_string(existing_paths)?;
    let result = sqlx::query(
        "DELETE FROM media_files
         WHERE folder_id IN (SELECT id FROM folders WHERE id = ?1 OR parent_id = ?1)
           AND relative_path NOT IN (SELECT value FROM json_each(?2))"
    )
    .bind(root_folder_id)
    .bind(paths_json)
    .execute(pool)
    .await?;

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
        search_score: None,
    }
}
