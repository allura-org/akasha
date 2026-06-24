use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, SqlitePool};

/// A configured Searchable property, persisted in `searchable_configs`.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct SearchableConfig {
    pub id: i64,
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    #[sqlx(json)]
    pub options: serde_json::Value,
    pub created_at: chrono::NaiveDateTime,
}

/// Load every Searchable configuration from the database.
pub async fn list_searchable_configs(pool: &SqlitePool) -> Result<Vec<SearchableConfig>> {
    let rows = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at FROM searchable_configs ORDER BY name"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Load only the enabled Searchable configurations.
pub async fn list_enabled_configs(pool: &SqlitePool) -> Result<Vec<SearchableConfig>> {
    let rows = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at FROM searchable_configs WHERE enabled = 1 ORDER BY name"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Insert a new Searchable configuration. Used by migrations and future model installers.
pub async fn insert_config(
    pool: &SqlitePool,
    name: &str,
    kind: &str,
    enabled: bool,
    options: serde_json::Value,
) -> Result<i64> {
    let id = sqlx::query(
        "INSERT INTO searchable_configs (name, kind, enabled, options) VALUES (?1, ?2, ?3, ?4)"
    )
    .bind(name)
    .bind(kind)
    .bind(enabled)
    .bind(options)
    .execute(pool)
    .await?
    .last_insert_rowid();
    Ok(id)
}

/// Upsert a computed Searchable value for a media file.
pub async fn upsert_value(
    pool: &SqlitePool,
    media_file_id: i64,
    searchable_config_id: i64,
    value: serde_json::Value,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO searchable_values (media_file_id, searchable_config_id, value_json) VALUES (?1, ?2, ?3)
         ON CONFLICT(media_file_id, searchable_config_id) DO UPDATE SET
             value_json = excluded.value_json,
             updated_at = CURRENT_TIMESTAMP"
    )
    .bind(media_file_id)
    .bind(searchable_config_id)
    .bind(value)
    .execute(pool)
    .await?;
    Ok(())
}

/// Delete all computed Searchable values for a media file. Useful on rescan/refresh.
pub async fn delete_values_for_media(pool: &SqlitePool, media_file_id: i64) -> Result<u64> {
    let rows = sqlx::query("DELETE FROM searchable_values WHERE media_file_id = ?1")
        .bind(media_file_id)
        .execute(pool)
        .await?
        .rows_affected();
    Ok(rows)
}

/// Delete all computed values for a given Searchable config.
pub async fn delete_values_for_config(pool: &SqlitePool, searchable_config_id: i64) -> Result<u64> {
    let rows = sqlx::query("DELETE FROM searchable_values WHERE searchable_config_id = ?1")
        .bind(searchable_config_id)
        .execute(pool)
        .await?
        .rows_affected();
    Ok(rows)
}

/// Enqueue a background inference job for a media file / Searchable pair.
pub async fn enqueue_job(
    pool: &SqlitePool,
    media_file_id: i64,
    searchable_config_id: i64,
) -> Result<i64> {
    // Avoid duplicate pending jobs.
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT id FROM job_queue WHERE media_file_id = ?1 AND searchable_config_id = ?2 AND status = 'pending'"
    )
    .bind(media_file_id)
    .bind(searchable_config_id)
    .fetch_optional(pool)
    .await?;

    if let Some(id) = existing {
        return Ok(id);
    }

    let id = sqlx::query(
        "INSERT INTO job_queue (media_file_id, searchable_config_id, status, attempts) VALUES (?1, ?2, 'pending', 0)"
    )
    .bind(media_file_id)
    .bind(searchable_config_id)
    .execute(pool)
    .await?
    .last_insert_rowid();
    Ok(id)
}

/// Claim the next batch of pending jobs.
pub async fn claim_pending_jobs(pool: &SqlitePool, limit: i64) -> Result<Vec<JobRow>> {
    let rows = sqlx::query_as::<_, JobRow>(
        "UPDATE job_queue
         SET status = 'running', updated_at = CURRENT_TIMESTAMP
         WHERE id IN (
             SELECT id FROM job_queue WHERE status = 'pending' ORDER BY created_at LIMIT ?1
         )
         RETURNING id, media_file_id, searchable_config_id, status, attempts, error, created_at, updated_at"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Mark a job as completed.
pub async fn complete_job(pool: &SqlitePool, job_id: i64) -> Result<()> {
    sqlx::query(
        "UPDATE job_queue SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE id = ?1"
    )
    .bind(job_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Mark a job as failed and increment its attempt counter.
pub async fn fail_job(pool: &SqlitePool, job_id: i64, error: &str) -> Result<()> {
    sqlx::query(
        "UPDATE job_queue SET status = 'failed', attempts = attempts + 1, error = ?1, updated_at = CURRENT_TIMESTAMP WHERE id = ?2"
    )
    .bind(error)
    .bind(job_id)
    .execute(pool)
    .await?;
    Ok(())
}

#[derive(Debug, Clone, FromRow)]
pub struct JobRow {
    pub id: i64,
    pub media_file_id: i64,
    pub searchable_config_id: i64,
    pub status: String,
    pub attempts: i64,
    pub error: Option<String>,
    pub created_at: chrono::NaiveDateTime,
    pub updated_at: chrono::NaiveDateTime,
}
