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
    pub updated_at: chrono::NaiveDateTime,
}

/// Load every Searchable configuration from the database.
pub async fn list_searchable_configs(pool: &SqlitePool) -> Result<Vec<SearchableConfig>> {
    let rows = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at, updated_at FROM searchable_configs ORDER BY name"
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
        "INSERT INTO searchable_configs (name, kind, enabled, options, updated_at) VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)"
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

/// Insert or update a Searchable configuration, keyed by its (name, kind) pair.
/// Returns the configuration's row id, regardless of whether it was inserted or
/// updated.
pub async fn upsert_config(
    pool: &SqlitePool,
    name: &str,
    kind: &str,
    enabled: bool,
    options: serde_json::Value,
) -> Result<i64> {
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO searchable_configs (name, kind, enabled, options, updated_at)
         VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)
         ON CONFLICT(name, kind) DO UPDATE SET
             enabled = excluded.enabled,
             options = excluded.options,
             updated_at = CURRENT_TIMESTAMP
         RETURNING id"
    )
    .bind(name)
    .bind(kind)
    .bind(enabled)
    .bind(options)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

/// Fetch a Searchable configuration by its (name, kind) pair.
pub async fn get_config_by_name_kind(
    pool: &SqlitePool,
    name: &str,
    kind: &str,
) -> Result<Option<SearchableConfig>> {
    let row = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at, updated_at FROM searchable_configs
         WHERE name = ?1 AND kind = ?2"
    )
    .bind(name)
    .bind(kind)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Fetch a Searchable configuration by its database id.
pub async fn get_config_by_id(pool: &SqlitePool, id: i64) -> Result<Option<SearchableConfig>> {
    let row = sqlx::query_as::<_, SearchableConfig>(
        "SELECT id, name, kind, enabled, options, created_at, updated_at FROM searchable_configs
         WHERE id = ?1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
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

/// Update the `tags_json` column for a media file and mirror the tags into
/// `searchable_tags`. Both writes happen in a single transaction.
pub async fn update_tags_json(
    pool: &SqlitePool,
    media_file_id: i64,
    source: &str,
    tags: std::collections::HashMap<String, f32>,
) -> Result<()> {
    let mut tx = pool.begin().await?;

    // Read existing tags_json, update the source entry.
    let existing: Option<String> = sqlx::query_scalar(
        "SELECT tags_json FROM media_files WHERE id = ?1"
    )
    .bind(media_file_id)
    .fetch_optional(&mut *tx)
    .await?;

    let mut map: serde_json::Map<String, serde_json::Value> = existing
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    map.insert(source.to_string(), serde_json::to_value(&tags)?);

    sqlx::query("UPDATE media_files SET tags_json = ?1 WHERE id = ?2")
        .bind(serde_json::to_string(&map)?)
        .bind(media_file_id)
        .execute(&mut *tx)
        .await?;

    // Mirror into searchable_tags.
    sqlx::query("DELETE FROM searchable_tags WHERE media_file_id = ?1 AND source = ?2")
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;

    // Keep the FTS5 trigram side table in sync with searchable_tags.
    sqlx::query("DELETE FROM searchable_tags_fts WHERE media_file_id = ?1 AND source = ?2")
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;

    for (tag, score) in tags {
        sqlx::query(
            "INSERT INTO searchable_tags (media_file_id, source, tag, score)
             VALUES (?1, ?2, ?3, ?4)"
        )
        .bind(media_file_id)
        .bind(source)
        .bind(tag.as_str())
        .bind(score)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO searchable_tags_fts (tag, media_file_id, source) VALUES (?1, ?2, ?3)"
        )
        .bind(tag.as_str())
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

/// Update the `descriptions_json` column for a media file and mirror the
/// description into the FTS5 side table. Both writes happen in a single
/// transaction.
pub async fn update_description_json(
    pool: &SqlitePool,
    media_file_id: i64,
    source: &str,
    description: &str,
) -> Result<()> {
    let mut tx = pool.begin().await?;

    let existing: Option<String> = sqlx::query_scalar(
        "SELECT descriptions_json FROM media_files WHERE id = ?1"
    )
    .bind(media_file_id)
    .fetch_optional(&mut *tx)
    .await?;

    let mut map: serde_json::Map<String, serde_json::Value> = existing
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    map.insert(source.to_string(), serde_json::Value::String(description.to_string()));

    sqlx::query("UPDATE media_files SET descriptions_json = ?1 WHERE id = ?2")
        .bind(serde_json::to_string(&map)?)
        .bind(media_file_id)
        .execute(&mut *tx)
        .await?;

    // Mirror into FTS5. UPSERT is not supported on virtual tables, so use
    // INSERT OR REPLACE (rowid is the FTS5 docid and matches media_file_id).
    // NOTE: Because rowid is tied directly to media_file_id, this design only
    // supports a single description source per media file; a later call with a
    // different `source` will overwrite the previous FTS5 row.
    sqlx::query(
        "INSERT OR REPLACE INTO searchable_text_fts (rowid, media_file_id, source, content)
         VALUES (?1, ?1, ?2, ?3)"
    )
    .bind(media_file_id)
    .bind(source)
    .bind(description)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(())
}

/// Delete all tags for a given media file and source, updating both
/// `searchable_tags` and `media_files.tags_json`.
pub async fn delete_tags_for_source(
    pool: &SqlitePool,
    media_file_id: i64,
    source: &str,
) -> Result<()> {
    let mut tx = pool.begin().await?;

    sqlx::query("DELETE FROM searchable_tags WHERE media_file_id = ?1 AND source = ?2")
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;

    sqlx::query("DELETE FROM searchable_tags_fts WHERE media_file_id = ?1 AND source = ?2")
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;

    let existing: Option<String> = sqlx::query_scalar(
        "SELECT tags_json FROM media_files WHERE id = ?1"
    )
    .bind(media_file_id)
    .fetch_optional(&mut *tx)
    .await?;

    let mut map: serde_json::Map<String, serde_json::Value> = existing
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    map.remove(source);

    sqlx::query("UPDATE media_files SET tags_json = ?1 WHERE id = ?2")
        .bind(serde_json::to_string(&map)?)
        .bind(media_file_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(())
}

/// Delete a description for a given media file and source, updating both
/// `searchable_text_fts` and `media_files.descriptions_json`.
pub async fn delete_description_for_source(
    pool: &SqlitePool,
    media_file_id: i64,
    source: &str,
) -> Result<()> {
    let mut tx = pool.begin().await?;

    sqlx::query("DELETE FROM searchable_text_fts WHERE media_file_id = ?1 AND source = ?2")
        .bind(media_file_id)
        .bind(source)
        .execute(&mut *tx)
        .await?;

    let existing: Option<String> = sqlx::query_scalar(
        "SELECT descriptions_json FROM media_files WHERE id = ?1"
    )
    .bind(media_file_id)
    .fetch_optional(&mut *tx)
    .await?;

    let mut map: serde_json::Map<String, serde_json::Value> = existing
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    map.remove(source);

    sqlx::query("UPDATE media_files SET descriptions_json = ?1 WHERE id = ?2")
        .bind(serde_json::to_string(&map)?)
        .bind(media_file_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
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

/// Read the current `[[models]]` registry and upsert a `searchable_configs`
/// row for each output kind declared by each model. Any existing rows that are
/// no longer present in the registry are disabled (not deleted) so their values
/// remain queryable but are no longer shown as active sources.
pub async fn sync_model_configs(
    pool: &SqlitePool,
    models: &[crate::config::ModelConfig],
) -> Result<()> {
    let mut wanted = std::collections::HashSet::new();

    for model in models {
        let base_options = serde_json::json!({
            "path": model.path,
            "base_url": model.base_url,
            "model_id": model.model_id,
            "api_key": model.api_key,
            "backend": model.backend,
            "remote": model.remote,
            "onnx": model.onnx,
            "kind": model.kind,
        });

        if let Some(tags) = &model.tags {
            let mut options = serde_json::to_value(tags)?;
            merge_json(&mut options, base_options.clone());
            upsert_config(pool, &model.name, "tags", true, options).await?;
            wanted.insert((model.name.clone(), "tags".to_string()));
        }
        if let Some(description) = &model.description {
            let mut options = serde_json::to_value(description)?;
            merge_json(&mut options, base_options.clone());
            upsert_config(pool, &model.name, "description", true, options).await?;
            wanted.insert((model.name.clone(), "description".to_string()));
        }
        if let Some(classification) = &model.classification {
            let mut options = serde_json::to_value(classification)?;
            merge_json(&mut options, base_options);
            upsert_config(pool, &model.name, "classification", true, options).await?;
            wanted.insert((model.name.clone(), "classification".to_string()));
        }
    }

    fn merge_json(target: &mut serde_json::Value, source: serde_json::Value) {
        if let (serde_json::Value::Object(t), serde_json::Value::Object(s)) = (target, source) {
            for (k, v) in s {
                t.insert(k, v);
            }
        }
    }

    let existing = list_searchable_configs(pool).await?;
    for cfg in existing {
        // Never disable built-in text Searchables (e.g. the seeded `filename`
        // row). They are not part of the user-configured [[models]] registry but
        // must remain active.
        if cfg.kind == "text" {
            continue;
        }
        if !wanted.contains(&(cfg.name.clone(), cfg.kind.clone())) {
            sqlx::query("UPDATE searchable_configs SET enabled = 0 WHERE id = ?1")
                .bind(cfg.id)
                .execute(pool)
                .await?;
        }
    }

    Ok(())
}

/// Reconstruct a `ModelConfig` from a persisted `SearchableConfig` row.
///
/// The `SearchableConfig` options are expected to contain the base model
/// fields written by `sync_model_configs`. Output-kind specific options such
/// as `threshold` and `prompt` are mapped back to their respective option
/// structs.
pub fn model_config_from_searchable_config(cfg: &SearchableConfig) -> Result<crate::config::ModelConfig> {
    let opts = &cfg.options;

    let kind = opts
        .get("kind")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(crate::config::ModelKind::Local);

    let remote: Option<crate::config::ModelRemoteOptions> = opts
        .get("remote")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    let onnx: Option<crate::config::ModelOnnxOptions> = opts
        .get("onnx")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    let tags: Option<crate::config::ModelTagsOptions> = opts
        .get("threshold")
        .and_then(|_| serde_json::from_value(opts.clone()).ok());

    let description: Option<crate::config::ModelDescriptionOptions> = opts
        .get("prompt")
        .and_then(|_| serde_json::from_value(opts.clone()).ok());

    let classification: Option<crate::config::ModelClassificationOptions> =
        if cfg.kind == "classification" {
            Some(crate::config::ModelClassificationOptions {})
        } else {
            None
        };

    Ok(crate::config::ModelConfig {
        name: cfg.name.clone(),
        kind,
        backend: opts.get("backend").and_then(|v| v.as_str()).map(|s| s.to_string()),
        path: opts.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()),
        base_url: opts.get("base_url").and_then(|v| v.as_str()).map(|s| s.to_string()),
        model_id: opts.get("model_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
        api_key: opts.get("api_key").and_then(|v| v.as_str()).map(|s| s.to_string()),
        remote,
        onnx,
        tags,
        description,
        classification,
    })
}

/// Enqueue a background processing job for a media file.
pub async fn enqueue_job(
    pool: &SqlitePool,
    media_file_id: i64,
    job_kind: &str,
    params_json: &str,
    searchable_config_id: Option<i64>,
) -> Result<i64> {
    // Avoid duplicate pending jobs for the same media/config/kind/params combination.
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT id FROM job_queue
         WHERE media_file_id = ?1 AND job_kind = ?2 AND params_json = ?3
           AND searchable_config_id IS ?4 AND status = 'pending'"
    )
    .bind(media_file_id)
    .bind(job_kind)
    .bind(params_json)
    .bind(searchable_config_id)
    .fetch_optional(pool)
    .await?;

    if let Some(id) = existing {
        return Ok(id);
    }

    let id = sqlx::query(
        "INSERT INTO job_queue (media_file_id, searchable_config_id, job_kind, params_json, status, attempts)
         VALUES (?1, ?2, ?3, ?4, 'pending', 0)"
    )
    .bind(media_file_id)
    .bind(searchable_config_id)
    .bind(job_kind)
    .bind(params_json)
    .execute(pool)
    .await?
    .last_insert_rowid();
    Ok(id)
}

/// Claim the next batch of pending jobs for media files that are still present.
/// Jobs for missing files are left pending; they will be retried automatically
/// if the file reappears, or dropped when the record is purged.
pub async fn claim_pending_jobs(pool: &SqlitePool, limit: i64) -> Result<Vec<JobRow>> {
    let rows = sqlx::query_as::<_, JobRow>(
        "UPDATE job_queue
         SET status = 'running', updated_at = CURRENT_TIMESTAMP
         WHERE id IN (
             SELECT j.id
             FROM job_queue j
             JOIN media_files m ON m.id = j.media_file_id
             WHERE j.status = 'pending' AND m.is_present = 1
             ORDER BY j.searchable_config_id, j.created_at
             LIMIT ?1
         )
         RETURNING id, media_file_id, searchable_config_id, job_kind, params_json, status, attempts, error, created_at, updated_at"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Count jobs that are either pending or running.
pub async fn count_pending_jobs(pool: &SqlitePool) -> Result<i64> {
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM job_queue WHERE status IN ('pending', 'running')"
    )
    .fetch_one(pool)
    .await?;
    Ok(row.0)
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

/// Reset any jobs left in the `running` state back to `pending`. This is
/// intended to be called once at startup so that jobs abandoned by a previous
/// process crash or unclean shutdown are retried instead of staying stuck.
pub async fn reset_running_jobs(pool: &SqlitePool) -> Result<u64> {
    let result = sqlx::query(
        "UPDATE job_queue SET status = 'pending', updated_at = CURRENT_TIMESTAMP WHERE status = 'running'"
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

#[derive(Debug, Clone, FromRow)]
pub struct JobRow {
    pub id: i64,
    pub media_file_id: i64,
    pub searchable_config_id: Option<i64>,
    pub job_kind: String,
    pub params_json: Option<String>,
    pub status: String,
    pub attempts: i64,
    pub error: Option<String>,
    pub created_at: chrono::NaiveDateTime,
    pub updated_at: chrono::NaiveDateTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn enqueue_and_claim_ai_job_round_trip() {
        let pool = setup_pool().await;

        let fid = crate::db::folder::insert(
            &pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable",
        )
        .await
        .unwrap();

        let media_id = crate::db::media::upsert(
            &pool,
            fid,
            "foo.jpg",
            "/tmp/root/foo.jpg",
            "hash",
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let params = r#"{"model_name":"wd14"}"#;
        let job_id = enqueue_job(&pool, media_id, "tagger", params, None)
            .await
            .unwrap();

        let pending = count_pending_jobs(&pool).await.unwrap();
        assert_eq!(pending, 1);

        let jobs = claim_pending_jobs(&pool, 10).await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, job_id);
        assert_eq!(jobs[0].media_file_id, media_id);
        assert_eq!(jobs[0].job_kind, "tagger");
        assert_eq!(jobs[0].params_json.as_deref(), Some(params));
        assert!(jobs[0].searchable_config_id.is_none());
    }

    #[tokio::test]
    async fn upsert_config_round_trip() {
        let pool = setup_pool().await;

        let id1 = upsert_config(
            &pool,
            "filename",
            "text",
            true,
            serde_json::json!({"boost": 1.0}),
        )
        .await
        .unwrap();

        let id2 = upsert_config(
            &pool,
            "filename",
            "text",
            false,
            serde_json::json!({"boost": 2.0}),
        )
        .await
        .unwrap();

        assert_eq!(id1, id2);

        let cfg = get_config_by_name_kind(&pool, "filename", "text")
            .await
            .unwrap()
            .expect("config should exist");
        assert_eq!(cfg.id, id1);
        assert!(!cfg.enabled);
        assert_eq!(cfg.options["boost"], 2.0);
    }

    #[tokio::test]
    async fn update_tags_json_writes_column_and_side_table() {
        let pool = setup_pool().await;
        let fid = crate::db::folder::insert(
            &pool, None, "/tmp", true, false, &[], &[], None, None, "disable",
        )
        .await
        .unwrap();
        let mid = crate::db::media::upsert(
            &pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None,
        )
        .await
        .unwrap();

        let mut tags = std::collections::HashMap::new();
        tags.insert("cat".to_string(), 0.9f32);
        update_tags_json(&pool, mid, "wd-vit", tags).await.unwrap();

        let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
            .bind(mid)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(row.0.contains("cat"));

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM searchable_tags WHERE media_file_id = ?1")
                .bind(mid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 1);

        let fts_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM searchable_tags_fts WHERE media_file_id = ?1 AND source = ?2"
        )
        .bind(mid)
        .bind("wd-vit")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(fts_count.0, 1);

        let fts_tag: (String,) = sqlx::query_as(
            "SELECT tag FROM searchable_tags_fts WHERE media_file_id = ?1 AND source = ?2"
        )
        .bind(mid)
        .bind("wd-vit")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(fts_tag.0, "cat");
    }

    #[tokio::test]
    async fn update_description_json_writes_column_and_fts() {
        let pool = setup_pool().await;
        let fid = crate::db::folder::insert(
            &pool, None, "/tmp", true, false, &[], &[], None, None, "disable",
        )
        .await
        .unwrap();
        let mid = crate::db::media::upsert(
            &pool, fid, "b.jpg", "/tmp/b.jpg", "hash", None, None, None, None, None,
        )
        .await
        .unwrap();

        update_description_json(&pool, mid, "blip", "a cat on a mat")
            .await
            .unwrap();

        let row: (String,) =
            sqlx::query_as("SELECT descriptions_json FROM media_files WHERE id = ?1")
                .bind(mid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(row.0.contains("cat"));

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM searchable_text_fts WHERE media_file_id = ?1"
        )
        .bind(mid)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count.0, 1);
    }

    #[tokio::test]
    async fn delete_tags_for_source_cleans_column_and_side_table() {
        let pool = setup_pool().await;
        let fid = crate::db::folder::insert(
            &pool, None, "/tmp", true, false, &[], &[], None, None, "disable",
        )
        .await
        .unwrap();
        let mid = crate::db::media::upsert(
            &pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None,
        )
        .await
        .unwrap();

        let mut tags = std::collections::HashMap::new();
        tags.insert("cat".to_string(), 0.9f32);
        update_tags_json(&pool, mid, "wd-vit", tags).await.unwrap();

        delete_tags_for_source(&pool, mid, "wd-vit").await.unwrap();

        let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
            .bind(mid)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(!row.0.contains("wd-vit"));

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM searchable_tags WHERE media_file_id = ?1")
                .bind(mid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 0);

        let fts_count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM searchable_tags_fts WHERE media_file_id = ?1")
                .bind(mid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(fts_count.0, 0);
    }

    #[tokio::test]
    async fn delete_description_for_source_cleans_column_and_fts() {
        let pool = setup_pool().await;
        let fid = crate::db::folder::insert(
            &pool, None, "/tmp", true, false, &[], &[], None, None, "disable",
        )
        .await
        .unwrap();
        let mid = crate::db::media::upsert(
            &pool, fid, "b.jpg", "/tmp/b.jpg", "hash", None, None, None, None, None,
        )
        .await
        .unwrap();

        update_description_json(&pool, mid, "blip", "a cat on a mat")
            .await
            .unwrap();

        delete_description_for_source(&pool, mid, "blip").await.unwrap();

        let row: (String,) =
            sqlx::query_as("SELECT descriptions_json FROM media_files WHERE id = ?1")
                .bind(mid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(!row.0.contains("blip"));

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM searchable_text_fts WHERE media_file_id = ?1"
        )
        .bind(mid)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count.0, 0);
    }

    #[tokio::test]
    async fn sync_model_configs_upserts_and_disables_stale() {
        use crate::config::{
            ModelClassificationOptions, ModelConfig, ModelDescriptionOptions, ModelKind,
            ModelTagsOptions,
        };

        let pool = setup_pool().await;

        let model = ModelConfig {
            name: "test-model".into(),
            kind: ModelKind::Local,
            backend: None,
            path: Some("/models/test".into()),
            base_url: None,
            model_id: Some("id1".into()),
            api_key: None,
            tags: Some(ModelTagsOptions::default()),
            description: None,
            classification: Some(ModelClassificationOptions {}),
            remote: None,
            onnx: None,
        };
        sync_model_configs(&pool, &[model]).await.unwrap();

        let configs = list_searchable_configs(&pool).await.unwrap();
        // The seed `filename` config is also present.
        assert_eq!(configs.len(), 3);

        let tags = configs.iter().find(|c| c.kind == "tags").expect("tags config");
        assert_eq!(tags.name, "test-model");
        assert!(tags.enabled);
        assert_eq!(tags.options["path"].as_str(), Some("/models/test"));
        assert_eq!(tags.options["model_id"].as_str(), Some("id1"));
        assert!(tags.options["threshold"].as_f64().is_some());

        let classification = configs
            .iter()
            .find(|c| c.kind == "classification")
            .expect("classification config");
        assert!(classification.enabled);
        assert_eq!(classification.options["model_id"].as_str(), Some("id1"));
        assert_eq!(classification.options["path"].as_str(), Some("/models/test"));

        // Re-sync with a different output kind; previous rows should be disabled.
        let model2 = ModelConfig {
            name: "test-model".into(),
            kind: ModelKind::Local,
            backend: None,
            path: Some("/models/test".into()),
            base_url: None,
            model_id: Some("id1".into()),
            api_key: None,
            tags: None,
            description: Some(ModelDescriptionOptions {
                prompt: Some("describe".into()),
                ..Default::default()
            }),
            classification: None,
            remote: None,
            onnx: None,
        };
        sync_model_configs(&pool, &[model2]).await.unwrap();

        let configs = list_searchable_configs(&pool).await.unwrap();
        // Stale rows are disabled but remain in the table.
        assert_eq!(configs.len(), 4);

        let tags = configs
            .iter()
            .find(|c| c.kind == "tags" && c.name == "test-model")
            .unwrap();
        assert!(!tags.enabled);

        let classification = configs
            .iter()
            .find(|c| c.kind == "classification" && c.name == "test-model")
            .unwrap();
        assert!(!classification.enabled);

        let description = configs
            .iter()
            .find(|c| c.kind == "description" && c.name == "test-model")
            .unwrap();
        assert!(description.enabled);
        assert_eq!(description.options["prompt"].as_str(), Some("describe"));
        assert_eq!(description.options["path"].as_str(), Some("/models/test"));
    }

    #[tokio::test]
    async fn sync_model_configs_keeps_filename_enabled() {
        use crate::config::{ModelConfig, ModelKind};

        let pool = setup_pool().await;

        let model = ModelConfig {
            name: "test-model".into(),
            kind: ModelKind::Local,
            backend: None,
            path: Some("/models/test".into()),
            base_url: None,
            model_id: Some("id1".into()),
            api_key: None,
            tags: None,
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        };
        sync_model_configs(&pool, &[model]).await.unwrap();

        let filename = get_config_by_name_kind(&pool, "filename", "text")
            .await
            .unwrap()
            .expect("filename config should exist");
        assert!(filename.enabled);
    }

    #[tokio::test]
    async fn model_config_from_searchable_config_reconstructs_remote_options() {
        use crate::config::{
            ModelConfig, ModelKind, ModelRemoteOptions, ModelTagsOptions,
        };

        let pool = setup_pool().await;

        let model = ModelConfig {
            name: "remote-test".into(),
            kind: ModelKind::Remote,
            backend: Some("remote".into()),
            path: None,
            base_url: Some("https://example.com".into()),
            model_id: Some("m1".into()),
            api_key: Some("secret".into()),
            tags: Some(ModelTagsOptions::default()),
            description: None,
            classification: None,
            remote: Some(ModelRemoteOptions {
                chat_endpoint: "/v1/chat".into(),
                tag_endpoint: "/v1/tag".into(),
                classify_endpoint: "/v1/classify".into(),
            }),
            onnx: None,
        };
        sync_model_configs(&pool, &[model]).await.unwrap();

        let cfg = get_config_by_name_kind(&pool, "remote-test", "tags")
            .await
            .unwrap()
            .expect("tags config should exist");

        let reconstructed = model_config_from_searchable_config(&cfg).unwrap();
        assert_eq!(reconstructed.name, "remote-test");
        assert_eq!(reconstructed.kind, ModelKind::Remote);
        assert_eq!(reconstructed.backend.as_deref(), Some("remote"));
        assert_eq!(reconstructed.base_url.as_deref(), Some("https://example.com"));
        assert_eq!(reconstructed.model_id.as_deref(), Some("m1"));
        assert_eq!(reconstructed.api_key.as_deref(), Some("secret"));
        let remote = reconstructed.remote.expect("remote options should exist");
        assert_eq!(remote.chat_endpoint, "/v1/chat");
        assert_eq!(remote.tag_endpoint, "/v1/tag");
        assert_eq!(remote.classify_endpoint, "/v1/classify");
        assert!(reconstructed.tags.is_some());
    }

    #[tokio::test]
    async fn model_config_from_searchable_config_preserves_top_k() {
        use crate::config::{ModelConfig, ModelKind, ModelTagsOptions};

        let pool = setup_pool().await;

        let model = ModelConfig {
            name: "topk-test".into(),
            kind: ModelKind::Local,
            backend: None,
            path: Some("/models/test".into()),
            base_url: None,
            model_id: None,
            api_key: None,
            tags: Some(ModelTagsOptions {
                threshold: 0.25,
                top_k: Some(42),
            }),
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        };
        sync_model_configs(&pool, &[model]).await.unwrap();

        let cfg = get_config_by_name_kind(&pool, "topk-test", "tags")
            .await
            .unwrap()
            .expect("tags config should exist");

        let reconstructed = model_config_from_searchable_config(&cfg).unwrap();
        let tags = reconstructed.tags.expect("tags options should exist");
        assert!((tags.threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(tags.top_k, Some(42));
    }
}
