use sqlx::SqlitePool;

#[derive(Debug, Clone)]
pub struct Folder {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub path: String,
    pub recursive: bool,
    pub scan_complete: bool,
    pub exclude: Vec<String>,
    pub include: Vec<String>,
    pub thumbnail_cache_mode: Option<String>,
    pub thumbnail_cache_folder: Option<String>,
    pub thumbnail_cache_fallback: String,
    pub is_present: bool,
}

const FOLDER_COLUMNS: &str = "id, parent_id, path, recursive, scan_complete, exclude, include, thumbnail_cache_mode, thumbnail_cache_folder, thumbnail_cache_fallback, is_present";

pub async fn list_all(pool: &SqlitePool) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>(
        &format!("SELECT {FOLDER_COLUMNS} FROM folders ORDER BY path")
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn list_roots(pool: &SqlitePool) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>(
        &format!("SELECT {FOLDER_COLUMNS} FROM folders WHERE parent_id IS NULL")
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn list_children(pool: &SqlitePool, parent_id: i64) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>(
        &format!("SELECT {FOLDER_COLUMNS} FROM folders WHERE parent_id = ?1 ORDER BY path")
    )
    .bind(parent_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn get_by_path(pool: &SqlitePool, path: &str) -> anyhow::Result<Option<Folder>> {
    let row = sqlx::query_as::<_, FolderRow>(
        &format!("SELECT {FOLDER_COLUMNS} FROM folders WHERE path = ?1")
    )
    .bind(path)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(into_folder))
}

pub async fn insert(
    pool: &SqlitePool,
    parent_id: Option<i64>,
    path: &str,
    recursive: bool,
    scan_complete: bool,
    exclude: &[String],
    include: &[String],
    cache_mode: Option<&str>,
    cache_folder: Option<&str>,
    cache_fallback: &str,
) -> anyhow::Result<i64> {
    let exclude_json = serde_json::to_string(exclude)?;
    let include_json = serde_json::to_string(include)?;
    let id = sqlx::query(
        "INSERT INTO folders (parent_id, path, recursive, scan_complete, exclude, include, thumbnail_cache_mode, thumbnail_cache_folder, thumbnail_cache_fallback)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
    )
    .bind(parent_id)
    .bind(path)
    .bind(recursive)
    .bind(scan_complete)
    .bind(exclude_json)
    .bind(include_json)
    .bind(cache_mode)
    .bind(cache_folder)
    .bind(cache_fallback)
    .execute(pool)
    .await?
    .last_insert_rowid();

    Ok(id)
}

pub async fn get_or_create(
    pool: &SqlitePool,
    parent_id: Option<i64>,
    path: &str,
    recursive: bool,
    scan_complete: bool,
    exclude: &[String],
    include: &[String],
    cache_mode: Option<&str>,
    cache_folder: Option<&str>,
    cache_fallback: &str,
) -> anyhow::Result<i64> {
    if let Some(folder) = get_by_path(pool, path).await? {
        // Config is the source of truth for exclude/include; the stored row is
        // only written at insert time, so resync it when the filters changed.
        if folder.exclude != exclude || folder.include != include {
            sqlx::query("UPDATE folders SET exclude = ?1, include = ?2 WHERE id = ?3")
                .bind(serde_json::to_string(exclude)?)
                .bind(serde_json::to_string(include)?)
                .bind(folder.id)
                .execute(pool)
                .await?;
        }
        return Ok(folder.id);
    }
    insert(
        pool,
        parent_id,
        path,
        recursive,
        scan_complete,
        exclude,
        include,
        cache_mode,
        cache_folder,
        cache_fallback,
    )
    .await
}

pub async fn update_scan_complete(
    pool: &SqlitePool,
    folder_id: i64,
    scan_complete: bool,
) -> anyhow::Result<()> {
    sqlx::query("UPDATE folders SET scan_complete = ?1 WHERE id = ?2")
        .bind(scan_complete)
        .bind(folder_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_scan_complete_recursive(
    pool: &SqlitePool,
    folder_id: i64,
    scan_complete: bool,
) -> anyhow::Result<u64> {
    let result = sqlx::query(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         UPDATE folders SET scan_complete = ?2 WHERE id IN (SELECT id FROM subtree)"
    )
    .bind(folder_id)
    .bind(scan_complete)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

/// List `(id, path, is_present)` for a folder and all its descendants. Used by
/// the scanner to reconcile folder rows against what's actually on disk.
pub async fn list_subtree(pool: &SqlitePool, root_id: i64) -> anyhow::Result<Vec<(i64, String, bool)>> {
    let rows = sqlx::query_as::<_, (i64, String, i64)>(
        "WITH RECURSIVE subtree(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.id FROM folders JOIN subtree ON folders.parent_id = subtree.id
         )
         SELECT id, path, is_present FROM folders WHERE id IN (SELECT id FROM subtree)"
    )
    .bind(root_id)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(id, path, p)| (id, path, p != 0)).collect())
}

/// Mark a folder missing along with all of its media. Subfolders are expected
/// to be marked separately (their paths vanish with the parent).
pub async fn mark_missing(pool: &SqlitePool, folder_id: i64) -> anyhow::Result<u64> {
    let mut tx = pool.begin().await?;

    sqlx::query(
        "UPDATE media_files SET is_present = 0, missing_since = CURRENT_TIMESTAMP
         WHERE folder_id = ?1 AND is_present = 1"
    )
    .bind(folder_id)
    .execute(&mut *tx)
    .await?;

    let result = sqlx::query(
        "UPDATE folders SET is_present = 0, missing_since = CURRENT_TIMESTAMP
         WHERE id = ?1 AND is_present = 1"
    )
    .bind(folder_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(result.rows_affected())
}

/// Mark the folder at `path` (and everything under it, folders and media)
/// missing. Used by the file watcher when a directory is removed; a no-op for
/// paths that aren't tracked folders (e.g. individual removed files).
pub async fn mark_missing_by_path(pool: &SqlitePool, path: &str) -> anyhow::Result<u64> {
    let mut tx = pool.begin().await?;

    // substr-based prefix match avoids LIKE wildcard escaping issues.
    let prefix = format!("{path}/");
    let prefix_len = prefix.len() as i64;
    const FOLDER_MATCH: &str = "(path = ?1 OR substr(path, 1, ?2) = ?3)";

    sqlx::query(
        &format!(
            "UPDATE media_files SET is_present = 0, missing_since = CURRENT_TIMESTAMP
             WHERE is_present = 1 AND folder_id IN (SELECT id FROM folders WHERE {FOLDER_MATCH})"
        )
    )
    .bind(path)
    .bind(prefix_len)
    .bind(&prefix)
    .execute(&mut *tx)
    .await?;

    let result = sqlx::query(
        &format!(
            "UPDATE folders SET is_present = 0, missing_since = CURRENT_TIMESTAMP
             WHERE is_present = 1 AND {FOLDER_MATCH}"
        )
    )
    .bind(path)
    .bind(prefix_len)
    .bind(&prefix)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(result.rows_affected())
}

/// Restore a folder (and any missing ancestors) to present. Used when a file
/// event or scan sees a path that was previously marked missing.
pub async fn mark_present_with_ancestors(pool: &SqlitePool, folder_id: i64) -> anyhow::Result<u64> {
    let result = sqlx::query(
        "WITH RECURSIVE ancestors(id) AS (
            SELECT ?1
            UNION ALL
            SELECT folders.parent_id FROM folders
            JOIN ancestors ON folders.id = ancestors.id
            WHERE folders.parent_id IS NOT NULL
         )
         UPDATE folders SET is_present = 1, missing_since = NULL
         WHERE id IN (SELECT id FROM ancestors) AND is_present = 0"
    )
    .bind(folder_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

#[derive(sqlx::FromRow)]
struct FolderRow {
    id: i64,
    parent_id: Option<i64>,
    path: String,
    recursive: i64,
    scan_complete: i64,
    exclude: String,
    include: String,
    thumbnail_cache_mode: Option<String>,
    thumbnail_cache_folder: Option<String>,
    thumbnail_cache_fallback: String,
    is_present: i64,
}

fn into_folder(row: FolderRow) -> Folder {
    Folder {
        id: row.id,
        parent_id: row.parent_id,
        path: row.path,
        recursive: row.recursive != 0,
        scan_complete: row.scan_complete != 0,
        exclude: serde_json::from_str(&row.exclude).unwrap_or_default(),
        include: serde_json::from_str(&row.include).unwrap_or_default(),
        thumbnail_cache_mode: row.thumbnail_cache_mode,
        thumbnail_cache_folder: row.thumbnail_cache_folder,
        thumbnail_cache_fallback: row.thumbnail_cache_fallback,
        is_present: row.is_present != 0,
    }
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
    async fn mark_missing_by_path_covers_subtree_and_media() {
        let pool = setup_pool().await;
        let root = insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let sub = insert(&pool, Some(root), "/tmp/root/sub", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let nested = insert(&pool, Some(sub), "/tmp/root/sub/nested", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let other = insert(&pool, Some(root), "/tmp/root/other", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();

        let mid = crate::db::media::upsert(
            &pool, nested, "a.jpg", "/tmp/root/sub/nested/a.jpg", "hash",
            None, None, None, None, None,
        )
        .await
        .unwrap();

        // Removing "/tmp/root/sub" marks the whole subtree missing.
        let marked = mark_missing_by_path(&pool, "/tmp/root/sub").await.unwrap();
        assert_eq!(marked, 2);

        for path in ["/tmp/root/sub", "/tmp/root/sub/nested"] {
            let f = get_by_path(&pool, path).await.unwrap().unwrap();
            assert!(!f.is_present, "{path} should be missing");
        }
        assert!(get_by_path(&pool, "/tmp/root/other").await.unwrap().unwrap().is_present);
        assert!(get_by_path(&pool, "/tmp/root").await.unwrap().unwrap().is_present);

        // Media under the subtree is marked missing too.
        let media = crate::db::media::get_by_id(&pool, mid).await.unwrap().unwrap();
        assert!(!media.is_present);

        // A path that isn't a tracked folder is a no-op.
        assert_eq!(mark_missing_by_path(&pool, "/tmp/root/other/file.jpg").await.unwrap(), 0);

        // Restore: bringing "nested" back also restores its missing ancestors.
        let restored = mark_present_with_ancestors(&pool, nested).await.unwrap();
        assert_eq!(restored, 2);
        assert!(get_by_path(&pool, "/tmp/root/sub").await.unwrap().unwrap().is_present);
        assert!(get_by_path(&pool, "/tmp/root/sub/nested").await.unwrap().unwrap().is_present);
        let _ = other;
    }

    #[tokio::test]
    async fn list_subtree_reports_presence() {
        let pool = setup_pool().await;
        let root = insert(&pool, None, "/tmp/root", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let sub = insert(&pool, Some(root), "/tmp/root/sub", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        mark_missing(&pool, sub).await.unwrap();

        let subtree = list_subtree(&pool, root).await.unwrap();
        assert_eq!(subtree.len(), 2);
        let sub_row = subtree.iter().find(|(id, _, _)| *id == sub).unwrap();
        assert!(!sub_row.2);
        let root_row = subtree.iter().find(|(id, _, _)| *id == root).unwrap();
        assert!(root_row.2);
    }
}
