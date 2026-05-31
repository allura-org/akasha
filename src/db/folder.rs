use sqlx::SqlitePool;

#[derive(Debug, Clone)]
pub struct Folder {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub path: String,
    pub recursive: bool,
    pub show_recursive: bool,
    pub scan_complete: bool,
    pub blacklist: Vec<String>,
    pub thumbnail_cache_mode: Option<String>,
}

pub async fn list_all(pool: &SqlitePool) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>("SELECT * FROM folders ORDER BY path")
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn list_roots(pool: &SqlitePool) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>(
        "SELECT * FROM folders WHERE parent_id IS NULL"
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn list_children(pool: &SqlitePool, parent_id: i64) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>(
        "SELECT * FROM folders WHERE parent_id = ?1 ORDER BY path"
    )
    .bind(parent_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn get_by_path(pool: &SqlitePool, path: &str) -> anyhow::Result<Option<Folder>> {
    let row = sqlx::query_as::<_, FolderRow>("SELECT * FROM folders WHERE path = ?1")
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
    show_recursive: bool,
    scan_complete: bool,
    blacklist: &[String],
    cache_mode: Option<&str>,
) -> anyhow::Result<i64> {
    let blacklist_json = serde_json::to_string(blacklist)?;
    let id = sqlx::query(
        "INSERT INTO folders (parent_id, path, recursive, show_recursive, scan_complete, blacklist, thumbnail_cache_mode)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
    )
    .bind(parent_id)
    .bind(path)
    .bind(recursive)
    .bind(show_recursive)
    .bind(scan_complete)
    .bind(blacklist_json)
    .bind(cache_mode)
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
    show_recursive: bool,
    scan_complete: bool,
    blacklist: &[String],
    cache_mode: Option<&str>,
) -> anyhow::Result<i64> {
    if let Some(folder) = get_by_path(pool, path).await? {
        return Ok(folder.id);
    }
    insert(pool, parent_id, path, recursive, show_recursive, scan_complete, blacklist, cache_mode).await
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

pub async fn update_show_recursive(
    pool: &SqlitePool,
    folder_id: i64,
    show_recursive: bool,
) -> anyhow::Result<()> {
    sqlx::query("UPDATE folders SET show_recursive = ?1 WHERE id = ?2")
        .bind(show_recursive)
        .bind(folder_id)
        .execute(pool)
        .await?;
    Ok(())
}

#[derive(sqlx::FromRow)]
struct FolderRow {
    id: i64,
    parent_id: Option<i64>,
    path: String,
    recursive: i64,
    show_recursive: i64,
    scan_complete: i64,
    blacklist: String,
    thumbnail_cache_mode: Option<String>,
}

fn into_folder(row: FolderRow) -> Folder {
    Folder {
        id: row.id,
        parent_id: row.parent_id,
        path: row.path,
        recursive: row.recursive != 0,
        show_recursive: row.show_recursive != 0,
        scan_complete: row.scan_complete != 0,
        blacklist: serde_json::from_str(&row.blacklist).unwrap_or_default(),
        thumbnail_cache_mode: row.thumbnail_cache_mode,
    }
}
