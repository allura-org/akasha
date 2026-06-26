use sqlx::SqlitePool;

#[derive(Debug, Clone)]
pub struct Folder {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub path: String,
    pub recursive: bool,
    pub flatten: bool,
    pub scan_complete: bool,
    pub exclude: Vec<String>,
    pub include: Vec<String>,
    pub thumbnail_cache_mode: Option<String>,
    pub thumbnail_cache_folder: Option<String>,
    pub thumbnail_cache_fallback: String,
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
    flatten: bool,
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
        "INSERT INTO folders (parent_id, path, recursive, flatten, scan_complete, exclude, include, thumbnail_cache_mode, thumbnail_cache_folder, thumbnail_cache_fallback)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"
    )
    .bind(parent_id)
    .bind(path)
    .bind(recursive)
    .bind(flatten)
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
    flatten: bool,
    scan_complete: bool,
    exclude: &[String],
    include: &[String],
    cache_mode: Option<&str>,
    cache_folder: Option<&str>,
    cache_fallback: &str,
) -> anyhow::Result<i64> {
    if let Some(folder) = get_by_path(pool, path).await? {
        return Ok(folder.id);
    }
    insert(
        pool,
        parent_id,
        path,
        recursive,
        flatten,
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

pub async fn update_flatten(
    pool: &SqlitePool,
    folder_id: i64,
    flatten: bool,
) -> anyhow::Result<()> {
    sqlx::query("UPDATE folders SET flatten = ?1 WHERE id = ?2")
        .bind(flatten)
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
    flatten: i64,
    scan_complete: i64,
    exclude: String,
    include: String,
    thumbnail_cache_mode: Option<String>,
    thumbnail_cache_folder: Option<String>,
    thumbnail_cache_fallback: String,
}

fn into_folder(row: FolderRow) -> Folder {
    Folder {
        id: row.id,
        parent_id: row.parent_id,
        path: row.path,
        recursive: row.recursive != 0,
        flatten: row.flatten != 0,
        scan_complete: row.scan_complete != 0,
        exclude: serde_json::from_str(&row.exclude).unwrap_or_default(),
        include: serde_json::from_str(&row.include).unwrap_or_default(),
        thumbnail_cache_mode: row.thumbnail_cache_mode,
        thumbnail_cache_folder: row.thumbnail_cache_folder,
        thumbnail_cache_fallback: row.thumbnail_cache_fallback,
    }
}
