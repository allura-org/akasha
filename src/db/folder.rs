use sqlx::SqlitePool;

#[derive(Debug, Clone)]
pub struct Folder {
    pub id: i64,
    pub path: String,
    pub recursive: bool,
    pub blacklist: Vec<String>,
    pub thumbnail_cache_mode: Option<String>,
}

pub async fn list_all(pool: &SqlitePool) -> anyhow::Result<Vec<Folder>> {
    let rows = sqlx::query_as::<_, FolderRow>("SELECT * FROM folders")
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(into_folder).collect())
}

pub async fn insert(
    pool: &SqlitePool,
    path: &str,
    recursive: bool,
    blacklist: &[String],
    cache_mode: Option<&str>,
) -> anyhow::Result<i64> {
    let blacklist_json = serde_json::to_string(blacklist)?;
    let id = sqlx::query(
        "INSERT INTO folders (path, recursive, blacklist, thumbnail_cache_mode)
         VALUES (?1, ?2, ?3, ?4)"
    )
    .bind(path)
    .bind(recursive)
    .bind(blacklist_json)
    .bind(cache_mode)
    .execute(pool)
    .await?
    .last_insert_rowid();

    Ok(id)
}

#[derive(sqlx::FromRow)]
struct FolderRow {
    id: i64,
    path: String,
    recursive: i64,
    blacklist: String,
    thumbnail_cache_mode: Option<String>,
}

fn into_folder(row: FolderRow) -> Folder {
    Folder {
        id: row.id,
        path: row.path,
        recursive: row.recursive != 0,
        blacklist: serde_json::from_str(&row.blacklist).unwrap_or_default(),
        thumbnail_cache_mode: row.thumbnail_cache_mode,
    }
}
