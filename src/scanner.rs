use std::path::Path;

pub struct ScanResult {
    pub path: String,
    pub hash: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub format: Option<String>,
    pub file_size: u64,
}

pub async fn scan_folder(
    _pool: &sqlx::SqlitePool,
    _folder_id: i64,
    _folder_path: &Path,
    _recursive: bool,
    _blacklist: &[String],
) -> anyhow::Result<Vec<ScanResult>> {
    // TODO: implement directory walk, hash, and dimension extraction
    Ok(Vec::new())
}
