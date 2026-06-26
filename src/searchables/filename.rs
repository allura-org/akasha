use std::collections::HashMap;

use anyhow::Result;
use sqlx::SqlitePool;

use super::{Searchable, SearchableKind};

const TOKEN_MATCH_SCORE: f32 = 1.0;
const EXACT_PATH_BONUS: f32 = 5.0;

/// Built-in Searchable that matches the `relative_path` of media files.
///
/// The query is split on whitespace into tokens. Each token that appears
/// anywhere in the relative path contributes to the score. A case-insensitive
/// exact path match adds a large bonus.
#[derive(Debug, Clone, Copy, Default)]
pub struct FilenameSearchable;

#[async_trait::async_trait]
impl Searchable for FilenameSearchable {
    fn name(&self) -> &str {
        "filename"
    }

    fn kind(&self) -> SearchableKind {
        SearchableKind::Text
    }

    async fn search(
        &self,
        pool: &SqlitePool,
        folder_id: i64,
        recursive: bool,
        query: &str,
    ) -> Result<Vec<(i64, f32)>> {
        let tokens: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let rows: Vec<(i64, String)> = if recursive {
            sqlx::query_as(
                r#"
                SELECT m.id, m.relative_path
                FROM media_files m
                JOIN folders f ON m.folder_id = f.id
                WHERE f.id = ?1
                   OR (f.path LIKE (
                       SELECT path || '/%'
                       FROM folders
                       WHERE id = ?1
                   ))
                "#,
            )
            .bind(folder_id)
            .fetch_all(pool)
            .await?
        } else {
            sqlx::query_as(
                r#"
                SELECT id, relative_path
                FROM media_files
                WHERE folder_id = ?1
                "#,
            )
            .bind(folder_id)
            .fetch_all(pool)
            .await?
        };

        let mut scores: HashMap<i64, f32> = HashMap::new();
        let full_query = query.trim().to_lowercase();

        for (id, path) in rows {
            let path_lower = path.to_lowercase();
            let mut contribution = 0.0f32;

            for token in &tokens {
                if path_lower.contains(token) {
                    contribution += TOKEN_MATCH_SCORE;
                }
            }

            if path_lower == full_query {
                contribution += EXACT_PATH_BONUS;
            }

            if contribution > 0.0 {
                *scores.entry(id).or_insert(0.0) += contribution;
            }
        }

        Ok(scores.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{folder, media};
    use sqlx::SqlitePool;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    async fn insert_test_folder(pool: &SqlitePool, path: &str) -> i64 {
        folder::insert(pool, None, path, true, false, &[], &[], None, None, "disable")
            .await
            .unwrap()
    }

    async fn insert_test_media(pool: &SqlitePool, folder_id: i64, relative: &str) -> i64 {
        let absolute = format!("/tmp/{relative}");
        media::upsert(
            pool,
            folder_id,
            relative,
            &absolute,
            "hash",
            None,
            None,
            None,
            Some(0),
            Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn matches_single_token() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "foo/bar.jpg").await;
        let m2 = insert_test_media(&pool, fid, "baz/qux.png").await;

        let searchable = FilenameSearchable;
        let hits = searchable.search(&pool, fid, false, "bar").await.unwrap();

        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&m1));
        assert!(!by_id.contains_key(&m2));
    }

    #[tokio::test]
    async fn multiple_tokens_add_score() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "foo/bar baz.jpg").await;
        let m2 = insert_test_media(&pool, fid, "foo/bar.jpg").await;

        let searchable = FilenameSearchable;
        let hits = searchable.search(&pool, fid, false, "bar baz").await.unwrap();

        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert_eq!(by_id[&m1], 2.0);
        assert_eq!(by_id[&m2], 1.0);
    }

    #[tokio::test]
    async fn recursive_search_follows_subfolders() {
        let pool = setup_pool().await;
        let root = insert_test_folder(&pool, "/tmp/root").await;
        let child = folder::insert(&pool, Some(root), "/tmp/root/child", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let _ = insert_test_media(&pool, root, "root.jpg").await;
        let child_media = insert_test_media(&pool, child, "child/nested.jpg").await;

        let searchable = FilenameSearchable;
        let hits = searchable.search(&pool, root, true, "nested").await.unwrap();

        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&child_media));
    }
}
