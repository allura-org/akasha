use anyhow::Result;
use sqlx::SqlitePool;

use super::{Searchable, SearchableKind};

/// Built-in Searchable that matches descriptions stored in `searchable_text_fts`.
///
/// Uses the FTS5 `bm25()` score for ranking. Lower `bm25` values indicate a
/// better match, so the score is negated so that higher values rank higher.
#[derive(Debug, Clone, Copy, Default)]
pub struct DescriptionSearchable;

#[async_trait::async_trait]
impl Searchable for DescriptionSearchable {
    fn name(&self) -> &str {
        "descriptions"
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
        let q = query.trim();
        if q.is_empty() {
            return Ok(Vec::new());
        }

        let sql = if recursive {
            r#"
            SELECT f.media_file_id, bm25(searchable_text_fts) AS score
            FROM searchable_text_fts f
            JOIN media_files m ON m.id = f.media_file_id
            JOIN folders fld ON fld.id = m.folder_id
            WHERE searchable_text_fts MATCH ?2
              AND (fld.id = ?1 OR fld.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
            ORDER BY score
            "#
        } else {
            r#"
            SELECT f.media_file_id, bm25(searchable_text_fts) AS score
            FROM searchable_text_fts f
            JOIN media_files m ON m.id = f.media_file_id
            WHERE m.folder_id = ?1 AND searchable_text_fts MATCH ?2
            ORDER BY score
            "#
        };

        let rows = sqlx::query_as::<_, (i64, f64)>(sql)
            .bind(folder_id)
            .bind(q)
            .fetch_all(pool)
            .await?;

        Ok(rows.into_iter().map(|(id, score)| (id, -score as f32)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{folder, media, searchable};
    use std::collections::HashMap;

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
    async fn matches_description_word() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;
        let m2 = insert_test_media(&pool, fid, "b.jpg").await;

        searchable::update_description_json(&pool, m1, "blip", "a cat on a mat")
            .await
            .unwrap();

        let searchable = DescriptionSearchable;
        let hits = searchable.search(&pool, fid, false, "cat").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&m1));
        assert!(!by_id.contains_key(&m2));
    }

    #[tokio::test]
    async fn returns_negated_bm25_score() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;

        searchable::update_description_json(&pool, m1, "blip", "a cat on a mat")
            .await
            .unwrap();

        let searchable = DescriptionSearchable;
        let hits = searchable.search(&pool, fid, false, "cat").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].1 >= 0.0, "score should be non-negative after negation");
    }

    #[tokio::test]
    async fn recursive_search_follows_subfolders() {
        let pool = setup_pool().await;
        let root = insert_test_folder(&pool, "/tmp/root").await;
        let child = folder::insert(
            &pool,
            Some(root),
            "/tmp/root/child",
            true,
            false,
            &[],
            &[],
            None,
            None,
            "disable",
        )
        .await
        .unwrap();
        let _ = insert_test_media(&pool, root, "root.jpg").await;
        let child_media = insert_test_media(&pool, child, "child/nested.jpg").await;

        searchable::update_description_json(&pool, child_media, "blip", "a cat on a mat")
            .await
            .unwrap();

        let searchable = DescriptionSearchable;
        let hits = searchable.search(&pool, root, true, "cat").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&child_media));
    }
}
