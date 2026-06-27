use anyhow::Result;
use sqlx::SqlitePool;

use super::{Searchable, SearchableKind};

/// Built-in Searchable that matches tags stored in `searchable_tags`.
///
/// The query is split on whitespace into lowercase tokens. Each matching tag
/// contributes `1.0` to the score, so files matching more tokens rank higher.
#[derive(Debug, Clone, Copy, Default)]
pub struct TagsSearchable;

#[async_trait::async_trait]
impl Searchable for TagsSearchable {
    fn name(&self) -> &str {
        "tags"
    }

    fn kind(&self) -> SearchableKind {
        SearchableKind::Tags
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

        let placeholders: Vec<String> = tokens
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 2))
            .collect();
        let sql = if recursive {
            format!(
                "SELECT t.media_file_id, COUNT(*) AS matches
                 FROM searchable_tags t
                 JOIN media_files m ON m.id = t.media_file_id
                 JOIN folders f ON f.id = m.folder_id
                 WHERE (f.id = ?1 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?1))
                   AND t.tag IN ({})
                 GROUP BY t.media_file_id",
                placeholders.join(",")
            )
        } else {
            format!(
                "SELECT t.media_file_id, COUNT(*) AS matches
                 FROM searchable_tags t
                 JOIN media_files m ON m.id = t.media_file_id
                 WHERE m.folder_id = ?1 AND t.tag IN ({})
                 GROUP BY t.media_file_id",
                placeholders.join(",")
            )
        };

        let mut q = sqlx::query_as::<_, (i64, i64)>(&sql).bind(folder_id);
        for token in &tokens {
            q = q.bind(token);
        }
        let rows = q.fetch_all(pool).await?;
        Ok(rows
            .into_iter()
            .map(|(id, matches)| (id, matches as f32 * 1.0))
            .collect())
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
    async fn matches_single_tag() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;
        let m2 = insert_test_media(&pool, fid, "b.jpg").await;

        let mut tags = HashMap::new();
        tags.insert("cat".to_string(), 0.9f32);
        searchable::update_tags_json(&pool, m1, "wd-vit", tags)
            .await
            .unwrap();

        let searchable = TagsSearchable;
        let hits = searchable.search(&pool, fid, false, "cat").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&m1));
        assert!(!by_id.contains_key(&m2));
    }

    #[tokio::test]
    async fn multiple_tags_add_score() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;
        let m2 = insert_test_media(&pool, fid, "b.jpg").await;

        let mut tags1 = HashMap::new();
        tags1.insert("cat".to_string(), 0.9f32);
        tags1.insert("dog".to_string(), 0.8f32);
        searchable::update_tags_json(&pool, m1, "wd-vit", tags1)
            .await
            .unwrap();

        let mut tags2 = HashMap::new();
        tags2.insert("cat".to_string(), 0.9f32);
        searchable::update_tags_json(&pool, m2, "wd-vit", tags2)
            .await
            .unwrap();

        let searchable = TagsSearchable;
        let hits = searchable
            .search(&pool, fid, false, "cat dog")
            .await
            .unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert_eq!(by_id[&m1], 2.0);
        assert_eq!(by_id[&m2], 1.0);
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

        let mut tags = HashMap::new();
        tags.insert("cat".to_string(), 0.9f32);
        searchable::update_tags_json(&pool, child_media, "wd-vit", tags)
            .await
            .unwrap();

        let searchable = TagsSearchable;
        let hits = searchable.search(&pool, root, true, "cat").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&child_media));
    }
}
