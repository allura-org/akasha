use anyhow::Result;
use sqlx::SqlitePool;

use super::{Searchable, SearchableKind};

/// Escape LIKE metacharacters so a token matches literally.
fn like_escape(token: &str) -> String {
    token.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// Built-in Searchable that matches tags stored in `searchable_tags`.
///
/// The query is split on whitespace into lowercase tokens. Each matching tag
/// contributes `1.0` to the score, so files matching more tokens rank higher.
/// Tokens of three or more characters match substrings; shorter tokens match
/// whole tags only (LIKE is case-insensitive for ASCII).
///
/// Substring resolution runs against the materialized distinct-tag lexicon
/// (`searchable_tag_lexicon`, a few MB), not the full tag table: milliseconds
/// per token regardless of cache state, versus tens of seconds of doclist I/O
/// for the trigram FTS side table (dropped in migration 022) or seconds for
/// a LIKE scan of the full covering index.
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
        let patterns: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .map(|t| {
                let escaped = like_escape(&t);
                if t.chars().count() >= 3 {
                    format!("%{escaped}%")
                } else {
                    escaped
                }
            })
            .collect();
        if patterns.is_empty() {
            return Ok(Vec::new());
        }

        let folder_param = patterns.len() + 1;
        let like_conditions = patterns
            .iter()
            .enumerate()
            .map(|(i, _)| format!("tag LIKE ?{} ESCAPE '\\'", i + 1))
            .collect::<Vec<_>>()
            .join(" OR ");

        // The CROSS JOIN + INDEXED BY forces the planner to drive from the
        // (tiny) matching tag list into the (tag, media_file_id) covering
        // index; left to itself it iterates every media file in the subtree
        // instead (~4x slower at collection scale).
        let sql = if recursive {
            format!(
                "WITH RECURSIVE subtree(id) AS (
                     SELECT ?{folder_param} UNION ALL
                     SELECT f.id FROM folders f JOIN subtree s ON f.parent_id = s.id
                 ),
                 matching_tags(tag) AS (
                     SELECT tag FROM searchable_tag_lexicon WHERE {like_conditions}
                 )
                 SELECT t.media_file_id, COUNT(*) AS matches
                 FROM matching_tags mt
                 CROSS JOIN searchable_tags t INDEXED BY idx_searchable_tags_tag_media
                 WHERE t.tag = mt.tag
                   AND t.media_file_id IN (
                       SELECT m.id FROM media_files m JOIN subtree s ON m.folder_id = s.id
                   )
                 GROUP BY t.media_file_id"
            )
        } else {
            format!(
                "WITH matching_tags(tag) AS (
                     SELECT tag FROM searchable_tag_lexicon WHERE {like_conditions}
                 )
                 SELECT t.media_file_id, COUNT(*) AS matches
                 FROM matching_tags mt
                 CROSS JOIN searchable_tags t INDEXED BY idx_searchable_tags_tag_media
                 WHERE t.tag = mt.tag
                   AND t.media_file_id IN (
                       SELECT m.id FROM media_files m WHERE m.folder_id = ?{folder_param}
                   )
                 GROUP BY t.media_file_id"
            )
        };

        let mut q = sqlx::query_as::<_, (i64, i64)>(&sql);
        for pattern in &patterns {
            q = q.bind(pattern);
        }
        let rows = q.bind(folder_id).fetch_all(pool).await?;
        Ok(rows.into_iter().map(|(id, n)| (id, n as f32)).collect())
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

    #[tokio::test]
    async fn substring_match_finds_underscored_tag() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;
        let m2 = insert_test_media(&pool, fid, "b.jpg").await;

        let mut tags = HashMap::new();
        tags.insert("blue_sky".to_string(), 0.9f32);
        searchable::update_tags_json(&pool, m1, "wd-vit", tags)
            .await
            .unwrap();

        let searchable = TagsSearchable;
        let hits = searchable.search(&pool, fid, false, "sky").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&m1));
        assert!(!by_id.contains_key(&m2));
    }

    #[tokio::test]
    async fn short_token_exact_match() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;
        let m2 = insert_test_media(&pool, fid, "b.jpg").await;

        let mut tags = HashMap::new();
        tags.insert("ox".to_string(), 0.9f32);
        searchable::update_tags_json(&pool, m1, "wd-vit", tags)
            .await
            .unwrap();

        let searchable = TagsSearchable;
        let hits = searchable.search(&pool, fid, false, "ox").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert!(by_id.contains_key(&m1));
        assert!(!by_id.contains_key(&m2));
    }

    #[tokio::test]
    async fn mixed_short_and_long_tokens() {
        let pool = setup_pool().await;
        let fid = insert_test_folder(&pool, "/tmp").await;
        let m1 = insert_test_media(&pool, fid, "a.jpg").await;
        let m2 = insert_test_media(&pool, fid, "b.jpg").await;

        let mut tags1 = HashMap::new();
        tags1.insert("ox".to_string(), 0.9f32);
        tags1.insert("blue_sky".to_string(), 0.8f32);
        searchable::update_tags_json(&pool, m1, "wd-vit", tags1)
            .await
            .unwrap();

        let mut tags2 = HashMap::new();
        tags2.insert("blue_sky".to_string(), 0.8f32);
        searchable::update_tags_json(&pool, m2, "wd-vit", tags2)
            .await
            .unwrap();

        let searchable = TagsSearchable;
        let hits = searchable.search(&pool, fid, false, "ox sky").await.unwrap();
        let by_id: HashMap<i64, f32> = hits.into_iter().collect();
        assert_eq!(by_id[&m1], 2.0);
        assert_eq!(by_id[&m2], 1.0);
    }
}
