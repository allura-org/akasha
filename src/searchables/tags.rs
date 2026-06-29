use anyhow::Result;
use sqlx::SqlitePool;
use std::collections::HashMap;

use super::{Searchable, SearchableKind};

/// Escape a single token into a quoted FTS5 phrase.
fn fts5_phrase(token: &str) -> String {
    let escaped = token.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

/// Build an FTS5 `MATCH` expression that matches any of the supplied tokens.
fn fts5_match_expr(tokens: &[String]) -> String {
    tokens
        .iter()
        .map(|t| fts5_phrase(t))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Built-in Searchable that matches tags stored in `searchable_tags`.
///
/// The query is split on whitespace into lowercase tokens. Each matching tag
/// contributes `1.0` to the score, so files matching more tokens rank higher.
/// Tokens of three or more characters use the FTS5 trigram side table for
/// substring matches; shorter tokens fall back to exact matches in
/// `searchable_tags`.
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

        let (short, long): (Vec<_>, Vec<_>) = tokens.into_iter().partition(|t| t.len() < 3);

        let mut scores: HashMap<i64, f32> = HashMap::new();

        if !long.is_empty() {
            let match_expr = fts5_match_expr(&long);
            let sql = if recursive {
                format!(
                    "SELECT fts.media_file_id, COUNT(*) AS matches
                     FROM searchable_tags_fts fts
                     JOIN media_files m ON m.id = fts.media_file_id
                     JOIN folders f ON f.id = m.folder_id
                     WHERE fts.tag MATCH ?1
                       AND (f.id = ?2 OR f.path LIKE (SELECT path || '/%' FROM folders WHERE id = ?2))
                     GROUP BY fts.media_file_id"
                )
            } else {
                format!(
                    "SELECT fts.media_file_id, COUNT(*) AS matches
                     FROM searchable_tags_fts fts
                     JOIN media_files m ON m.id = fts.media_file_id
                     WHERE m.folder_id = ?2 AND fts.tag MATCH ?1
                     GROUP BY fts.media_file_id"
                )
            };

            let rows: Vec<(i64, i64)> = sqlx::query_as(&sql)
                .bind(&match_expr)
                .bind(folder_id)
                .fetch_all(pool)
                .await?;

            for (id, matches) in rows {
                *scores.entry(id).or_insert(0.0) += matches as f32;
            }
        }

        if !short.is_empty() {
            let placeholders: Vec<String> = short
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
                       AND LOWER(t.tag) IN ({})
                     GROUP BY t.media_file_id",
                    placeholders.join(",")
                )
            } else {
                format!(
                    "SELECT t.media_file_id, COUNT(*) AS matches
                     FROM searchable_tags t
                     JOIN media_files m ON m.id = t.media_file_id
                     WHERE m.folder_id = ?1 AND LOWER(t.tag) IN ({})
                     GROUP BY t.media_file_id",
                    placeholders.join(",")
                )
            };

            let mut q = sqlx::query_as::<_, (i64, i64)>(&sql).bind(folder_id);
            for token in &short {
                q = q.bind(token);
            }
            let rows = q.fetch_all(pool).await?;
            for (id, matches) in rows {
                *scores.entry(id).or_insert(0.0) += matches as f32;
            }
        }

        Ok(scores.into_iter().collect())
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
