use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use sqlx::SqlitePool;

use crate::db::{media, searchable};

use super::{SearchHit, SearchQuery, SearchableRegistry};

/// Aggregates per-Searchable results into scored, sorted `MediaSummary` hits.
#[derive(Clone)]
pub struct SearchEngine {
    registry: Arc<SearchableRegistry>,
}

impl SearchEngine {
    pub fn new(registry: Arc<SearchableRegistry>) -> Self {
        Self { registry }
    }

    pub fn with_defaults() -> Self {
        Self::new(Arc::new(SearchableRegistry::with_defaults()))
    }

    /// Execute a search query within the requested folder scope.
    ///
    /// Returns an empty vector if the query has no text or no enabled Searchables.
    pub async fn execute(
        &self,
        pool: &SqlitePool,
        folder_id: i64,
        recursive: bool,
        query: &SearchQuery,
    ) -> Result<Vec<SearchHit>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        // Load enabled configs from the DB and intersect with the registry.
        let configs = searchable::list_enabled_configs(pool).await?;
        let enabled_names: std::collections::HashSet<&str> = configs
            .iter()
            .map(|c| c.name.as_str())
            .filter(|name| query.enabled_searchables.iter().any(|e| e == *name))
            .collect();

        if enabled_names.is_empty() {
            return Ok(Vec::new());
        }

        // Run each enabled Searchable and aggregate scores per media file.
        let mut scores: HashMap<i64, f32> = HashMap::new();
        for searchable in self.registry.iter() {
            if !enabled_names.contains(searchable.name()) {
                continue;
            }

            let contributions = searchable
                .search(pool, folder_id, recursive, &query.text)
                .await?;
            for (id, contribution) in contributions {
                *scores.entry(id).or_insert(0.0) += contribution;
            }
        }

        if scores.is_empty() {
            return Ok(Vec::new());
        }

        // Hydrate matching IDs into MediaSummary rows, still scoped to the folder.
        let ids: Vec<i64> = scores.keys().copied().collect();
        let ids_json = serde_json::to_string(&ids)?;
        let summaries = media::search_summaries(pool, folder_id, recursive, &ids_json).await?;

        // Combine summaries with their scores and sort.
        let mut hits: Vec<SearchHit> = summaries
            .into_iter()
            .filter_map(|summary| {
                scores
                    .get(&summary.id)
                    .map(|&score| SearchHit { media_summary: summary, score })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.media_summary.relative_path.cmp(&b.media_summary.relative_path))
        });

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{folder, media, searchable};
    use crate::searchables::{Searchable, SearchableKind};

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn engine_scores_and_sorts() {
        let pool = setup_pool().await;
        let fid = folder::insert(&pool, None, "/tmp", true, false, true, &[], &[], None, None, "disable")
            .await
            .unwrap();

        media::upsert(
            &pool, fid, "foo/bar.jpg", "/tmp/foo/bar.jpg", "hash",
            None, None, None, Some(0), Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();
        media::upsert(
            &pool, fid, "foo/baz.jpg", "/tmp/foo/baz.jpg", "hash",
            None, None, None, Some(0), Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();

        // The built-in filename Searchable is already seeded by migration 008.
        let engine = SearchEngine::with_defaults();
        let query = SearchQuery {
            text: "bar".into(),
            enabled_searchables: vec!["filename".into()],
        };
        let hits = engine.execute(&pool, fid, false, &query).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert!(hits[0].media_summary.relative_path.contains("bar"));
    }

    #[derive(Debug, Clone, Default)]
    struct ConstantSearchable {
        name: &'static str,
        ids: Vec<i64>,
    }

    #[async_trait::async_trait]
    impl Searchable for ConstantSearchable {
        fn name(&self) -> &str {
            self.name
        }

        fn kind(&self) -> SearchableKind {
            SearchableKind::Text
        }

        async fn search(
            &self,
            _pool: &SqlitePool,
            _folder_id: i64,
            _recursive: bool,
            _query: &str,
        ) -> anyhow::Result<Vec<(i64, f32)>> {
            Ok(self.ids.iter().copied().map(|id| (id, 1.0)).collect())
        }
    }

    #[tokio::test]
    async fn engine_aggregates_multiple_searchables() {
        let pool = setup_pool().await;
        let fid = folder::insert(&pool, None, "/tmp", true, false, true, &[], &[], None, None, "disable")
            .await
            .unwrap();

        let m1 = media::upsert(
            &pool, fid, "a.jpg", "/tmp/a.jpg", "hash",
            None, None, None, Some(0), Some(chrono::Local::now().naive_local()),
        )
        .await
        .unwrap();
        let m2 = media::upsert(
            &pool, fid, "b.jpg", "/tmp/b.jpg", "hash",
            None, None, None, Some(0), Some(chrono::Local::now().naive_local()),
        )
        .await
            .unwrap();

        // Register two fake text Searchables.
        searchable::insert_config(&pool, "alpha", "text", true, serde_json::json!({}))
            .await
            .unwrap();
        searchable::insert_config(&pool, "beta", "text", true, serde_json::json!({}))
            .await
            .unwrap();

        let mut registry = crate::searchables::SearchableRegistry::new();
        registry.register(Arc::new(ConstantSearchable {
            name: "alpha",
            ids: vec![m1, m2],
        }));
        registry.register(Arc::new(ConstantSearchable {
            name: "beta",
            ids: vec![m2],
        }));

        let engine = SearchEngine::new(Arc::new(registry));
        let query = SearchQuery {
            text: "x".into(),
            enabled_searchables: vec!["alpha".into(), "beta".into()],
        };
        let hits = engine.execute(&pool, fid, false, &query).await.unwrap();

        assert_eq!(hits.len(), 2);
        assert!(hits[0].score >= hits[1].score);
        let by_id: std::collections::HashMap<i64, f32> = hits
            .into_iter()
            .map(|h| (h.media_summary.id, h.score))
            .collect();
        assert_eq!(by_id[&m1], 1.0);
        assert_eq!(by_id[&m2], 2.0);
    }
}
