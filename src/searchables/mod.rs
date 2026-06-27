use std::sync::Arc;

use anyhow::Result;
use sqlx::SqlitePool;

/// A property attached to media items that Akasha can search against.
///
/// Each implementation knows how to turn a plaintext query into a set of
/// `(media_file_id, score_contribution)` matches for a given folder scope.
/// The engine aggregates contributions from all enabled Searchables and sorts
/// the final results by score.
#[async_trait::async_trait]
pub trait Searchable: Send + Sync {
    /// Unique machine-facing name, e.g. `"filename"`, `"wd14-tags"`.
    fn name(&self) -> &str;

    /// What kind of data this Searchable produces.
    fn kind(&self) -> SearchableKind;

    /// Execute a search within the requested folder scope.
    ///
    /// Returns pairs of `(media_file_id, score_contribution)`. Higher values
    /// mean the item matched the query more strongly for this Searchable.
    async fn search(
        &self,
        pool: &SqlitePool,
        folder_id: i64,
        recursive: bool,
        query: &str,
    ) -> Result<Vec<(i64, f32)>>;
}

/// The data shape of a Searchable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchableKind {
    /// Free-form text, e.g. filename, description, sidecar contents.
    Text,
    /// Tokenized tags. The query is split by delimiters before searching.
    Tags,
    /// Dense embedding vector. Each model gets its own Searchable property.
    Vector { dims: usize },
    /// Classification label + confidence.
    Classification,
}

impl SearchableKind {
    /// Textual representation stored in the database.
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchableKind::Text => "text",
            SearchableKind::Tags => "tags",
            SearchableKind::Vector { .. } => "vector",
            SearchableKind::Classification => "classification",
        }
    }
}

impl std::str::FromStr for SearchableKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "text" => Ok(SearchableKind::Text),
            "tags" => Ok(SearchableKind::Tags),
            "classification" => Ok(SearchableKind::Classification),
            _ if s.starts_with("vector:") => {
                let dims = s
                    .strip_prefix("vector:")
                    .unwrap()
                    .parse::<usize>()
                    .map_err(|e| anyhow::anyhow!("invalid vector dims: {e}"))?;
                Ok(SearchableKind::Vector { dims })
            }
            _ => Err(anyhow::anyhow!("unknown searchable kind: {s}")),
        }
    }
}

/// A user-entered query plus the set of enabled Searchables.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchQuery {
    pub text: String,
    pub enabled_searchables: Vec<String>,
}

impl SearchQuery {
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() || self.enabled_searchables.is_empty()
    }
}

/// A media item returned by the search engine with its aggregate score.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub media_summary: crate::db::media::MediaSummary,
    pub score: f32,
}

/// Collection of all built-in Searchable implementations.
#[derive(Default, Clone)]
pub struct SearchableRegistry {
    searchables: Vec<Arc<dyn Searchable>>,
}

impl SearchableRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(filename::FilenameSearchable));
        reg.register(Arc::new(tags::TagsSearchable));
        reg.register(Arc::new(description::DescriptionSearchable));
        reg
    }

    pub fn register(&mut self, searchable: Arc<dyn Searchable>) {
        self.searchables.push(searchable);
    }

    pub fn by_name(&self, name: &str) -> Option<&Arc<dyn Searchable>> {
        self.searchables.iter().find(|s| s.name() == name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.searchables.iter().map(|s| s.name())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Searchable>> {
        self.searchables.iter()
    }
}

pub mod description;
pub mod engine;
pub mod filename;
pub mod tags;
pub mod worker;

pub use engine::SearchEngine;
pub use worker::SearchWorker;
