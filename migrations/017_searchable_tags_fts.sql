-- FTS5 trigram side table for fast partial tag search.
CREATE VIRTUAL TABLE searchable_tags_fts USING fts5(
    tag,
    media_file_id UNINDEXED,
    source UNINDEXED,
    tokenize='trigram'
);
