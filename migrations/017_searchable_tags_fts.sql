-- FTS5 trigram side table for fast partial tag search.
CREATE VIRTUAL TABLE searchable_tags_fts USING fts5(
    tag,
    media_file_id UNINDEXED,
    source UNINDEXED,
    tokenize='trigram'
);

-- Backfill existing tags so partial search works immediately on upgrade.
INSERT INTO searchable_tags_fts (tag, media_file_id, source)
SELECT tag, media_file_id, source FROM searchable_tags;
