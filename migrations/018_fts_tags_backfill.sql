-- Backfill existing tags into the FTS5 trigram side table so partial
-- search works immediately on upgrade.
INSERT INTO searchable_tags_fts (tag, media_file_id, source)
SELECT tag, media_file_id, source FROM searchable_tags;
