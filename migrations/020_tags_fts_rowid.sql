-- Rebuild searchable_tags_fts with rowids aligned to searchable_tags.rowid.
-- Tag writes delete the previous rows for a (media_file_id, source) pair, and
-- those columns are UNINDEXED in the FTS5 table, so every delete was a full
-- scan of the FTS table (measured 190 ms at 2.3M rows, growing linearly).
-- Aligning rowids lets deletes go through the FTS docid index instead.
DROP TABLE searchable_tags_fts;
CREATE VIRTUAL TABLE searchable_tags_fts USING fts5(
    tag,
    media_file_id UNINDEXED,
    source UNINDEXED,
    tokenize='trigram'
);
INSERT INTO searchable_tags_fts (rowid, tag, media_file_id, source)
SELECT rowid, tag, media_file_id, source FROM searchable_tags;
