-- Covering index for MediaSummary queries.
-- SQLite can satisfy count queries and the lightweight portion of summary
-- queries from this index. relative_path and absolute_path are omitted to
-- keep the index size bounded; SQLite will do a rowid lookup for those.
CREATE INDEX IF NOT EXISTS idx_media_summary ON media_files(folder_id, id, blake3_hash, width, height, format);
