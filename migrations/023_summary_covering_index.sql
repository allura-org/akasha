-- Rebuild the MediaSummary covering index with the full summary column set.
--
-- The old idx_media_summary (folder_id, id, blake3_hash, width, height,
-- format) predates several MediaSummary columns (file_size, timestamps,
-- is_present, missing_since) and omits the path columns, so folder loads
-- degenerated into rowid probes against media_files — whose rows are now
-- several KB each thanks to tags_json. At collection scale the grid query
-- read multiple GB of table pages to return ~100MB of summary data
-- (measured 36s for 510k rows). With all summary columns covered, the same
-- query is a sequential index-only scan.
DROP INDEX idx_media_summary;
CREATE INDEX idx_media_summary ON media_files(
    folder_id, id, relative_path, absolute_path, blake3_hash,
    width, height, format, file_size, created_at, modified_at,
    is_present, missing_since
);
