-- Track files that have disappeared from disk without losing their metadata.
ALTER TABLE media_files ADD COLUMN is_present BOOLEAN NOT NULL DEFAULT 1;
ALTER TABLE media_files ADD COLUMN missing_since DATETIME;

-- Speed up orphan passes and any future "show/hide missing" filtering.
CREATE INDEX IF NOT EXISTS idx_media_present ON media_files(folder_id, is_present);
