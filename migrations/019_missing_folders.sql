-- Track folders that have disappeared from disk without losing their metadata,
-- mirroring media_files.is_present (013_missing_files.sql). A missing folder's
-- media is marked missing too, so all media-level filters keep working; the
-- rows are preserved so everything is restored if the folder reappears (e.g.
-- an unmounted drive comes back).
ALTER TABLE folders ADD COLUMN is_present BOOLEAN NOT NULL DEFAULT 1;
ALTER TABLE folders ADD COLUMN missing_since DATETIME;

-- Speed up "hide missing folders" tree filtering and missing-folder purges.
CREATE INDEX IF NOT EXISTS idx_folder_present ON folders(parent_id, is_present);
