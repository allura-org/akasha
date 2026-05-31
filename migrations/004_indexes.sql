-- Speed up mtime-based fast path for skipping unchanged files during scans
CREATE INDEX IF NOT EXISTS idx_media_modified_at ON media_files(modified_at);
