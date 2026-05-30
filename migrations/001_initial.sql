-- Base schema for Akasha MVP

CREATE TABLE folders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    recursive   BOOLEAN NOT NULL DEFAULT 1,
    blacklist   TEXT NOT NULL DEFAULT '[]',
    thumbnail_cache_mode TEXT,  -- overrides global: 'disabled', 'global', 'per_folder', 'custom'
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE media_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id   INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    absolute_path TEXT NOT NULL,
    blake3_hash TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    format      TEXT,
    file_size   INTEGER,
    modified_at DATETIME,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(folder_id, relative_path)
);

CREATE INDEX idx_media_hash ON media_files(blake3_hash);
CREATE INDEX idx_media_folder ON media_files(folder_id);
