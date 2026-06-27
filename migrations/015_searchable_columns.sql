PRAGMA foreign_keys = OFF;

-- Add raw Searchable storage columns to media_files.
ALTER TABLE media_files ADD COLUMN tags_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN descriptions_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN classifications_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE media_files ADD COLUMN embeddings_json TEXT DEFAULT NULL;

-- Side table for fast tag search.
CREATE TABLE searchable_tags (
    media_file_id INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    tag TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (media_file_id, source, tag)
);
CREATE INDEX idx_searchable_tags_tag ON searchable_tags(tag, score);
CREATE INDEX idx_searchable_tags_media ON searchable_tags(media_file_id);

-- FTS5 table for description search.
CREATE VIRTUAL TABLE searchable_text_fts USING fts5(
    media_file_id UNINDEXED,
    source UNINDEXED,
    content
);

-- Recreate searchable_configs with UNIQUE(name, kind) while preserving IDs.
CREATE TABLE searchable_configs_new (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    options TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, kind)
);

INSERT INTO searchable_configs_new (id, name, kind, enabled, options, created_at)
SELECT id, name, kind, enabled, options, created_at FROM searchable_configs;

DROP TABLE searchable_configs;
ALTER TABLE searchable_configs_new RENAME TO searchable_configs;

PRAGMA foreign_keys = ON;
