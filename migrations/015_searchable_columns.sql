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

-- Recreate searchable_configs with UNIQUE(name, kind) while preserving IDs and
-- all child rows in searchable_values and job_queue.
-- NOTE: sqlx runs migrations inside a transaction, so PRAGMA foreign_keys is a
-- no-op. We therefore stage child rows in tables without foreign keys, drop the
-- originals, recreate them with the new parent schema, and copy the rows back.

CREATE TABLE _searchable_configs_new (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    options TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, kind)
);

INSERT INTO _searchable_configs_new (id, name, kind, enabled, options, created_at)
SELECT id, name, kind, enabled, options, created_at FROM searchable_configs;

CREATE TABLE _searchable_values_temp (
    id INTEGER PRIMARY KEY,
    media_file_id INTEGER NOT NULL,
    searchable_config_id INTEGER NOT NULL,
    value_json TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO _searchable_values_temp (
    id, media_file_id, searchable_config_id, value_json, created_at, updated_at
)
SELECT
    id, media_file_id, searchable_config_id, value_json, created_at, updated_at
FROM searchable_values;

CREATE TABLE _job_queue_temp (
    id INTEGER PRIMARY KEY,
    media_file_id INTEGER NOT NULL,
    searchable_config_id INTEGER,
    job_kind TEXT NOT NULL DEFAULT 'tagger',
    params_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO _job_queue_temp (
    id, media_file_id, searchable_config_id, job_kind, params_json,
    status, attempts, error, created_at, updated_at
)
SELECT
    id, media_file_id, searchable_config_id, job_kind, params_json,
    status, attempts, error, created_at, updated_at
FROM job_queue;

-- Drop child tables before the parent so SQLite allows the drop without
-- requiring foreign-key enforcement to be disabled.
DROP TABLE searchable_values;
DROP TABLE job_queue;
DROP TABLE searchable_configs;

ALTER TABLE _searchable_configs_new RENAME TO searchable_configs;

CREATE TABLE searchable_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_file_id INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
    searchable_config_id INTEGER NOT NULL REFERENCES searchable_configs(id) ON DELETE CASCADE,
    value_json TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_file_id, searchable_config_id)
);

CREATE INDEX idx_searchable_values_media ON searchable_values(media_file_id);
CREATE INDEX idx_searchable_values_config ON searchable_values(searchable_config_id);

CREATE TABLE job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_file_id INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
    searchable_config_id INTEGER REFERENCES searchable_configs(id) ON DELETE CASCADE,
    job_kind TEXT NOT NULL DEFAULT 'tagger',
    params_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_job_queue_pending ON job_queue(status, job_kind, created_at);

INSERT INTO searchable_values (
    id, media_file_id, searchable_config_id, value_json, created_at, updated_at
)
SELECT
    id, media_file_id, searchable_config_id, value_json, created_at, updated_at
FROM _searchable_values_temp;

INSERT INTO job_queue (
    id, media_file_id, searchable_config_id, job_kind, params_json,
    status, attempts, error, created_at, updated_at
)
SELECT
    id, media_file_id, searchable_config_id, job_kind, params_json,
    status, attempts, error, created_at, updated_at
FROM _job_queue_temp;

DROP TABLE _searchable_values_temp;
DROP TABLE _job_queue_temp;
