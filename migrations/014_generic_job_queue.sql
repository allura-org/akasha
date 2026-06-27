-- Generalize job_queue so it can hold any kind of processing job, not just
-- Searchables inference. AI inference jobs produce searchable data, but they
-- are not a separate "searchable" job type — the Searchable is the output.

CREATE TABLE job_queue_new (
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

INSERT INTO job_queue_new (
    id, media_file_id, searchable_config_id, job_kind, params_json,
    status, attempts, error, created_at, updated_at
)
SELECT
    id, media_file_id, searchable_config_id, 'tagger', NULL,
    status, attempts, error, created_at, updated_at
FROM job_queue;

DROP TABLE job_queue;
ALTER TABLE job_queue_new RENAME TO job_queue;

CREATE INDEX idx_job_queue_pending ON job_queue(status, job_kind, created_at);
