-- Add updated_at to searchable_configs for upsert tracking.
-- SQLite ALTER TABLE cannot add a DATETIME column with CURRENT_TIMESTAMP default,
-- so recreate the table while preserving IDs and data.
CREATE TABLE searchable_configs_new (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    options TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, kind)
);

INSERT INTO searchable_configs_new (id, name, kind, enabled, options, created_at, updated_at)
SELECT id, name, kind, enabled, options, created_at, CURRENT_TIMESTAMP FROM searchable_configs;

DROP TABLE searchable_configs;
ALTER TABLE searchable_configs_new RENAME TO searchable_configs;
