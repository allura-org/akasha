-- Add updated_at to searchable_configs for upsert tracking.
-- SQLite ALTER TABLE cannot accept a non-constant DEFAULT expression, so the
-- column is added without a default and existing rows are back-filled.
ALTER TABLE searchable_configs ADD COLUMN updated_at DATETIME;
UPDATE searchable_configs SET updated_at = CURRENT_TIMESTAMP;

-- Recreate the name index that was dropped when the table was recreated in migration 015.
CREATE INDEX idx_searchable_config_name ON searchable_configs(name);
