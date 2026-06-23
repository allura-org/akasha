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
