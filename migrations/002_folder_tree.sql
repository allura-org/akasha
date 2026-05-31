-- Add tree structure to folders and per-folder display mode

ALTER TABLE folders ADD COLUMN parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE;
ALTER TABLE folders ADD COLUMN show_recursive BOOLEAN NOT NULL DEFAULT 0;

-- Existing folders are roots (no parent)
-- (parent_id is already NULL for existing rows)

-- Index for fast subtree queries
CREATE INDEX idx_folder_parent ON folders(parent_id);
