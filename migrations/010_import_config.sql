-- Restructure folder config to match new import config shape.

ALTER TABLE folders ADD COLUMN flatten BOOLEAN NOT NULL DEFAULT 0;
ALTER TABLE folders ADD COLUMN exclude TEXT NOT NULL DEFAULT '[]';
ALTER TABLE folders ADD COLUMN include TEXT NOT NULL DEFAULT '[]';
ALTER TABLE folders ADD COLUMN thumbnail_cache_folder TEXT;
ALTER TABLE folders ADD COLUMN thumbnail_cache_fallback TEXT NOT NULL DEFAULT 'disable';

-- Migrate old show_recursive semantics to new flatten semantics (flipped).
UPDATE folders SET flatten = NOT show_recursive;

-- Migrate old blacklist JSON to new exclude JSON.
UPDATE folders SET exclude = blacklist;

-- Leave old columns (show_recursive, blacklist, thumbnail_cache_mode) in place
-- for backwards compatibility; new code reads the new columns.
