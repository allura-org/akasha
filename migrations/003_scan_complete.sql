-- Add scan completion tracking per folder

ALTER TABLE folders ADD COLUMN scan_complete BOOLEAN NOT NULL DEFAULT 0;
