-- Materialized distinct-tag lexicon for fast substring tag search.
--
-- Resolving '%token%' against the (tag, media_file_id) covering index scans
-- 2.7M entries (~1.7s warm, ~20s cold when the index pages are scattered
-- across the file). The lexicon is a few MB, so substring resolution is
-- milliseconds. Rows are never deleted — a lexicon tag with no tag rows
-- simply matches nothing — so maintenance is a single INSERT OR IGNORE per
-- tag written.
CREATE TABLE searchable_tag_lexicon (
    tag TEXT PRIMARY KEY
);
INSERT OR IGNORE INTO searchable_tag_lexicon (tag)
SELECT DISTINCT tag FROM searchable_tags;
