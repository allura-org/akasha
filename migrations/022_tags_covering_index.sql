-- Replace the trigram FTS tag index with a covering index on searchable_tags.
--
-- At collection scale (2.7M tag rows), trigram FTS MATCH queries scan
-- enormous doclists (measured 9-45s for common tags) and force hundreds of
-- thousands of semi-random rowid probes. Tag substring search only needs the
-- distinct tag lexicon, which a LIKE scan over this covering index serves in
-- ~1s; per-media aggregation then runs as pure index range scans. Dropping
-- the FTS table also removes its write-time maintenance cost and several GB
-- of storage (reclaim with VACUUM, optional).
DROP TABLE searchable_tags_fts;
CREATE INDEX idx_searchable_tags_tag_media ON searchable_tags(tag, media_file_id);
