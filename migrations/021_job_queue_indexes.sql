-- Fix job_queue hot paths for very large queues (hundreds of thousands of
-- pending jobs):
--
-- 1. The enqueue dedup check filters on (media_file_id, job_kind, status,
-- searchable_config_id). Without media_file_id leading an index, each insert
-- scanned every pending row, making bulk enqueue O(n^2).
CREATE INDEX idx_job_queue_dedup
    ON job_queue(media_file_id, job_kind, status, searchable_config_id);
--
-- 2. claim_pending_jobs orders pending rows by (searchable_config_id,
-- created_at). The old (status, job_kind, created_at) index could not
-- provide that order, so every claim sorted the whole pending queue.
-- This partial index serves rows in claim order, so SQLite stops after
-- the first LIMIT matches.
CREATE INDEX idx_job_queue_claim
    ON job_queue(searchable_config_id, created_at)
    WHERE status = 'pending';
