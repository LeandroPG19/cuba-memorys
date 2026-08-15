-- 0058: mark the historical backlog as already processed before auto-extraction ever runs.
--
-- brain_observations.extracted_at mirrors brain_entities.relations_scanned_at (0039): a
-- progress mark the REM cycle uses to find what still needs a pass, not a flag anyone sets by
-- hand. Without a backfill the queue at install time is every observation this database has
-- ever held — 1890 rows measured on the live corpus, each one an LLM call carrying an 18s
-- budget, which is months of 4-hour cycles and a CLI's quota spent re-reading a year-old
-- backlog nobody asked to revisit. The real write rate is ~125 observations every 7 days,
-- which a five-per-cycle batch clears with room to spare — only rows written AFTER this
-- migration should ever queue.
ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS extracted_at TIMESTAMPTZ;

UPDATE brain_observations SET extracted_at = NOW() WHERE extracted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_observations_unextracted
    ON brain_observations (created_at)
    WHERE extracted_at IS NULL;
