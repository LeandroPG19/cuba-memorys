-- The REM cycle became the thing that grows the graph — decay, autolink, embedding
-- backfill, chunking, an LLM relation scan, LLM auto-extraction, PageRank, community
-- detection, duplicate counting — and none of it left a trace anywhere but
-- tracing::info!, which goes to the process's stderr and is gone at the next restart.
-- Its two LLM steps give up after two consecutive failures and say so only to that same
-- log: if the `claude` CLI falls off PATH, the graph stops growing and nothing notices.
--
-- One row per cycle, not a running/finished pair: the daemon holds an advisory lock for
-- the whole cycle (REM_LOCK in protocol.rs), so there is never a row to update concurrently
-- and nothing reads this table while a cycle is still in flight — the last row is always
-- either the previous cycle or, once this one commits, this one.
--
-- Capped at 500 rows by trigger, same number as observability::RING_CAPACITY for the
-- in-memory call ring: this table is that ring's persisted counterpart, at a cadence of
-- one row per 4-hour cycle instead of one per call, so 500 rows is years of history.
CREATE TABLE IF NOT EXISTS brain_rem_cycles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL,
    duration_ms bigint NOT NULL,
    decayed_count bigint NOT NULL DEFAULT 0,
    autolink_edges bigint NOT NULL DEFAULT 0,
    embeddings_backfilled bigint NOT NULL DEFAULT 0,
    entities_scanned bigint NOT NULL DEFAULT 0,
    facts_extracted bigint NOT NULL DEFAULT 0,
    communities bigint NOT NULL DEFAULT 0,
    duplicate_candidates bigint NOT NULL DEFAULT 0,
    relation_scan_failed bigint NOT NULL DEFAULT 0,
    extraction_failed bigint NOT NULL DEFAULT 0,
    error text,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rem_cycles_recent ON brain_rem_cycles (finished_at DESC);

CREATE OR REPLACE FUNCTION brain_cap_rem_cycles() RETURNS trigger AS $$
BEGIN
    DELETE FROM brain_rem_cycles
    WHERE id IN (
        SELECT id FROM brain_rem_cycles
        ORDER BY finished_at DESC
        OFFSET 500
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER brain_rem_cycles_cap
    AFTER INSERT ON brain_rem_cycles
    FOR EACH ROW EXECUTE FUNCTION brain_cap_rem_cycles();

GRANT SELECT, INSERT, UPDATE, DELETE ON brain_rem_cycles TO cuba_app;

COMMENT ON TABLE brain_rem_cycles IS
    'One row per REM consolidation cycle: when it ran, how long it took, what it did, and whether either LLM step (relation scan, auto-extraction) gave up after two consecutive failures. Capped at 500 rows by trigger.';
