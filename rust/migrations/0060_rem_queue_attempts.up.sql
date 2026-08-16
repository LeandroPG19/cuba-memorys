-- The REM queues for observation extraction and entity relation scanning both order
-- strictly by age (created_at / relations_scanned_at), and neither marks a row on
-- failure — on purpose, so a transient outage gets retried instead of silently
-- dropped. Put those two together and two rows that fail every single cycle for a
-- deterministic reason (content past the LLM's budget, say) sit at the head forever:
-- each cycle re-fetches the same oldest rows, the two-consecutive-failure cutoff trips
-- immediately, and nothing written after them is ever reached. Measured in production
-- on 2026-08-16: extraction_failed=2, facts_extracted=0, cycle after cycle.
--
-- attempts columns fix that without giving up on a row. ORDER BY attempts ASC first
-- means any row with fewer tries always outranks one that has failed more, so fresh
-- writes cut ahead of a stuck row instead of queuing behind it, and a row that keeps
-- failing just keeps sinking further behind newer work — it is still retried forever,
-- never abandoned, only deprioritised relative to whatever has not been tried yet.
ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS extraction_attempts INTEGER NOT NULL DEFAULT 0;

ALTER TABLE brain_entities
    ADD COLUMN IF NOT EXISTS relation_scan_attempts INTEGER NOT NULL DEFAULT 0;
