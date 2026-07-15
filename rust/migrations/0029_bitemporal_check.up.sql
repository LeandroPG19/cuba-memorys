-- 0029: enforce the bitemporal invariants that were previously unchecked.
--
-- brain_facts (migration 0018) allowed states that contradict its own model:
--   - valid_from > valid_to  (a fact valid in negative time)
--   - is_current = TRUE together with a valid_to already in the past
-- Nothing prevented them; the observation→fact mirror is best-effort, so drift
-- was possible. On the reference DB there are 0 current violations, so these
-- constraints validate cleanly.
--
-- Data safety: read-only over existing rows (they already satisfy both). If a
-- future dataset violates one, the migration fails loudly instead of corrupting
-- silently — which is the point.

ALTER TABLE brain_facts
    ADD CONSTRAINT ck_facts_valid_interval
    CHECK (valid_to IS NULL OR valid_from <= valid_to);

-- A current fact is open-ended: superseding a fact must set valid_to AND flip
-- is_current to FALSE together, never leave is_current=TRUE with a closed window.
-- Immutable form (NOW() is not allowed in CHECK); verified 0/550 rows violate.
ALTER TABLE brain_facts
    ADD CONSTRAINT ck_facts_current_open
    CHECK (NOT is_current OR valid_to IS NULL);
