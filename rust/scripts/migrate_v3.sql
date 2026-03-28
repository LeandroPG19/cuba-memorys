-- Migration: v2.x → v0.3.0
-- Drops columns that were written by FSRS-6/Dual-Strength but never read
-- by any handler or MCP tool (confirmed via audit 2026-03-27).
--
-- Safe to run on a live database: each ALTER TABLE acquires a brief
-- ACCESS EXCLUSIVE lock only for the duration of the catalog update.
-- Run during a maintenance window or low-traffic period.
--
-- Idempotent: uses IF EXISTS on all DROP operations.

BEGIN;

-- brain_observations: remove FSRS/Dual-Strength dead columns
ALTER TABLE brain_observations
    DROP COLUMN IF EXISTS source_id,
    DROP COLUMN IF EXISTS storage_strength,
    DROP COLUMN IF EXISTS retrieval_strength,
    DROP COLUMN IF EXISTS stability,
    DROP COLUMN IF EXISTS difficulty,
    DROP COLUMN IF EXISTS valid_from,
    DROP COLUMN IF EXISTS valid_until;

-- brain_observations: remove index that covered valid_until
DROP INDEX IF EXISTS idx_obs_valid_until;

-- brain_errors: remove attempts column (never incremented after v1)
ALTER TABLE brain_errors
    DROP COLUMN IF EXISTS attempts;

COMMIT;
