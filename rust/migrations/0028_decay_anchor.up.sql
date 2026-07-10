-- 0028: fix compounding decay by anchoring on last_decayed_at.
--
-- Bug: the REM daemon runs every 4h (protocol.rs REM_INTERVAL) and applied
--     importance *= exp(-ln2 * (NOW - last_accessed) / H)
-- on every cycle, WITHOUT advancing any anchor. Because last_accessed does not
-- move, each cycle re-applied the full time-since-access decay on top of the
-- already-decayed value. Simulated: a `fact` with a "30-day half-life" hits the
-- 0.01 floor by day ~10 (79x over-decay) instead of retaining 0.79. Importance
-- feeds ranking (score*0.7 + importance*0.3), so anything untouched for a week
-- silently sinks.
--
-- Fix: track `last_decayed_at` and decay only the *incremental* time since the
-- later of {last access, last decay}:
--     Δt   = NOW - GREATEST(last_accessed, last_decayed_at)
--     imp *= exp(-ln2 * Δt / H_eff)   ; last_decayed_at = NOW
-- The product of per-cycle factors telescopes to exp(-ln2 * total_idle / H),
-- so the result is independent of how often the daemon runs. GREATEST restarts
-- the idle clock on access without touching the (scattered) access-tracking
-- sites: after an access, last_accessed jumps past last_decayed_at.
--
-- Data safety: additive column, backfilled to NOW() so existing (already
-- over-decayed) rows take NO retroactive decay — the first post-migration cycle
-- sees Δt ≈ 0. No importance value is modified by this migration.

ALTER TABLE brain_observations ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMPTZ;
UPDATE brain_observations SET last_decayed_at = NOW() WHERE last_decayed_at IS NULL;
ALTER TABLE brain_observations ALTER COLUMN last_decayed_at SET DEFAULT NOW();
