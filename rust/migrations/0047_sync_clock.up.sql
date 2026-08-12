-- The conflict tiebreak needs to know which side changed last, and neither column
-- that could answer was trustworthy.
--
-- `updated_at` has no trigger: it is set by hand in eleven places across six
-- handlers, and any write that forgets leaves it stale. `version` is worse — it is
-- incremented in exactly one place, eco.rs's correct(), and in the live corpus all
-- 1880 observations sit at version 1 while 1258 of them have been modified. As a
-- logical clock it has never ticked.
--
-- The obvious fix breaks more than it repairs. A blanket BEFORE UPDATE trigger would
-- also fire for the REM decay pass, which writes updated_at itself on 1097 of 1880
-- rows every four hours, and for reembed and for the Robbins-Monro reinforcement —
-- routes that deliberately do not touch it. Every one of those would start looking
-- like a change worth syncing, and each export would ship a graph that had not
-- changed.
--
-- So the trigger fires on named columns only, and only when their values actually
-- differ. Decay moves importance and last_accessed; reembed moves embedding. Neither
-- is in this list, so neither wakes the clock — by construction, not by auditing
-- eleven call sites every time someone adds a twelfth.

CREATE OR REPLACE FUNCTION brain_bump_sync_clock()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at := NOW();
    NEW.version := OLD.version + 1;
    RETURN NEW;
END;
$func$ LANGUAGE plpgsql;

CREATE TRIGGER brain_observations_sync_clock
    BEFORE UPDATE OF content, observation_type, trust, evidence, tags
    ON brain_observations
    FOR EACH ROW
    WHEN (
        OLD.content          IS DISTINCT FROM NEW.content
     OR OLD.observation_type IS DISTINCT FROM NEW.observation_type
     OR OLD.trust            IS DISTINCT FROM NEW.trust
     OR OLD.evidence         IS DISTINCT FROM NEW.evidence
     OR OLD.tags             IS DISTINCT FROM NEW.tags
    )
    EXECUTE FUNCTION brain_bump_sync_clock();

COMMENT ON COLUMN brain_observations.version IS
    'Ticks once per change that another machine would care about — content, type, trust, evidence, tags. Deliberately NOT moved by decay, reinforcement or reembedding: those are local telemetry, and a clock that ticks for them makes every export look like new work.';
