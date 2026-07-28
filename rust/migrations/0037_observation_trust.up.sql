ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS trust text NOT NULL DEFAULT 'trusted';

ALTER TABLE brain_observations
    ADD CONSTRAINT brain_observations_trust_check
    CHECK (trust IN ('trusted', 'quarantined'));

CREATE INDEX IF NOT EXISTS idx_obs_quarantined
    ON brain_observations (created_at DESC)
    WHERE trust = 'quarantined';

COMMENT ON COLUMN brain_observations.trust IS
    'trusted = retrievable; quarantined = written from untrusted text, withheld from cuba_faro until promoted via cuba_eco action=promote.';
