DROP INDEX IF EXISTS idx_obs_quarantined;
ALTER TABLE brain_observations DROP CONSTRAINT IF EXISTS brain_observations_trust_check;
ALTER TABLE brain_observations DROP COLUMN IF EXISTS trust;
