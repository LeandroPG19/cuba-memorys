DROP INDEX IF EXISTS idx_obs_evidence;
ALTER TABLE brain_observations DROP CONSTRAINT IF EXISTS brain_observations_verified_is_stamped;
ALTER TABLE brain_observations DROP CONSTRAINT IF EXISTS brain_observations_evidence_needs_a_check;
ALTER TABLE brain_observations DROP CONSTRAINT IF EXISTS brain_observations_evidence_check;
ALTER TABLE brain_observations DROP COLUMN IF EXISTS verified_at;
ALTER TABLE brain_observations DROP COLUMN IF EXISTS verification;
ALTER TABLE brain_observations DROP COLUMN IF EXISTS evidence;
