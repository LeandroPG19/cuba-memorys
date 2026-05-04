DROP INDEX IF EXISTS idx_obs_tags;
ALTER TABLE brain_observations DROP COLUMN IF EXISTS tags;
