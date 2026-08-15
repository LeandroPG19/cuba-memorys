DROP INDEX IF EXISTS idx_observations_unextracted;

ALTER TABLE brain_observations
    DROP COLUMN IF EXISTS extracted_at;
