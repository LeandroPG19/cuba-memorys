ALTER TABLE brain_entities
    DROP COLUMN IF EXISTS relation_scan_attempts;

ALTER TABLE brain_observations
    DROP COLUMN IF EXISTS extraction_attempts;
