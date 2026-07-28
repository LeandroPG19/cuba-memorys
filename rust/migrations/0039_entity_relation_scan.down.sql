DROP INDEX IF EXISTS idx_entities_unscanned;

ALTER TABLE brain_entities
    DROP COLUMN IF EXISTS relations_scanned_at;
