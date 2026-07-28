ALTER TABLE brain_entities
    ADD COLUMN IF NOT EXISTS relations_scanned_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_entities_unscanned
    ON brain_entities (created_at)
    WHERE relations_scanned_at IS NULL;
