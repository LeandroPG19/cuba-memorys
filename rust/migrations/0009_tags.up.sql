-- V0.6: Auto-tagging — keyword tags array on observations.
-- Lets cuba_faro filter by tag (exact match against auto-extracted tags).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_observations' AND column_name = 'tags'
    ) THEN
        ALTER TABLE brain_observations ADD COLUMN tags TEXT[] DEFAULT '{}';
        CREATE INDEX IF NOT EXISTS idx_obs_tags ON brain_observations USING GIN(tags);
    END IF;
END $$;
