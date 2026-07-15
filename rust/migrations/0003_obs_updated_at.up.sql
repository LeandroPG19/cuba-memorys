-- FIX-OBS-001: Add updated_at to brain_observations.
-- The column was missing from schema but referenced in decay/eco/REM queries,
-- causing all those operations to fail silently.
-- (Already declared in 0001 for fresh installs; this migration covers legacy DBs.)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_observations' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE brain_observations ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;
