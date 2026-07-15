-- BCM θ_M EMA migration — adds bcm_theta column for persistent sliding threshold.
-- V3: Deep Research 2026-03-14. Bienenstock-Cooper-Munro 1982.
-- (Already declared in 0001 for fresh installs; this migration covers legacy DBs.)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_entities'
        AND column_name = 'bcm_theta'
    ) THEN
        ALTER TABLE brain_entities
            ADD COLUMN bcm_theta FLOAT DEFAULT 10.0;
    END IF;
END $$;
