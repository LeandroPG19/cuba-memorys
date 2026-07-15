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
