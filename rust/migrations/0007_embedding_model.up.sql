DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_observations' AND column_name = 'embedding_model'
    ) THEN
        ALTER TABLE brain_observations ADD COLUMN embedding_model TEXT DEFAULT 'multilingual-e5-small';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_episodes' AND column_name = 'embedding_model'
    ) THEN
        ALTER TABLE brain_episodes ADD COLUMN embedding_model TEXT DEFAULT 'multilingual-e5-small';
    END IF;
END $$;
