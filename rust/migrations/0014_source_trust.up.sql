DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_source_trust'
    ) THEN
        CREATE TABLE brain_source_trust (
            source TEXT PRIMARY KEY,
            alpha FLOAT NOT NULL DEFAULT 1.0 CHECK (alpha >= 1.0),
            beta FLOAT NOT NULL DEFAULT 1.0 CHECK (beta >= 1.0),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        INSERT INTO brain_source_trust (source) VALUES
            ('agent'),
            ('error_detection'),
            ('user'),
            ('consolidation'),
            ('inference')
        ON CONFLICT (source) DO NOTHING;
    END IF;
END $$;
