DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_verify_log'
    ) THEN
        CREATE TABLE brain_verify_log (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            claim TEXT NOT NULL,
            entity_name TEXT,
            confidence FLOAT NOT NULL,
            grounding_level TEXT NOT NULL,
            outcome TEXT DEFAULT 'pending'
                CHECK (outcome IN ('pending', 'correct', 'incorrect', 'unknown')),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_verify_log_entity ON brain_verify_log(entity_name);
        CREATE INDEX idx_verify_log_outcome ON brain_verify_log(outcome);
    END IF;
END $$;
