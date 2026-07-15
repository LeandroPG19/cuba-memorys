-- Prospective memory triggers (cuba_centinela).
-- "Remember to remind me about X when Y happens."
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_triggers'
    ) THEN
        CREATE TABLE brain_triggers (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            entity_pattern TEXT NOT NULL,
            condition_type TEXT NOT NULL
                CHECK (condition_type IN ('on_access', 'on_session_start', 'on_error_match')),
            message TEXT NOT NULL,
            observation_id UUID REFERENCES brain_observations(id) ON DELETE SET NULL,
            active BOOLEAN DEFAULT TRUE,
            fire_count INT DEFAULT 0,
            max_fires INT DEFAULT 1,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
        CREATE INDEX idx_triggers_active ON brain_triggers(active) WHERE active = TRUE;
        CREATE INDEX idx_triggers_pattern ON brain_triggers USING GIN(entity_pattern gin_trgm_ops);
    END IF;
END $$;
