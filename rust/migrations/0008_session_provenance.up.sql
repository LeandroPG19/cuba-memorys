-- V0.6: Session provenance — link observations/episodes to the session that created them.
-- Enables session_diff in cuba_jornada end and provenance tracking.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_observations' AND column_name = 'session_id'
    ) THEN
        ALTER TABLE brain_observations ADD COLUMN session_id UUID REFERENCES brain_sessions(id) ON DELETE SET NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_episodes' AND column_name = 'session_id'
    ) THEN
        ALTER TABLE brain_episodes ADD COLUMN session_id UUID REFERENCES brain_sessions(id) ON DELETE SET NULL;
    END IF;
END $$;
