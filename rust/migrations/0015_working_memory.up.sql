DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_wm'
    ) THEN
        CREATE TABLE brain_wm (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            session_id UUID REFERENCES brain_sessions(id) ON DELETE CASCADE,
            project_id UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
            content TEXT NOT NULL,
            tag TEXT,
            ttl_seconds INT NOT NULL DEFAULT 3600 CHECK (ttl_seconds > 0),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX idx_wm_session ON brain_wm(session_id, created_at DESC);
        CREATE INDEX idx_wm_expires ON brain_wm(expires_at);
        CREATE INDEX idx_wm_project
            ON brain_wm(project_id) WHERE project_id IS NOT NULL;
    END IF;
END $$;
