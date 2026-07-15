-- V0.9: Working memory buffer (Baddeley 1992 working memory model).
--
-- Separates "what the agent has on its mental scratchpad RIGHT NOW" from
-- the episodic store. TTL-based — entries expire automatically and are
-- purged by `cuba_zafra` REM cycle. Useful for:
--   - inter-step plan state during long-horizon agent tasks
--   - tentative observations the agent is not yet ready to commit
--   - cross-tool-call reminders within a single session
--
-- Distinct from `brain_compaction_snapshots` (post-/compact recovery) and
-- `brain_observations` (long-term semantic memory).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_wm'
    ) THEN
        -- expires_at is a normal column populated by the handler
        -- (`pizarra::write` computes `now() + ttl_seconds`). PostgreSQL
        -- rejects GENERATED ... STORED with interval arithmetic because
        -- the make_interval/text-to-interval coercion is not declared
        -- IMMUTABLE. Computing expires_at at INSERT time avoids that and
        -- keeps the index-able TIMESTAMPTZ semantics.
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
