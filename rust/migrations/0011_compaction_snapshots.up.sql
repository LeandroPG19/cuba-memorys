-- V0.8: Compaction-survival snapshots — agents call cuba_pre_compact(snapshot)
-- before /compact and cuba_pre_compact(restore) post-compact to reinject context.
-- Depends on 0010 for brain_projects FK.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_compaction_snapshots'
    ) THEN
        CREATE TABLE brain_compaction_snapshots (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            session_id UUID REFERENCES brain_sessions(id) ON DELETE CASCADE,
            project_id UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
            summary_md TEXT NOT NULL,
            key_observations JSONB DEFAULT '[]',
            decisions JSONB DEFAULT '[]',
            unresolved_errors JSONB DEFAULT '[]',
            pending_embeddings JSONB DEFAULT '[]',
            active_goals JSONB DEFAULT '[]',
            obs_count INT DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_snapshots_session
            ON brain_compaction_snapshots(session_id, created_at DESC);
        CREATE INDEX idx_snapshots_project
            ON brain_compaction_snapshots(project_id)
            WHERE project_id IS NOT NULL;
    END IF;
END $$;
