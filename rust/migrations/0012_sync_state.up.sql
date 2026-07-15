-- V0.8: Sync state tracker — records which export manifests have been
-- imported locally so re-imports are no-ops (idempotent dedup).
-- Depends on 0010 for brain_projects FK.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_sync_state'
    ) THEN
        CREATE TABLE brain_sync_state (
            manifest_hash TEXT PRIMARY KEY,
            project_id UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
            imported_at TIMESTAMPTZ DEFAULT NOW(),
            rows_inserted INT NOT NULL DEFAULT 0,
            source_path TEXT
        );
        CREATE INDEX idx_sync_project
            ON brain_sync_state(project_id) WHERE project_id IS NOT NULL;
    END IF;
END $$;
