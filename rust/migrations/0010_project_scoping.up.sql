-- V0.8: Project scoping — brain_projects table + project_id NULL columns
-- across the 6 main tables. NULL = global (backward compat for legacy rows).
-- Filter activates only when CUBA_PROJECT_FILTER != "off" AND a session has a
-- non-null project_id.
-- Must run BEFORE the v0.8 satellite tables (snapshots, sync_state, judgments)
-- because they FK to brain_projects.
DO $$
DECLARE t TEXT;
BEGIN
    -- M1.1: brain_projects table
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_projects'
    ) THEN
        CREATE TABLE brain_projects (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_active_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_projects_name ON brain_projects(name);
    END IF;

    -- M1.2: project_id NULL FK in 6 scoped tables
    FOREACH t IN ARRAY ARRAY[
        'brain_entities','brain_observations',
        'brain_episodes','brain_sessions',
        'brain_errors','brain_relations'
    ]
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = t AND column_name = 'project_id'
        ) THEN
            EXECUTE format(
                'ALTER TABLE %I ADD COLUMN project_id UUID NULL
                 REFERENCES brain_projects(id) ON DELETE SET NULL', t);
            EXECUTE format(
                'CREATE INDEX IF NOT EXISTS idx_%s_project ON %I(project_id)
                 WHERE project_id IS NOT NULL', t, t);
        END IF;
    END LOOP;
END $$;
