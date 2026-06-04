-- Enable Row-Level Security on project-scoped tables (v0.9+).
-- Safe to re-run: drops and recreates tenant_isolation policies.
--
-- Usage:
--   psql "$DATABASE_URL" -f scripts/enable-rls.sql
--
-- Requires superuser or table owner. Complements migration 0017_rls_policies.up.sql
-- and bitemporal RLS in 0018_bitemporal_core.up.sql.

DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities','brain_observations',
        'brain_episodes','brain_sessions',
        'brain_errors','brain_relations',
        'brain_facts'
    ]
    LOOP
        IF EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = t
        ) THEN
            EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
            EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
            EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
            EXECUTE format(
                'CREATE POLICY tenant_isolation ON %I
                 USING (
                     current_setting(''app.current_project'', true) IS NULL
                  OR current_setting(''app.current_project'', true) = ''''
                  OR current_setting(''app.current_project'', true) = ''*''
                  OR project_id IS NULL
                  OR project_id::text = current_setting(''app.current_project'', true)
                 )', t);
            RAISE NOTICE 'RLS enabled on %', t;
        ELSE
            RAISE NOTICE 'Skipping missing table %', t;
        END IF;
    END LOOP;
END $$;