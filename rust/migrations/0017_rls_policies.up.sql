-- V0.9: PostgreSQL Row-Level Security per project (defense in depth).
--
-- The handlers already enforce project filtering at the SQL `WHERE` level,
-- but RLS adds a second wall: even if a future caller queries a table
-- directly without going through the handler, RLS clamps it to the active
-- project resolved by `current_setting('app.current_project', true)`.
--
-- Policies coexist with `cuba::project::current_project_id` — that helper
-- continues to set both the WHERE-clause bind AND issues `SET LOCAL
-- app.current_project = $1` (added in this migration's companion code).
--
-- NULL active project → policy returns ALL rows (back-compat with v0.7).
-- Sentinel `*` → admin bypass (matches handler-side kill-switch).
DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities','brain_observations',
        'brain_episodes','brain_sessions',
        'brain_errors','brain_relations'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        -- FORCE = applies to the table owner too (defense in depth).
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);

        -- Drop existing policy if re-running (idempotent)
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
    END LOOP;
END $$;
