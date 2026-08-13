-- Migration 0017 put row-level security on the six tables that were the graph at
-- the time. Everything added since that holds text — snapshots, working memory,
-- chunks, procedures, judgments, and the sync conflict record from 0052 — was
-- left outside, and the gap is not theoretical: measured on a throwaway database,
-- a session scoped to one project reads another project's observation text
-- straight out of brain_sync_conflicts, which stores it verbatim.
--
-- Read this next part before concluding anything from a measurement: on a default
-- install RLS is INERT. .env.example connects as `cuba`, which is SUPERUSER with
-- BYPASSRLS, so no policy in this database is enforced at all. These policies bite
-- only where scripts/create-app-role.sql has been run and the runtime connects as
-- cuba_app — the recommended setup, and the one `doctor` pushes people towards.
-- Somebody who measures "zero rows leaked" as `cuba` has measured nothing.
--
-- The policy is copied from 0017 verbatim rather than written afresh, and the
-- three escapes in it are load-bearing, not sloppiness:
--   ''       the pool's resting scope (db.rs sets it on acquire and release), so a
--            single-project install behaves exactly as it did before
--   '*'      the documented kill switch
--   NULL     rows shared across projects, and connections that never saw the pool

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_compaction_snapshots',
        'brain_wm',
        'brain_observation_chunks',
        'brain_procedures',
        'brain_judgments',
        'brain_embedding_stats'
    ]
    LOOP
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
    END LOOP;
END $$;

-- brain_sync_conflicts has no project_id of its own — it points at the observation
-- it is about — so it delegates to the policy that already guards that table.
--
-- Why this cannot lock the import out of its own inserts, which is the question
-- that decides whether this migration is safe: the INSERT in handlers/sync.rs
-- already reads `JOIN brain_observations o ON o.id = u.id`. That join is filtered
-- by the same policy under the same app.current_project the import sets with SET
-- LOCAL, so every row the INSERT can produce names an observation that is visible
-- in that scope. The WITH CHECK — which PostgreSQL derives from USING when no
-- explicit one is given — is therefore true by construction. A conflict about an
-- out-of-scope observation is not rejected; it is never generated, which is
-- already today's behaviour.
--
-- That safety rests entirely on that JOIN, and the place it would break is not
-- this file. If anyone ever rewrites the conflict INSERT to take rows straight
-- from the bundle without checking what is here, this policy starts refusing them.
--
-- The foreign key is ON DELETE CASCADE, so the EXISTS can never be false for a
-- surviving row: a conflict cannot outlive the observation it describes.

ALTER TABLE brain_sync_conflicts ENABLE ROW LEVEL SECURITY;
ALTER TABLE brain_sync_conflicts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON brain_sync_conflicts;
CREATE POLICY tenant_isolation ON brain_sync_conflicts
    USING (EXISTS (SELECT 1 FROM brain_observations o WHERE o.id = observation_id));

-- What this migration deliberately leaves alone, so the next person does not
-- spend a round rediscovering why:
--
--   brain_audit_log        The hash chain is linear over every row. Hiding one
--                          makes `cuba_archivo verify` read a truncated chain and
--                          report tampering where there is none. 0041 already
--                          revokes UPDATE and DELETE from cuba_app, which is the
--                          right protection for an append-only log.
--   brain_peer_notices     Written by a remote peer that has no active project
--                          here, so project_id would always be NULL and the policy
--                          a no-op with a cost.
--   brain_handler_failures Its only reader is admin/traffic, which runs outside
--                          session::with_client with an empty scope, so the policy
--                          would not change a single row it returns.
--   brain_triggers,        Both return content and neither has a project column.
--   brain_verify_log       Adding one is its own migration and its own risk; it is
--                          recorded as debt rather than smuggled in here.

COMMENT ON POLICY tenant_isolation ON brain_sync_conflicts IS
    'Delegates to the policy on brain_observations through the row this conflict is about. Safe only because the INSERT in handlers/sync.rs joins that table first — see the header of migration 0055.';
