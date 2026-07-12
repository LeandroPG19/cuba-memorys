-- 0031: principals × grants — the minimum real RBAC.
--
-- Today "scope" is a search filter: every query adds `project_id = ...`, and the
-- `tenant_isolation` RLS policy trusts `app.current_project`, a session variable
-- the application itself sets. That keeps two projects from bleeding into each
-- other by accident. It is not an authorization model: nothing says *who* is
-- allowed to read a project, because until now there was only ever one who.
--
-- This adds the missing nouns:
--
--   brain_principals  — who (a person, an agent, a service)
--   brain_grants      — principal × project × role
--
-- and one RESTRICTIVE policy per protected table. RESTRICTIVE matters: permissive
-- policies combine with OR, so adding one would *widen* access. Restrictive
-- policies combine with AND, so this can only ever narrow it.
--
-- CERO REGRESIÓN, and that is deliberate: `brain_principal_can()` returns TRUE
-- while `brain_principals` is empty. A single-user install (today's install)
-- behaves exactly as before, because there is no principal to deny. The moment
-- you insert the first principal, the model switches to fail-closed: an
-- unidentified session gets nothing.
--
-- IMPORTANT — this is inert while the app connects as a SUPERUSER, because
-- Postgres skips RLS entirely for superusers. Same caveat as the audit log.
-- Run scripts/create-app-role.sql and connect as `cuba_app`, or none of this is
-- enforced. `cuba-memorys doctor` reports it (check: runtime_role).

CREATE TABLE IF NOT EXISTS brain_principals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'human'
                CHECK (kind IN ('human', 'agent', 'service')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Disable rather than delete: a grant history that vanishes is not an
    -- audit trail. Revocation must be visible.
    disabled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS brain_grants (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES brain_principals(id) ON DELETE CASCADE,
    -- NULL project = every project. The escape hatch for an owner/admin.
    project_id   UUID REFERENCES brain_projects(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('reader', 'writer', 'admin')),
    granted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    granted_by   TEXT
);

-- A NULL project_id must not be able to slip past a UNIQUE constraint twice:
-- in SQL, NULL <> NULL, so a plain UNIQUE would happily store the same
-- global grant a hundred times.
CREATE UNIQUE INDEX IF NOT EXISTS uq_grant_scoped
    ON brain_grants (principal_id, project_id, role) WHERE project_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_grant_global
    ON brain_grants (principal_id, role) WHERE project_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_grants_principal ON brain_grants (principal_id);

-- Can the current session touch this project?
--
-- STABLE, not VOLATILE: it is called once per row by RLS, and a volatile
-- function would defeat the planner on every scan.
CREATE OR REPLACE FUNCTION brain_principal_can(target_project UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER          -- must read the grant tables even when the caller cannot
SET search_path = public  -- a SECURITY DEFINER without this is a privilege-escalation bug
AS $$
DECLARE
    current_principal TEXT := NULLIF(current_setting('app.current_principal', true), '');
    principal         UUID;
BEGIN
    -- Single-user install: nobody is defined, so nobody is denied. This is what
    -- makes the migration a no-op for every existing setup.
    IF NOT EXISTS (SELECT 1 FROM brain_principals WHERE disabled_at IS NULL) THEN
        RETURN TRUE;
    END IF;

    -- Principals exist, so the model is on: an unidentified session gets nothing.
    IF current_principal IS NULL THEN
        RETURN FALSE;
    END IF;

    SELECT id INTO principal
    FROM brain_principals
    WHERE name = current_principal AND disabled_at IS NULL;

    IF principal IS NULL THEN
        RETURN FALSE;
    END IF;

    RETURN EXISTS (
        SELECT 1 FROM brain_grants g
        WHERE g.principal_id = principal
          AND (g.project_id IS NULL OR g.project_id = target_project)
    );
END;
$$;

-- Rows with no project (project_id IS NULL) stay visible: they are the shared,
-- unscoped memories, and hiding them would break every existing install that
-- never used project scoping at all.
DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'brain_entities', 'brain_observations', 'brain_episodes',
        'brain_errors', 'brain_relations'
    ] LOOP
        IF EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = t) THEN
            EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
            EXECUTE format('DROP POLICY IF EXISTS principal_grants ON %I', t);
            EXECUTE format(
                'CREATE POLICY principal_grants ON %I AS RESTRICTIVE
                 USING (project_id IS NULL OR brain_principal_can(project_id))', t);
        END IF;
    END LOOP;
END $$;
