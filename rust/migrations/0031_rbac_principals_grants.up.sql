
CREATE TABLE IF NOT EXISTS brain_principals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'human'
                CHECK (kind IN ('human', 'agent', 'service')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    disabled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS brain_grants (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES brain_principals(id) ON DELETE CASCADE,
    project_id   UUID REFERENCES brain_projects(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('reader', 'writer', 'admin')),
    granted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    granted_by   TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_grant_scoped
    ON brain_grants (principal_id, project_id, role) WHERE project_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_grant_global
    ON brain_grants (principal_id, role) WHERE project_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_grants_principal ON brain_grants (principal_id);

CREATE OR REPLACE FUNCTION brain_principal_can(target_project UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_principal TEXT := NULLIF(current_setting('app.current_principal', true), '');
    principal         UUID;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM brain_principals WHERE disabled_at IS NULL) THEN
        RETURN TRUE;
    END IF;

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
