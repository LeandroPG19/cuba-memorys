-- v0.10: bitemporal facts + performance indexes on legacy observations

CREATE INDEX IF NOT EXISTS idx_obs_session ON brain_observations(session_id)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_obs_last_accessed ON brain_observations(last_accessed);

CREATE TABLE IF NOT EXISTS brain_facts (
    fact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_entity_id UUID REFERENCES brain_entities(id) ON DELETE CASCADE,
    project_id UUID REFERENCES brain_projects(id) ON DELETE SET NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMPTZ,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS brain_fact_supersedes (
    supersedes_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    old_fact_id UUID NOT NULL REFERENCES brain_facts(fact_id) ON DELETE CASCADE,
    new_fact_id UUID NOT NULL REFERENCES brain_facts(fact_id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    superseded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS brain_fact_provenance (
    provenance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_id UUID NOT NULL REFERENCES brain_facts(fact_id) ON DELETE CASCADE,
    episode_id UUID,
    source_type TEXT,
    extractor_version TEXT,
    source_span TEXT,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facts_validity ON brain_facts(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_current ON brain_facts(is_current) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_supersedes_old ON brain_fact_supersedes(old_fact_id);
CREATE INDEX IF NOT EXISTS idx_supersedes_new ON brain_fact_supersedes(new_fact_id);
CREATE INDEX IF NOT EXISTS idx_facts_project ON brain_facts(project_id) WHERE project_id IS NOT NULL;

CREATE OR REPLACE FUNCTION get_facts_as_of(query_date TIMESTAMPTZ)
RETURNS TABLE (
    fact_id UUID, subject TEXT, predicate TEXT, object TEXT, confidence FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT f.fact_id, f.subject, f.predicate, f.object, f.confidence
    FROM brain_facts f
    WHERE f.valid_from <= query_date
      AND (f.valid_to IS NULL OR f.valid_to > query_date)
      AND f.is_current = TRUE;
END;
$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'brain_facts') THEN
        ALTER TABLE brain_facts ENABLE ROW LEVEL SECURITY;
        ALTER TABLE brain_facts FORCE ROW LEVEL SECURITY;
        EXECUTE 'DROP POLICY IF EXISTS tenant_isolation ON brain_facts';
        EXECUTE '
            CREATE POLICY tenant_isolation ON brain_facts
            USING (
                current_setting(''app.current_project'', true) IS NULL
             OR current_setting(''app.current_project'', true) = ''''
             OR current_setting(''app.current_project'', true) = ''*''
             OR project_id IS NULL
             OR project_id::text = current_setting(''app.current_project'', true)
            )';
    END IF;
END $$;