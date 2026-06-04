DROP POLICY IF EXISTS tenant_isolation ON brain_facts;
ALTER TABLE brain_facts DISABLE ROW LEVEL SECURITY;
DROP FUNCTION IF EXISTS get_facts_as_of(TIMESTAMPTZ);
DROP TABLE IF EXISTS brain_fact_provenance;
DROP TABLE IF EXISTS brain_fact_supersedes;
DROP TABLE IF EXISTS brain_facts;
DROP INDEX IF EXISTS idx_obs_last_accessed;
DROP INDEX IF EXISTS idx_obs_session;