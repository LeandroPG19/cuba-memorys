
CREATE INDEX IF NOT EXISTS idx_observations_created_at ON brain_observations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entities_created_at     ON brain_entities (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_errors_created_at       ON brain_errors (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_created_at     ON brain_episodes (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created_at        ON brain_facts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_session_id     ON brain_episodes (session_id);
CREATE INDEX IF NOT EXISTS idx_node_metrics_community  ON brain_node_metrics (community_id);
