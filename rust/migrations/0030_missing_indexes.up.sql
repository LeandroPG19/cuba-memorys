-- 0030: indexes on frequently-filtered columns that had none.
--
-- Audit found no index on created_at anywhere, despite cuba_faro filtering on a
-- created_at range in every text branch and several queries doing
-- ORDER BY created_at DESC. Also brain_episodes.session_id (session diff) and
-- brain_node_metrics.community_id (community grouping) were unindexed.
--
-- Irrelevant at hundreds of rows, real as the corpus grows. All are additive
-- and IF NOT EXISTS, so re-running is safe and no data is touched.
--
-- Note: not CONCURRENTLY — sqlx runs each migration in a transaction and
-- CREATE INDEX CONCURRENTLY cannot run inside one. At this table size the brief
-- lock is negligible.

CREATE INDEX IF NOT EXISTS idx_observations_created_at ON brain_observations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entities_created_at     ON brain_entities (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_errors_created_at       ON brain_errors (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_created_at     ON brain_episodes (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created_at        ON brain_facts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_session_id     ON brain_episodes (session_id);
CREATE INDEX IF NOT EXISTS idx_node_metrics_community  ON brain_node_metrics (community_id);
