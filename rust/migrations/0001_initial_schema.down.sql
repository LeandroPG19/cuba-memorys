-- Rollback v0.3.0 base schema. Order matters: drop FK children first.
DROP TABLE IF EXISTS brain_sessions CASCADE;
DROP TABLE IF EXISTS brain_errors CASCADE;
DROP TABLE IF EXISTS brain_relations CASCADE;
DROP TABLE IF EXISTS brain_observations CASCADE;
DROP TABLE IF EXISTS brain_entities CASCADE;
-- Extensions left intact (may be used by other schemas).
