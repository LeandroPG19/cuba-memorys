DROP INDEX IF EXISTS idx_obs_embedding_half_hnsw;
ALTER TABLE brain_observations DROP COLUMN IF EXISTS embedding_half;
