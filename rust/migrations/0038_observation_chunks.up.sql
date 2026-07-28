DO $$
DECLARE
    dim int;
BEGIN
    SELECT a.atttypmod INTO dim
    FROM pg_attribute a
    WHERE a.attrelid = 'brain_observations'::regclass
      AND a.attname = 'embedding'
      AND NOT a.attisdropped;

    IF dim IS NULL OR dim <= 0 THEN
        dim := 384;
    END IF;

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS brain_observation_chunks (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id uuid NOT NULL REFERENCES brain_observations(id) ON DELETE CASCADE,
            chunk_index int NOT NULL,
            content text NOT NULL,
            embedding vector(%s),
            embedding_model text,
            project_id uuid REFERENCES brain_projects(id) ON DELETE SET NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (observation_id, chunk_index)
        )', dim);
END $$;

CREATE INDEX IF NOT EXISTS idx_obs_chunks_observation
    ON brain_observation_chunks (observation_id);

CREATE INDEX IF NOT EXISTS idx_obs_chunks_embedding_hnsw
    ON brain_observation_chunks USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE brain_observation_chunks IS
    'Overlapping slices of observations longer than the embedder truncation limit. The parent row keeps the full text; these carry the embeddings that would otherwise be cut off at 512 tokens.';
