DO $halfvec$
DECLARE
    dim integer;
BEGIN
    SELECT atttypmod INTO dim
    FROM pg_attribute
    WHERE attrelid = 'brain_observations'::regclass AND attname = 'embedding';

    IF dim IS NULL OR dim < 1 THEN
        RAISE NOTICE 'brain_observations.embedding has no dimension — skipping halfvec';
        RETURN;
    END IF;

    EXECUTE format(
        'ALTER TABLE brain_observations ADD COLUMN IF NOT EXISTS embedding_half halfvec(%s)',
        dim
    );

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_obs_embedding_half_hnsw
         ON brain_observations USING hnsw (embedding_half halfvec_cosine_ops)
         WITH (m = 16, ef_construction = 128)'
    );
END
$halfvec$;
