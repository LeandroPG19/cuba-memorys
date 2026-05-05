-- v0.4.0: Episodic memory (Tulving 1972) — separate from semantic facts.
-- Specific temporal events with actors/artifacts. Decay via Wixted power-law.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_episodes'
    ) THEN
        CREATE TABLE brain_episodes (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            entity_id UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            ended_at TIMESTAMPTZ,
            actors TEXT[] DEFAULT '{}',
            artifacts TEXT[] DEFAULT '{}',
            importance FLOAT DEFAULT 0.5
                CHECK (importance >= 0.0 AND importance <= 1.0),
            embedding vector(384),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', content)
            ) STORED
        );
        CREATE INDEX idx_episodes_entity ON brain_episodes(entity_id);
        CREATE INDEX idx_episodes_search ON brain_episodes USING GIN(search_vector);
        CREATE INDEX idx_episodes_trgm   ON brain_episodes USING GIN(content gin_trgm_ops);
        CREATE INDEX idx_episodes_time   ON brain_episodes(started_at DESC);
        -- HNSW vector index for episode semantic search
        CREATE INDEX idx_episodes_embedding_hnsw
            ON brain_episodes USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 128);
    END IF;
END $$;
