CREATE TABLE IF NOT EXISTS brain_entity_aliases (
    alias_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    alias_text TEXT NOT NULL,
    language_code VARCHAR(5) DEFAULT 'es',
    frequency_count BIGINT DEFAULT 0,
    last_used TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(entity_id, alias_text)
);

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_entity_alias_trgm ON brain_entity_aliases USING gin (alias_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entity_lookup ON brain_entity_aliases(entity_id);

CREATE OR REPLACE FUNCTION find_similar_entities(search_term TEXT, threshold FLOAT DEFAULT 0.4)
RETURNS TABLE (entity_id UUID, alias_text TEXT, similarity_score FLOAT) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT e.entity_id, e.alias_text, similarity(e.alias_text, search_term) AS score
    FROM brain_entity_aliases e
    WHERE similarity(e.alias_text, search_term) > threshold
    ORDER BY score DESC
    LIMIT 5;
END;
$$;