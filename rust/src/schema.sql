-- cuba-memorys schema v2.2.0 (Deep Research V3: SAC calibration, BCM EMA, configurable γ/TTL)
-- Based on v2.1.0 + BCM θ_M persistence (EMA sliding threshold)

-- Extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Core Tables ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS brain_entities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    importance FLOAT DEFAULT 0.5
        CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count INT DEFAULT 0,
    -- V3: BCM EMA sliding threshold (Deep Research 2026-03-14)
    bcm_theta FLOAT DEFAULT 10.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(entity_type, ''))
    ) STORED
);

CREATE TABLE IF NOT EXISTS brain_observations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_id UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    observation_type TEXT DEFAULT 'fact'
        CHECK (observation_type IN (
            'fact', 'decision', 'lesson', 'preference',
            'error', 'solution', 'context', 'tool_usage', 'superseded'
        )),
    importance FLOAT DEFAULT 0.5
        CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    source TEXT DEFAULT 'agent'
        CHECK (source IN ('agent', 'error_detection', 'user', 'consolidation', 'inference')),
    source_id TEXT,
    version INT DEFAULT 1,
    previous_versions JSONB DEFAULT '[]',
    -- FSRS-6 decay model (Ye 2024, ADR-001: custom implementation)
    stability FLOAT DEFAULT 1.0,
    difficulty FLOAT DEFAULT 5.0,
    -- Dual-Strength Model (Bjork & Bjork 1992, ADR-002: v1.0)
    storage_strength FLOAT DEFAULT 0.5,
    retrieval_strength FLOAT DEFAULT 1.0,
    -- Temporal validity for data lifecycle
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    -- Semantic embedding (pgvector — 384d BGE-small)
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED
);

CREATE TABLE IF NOT EXISTS brain_relations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    from_entity UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    to_entity UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0
        CHECK (strength >= 0.0 AND strength <= 1.0),
    bidirectional BOOLEAN DEFAULT FALSE,
    last_traversed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(from_entity, to_entity, relation_type)
);

CREATE TABLE IF NOT EXISTS brain_errors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    project TEXT DEFAULT 'default',
    solution TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    attempts INT DEFAULT 0,
    synapse_weight FLOAT DEFAULT 1.0
        CHECK (synapse_weight >= 0.0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple',
            coalesce(error_type, '') || ' ' ||
            coalesce(error_message, '') || ' ' ||
            coalesce(solution, ''))
    ) STORED
);

CREATE TABLE IF NOT EXISTS brain_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_name TEXT,
    goals JSONB DEFAULT '[]',
    summary TEXT,
    outcome TEXT CHECK (outcome IN ('success', 'partial', 'failed', 'abandoned', NULL)),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- ── Indexes ──────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_entities_search ON brain_entities USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_entities_trgm ON brain_entities USING GIN(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_type ON brain_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_obs_search ON brain_observations USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_obs_trgm ON brain_observations USING GIN(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_obs_entity ON brain_observations(entity_id);
CREATE INDEX IF NOT EXISTS idx_obs_type ON brain_observations(observation_type);
CREATE INDEX IF NOT EXISTS idx_obs_importance ON brain_observations(importance DESC);
CREATE INDEX IF NOT EXISTS idx_errors_search ON brain_errors USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_errors_trgm ON brain_errors USING GIN(error_message gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_errors_project ON brain_errors(project);
CREATE INDEX IF NOT EXISTS idx_errors_resolved ON brain_errors(resolved);
CREATE INDEX IF NOT EXISTS idx_relations_from ON brain_relations(from_entity);
CREATE INDEX IF NOT EXISTS idx_relations_to ON brain_relations(to_entity);

-- Partial index for temporal validity filter
CREATE INDEX IF NOT EXISTS idx_obs_valid_until
    ON brain_observations(valid_until)
    WHERE valid_until IS NOT NULL;

-- Partial GIN index excluding superseded — avoids scanning obsolete rows
CREATE INDEX IF NOT EXISTS idx_obs_active_search
    ON brain_observations USING GIN(search_vector)
    WHERE observation_type != 'superseded';

-- HNSW index for ANN vector search — O(log n) cosine similarity
-- m=16: connections/node (optimal for 384d BGE-small)
-- ef_construction=128: build quality (Google Cloud + pgvector benchmarks 2025)
CREATE INDEX IF NOT EXISTS idx_obs_embedding_hnsw
    ON brain_observations USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
