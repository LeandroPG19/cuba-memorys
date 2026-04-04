-- cuba-memorys schema v0.3.0 (exponential decay, BCM EMA, pgvector embeddings)
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
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    source TEXT DEFAULT 'agent'
        CHECK (source IN ('agent', 'error_detection', 'user', 'consolidation', 'inference')),
    version INT DEFAULT 1,
    previous_versions JSONB DEFAULT '[]',
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

-- Partial GIN index excluding superseded — avoids scanning obsolete rows
CREATE INDEX IF NOT EXISTS idx_obs_active_search
    ON brain_observations USING GIN(search_vector)
    WHERE observation_type != 'superseded';

-- HNSW index for ANN vector search — O(log n) cosine similarity
-- m=16: connections/node (optimal for 384d multilingual-e5-small)
-- ef_construction=128: build quality (Google Cloud + pgvector benchmarks 2025)
CREATE INDEX IF NOT EXISTS idx_obs_embedding_hnsw
    ON brain_observations USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- ── Episodic Memory ──────────────────────────────────────────────
-- Separate from semantic memory (brain_observations) per Tulving (1972).
-- Episodic = specific temporal events with actors/artifacts.
-- Decay: power-law I(t) = I₀ / (1 + c·t)^β, halflife ~3d (vs 30d semantic).
CREATE TABLE IF NOT EXISTS brain_episodes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_id UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    actors TEXT[] DEFAULT '{}',
    artifacts TEXT[] DEFAULT '{}',
    importance FLOAT DEFAULT 0.5
        CHECK (importance >= 0.0 AND importance <= 1.0),
    -- Semantic embedding for vector search (same model as observations)
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_episodes_entity   ON brain_episodes(entity_id);
CREATE INDEX IF NOT EXISTS idx_episodes_search   ON brain_episodes USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_episodes_trgm     ON brain_episodes USING GIN(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_episodes_time     ON brain_episodes(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_embedding_hnsw
    ON brain_episodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- ── Prospective Memory Triggers (cuba_centinela) ────────────────
CREATE TABLE IF NOT EXISTS brain_triggers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_pattern TEXT NOT NULL,
    condition_type TEXT NOT NULL
        CHECK (condition_type IN ('on_access', 'on_session_start', 'on_error_match')),
    message TEXT NOT NULL,
    observation_id UUID REFERENCES brain_observations(id) ON DELETE SET NULL,
    active BOOLEAN DEFAULT TRUE,
    fire_count INT DEFAULT 0,
    max_fires INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_triggers_active ON brain_triggers(active) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_triggers_pattern ON brain_triggers USING GIN(entity_pattern gin_trgm_ops);

-- ── Bayesian Calibration Log (cuba_calibrar) ────────────────────
CREATE TABLE IF NOT EXISTS brain_verify_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    claim TEXT NOT NULL,
    entity_name TEXT,
    confidence FLOAT NOT NULL,
    grounding_level TEXT NOT NULL,
    outcome TEXT DEFAULT 'pending'
        CHECK (outcome IN ('pending', 'correct', 'incorrect', 'unknown')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_verify_log_entity ON brain_verify_log(entity_name);
CREATE INDEX IF NOT EXISTS idx_verify_log_outcome ON brain_verify_log(outcome);
