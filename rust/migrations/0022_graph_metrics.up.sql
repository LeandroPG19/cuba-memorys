CREATE TABLE IF NOT EXISTS brain_node_metrics (
    node_id UUID PRIMARY KEY REFERENCES brain_entities(id) ON DELETE CASCADE,
    pagerank_score FLOAT DEFAULT 0.0,
    betweenness_centrality FLOAT DEFAULT 0.0,
    energy_score FLOAT DEFAULT 0.0,
    community_id UUID,
    last_calculated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_node_energy ON brain_node_metrics(energy_score DESC);
CREATE INDEX IF NOT EXISTS idx_node_pagerank ON brain_node_metrics(pagerank_score DESC);