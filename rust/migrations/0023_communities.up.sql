CREATE TABLE IF NOT EXISTS brain_communities (
    community_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_name TEXT,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    algorithm_version TEXT DEFAULT 'leiden_v1',
    modularity_score FLOAT
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_community' AND table_name = 'brain_node_metrics'
    ) THEN
        ALTER TABLE brain_node_metrics
        ADD CONSTRAINT fk_community
        FOREIGN KEY (community_id) REFERENCES brain_communities(community_id) ON DELETE SET NULL;
    END IF;
END $$;