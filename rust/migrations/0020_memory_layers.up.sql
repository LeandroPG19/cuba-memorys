DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memory_layer_type') THEN
        CREATE TYPE memory_layer_type AS ENUM ('episodic', 'semantic', 'working', 'project');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS brain_memory_layers (
    layer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer_name memory_layer_type NOT NULL UNIQUE,
    decay_rate FLOAT NOT NULL DEFAULT 0.05,
    decay_model TEXT NOT NULL DEFAULT 'exponential',
    ttl_days INT,
    retrieval_weight FLOAT DEFAULT 1.0,
    config_json JSONB DEFAULT '{}'::jsonb
);

INSERT INTO brain_memory_layers (layer_name, decay_rate, decay_model, ttl_days, retrieval_weight) VALUES
    ('episodic', 0.05, 'exponential', 90, 0.8),
    ('semantic', 0.01, 'power_law', NULL, 1.0),
    ('working', 0.15, 'exponential', 1, 0.6),
    ('project', 0.001, 'exponential', NULL, 1.0)
ON CONFLICT (layer_name) DO NOTHING;

ALTER TABLE brain_facts ADD COLUMN IF NOT EXISTS layer_id UUID REFERENCES brain_memory_layers(layer_id);