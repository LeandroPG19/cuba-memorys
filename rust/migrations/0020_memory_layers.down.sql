ALTER TABLE brain_facts DROP COLUMN IF EXISTS layer_id;
DROP TABLE IF EXISTS brain_memory_layers;
DROP TYPE IF EXISTS memory_layer_type;