
ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS origin_node text
    DEFAULT NULLIF(current_setting('cuba.node_name', true), '');

COMMENT ON COLUMN brain_observations.origin_node IS
    'Machine that created this observation (CUBA_NODE_NAME at write time). NULL = unknown/pre-provenance.';
