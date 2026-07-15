-- 0034: provenance — which machine wrote this memory.
--
-- Two computers sharing one database (see the cloud mode) need to tell their memories
-- apart: "did I decide this on the laptop or the workshop PC?". `source` cannot answer
-- that — it already means agent-vs-user. So a new column, filled from a per-session GUC
-- the pool sets on connect (`SET cuba.node_name = '<CUBA_NODE_NAME>'`), the same way
-- audit context is injected. Every INSERT inherits it through the column DEFAULT, so no
-- write path has to be touched.
--
-- NULL is a legitimate value: existing rows predate provenance, and a node that never
-- set CUBA_NODE_NAME simply has none. NULLIF collapses the empty-string GUC to NULL so
-- "unset" and "" are the same thing.

ALTER TABLE brain_observations
    ADD COLUMN IF NOT EXISTS origin_node text
    DEFAULT NULLIF(current_setting('cuba.node_name', true), '');

COMMENT ON COLUMN brain_observations.origin_node IS
    'Machine that created this observation (CUBA_NODE_NAME at write time). NULL = unknown/pre-provenance.';
