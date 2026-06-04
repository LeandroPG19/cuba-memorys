CREATE OR REPLACE VIEW v_observations_compat AS
SELECT
    f.fact_id AS observation_id,
    e.name AS entity_name,
    f.predicate AS observation_type,
    f.object AS content,
    f.confidence AS importance,
    f.valid_from AS created_at,
    f.layer_id,
    f.is_current
FROM brain_facts f
LEFT JOIN brain_entities e ON e.id = f.subject_entity_id
WHERE f.is_current = TRUE;