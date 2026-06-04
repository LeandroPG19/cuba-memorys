CREATE OR REPLACE VIEW v_unified_memory_search AS
SELECT
    f.fact_id,
    f.subject,
    f.predicate,
    f.object,
    f.confidence,
    f.layer_id,
    l.layer_name::text AS layer_name,
    m.energy_score,
    m.pagerank_score,
    st.alpha / NULLIF(st.alpha + st.beta, 0) AS calibrated_confidence
FROM brain_facts f
LEFT JOIN brain_memory_layers l ON f.layer_id = l.layer_id
LEFT JOIN brain_entities e ON e.id = f.subject_entity_id
LEFT JOIN brain_node_metrics m ON m.node_id = e.id
LEFT JOIN brain_source_trust st ON st.source = COALESCE(f.predicate, 'agent')
WHERE f.is_current = TRUE;