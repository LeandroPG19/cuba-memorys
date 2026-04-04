//! Handler: cuba_hipotesis — Abductive inference on the knowledge graph.
//!
//! Given an observed effect E, finds plausible causes by traversing
//! causal relations BACKWARDS through the knowledge graph.
//!
//! Abductive reasoning: if H → E (H causes E) and E is observed,
//! then H is a plausible hypothesis. Score = path_strength × entity_importance.
//!
//! Supported relation types for backward traversal:
//!   - causes: direct causal link
//!   - related_to: associative link
//!   - depends_on: dependency (reversed → dependent may cause issues)
//!
//! All operations are READ-ONLY.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("explain");
    match action {
        "explain" => explain(pool, &args).await,
        _ => anyhow::bail!("Invalid action: {action}. Use 'explain'"),
    }
}

/// Find plausible causes for an observed effect via backward graph traversal.
async fn explain(pool: &PgPool, args: &Value) -> Result<Value> {
    let effect = args.get("effect").and_then(|v| v.as_str()).unwrap_or("");
    if effect.is_empty() {
        anyhow::bail!("'effect' parameter is required");
    }

    let max_depth = args
        .get("max_depth")
        .and_then(|v| v.as_i64())
        .unwrap_or(3)
        .clamp(1, 5);

    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10)
        .min(50);

    // Check effect entity exists
    let effect_exists: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1 LIMIT 1")
            .bind(effect)
            .fetch_optional(pool)
            .await?;

    if effect_exists.is_none() {
        return Ok(serde_json::json!({
            "action": "explain",
            "effect": effect,
            "hypotheses": [],
            "count": 0,
            "note": "Effect entity not found in knowledge graph"
        }));
    }

    // CTE recursive backward traversal over causal relations.
    // path_strength = product of relation strengths along the chain.
    // plausibility = path_strength × entity_importance (incorporates PageRank + Hebbian boosts).
    // Cycle prevention via path array.
    let rows: Vec<(String, String, f64, i64, f64, f64)> = sqlx::query_as(
        "WITH RECURSIVE causal_chain AS (
            -- Base: direct causes of the effect
            SELECT
                r.from_entity AS current_node,
                r.relation_type,
                1::bigint AS depth,
                r.strength::float8 AS path_strength,
                ARRAY[target.id, r.from_entity] AS path
            FROM brain_relations r
            JOIN brain_entities target ON r.to_entity = target.id
            WHERE target.name = $1
              AND r.relation_type IN ('causes', 'related_to', 'depends_on')

            UNION ALL

            -- Recurse: causes of causes
            SELECT
                r.from_entity,
                r.relation_type,
                cc.depth + 1,
                (cc.path_strength * r.strength)::float8,
                cc.path || r.from_entity
            FROM brain_relations r
            JOIN causal_chain cc ON r.to_entity = cc.current_node
            WHERE cc.depth < $2
              AND r.relation_type IN ('causes', 'related_to', 'depends_on')
              AND NOT (r.from_entity = ANY(cc.path))
        )
        SELECT
            e.name,
            e.entity_type,
            e.importance::float8,
            cc.depth,
            cc.path_strength,
            (cc.path_strength * e.importance)::float8 AS plausibility_score
        FROM causal_chain cc
        JOIN brain_entities e ON cc.current_node = e.id
        ORDER BY plausibility_score DESC
        LIMIT $3",
    )
    .bind(effect)
    .bind(max_depth)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let hypotheses: Vec<Value> = rows
        .iter()
        .map(
            |(name, entity_type, importance, depth, path_strength, plausibility)| {
                serde_json::json!({
                    "cause": name,
                    "entity_type": entity_type,
                    "importance": importance,
                    "hops": depth,
                    "path_strength": path_strength,
                    "plausibility": plausibility
                })
            },
        )
        .collect();

    let count = hypotheses.len();

    Ok(serde_json::json!({
        "action": "explain",
        "effect": effect,
        "hypotheses": hypotheses,
        "count": count,
        "max_depth": max_depth,
        "scoring": "plausibility = path_strength × entity_importance (incorporates PageRank + Hebbian)"
    }))
}
