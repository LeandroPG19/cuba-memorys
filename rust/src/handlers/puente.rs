//! Handler: cuba_puente — Relations management.
//!
//! CTE-based traversal and inference for graph navigation.
//! Hebbian: relations strengthen with use (traversal).

use crate::constants::VALID_RELATION_TYPES;
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::{PgPool, Row};

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "create" => create(pool, &args).await,
        "delete" => delete(pool, &args).await,
        "traverse" => traverse(pool, &args).await,
        "infer" => infer(pool, &args).await,
        "predict" => predict_links(pool, &args).await,
        _ => anyhow::bail!(
            "Invalid action: {action}. Use create/delete/traverse/infer/predict"
        ),
    }
}

/// Create a relation between entities.
async fn create(pool: &PgPool, args: &Value) -> Result<Value> {
    let from = args
        .get("from_entity")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let to = args.get("to_entity").and_then(|v| v.as_str()).unwrap_or("");
    let rel_type = args
        .get("relation_type")
        .and_then(|v| v.as_str())
        .unwrap_or("related_to");
    let bidirectional = args
        .get("bidirectional")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if from.is_empty() || to.is_empty() {
        anyhow::bail!("from_entity and to_entity are required");
    }
    if !VALID_RELATION_TYPES.contains(&rel_type) {
        anyhow::bail!("Invalid relation_type: {rel_type}");
    }

    // Get entity IDs
    let from_id = get_entity_id(pool, from).await?;
    let to_id = get_entity_id(pool, to).await?;

    // Upsert relation (strengthen if exists, create if not)
    let result = sqlx::query(
        "INSERT INTO brain_relations (from_entity, to_entity, relation_type, bidirectional)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (from_entity, to_entity, relation_type)
         DO UPDATE SET strength = LEAST(brain_relations.strength + 0.1, 1.0),
                       last_traversed = NOW()
         RETURNING (xmax = 0) AS is_insert",
    )
    .bind(from_id)
    .bind(to_id)
    .bind(rel_type)
    .bind(bidirectional)
    .fetch_one(pool)
    .await?;

    let is_new: bool = result.get::<bool, _>("is_insert");
    if !is_new {
        tracing::info!(
            from = from, to = to, rel = rel_type,
            "relation already exists — strengthened (Hebbian)"
        );
    }

    // If bidirectional, create reverse
    if bidirectional {
        sqlx::query(
            "INSERT INTO brain_relations (from_entity, to_entity, relation_type, bidirectional)
             VALUES ($1, $2, $3, true)
             ON CONFLICT (from_entity, to_entity, relation_type)
             DO UPDATE SET strength = LEAST(brain_relations.strength + 0.1, 1.0),
                           last_traversed = NOW()",
        )
        .bind(to_id)
        .bind(from_id)
        .bind(rel_type)
        .execute(pool)
        .await?;
    }

    Ok(serde_json::json!({
        "action": "create",
        "from": from,
        "to": to,
        "relation_type": rel_type,
        "bidirectional": bidirectional
    }))
}

/// Delete a relation.
async fn delete(pool: &PgPool, args: &Value) -> Result<Value> {
    let from = args
        .get("from_entity")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let to = args.get("to_entity").and_then(|v| v.as_str()).unwrap_or("");
    let rel_type = args
        .get("relation_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let from_id = get_entity_id(pool, from).await?;
    let to_id = get_entity_id(pool, to).await?;

    let result = sqlx::query(
        "DELETE FROM brain_relations
         WHERE from_entity = $1 AND to_entity = $2 AND relation_type = $3",
    )
    .bind(from_id)
    .bind(to_id)
    .bind(rel_type)
    .execute(pool)
    .await?;

    Ok(serde_json::json!({
        "action": "delete",
        "deleted": result.rows_affected() > 0
    }))
}

/// Traverse graph from a starting entity using CTE.
async fn traverse(pool: &PgPool, args: &Value) -> Result<Value> {
    let start = args
        .get("start_entity")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let max_depth = args
        .get("max_depth")
        .and_then(|v| v.as_i64())
        .unwrap_or(3)
        .min(5);

    if start.is_empty() {
        anyhow::bail!("start_entity is required");
    }

    let start_id = get_entity_id(pool, start).await?;

    // CTE-based depth-first traversal
    let paths: Vec<(String, String, f64, i32)> = sqlx::query_as(
        r#"
        WITH RECURSIVE graph_walk AS (
            SELECT
                r.to_entity AS current_node,
                r.relation_type,
                e2.name AS node_name,
                r.strength,
                1 AS depth
            FROM brain_relations r
            JOIN brain_entities e2 ON r.to_entity = e2.id
            WHERE r.from_entity = $1

            UNION ALL

            SELECT
                r.to_entity,
                r.relation_type,
                e2.name,
                r.strength,
                gw.depth + 1
            FROM brain_relations r
            JOIN brain_entities e2 ON r.to_entity = e2.id
            JOIN graph_walk gw ON r.from_entity = gw.current_node
            WHERE gw.depth < $2
        )
        SELECT node_name, relation_type, strength, depth
        FROM graph_walk
        ORDER BY depth, strength DESC
        LIMIT 50
        "#,
    )
    .bind(start_id)
    .bind(max_depth)
    .fetch_all(pool)
    .await?;

    // Strengthen traversed edges (Hebbian)
    sqlx::query(
        "UPDATE brain_relations SET
            strength = LEAST(strength + 0.02, 1.0),
            last_traversed = NOW()
         WHERE from_entity = $1",
    )
    .bind(start_id)
    .execute(pool)
    .await?;

    let nodes: Vec<Value> = paths
        .iter()
        .map(|(name, rel_type, strength, depth)| {
            serde_json::json!({
                "name": name,
                "relation": rel_type,
                "strength": strength,
                "depth": depth
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "traverse",
        "start": start,
        "max_depth": max_depth,
        "nodes": nodes,
        "count": nodes.len()
    }))
}

/// Infer transitive connections (A→B→C).
async fn infer(pool: &PgPool, args: &Value) -> Result<Value> {
    let start = args
        .get("start_entity")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let max_depth = args
        .get("max_depth")
        .and_then(|v| v.as_i64())
        .unwrap_or(3)
        .min(5);

    if start.is_empty() {
        anyhow::bail!("start_entity is required");
    }

    let start_id = get_entity_id(pool, start).await?;

    // Find transitive paths using CTE
    let inferences: Vec<(String, i32, f64)> = sqlx::query_as(
        r#"
        WITH RECURSIVE transitive_closure AS (
            SELECT
                r.to_entity AS current_node,
                1 AS depth,
                r.strength AS path_strength
            FROM brain_relations r
            WHERE r.from_entity = $1

            UNION ALL

            SELECT
                r.to_entity,
                tc.depth + 1,
                tc.path_strength * r.strength
            FROM brain_relations r
            JOIN transitive_closure tc ON r.from_entity = tc.current_node
            WHERE tc.depth < $2
        )
        SELECT e.name, tc.depth, tc.path_strength
        FROM transitive_closure tc
        JOIN brain_entities e ON tc.current_node = e.id
        WHERE tc.depth > 1
        ORDER BY tc.path_strength DESC
        LIMIT 20
        "#,
    )
    .bind(start_id)
    .bind(max_depth)
    .fetch_all(pool)
    .await?;

    let inferred: Vec<Value> = inferences
        .iter()
        .map(|(name, depth, strength)| {
            serde_json::json!({
                "entity": name,
                "hops": depth,
                "inferred_strength": strength
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "infer",
        "start": start,
        "inferred_connections": inferred,
        "count": inferred.len()
    }))
}

/// V0.6: Adamic-Adar link prediction — predict missing relations via common neighbors.
///
/// AA(x,y) = Σ 1/log(|N(z)|) for z in common neighbors of x and y.
/// High AA score means two entities share many well-connected neighbors,
/// suggesting a missing direct relation. Simple, effective for small graphs.
async fn predict_links(pool: &PgPool, args: &Value) -> Result<Value> {
    let entity_name = args
        .get("entity_name")
        .or_else(|| args.get("start_entity"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required for predict");
    }

    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10)
        .min(50);

    let entity_id: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1 LIMIT 1")
            .bind(entity_name)
            .fetch_optional(pool)
            .await?;

    let entity_id = match entity_id {
        Some((id,)) => id,
        None => {
            return Ok(serde_json::json!({
                "action": "predict",
                "entity": entity_name,
                "predictions": [],
                "note": "Entity not found"
            }))
        }
    };

    // Adamic-Adar link prediction via SQL CTEs
    let predictions: Vec<(String, String, f64)> = sqlx::query_as(
        r#"
        WITH entity_neighbors AS (
            SELECT CASE WHEN from_entity = $1 THEN to_entity ELSE from_entity END AS neighbor
            FROM brain_relations
            WHERE from_entity = $1 OR to_entity = $1
        ),
        candidate_links AS (
            SELECT DISTINCT
                CASE WHEN r.from_entity = en.neighbor THEN r.to_entity ELSE r.from_entity END AS candidate,
                en.neighbor AS shared_neighbor
            FROM entity_neighbors en
            JOIN brain_relations r ON r.from_entity = en.neighbor OR r.to_entity = en.neighbor
            WHERE CASE WHEN r.from_entity = en.neighbor THEN r.to_entity ELSE r.from_entity END != $1
              AND CASE WHEN r.from_entity = en.neighbor THEN r.to_entity ELSE r.from_entity END
                  NOT IN (SELECT neighbor FROM entity_neighbors)
        ),
        neighbor_degrees AS (
            SELECT node, SUM(cnt) AS degree FROM (
                SELECT from_entity AS node, COUNT(*) AS cnt FROM brain_relations GROUP BY from_entity
                UNION ALL
                SELECT to_entity, COUNT(*) FROM brain_relations GROUP BY to_entity
            ) sub GROUP BY node
        ),
        aa_scores AS (
            SELECT cl.candidate, SUM(1.0 / LN(2.0 + COALESCE(nd.degree, 1))) AS aa_score
            FROM candidate_links cl
            LEFT JOIN neighbor_degrees nd ON cl.shared_neighbor = nd.node
            GROUP BY cl.candidate
        )
        SELECT e.name, e.entity_type, aa.aa_score::float8
        FROM aa_scores aa
        JOIN brain_entities e ON aa.candidate = e.id
        ORDER BY aa.aa_score DESC
        LIMIT $2
        "#,
    )
    .bind(entity_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let prediction_json: Vec<Value> = predictions
        .iter()
        .map(|(name, etype, score)| {
            serde_json::json!({
                "entity": name,
                "entity_type": etype,
                "adamic_adar_score": score,
                "recommendation": "Consider creating a relation between these entities"
            })
        })
        .collect();

    let count = prediction_json.len();
    Ok(serde_json::json!({
        "action": "predict",
        "entity": entity_name,
        "predictions": prediction_json,
        "count": count,
        "algorithm": "Adamic-Adar (sum of 1/log(degree) for common neighbors)"
    }))
}

/// Get entity_id by name.
async fn get_entity_id(pool: &PgPool, name: &str) -> Result<uuid::Uuid> {
    let row: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
            .bind(name)
            .fetch_optional(pool)
            .await?;

    row.map(|(id,)| id)
        .context(format!("Entity '{name}' not found"))
}
