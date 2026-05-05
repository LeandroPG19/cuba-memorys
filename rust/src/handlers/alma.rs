//! Handler: cuba_alma — Entity CRUD.
//!
//! FIX B3: create returns explicit "already_existed" flag instead of silent upsert.
//! V10: Upsert detection — AI knows if entity was created or re-found.
//! Hebbian: get/update auto-boost entity + neighbor importance.

use crate::cognitive::{dual_strength, hebbian};
use crate::constants::VALID_ENTITY_TYPES;
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "create" => create(pool, name, &args).await,
        "update" => update(pool, name, &args).await,
        "delete" => delete(pool, name).await,
        "get" => get(pool, name).await,
        _ => anyhow::bail!("Invalid action: {action}. Use create/update/delete/get"),
    }
}

/// Create entity — FIX B3: explicit "already_existed" detection.
async fn create(pool: &PgPool, name: &str, args: &Value) -> Result<Value> {
    if name.is_empty() || name.len() > 200 {
        anyhow::bail!("Entity name must be 1-200 characters");
    }

    let entity_type = args
        .get("entity_type")
        .and_then(|v| v.as_str())
        .unwrap_or("concept");

    if !VALID_ENTITY_TYPES.contains(&entity_type) {
        anyhow::bail!("Invalid entity_type: {entity_type}");
    }

    // FIX B3: Check existence FIRST, then insert if needed (no silent upsert)
    let existing: Option<(uuid::Uuid, String, String, f64, i32)> = sqlx::query_as(
        "SELECT id, name, entity_type, importance, access_count FROM brain_entities WHERE name = $1"
    )
    .bind(name)
    .fetch_optional(pool)
    .await?;

    if let Some((id, existing_name, existing_type, importance, access_count)) = existing {
        // V10: Return explicit flag that entity already existed
        tracing::info!(entity = %name, "entity already exists (V10 detection)");

        // Hebbian boost on re-access
        boost_entity_importance(pool, id).await?;

        return Ok(serde_json::json!({
            "action": "create",
            "already_existed": true,
            "entity": {
                "id": id.to_string(),
                "name": existing_name,
                "entity_type": existing_type,
                "importance": importance,
                "access_count": access_count + 1
            },
            "message": format!("Entity '{}' already exists (type: {}). Importance boosted.", name, existing_type)
        }));
    }

    // V0.8: Resolve current project (None = global)
    let project_id = crate::project::current_project_id(pool).await?;

    // Actually create
    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_entities (name, entity_type, project_id) VALUES ($1, $2, $3) RETURNING id",
    )
    .bind(name)
    .bind(entity_type)
    .bind(project_id)
    .fetch_one(pool)
    .await
    .context("failed to create entity")?;

    tracing::info!(entity = %name, entity_type = %entity_type, "entity created");

    Ok(serde_json::json!({
        "action": "create",
        "already_existed": false,
        "entity": {
            "id": row.0.to_string(),
            "name": name,
            "entity_type": entity_type,
            "importance": 0.5
        }
    }))
}

/// Update entity name.
async fn update(pool: &PgPool, name: &str, args: &Value) -> Result<Value> {
    let new_name = args.get("new_name").and_then(|v| v.as_str()).unwrap_or("");

    if new_name.is_empty() || new_name.len() > 200 {
        anyhow::bail!("new_name must be 1-200 characters");
    }

    let result =
        sqlx::query("UPDATE brain_entities SET name = $1, updated_at = NOW() WHERE name = $2")
            .bind(new_name)
            .bind(name)
            .execute(pool)
            .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Entity '{name}' not found");
    }

    tracing::info!(from = %name, to = %new_name, "entity renamed");

    Ok(serde_json::json!({
        "action": "update",
        "old_name": name,
        "new_name": new_name
    }))
}

/// Delete entity (cascades to observations and relations).
/// V0.8: respects project scope — refuses to delete an entity owned by another project.
async fn delete(pool: &PgPool, name: &str) -> Result<Value> {
    let project_id = crate::project::current_project_id(pool).await?;

    let result = sqlx::query(
        "DELETE FROM brain_entities
         WHERE name = $1
           AND ($2::uuid IS NULL OR project_id = $2 OR project_id IS NULL)",
    )
    .bind(name)
    .bind(project_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Entity '{name}' not found (within current project scope)");
    }

    tracing::info!(entity = %name, "entity deleted (cascaded)");

    Ok(serde_json::json!({
        "action": "delete",
        "name": name,
        "cascade": true
    }))
}

/// Get entity with observations — FIX B1: fresh data with FOR UPDATE.
/// V0.8: scope lookup to current project (None = global).
async fn get(pool: &PgPool, name: &str) -> Result<Value> {
    let project_id = crate::project::current_project_id(pool).await?;

    // Get entity (with FOR UPDATE to prevent stale reads — FIX B1 partial)
    let entity: Option<(uuid::Uuid, String, String, f64, i32)> = sqlx::query_as(
        "SELECT id, name, entity_type, importance, access_count
         FROM brain_entities
         WHERE name = $1
           AND ($2::uuid IS NULL OR project_id = $2 OR project_id IS NULL)",
    )
    .bind(name)
    .bind(project_id)
    .fetch_optional(pool)
    .await?;

    let (entity_id, entity_name, entity_type, importance, access_count) = match entity {
        Some(e) => e,
        None => anyhow::bail!("Entity '{name}' not found"),
    };

    // Hebbian boost on access
    boost_entity_importance(pool, entity_id).await?;

    // Update access tracking (last_accessed + access_count)
    dual_strength::on_entity_access(pool, entity_id).await?;

    // Get observations (non-superseded)
    let observations: Vec<(uuid::Uuid, String, String, f64, String)> = sqlx::query_as(
        "SELECT id, content, observation_type, importance, source
         FROM brain_observations
         WHERE entity_id = $1 AND observation_type != 'superseded'
         ORDER BY importance DESC, created_at DESC",
    )
    .bind(entity_id)
    .fetch_all(pool)
    .await?;

    let obs_json: Vec<Value> = observations
        .iter()
        .map(|(id, content, obs_type, imp, source)| {
            serde_json::json!({
                "id": id.to_string(),
                "content": content,
                "type": obs_type,
                "importance": imp,
                "source": source
            })
        })
        .collect();

    // Get relations
    let relations: Vec<(String, String, String, f64)> = sqlx::query_as(
        "SELECT e.name, r.relation_type,
                CASE WHEN r.from_entity = $1 THEN 'outgoing' ELSE 'incoming' END,
                r.strength
         FROM brain_relations r
         JOIN brain_entities e ON (
            CASE WHEN r.from_entity = $1 THEN r.to_entity ELSE r.from_entity END = e.id
         )
         WHERE r.from_entity = $1 OR r.to_entity = $1",
    )
    .bind(entity_id)
    .fetch_all(pool)
    .await?;

    let rel_json: Vec<Value> = relations
        .iter()
        .map(|(name, rel_type, direction, strength)| {
            serde_json::json!({
                "entity": name,
                "type": rel_type,
                "direction": direction,
                "strength": strength
            })
        })
        .collect();

    let mut response = serde_json::json!({
        "action": "get",
        "entity": {
            "name": entity_name,
            "entity_type": entity_type,
            "importance": importance,
            "access_count": access_count
        },
        "observations": obs_json,
        "observation_count": obs_json.len(),
        "relations": rel_json
    });

    // Centinela: check on_access triggers
    let triggered = crate::handlers::centinela::check_triggers(pool, name, "on_access")
        .await
        .unwrap_or_default();
    if !triggered.is_empty() {
        response["triggered_reminders"] = serde_json::json!(triggered);
    }

    Ok(response)
}

/// Boost entity importance via Hebbian learning with BCM throttling.
///
/// Delegates to cognitive::hebbian which applies BCM metaplastic throttling
/// (Bienenstock-Cooper-Munro 1982) to prevent winner-take-all dynamics.
async fn boost_entity_importance(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    hebbian::boost_on_access(pool, entity_id).await?;
    hebbian::boost_neighbors(pool, entity_id).await?;
    Ok(())
}
