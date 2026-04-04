//! Handler: cuba_centinela — Prospective memory triggers.
//!
//! Set triggers that fire when entities are accessed, sessions start,
//! or errors match a pattern. "Remember to remind me about X when Y happens."

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "create" => create(pool, &args).await,
        "list" => list(pool).await,
        "delete" => delete(pool, &args).await,
        "check" => {
            let entity_name = args
                .get("entity_pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let condition = args
                .get("condition_type")
                .and_then(|v| v.as_str())
                .unwrap_or("on_access");
            let triggered = check_triggers(pool, entity_name, condition).await?;
            Ok(serde_json::json!({
                "action": "check",
                "triggered": triggered,
                "count": triggered.len()
            }))
        }
        _ => anyhow::bail!("Invalid action: {action}. Use create/list/delete/check"),
    }
}

/// Create a new prospective memory trigger.
async fn create(pool: &PgPool, args: &Value) -> Result<Value> {
    let entity_pattern = args
        .get("entity_pattern")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let condition_type = args
        .get("condition_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let message = args.get("message").and_then(|v| v.as_str()).unwrap_or("");

    if entity_pattern.is_empty() {
        anyhow::bail!("entity_pattern is required");
    }
    if message.is_empty() {
        anyhow::bail!("message is required");
    }

    let valid_conditions = ["on_access", "on_session_start", "on_error_match"];
    if !valid_conditions.contains(&condition_type) {
        anyhow::bail!(
            "Invalid condition_type: {condition_type}. Use on_access/on_session_start/on_error_match"
        );
    }

    let max_fires = args
        .get("max_fires")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        .clamp(-1, 10_000) as i32;
    let expires_at = args
        .get("expires_at")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&chrono::Utc));

    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_triggers (entity_pattern, condition_type, message, max_fires, expires_at)
         VALUES ($1, $2, $3, $4, $5) RETURNING id"
    )
    .bind(entity_pattern)
    .bind(condition_type)
    .bind(message)
    .bind(max_fires)
    .bind(expires_at)
    .fetch_one(pool)
    .await
    .context("failed to create trigger")?;

    tracing::info!(
        trigger_id = %row.0,
        pattern = %entity_pattern,
        condition = %condition_type,
        "trigger created"
    );

    Ok(serde_json::json!({
        "action": "create",
        "trigger_id": row.0.to_string(),
        "entity_pattern": entity_pattern,
        "condition_type": condition_type,
        "message": message,
        "max_fires": max_fires
    }))
}

/// List all active triggers.
async fn list(pool: &PgPool) -> Result<Value> {
    type TriggerRow = (
        uuid::Uuid,
        String,
        String,
        String,
        bool,
        i32,
        i32,
        chrono::DateTime<chrono::Utc>,
        Option<chrono::DateTime<chrono::Utc>>,
    );
    let triggers: Vec<TriggerRow> = sqlx::query_as(
        "SELECT id, entity_pattern, condition_type, message, active,
                fire_count, max_fires, created_at, expires_at
         FROM brain_triggers
         WHERE active = TRUE AND (expires_at IS NULL OR expires_at > NOW())
         ORDER BY created_at DESC
         LIMIT 50",
    )
    .fetch_all(pool)
    .await?;

    let items: Vec<Value> = triggers
        .iter()
        .map(
            |(id, pattern, cond, msg, active, fires, max_f, created, expires)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "entity_pattern": pattern,
                    "condition_type": cond,
                    "message": msg,
                    "active": active,
                    "fire_count": fires,
                    "max_fires": max_f,
                    "created_at": created.to_rfc3339(),
                    "expires_at": expires.map(|e| e.to_rfc3339())
                })
            },
        )
        .collect();

    let count = items.len();
    Ok(serde_json::json!({
        "action": "list",
        "triggers": items,
        "count": count
    }))
}

/// Delete a trigger by ID.
async fn delete(pool: &PgPool, args: &Value) -> Result<Value> {
    let trigger_id = args
        .get("trigger_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let id: uuid::Uuid = trigger_id.parse().context("invalid trigger_id UUID")?;

    let result = sqlx::query("DELETE FROM brain_triggers WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Trigger not found");
    }

    Ok(serde_json::json!({
        "action": "delete",
        "trigger_id": trigger_id
    }))
}

/// Check and fire matching triggers for a given entity + condition.
///
/// Public so other handlers (alma, jornada, alarma) can call it.
/// Returns the list of fired trigger messages.
pub async fn check_triggers(
    pool: &PgPool,
    entity_name: &str,
    condition: &str,
) -> Result<Vec<Value>> {
    if entity_name.is_empty() {
        return Ok(vec![]);
    }

    type TrigRow = (uuid::Uuid, String, String, i32, i32);
    let triggers: Vec<TrigRow> = sqlx::query_as(
        "SELECT id, entity_pattern, message, fire_count, max_fires
         FROM brain_triggers
         WHERE active = TRUE
           AND (expires_at IS NULL OR expires_at > NOW())
           AND condition_type = $1
           AND (entity_pattern = $2 OR similarity(entity_pattern, $2) > 0.5)
         LIMIT 10",
    )
    .bind(condition)
    .bind(entity_name)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut fired: Vec<Value> = Vec::new();

    for (id, pattern, message, fire_count, max_fires) in &triggers {
        // Increment fire_count; deactivate if at max
        let new_count = fire_count + 1;
        let deactivate = *max_fires > 0 && new_count >= *max_fires;

        if let Err(e) = sqlx::query(
            "UPDATE brain_triggers SET fire_count = $2, active = $3 WHERE id = $1",
        )
        .bind(id)
        .bind(new_count)
        .bind(!deactivate)
        .execute(pool)
        .await
        {
            tracing::warn!(trigger_id = %id, error = %e, "failed to update trigger fire_count");
        }

        fired.push(serde_json::json!({
            "trigger_id": id.to_string(),
            "entity_pattern": pattern,
            "message": message,
            "fire_count": new_count,
            "deactivated": deactivate
        }));
    }

    Ok(fired)
}
