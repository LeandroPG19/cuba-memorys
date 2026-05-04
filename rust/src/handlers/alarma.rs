//! Handler: cuba_alarma — Error reporting with pattern detection.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

use super::zafra::safe_truncate;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let error_type = args
        .get("error_type")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");
    let error_message = args
        .get("error_message")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let context = args
        .get("context")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));
    let project = args
        .get("project")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    if error_message.is_empty() {
        anyhow::bail!("error_message is required");
    }

    // V0.8: Resolve current project_id (FK). Coexists with project TEXT column
    // (legacy filter); FK is the new canonical scoping primitive.
    let project_id = crate::project::current_project_id(pool).await?;

    // Insert error (FIX A-003: safe_truncate prevents UTF-8 panic)
    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_errors (error_type, error_message, context, project, project_id) VALUES ($1, $2, $3, $4, $5) RETURNING id"
    )
    .bind(error_type)
    .bind(safe_truncate(error_message, 5000))
    .bind(&context)
    .bind(project)
    .bind(project_id)
    .fetch_one(pool)
    .await
    .context("failed to insert error")?;

    // Pattern detection: check for similar errors (≥3 = warning).
    // Exclude the just-inserted error (id != $3) to avoid self-match.
    let similar_count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM brain_errors WHERE similarity(error_message, $1) > 0.5 AND project = $2 AND id != $3"
    )
    .bind(error_message)
    .bind(project)
    .bind(row.0)
    .fetch_one(pool)
    .await?;

    // Hebbian: boost synapse_weight on similar errors (excluding self)
    if similar_count.0 >= 3 {
        sqlx::query(
            "UPDATE brain_errors SET synapse_weight = LEAST(synapse_weight + 0.1, 5.0)
             WHERE similarity(error_message, $1) > 0.5 AND project = $2 AND id != $3",
        )
        .bind(error_message)
        .bind(project)
        .bind(row.0)
        .execute(pool)
        .await?;
    }

    let mut response = serde_json::json!({
        "error_id": row.0.to_string(),
        "error_type": error_type,
        "project": project,
        "similar_errors": similar_count.0
    });

    if similar_count.0 >= 3 {
        response["pattern_warning"] = serde_json::json!(format!(
            "⚠️ Pattern detected: {} similar errors in project '{}'",
            similar_count.0, project
        ));
    }

    // Centinela: check on_error_match triggers
    let triggered = crate::handlers::centinela::check_triggers(pool, error_type, "on_error_match")
        .await
        .unwrap_or_default();
    if !triggered.is_empty() {
        response["triggered_reminders"] = serde_json::json!(triggered);
    }

    Ok(response)
}
