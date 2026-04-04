//! Handler: cuba_remedio — Resolve errors with solution.
//!
//! FIX A-005: UTF-8 safe truncation via zafra::safe_truncate.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

use super::zafra::safe_truncate;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let error_id_str = args.get("error_id").and_then(|v| v.as_str()).unwrap_or("");
    let solution = args.get("solution").and_then(|v| v.as_str()).unwrap_or("");

    if error_id_str.is_empty() || solution.is_empty() {
        anyhow::bail!("error_id and solution are required");
    }

    let error_id: uuid::Uuid = error_id_str.parse().context("invalid error_id UUID")?;

    // Mark error as resolved
    let result = sqlx::query(
        "UPDATE brain_errors SET solution = $2, resolved = true, resolved_at = NOW() WHERE id = $1",
    )
    .bind(error_id)
    .bind(solution)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Error not found: {error_id_str}");
    }

    // Cross-reference: find similar unresolved errors
    let similar: Vec<(uuid::Uuid, String)> = sqlx::query_as(
        "SELECT e2.id, e2.error_message FROM brain_errors e1
         JOIN brain_errors e2 ON similarity(e1.error_message, e2.error_message) > 0.5
         WHERE e1.id = $1 AND e2.resolved = false AND e2.id != $1 LIMIT 5",
    )
    .bind(error_id)
    .fetch_all(pool)
    .await?;

    let cross_refs: Vec<Value> = similar.iter().map(|(id, msg)| {
        serde_json::json!({"id": id.to_string(), "error_message": safe_truncate(msg, 100)})
    }).collect();

    Ok(serde_json::json!({
        "error_id": error_id_str,
        "resolved": true,
        "solution": solution,
        "similar_unresolved": cross_refs,
        "similar_count": cross_refs.len()
    }))
}
