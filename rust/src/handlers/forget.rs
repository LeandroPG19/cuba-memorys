//! Handler: cuba_forget — GDPR Right to Erasure (ARCO).
//!
//! Performs cascading hard-delete across ALL tables for a given entity,
//! including brain_errors and brain_sessions which are NOT covered by
//! FK ON DELETE CASCADE (they reference entities by name, not FK).
//!
//! POST-AUDIT FIX: Gemini audit identified COMP-001 (GDPR non-compliance)
//! because cuba_alma(delete) only cascades via FK to observations + relations,
//! leaving orphaned references in errors/sessions.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let entity_name = args
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required");
    }

    let confirm = args
        .get("confirm")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if !confirm {
        anyhow::bail!(
            "GDPR erasure is IRREVERSIBLE. Set confirm=true to proceed. \
             This will delete entity '{}' and ALL references across \
             observations, relations, errors, and sessions.",
            entity_name
        );
    }

    // Use a transaction for atomicity
    let mut tx = pool.begin().await.context("failed to begin transaction")?;

    // 1. Delete errors that mention this entity (by name in context or message)
    let errors_deleted: (i64,) = sqlx::query_as(
        "WITH deleted AS (
            DELETE FROM brain_errors
            WHERE error_message ILIKE '%' || $1 || '%'
               OR context::text ILIKE '%' || $1 || '%'
            RETURNING 1
        ) SELECT COUNT(*) FROM deleted",
    )
    .bind(entity_name)
    .fetch_one(&mut *tx)
    .await
    .context("failed to delete errors")?;

    // 2. Delete sessions that mention this entity in goals or summary
    let sessions_deleted: (i64,) = sqlx::query_as(
        "WITH deleted AS (
            DELETE FROM brain_sessions
            WHERE goals::text ILIKE '%' || $1 || '%'
               OR session_name ILIKE '%' || $1 || '%'
               OR summary ILIKE '%' || $1 || '%'
            RETURNING 1
        ) SELECT COUNT(*) FROM deleted",
    )
    .bind(entity_name)
    .fetch_one(&mut *tx)
    .await
    .context("failed to delete sessions")?;

    // 3. Delete the entity itself (FK CASCADE handles observations + relations)
    let entity_deleted = sqlx::query("DELETE FROM brain_entities WHERE name = $1")
        .bind(entity_name)
        .execute(&mut *tx)
        .await
        .context("failed to delete entity")?;

    tx.commit().await.context("failed to commit transaction")?;

    let entity_found = entity_deleted.rows_affected() > 0;

    tracing::info!(
        entity = %entity_name,
        entity_deleted = entity_found,
        errors_purged = errors_deleted.0,
        sessions_purged = sessions_deleted.0,
        "GDPR erasure completed"
    );

    Ok(serde_json::json!({
        "action": "forget",
        "entity_name": entity_name,
        "entity_deleted": entity_found,
        "cascaded": {
            "observations": "via FK CASCADE",
            "relations": "via FK CASCADE",
            "errors_purged": errors_deleted.0,
            "sessions_purged": sessions_deleted.0
        },
        "gdpr_compliance": "Right to Erasure (ARCO) satisfied"
    }))
}
