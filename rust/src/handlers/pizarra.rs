//! Handler: cuba_pizarra — Working memory buffer (v0.9).
//!
//! Inspired by Baddeley 1992 working memory: a TTL-bounded scratchpad
//! orthogonal to episodic (`brain_episodes`) and semantic (`brain_observations`)
//! memory. Entries auto-expire on read and are bulk-purged by `cuba_zafra`.
//!
//! Use cases:
//! - Inter-step plan state during long-horizon agent tasks.
//! - Tentative observations the agent is not yet ready to commit.
//! - Cross-tool-call reminders inside a single session.
//!
//! Actions:
//! - `write {content, tag?, ttl_seconds?}` — store, default TTL 3600s.
//! - `read {tag?}` — return non-expired entries, optionally filtered by tag.
//! - `clear {tag?}` — delete entries (all in session, or by tag).

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

const DEFAULT_TTL_SECS: i32 = 3600;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    match action {
        "write" => write(pool, &args).await,
        "read" => read(pool, &args).await,
        "clear" => clear(pool, &args).await,
        _ => anyhow::bail!("Invalid action: {action}. Use write/read/clear"),
    }
}

async fn write(pool: &PgPool, args: &Value) -> Result<Value> {
    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("content is required"))?;
    let tag = args.get("tag").and_then(|v| v.as_str());
    let ttl_seconds = args
        .get("ttl_seconds")
        .and_then(|v| v.as_i64())
        .map(|t| t as i32)
        .unwrap_or(DEFAULT_TTL_SECS);
    if ttl_seconds <= 0 {
        anyhow::bail!("ttl_seconds must be > 0");
    }

    let project_id = crate::project::current_project_id(pool).await?;
    let session_id = crate::handlers::jornada::current_session_id(pool).await?;

    // expires_at computed in SQL with make_interval to avoid the GENERATED
    // ALWAYS … STORED limitation (interval coercion is not IMMUTABLE).
    let row: (Uuid,) = sqlx::query_as(
        "INSERT INTO brain_wm (session_id, project_id, content, tag, ttl_seconds, expires_at)
         VALUES ($1, $2, $3, $4, $5, NOW() + make_interval(secs => $5::float8))
         RETURNING id",
    )
    .bind(session_id)
    .bind(project_id)
    .bind(content)
    .bind(tag)
    .bind(ttl_seconds)
    .fetch_one(pool)
    .await
    .context("write working memory")?;

    Ok(serde_json::json!({
        "action": "write",
        "id": row.0.to_string(),
        "session_id": session_id.map(|s| s.to_string()),
        "tag": tag,
        "ttl_seconds": ttl_seconds,
    }))
}

async fn read(pool: &PgPool, args: &Value) -> Result<Value> {
    let tag = args.get("tag").and_then(|v| v.as_str());
    let session_id = crate::handlers::jornada::current_session_id(pool).await?;
    let project_id = crate::project::current_project_id(pool).await?;

    type Row = (
        Uuid,
        String,
        Option<String>,
        chrono::DateTime<chrono::Utc>,
        chrono::DateTime<chrono::Utc>,
    );
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, content, tag, created_at, expires_at FROM brain_wm
         WHERE expires_at > NOW()
           AND ($1::uuid IS NULL OR session_id = $1)
           AND ($2::text IS NULL OR tag = $2)
           AND ($3::uuid IS NULL OR project_id = $3 OR project_id IS NULL)
         ORDER BY created_at DESC
         LIMIT 100",
    )
    .bind(session_id)
    .bind(tag)
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let entries: Vec<Value> = rows
        .into_iter()
        .map(|(id, content, tag, created, expires)| {
            serde_json::json!({
                "id": id.to_string(),
                "content": content,
                "tag": tag,
                "created_at": created.to_rfc3339(),
                "expires_at": expires.to_rfc3339(),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "read",
        "entries": entries.clone(),
        "count": entries.len(),
        "session_id": session_id.map(|s| s.to_string()),
    }))
}

async fn clear(pool: &PgPool, args: &Value) -> Result<Value> {
    let tag = args.get("tag").and_then(|v| v.as_str());
    let session_id = crate::handlers::jornada::current_session_id(pool).await?;

    let result = sqlx::query(
        "DELETE FROM brain_wm
         WHERE ($1::uuid IS NULL OR session_id = $1)
           AND ($2::text IS NULL OR tag = $2)",
    )
    .bind(session_id)
    .bind(tag)
    .execute(pool)
    .await?;

    Ok(serde_json::json!({
        "action": "clear",
        "deleted": result.rows_affected(),
        "tag": tag,
        "session_id": session_id.map(|s| s.to_string()),
    }))
}

/// Garbage-collect expired entries. Called by `cuba_zafra` REM cycle.
pub async fn purge_expired(pool: &PgPool) -> Result<u64> {
    let result = sqlx::query("DELETE FROM brain_wm WHERE expires_at <= NOW()")
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}
