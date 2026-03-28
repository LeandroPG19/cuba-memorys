//! Handler: cuba_jornada — Working session management.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "start" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed");
            let goals = args.get("goals").cloned().unwrap_or(Value::Array(vec![]));
            let row: (uuid::Uuid,) = sqlx::query_as(
                "INSERT INTO brain_sessions (session_name, goals) VALUES ($1, $2) RETURNING id"
            ).bind(name).bind(&goals).fetch_one(pool).await.context("failed to start session")?;
            Ok(serde_json::json!({"action": "started", "session": {"id": row.0.to_string(), "session_name": name, "started_at": chrono::Utc::now().to_rfc3339()}}))
        }
        "end" => {
            let outcome = args.get("outcome").and_then(|v| v.as_str()).unwrap_or("success");
            let summary = args.get("summary").and_then(|v| v.as_str()).unwrap_or("");
            let result = sqlx::query(
                "UPDATE brain_sessions SET ended_at = NOW(), outcome = $1, summary = $2 WHERE id = (SELECT id FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1)"
            ).bind(outcome).bind(summary).execute(pool).await?;
            Ok(serde_json::json!({"action": "ended", "outcome": outcome, "updated": result.rows_affected() > 0}))
        }
        "current" => {
            let session: Option<(uuid::Uuid, Option<String>, Value)> = sqlx::query_as(
                "SELECT id, session_name, goals FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
            ).fetch_optional(pool).await?;
            match session {
                Some((id, name, goals)) => Ok(serde_json::json!({"action": "current", "session": {"id": id.to_string(), "name": name, "goals": goals}})),
                None => Ok(serde_json::json!({"action": "current", "session": null})),
            }
        }
        "list" => {
            type SessionRow = (uuid::Uuid, Option<String>, Option<String>, Option<String>);
            let sessions: Vec<SessionRow> = sqlx::query_as(
                "SELECT id, session_name, outcome, summary FROM brain_sessions ORDER BY started_at DESC LIMIT 20"
            ).fetch_all(pool).await?;
            let list: Vec<Value> = sessions.iter().map(|(id, name, outcome, summary)| {
                serde_json::json!({"id": id.to_string(), "name": name, "outcome": outcome, "summary": summary})
            }).collect();
            Ok(serde_json::json!({"action": "list", "sessions": list, "count": list.len()}))
        }
        _ => anyhow::bail!("Invalid action: {action}"),
    }
}
