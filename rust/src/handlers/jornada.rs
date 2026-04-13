//! Handler: cuba_jornada — Working session management.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "start" => {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed");
            let goals = args.get("goals").cloned().unwrap_or(Value::Array(vec![]));
            let row: (uuid::Uuid,) = sqlx::query_as(
                "INSERT INTO brain_sessions (session_name, goals) VALUES ($1, $2) RETURNING id",
            )
            .bind(name)
            .bind(&goals)
            .fetch_one(pool)
            .await
            .context("failed to start session")?;

            let mut response = serde_json::json!({"action": "started", "session": {"id": row.0.to_string(), "session_name": name, "started_at": chrono::Utc::now().to_rfc3339()}});

            // V0.6: Fetch previous session summary for context continuity
            let prev_session: Option<(Option<String>, Option<String>, Option<String>)> =
                sqlx::query_as(
                    "SELECT session_name, summary, outcome FROM brain_sessions
                     WHERE ended_at IS NOT NULL ORDER BY ended_at DESC LIMIT 1",
                )
                .fetch_optional(pool)
                .await
                .ok()
                .flatten();

            if let Some((prev_name, prev_summary, prev_outcome)) = prev_session {
                response["previous_session"] = serde_json::json!({
                    "name": prev_name,
                    "summary": prev_summary,
                    "outcome": prev_outcome
                });
            }

            // Centinela: check on_session_start triggers
            let triggered =
                crate::handlers::centinela::check_triggers(pool, name, "on_session_start")
                    .await
                    .unwrap_or_default();
            if !triggered.is_empty() {
                response["triggered_reminders"] = serde_json::json!(triggered);
            }

            Ok(response)
        }
        "end" => {
            let outcome = args
                .get("outcome")
                .and_then(|v| v.as_str())
                .unwrap_or("success");
            let summary = args.get("summary").and_then(|v| v.as_str()).unwrap_or("");

            // Atomic UPDATE RETURNING — eliminates TOCTOU race between SELECT and UPDATE.
            // A concurrent `end` call on the same session cannot both succeed.
            let active_session: Option<(uuid::Uuid,)> = sqlx::query_as(
                "UPDATE brain_sessions SET ended_at = NOW(), outcome = $1, summary = $2
                 WHERE id = (SELECT id FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1 FOR UPDATE SKIP LOCKED)
                 RETURNING id"
            ).bind(outcome).bind(summary).fetch_optional(pool).await?;

            let updated = active_session.is_some();

            let mut response = serde_json::json!({
                "action": "ended",
                "outcome": outcome,
                "updated": updated
            });

            // V0.6: Session diff — summarize what was created during this session
            if let Some((session_id,)) = active_session {
                let session_diff: Vec<(String, i64)> = sqlx::query_as(
                    "SELECT observation_type, COUNT(*) FROM brain_observations
                     WHERE session_id = $1 GROUP BY observation_type",
                )
                .bind(session_id)
                .fetch_all(pool)
                .await
                .unwrap_or_default();

                let episode_count: i64 =
                    sqlx::query_scalar("SELECT COUNT(*) FROM brain_episodes WHERE session_id = $1")
                        .bind(session_id)
                        .fetch_one(pool)
                        .await
                        .unwrap_or(0);

                let mut diff = serde_json::Map::new();
                for (obs_type, count) in &session_diff {
                    diff.insert(obs_type.clone(), serde_json::json!(count));
                }
                diff.insert("episodes".to_string(), serde_json::json!(episode_count));

                response["session_diff"] = Value::Object(diff);
            }

            Ok(response)
        }
        "current" => {
            let session: Option<(uuid::Uuid, Option<String>, Value)> = sqlx::query_as(
                "SELECT id, session_name, goals FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
            ).fetch_optional(pool).await?;
            match session {
                Some((id, name, goals)) => Ok(
                    serde_json::json!({"action": "current", "session": {"id": id.to_string(), "name": name, "goals": goals}}),
                ),
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
