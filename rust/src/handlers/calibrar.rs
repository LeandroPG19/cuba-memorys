//! Handler: cuba_calibrar — Bayesian confidence calibration.
//!
//! Tracks verify predictions from faro, marks outcomes, and computes
//! P(correct | grounding_level) via Beta distribution (Bayesian update).
//! Closes the feedback loop between `faro verify` and `eco correct`.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "stats" => stats(pool).await,
        "history" => history(pool, &args).await,
        "resolve" => resolve(pool, &args).await,
        _ => anyhow::bail!("Invalid action: {action}. Use stats/history/resolve"),
    }
}

/// Calibration statistics — P(correct) per grounding_level via Beta distribution.
///
/// Beta(alpha, beta): alpha = correct+1 (prior), beta = incorrect+1 (prior).
/// P(correct) = alpha / (alpha + beta).
async fn stats(pool: &PgPool) -> Result<Value> {
    type StatRow = (String, i64, i64, f64);
    let rows: Vec<StatRow> = sqlx::query_as(
        "SELECT grounding_level,
                COUNT(*) FILTER (WHERE outcome = 'correct') + 1 AS alpha,
                COUNT(*) FILTER (WHERE outcome = 'incorrect') + 1 AS beta,
                (COUNT(*) FILTER (WHERE outcome = 'correct') + 1)::float8 /
                (COUNT(*) FILTER (WHERE outcome = 'correct') + COUNT(*) FILTER (WHERE outcome = 'incorrect') + 2)::float8 AS p_correct
         FROM brain_verify_log
         WHERE outcome != 'pending'
         GROUP BY grounding_level
         ORDER BY p_correct DESC"
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let levels: Vec<Value> = rows
        .iter()
        .map(|(level, alpha, beta, p_correct)| {
            serde_json::json!({
                "grounding_level": level,
                "alpha": alpha,
                "beta": beta,
                "p_correct": p_correct,
                "total_resolved": alpha + beta - 2
            })
        })
        .collect();

    // Overall stats
    let total: Option<(i64, i64, i64)> = sqlx::query_as(
        "SELECT COUNT(*),
                COUNT(*) FILTER (WHERE outcome = 'correct'),
                COUNT(*) FILTER (WHERE outcome = 'incorrect')
         FROM brain_verify_log",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    let (total_count, correct, incorrect) = total.unwrap_or((0, 0, 0));

    Ok(serde_json::json!({
        "action": "stats",
        "levels": levels,
        "overall": {
            "total_predictions": total_count,
            "correct": correct,
            "incorrect": incorrect,
            "pending": total_count - correct - incorrect
        }
    }))
}

/// Show recent verify log entries with outcomes.
async fn history(pool: &PgPool, args: &Value) -> Result<Value> {
    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .min(100);

    type LogRow = (
        uuid::Uuid,
        String,
        Option<String>,
        f64,
        String,
        String,
        chrono::DateTime<chrono::Utc>,
    );
    let rows: Vec<LogRow> = sqlx::query_as(
        "SELECT id, claim, entity_name, confidence, grounding_level, outcome, created_at
         FROM brain_verify_log
         ORDER BY created_at DESC
         LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let items: Vec<Value> = rows
        .iter()
        .map(|(id, claim, entity, conf, level, outcome, created)| {
            serde_json::json!({
                "id": id.to_string(),
                "claim": claim,
                "entity_name": entity,
                "confidence": conf,
                "grounding_level": level,
                "outcome": outcome,
                "created_at": created.to_rfc3339()
            })
        })
        .collect();

    let count = items.len();
    Ok(serde_json::json!({
        "action": "history",
        "entries": items,
        "count": count
    }))
}

/// Resolve a verify log entry as correct or incorrect.
async fn resolve(pool: &PgPool, args: &Value) -> Result<Value> {
    let verify_id = args.get("verify_id").and_then(|v| v.as_str()).unwrap_or("");
    let outcome = args.get("outcome").and_then(|v| v.as_str()).unwrap_or("");

    let id: uuid::Uuid = verify_id.parse().context("invalid verify_id UUID")?;

    if outcome != "correct" && outcome != "incorrect" {
        anyhow::bail!("outcome must be 'correct' or 'incorrect'");
    }

    let result = sqlx::query(
        "UPDATE brain_verify_log SET outcome = $2 WHERE id = $1 AND outcome = 'pending'",
    )
    .bind(id)
    .bind(outcome)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Verify log entry not found or already resolved");
    }

    Ok(serde_json::json!({
        "action": "resolve",
        "verify_id": verify_id,
        "outcome": outcome
    }))
}
