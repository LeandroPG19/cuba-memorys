//! Handler: cuba_eco — RLHF feedback (Oja's rule).
//!
//! Positive: boost importance (Oja Hebbian).
//! Negative: decrease importance (anti-Hebbian).
//! Correct: update content with versioning.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let entity_name = args.get("entity_name").and_then(|v| v.as_str());
    let observation_id = args.get("observation_id").and_then(|v| v.as_str());

    match action {
        "positive" => positive(pool, entity_name, observation_id).await,
        "negative" => negative(pool, entity_name, observation_id).await,
        "correct" => correct(pool, observation_id, &args).await,
        _ => anyhow::bail!("Invalid action: {action}. Use positive/negative/correct"),
    }
}

/// Positive RLHF: Oja's rule — importance += η * (1 - importance).
async fn positive(pool: &PgPool, entity_name: Option<&str>, observation_id: Option<&str>) -> Result<Value> {
    let mut boosted = 0u32;

    if let Some(obs_id_str) = observation_id {
        let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
        let result = sqlx::query(
            "UPDATE brain_observations SET
                importance = LEAST(importance + 0.05 * (1.0 - importance), 1.0),
                access_count = access_count + 1,
                last_accessed = NOW()
             WHERE id = $1"
        )
        .bind(obs_id)
        .execute(pool)
        .await?;
        boosted += result.rows_affected() as u32;
    }

    if let Some(name) = entity_name {
        let result = sqlx::query(
            "UPDATE brain_entities SET
                importance = LEAST(importance + 0.05 * (1.0 - importance), 1.0),
                access_count = access_count + 1,
                updated_at = NOW()
             WHERE name = $1"
        )
        .bind(name)
        .execute(pool)
        .await?;
        boosted += result.rows_affected() as u32;
    }

    Ok(serde_json::json!({
        "action": "positive",
        "boosted_count": boosted,
        "rule": "oja_positive"
    }))
}

/// Negative RLHF: anti-Oja — importance -= η * importance.
async fn negative(pool: &PgPool, entity_name: Option<&str>, observation_id: Option<&str>) -> Result<Value> {
    let mut decreased = 0u32;

    if let Some(obs_id_str) = observation_id {
        let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
        let result = sqlx::query(
            "UPDATE brain_observations SET
                importance = GREATEST(importance - 0.05 * importance, 0.0),
                last_accessed = NOW()
             WHERE id = $1"
        )
        .bind(obs_id)
        .execute(pool)
        .await?;
        decreased += result.rows_affected() as u32;
    }

    if let Some(name) = entity_name {
        let result = sqlx::query(
            "UPDATE brain_entities SET
                importance = GREATEST(importance - 0.05 * importance, 0.0),
                updated_at = NOW()
             WHERE name = $1"
        )
        .bind(name)
        .execute(pool)
        .await?;
        decreased += result.rows_affected() as u32;
    }

    Ok(serde_json::json!({
        "action": "negative",
        "decreased_count": decreased,
        "rule": "oja_negative"
    }))
}

/// Content correction with version history.
async fn correct(pool: &PgPool, observation_id: Option<&str>, args: &Value) -> Result<Value> {
    let obs_id_str = observation_id.context("observation_id required for correct")?;
    let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
    let correction = args.get("correction").and_then(|v| v.as_str())
        .context("correction text is required")?;

    // Archive old content in previous_versions, then update
    let result = sqlx::query(
        "UPDATE brain_observations SET
            previous_versions = previous_versions || jsonb_build_array(
                jsonb_build_object('content', content, 'version', version, 'corrected_at', NOW()::text)
            ),
            content = $2,
            version = version + 1,
            last_accessed = NOW()
         WHERE id = $1"
    )
    .bind(obs_id)
    .bind(correction)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Observation not found");
    }

    Ok(serde_json::json!({
        "action": "correct",
        "observation_id": obs_id_str,
        "new_content": correction,
        "versioned": true
    }))
}
