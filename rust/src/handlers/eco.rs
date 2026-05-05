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

/// Positive RLHF: Oja's rule with V0.9 Robbins-Monro adaptive learning rate.
///
/// V0.9: η_t = η_0 / sqrt(1 + access_count/100) — Robbins-Monro 1951
/// stochastic approximation. Convergence O(1/√t) instead of constant rate.
/// As an observation accumulates feedback, the marginal effect of each
/// additional Oja step shrinks → bounded importance volatility, no
/// single noisy signal can radically swing a well-established memory.
///
/// At access_count=0: η = 0.05 (original behavior).
/// At access_count=100: η ≈ 0.0354.
/// At access_count=1000: η ≈ 0.0157.
async fn positive(
    pool: &PgPool,
    entity_name: Option<&str>,
    observation_id: Option<&str>,
) -> Result<Value> {
    let mut boosted = 0u32;

    if let Some(obs_id_str) = observation_id {
        let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
        let result = sqlx::query(
            "UPDATE brain_observations SET
                importance = LEAST(
                    importance + (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * (1.0 - importance),
                    1.0
                ),
                access_count = access_count + 1,
                last_accessed = NOW()
             WHERE id = $1",
        )
        .bind(obs_id)
        .execute(pool)
        .await?;
        boosted += result.rows_affected() as u32;
    }

    if let Some(name) = entity_name {
        let result = sqlx::query(
            "UPDATE brain_entities SET
                importance = LEAST(
                    importance + (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * (1.0 - importance),
                    1.0
                ),
                access_count = access_count + 1,
                updated_at = NOW()
             WHERE name = $1",
        )
        .bind(name)
        .execute(pool)
        .await?;
        boosted += result.rows_affected() as u32;
    }

    Ok(serde_json::json!({
        "action": "positive",
        "boosted_count": boosted,
        "rule": "oja_positive_robbins_monro"
    }))
}

/// Negative RLHF: anti-Oja with V0.9 Robbins-Monro adaptive learning rate.
async fn negative(
    pool: &PgPool,
    entity_name: Option<&str>,
    observation_id: Option<&str>,
) -> Result<Value> {
    let mut decreased = 0u32;

    if let Some(obs_id_str) = observation_id {
        let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
        let result = sqlx::query(
            "UPDATE brain_observations SET
                importance = GREATEST(
                    importance - (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * importance,
                    0.0
                ),
                last_accessed = NOW()
             WHERE id = $1",
        )
        .bind(obs_id)
        .execute(pool)
        .await?;
        decreased += result.rows_affected() as u32;
    }

    if let Some(name) = entity_name {
        let result = sqlx::query(
            "UPDATE brain_entities SET
                importance = GREATEST(
                    importance - (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * importance,
                    0.0
                ),
                updated_at = NOW()
             WHERE name = $1",
        )
        .bind(name)
        .execute(pool)
        .await?;
        decreased += result.rows_affected() as u32;
    }

    Ok(serde_json::json!({
        "action": "negative",
        "decreased_count": decreased,
        "rule": "oja_negative_robbins_monro"
    }))
}

/// Content correction with version history.
async fn correct(pool: &PgPool, observation_id: Option<&str>, args: &Value) -> Result<Value> {
    let obs_id_str = observation_id.context("observation_id required for correct")?;
    let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
    let correction = args
        .get("correction")
        .and_then(|v| v.as_str())
        .context("correction text is required")?;

    // Archive old content in previous_versions, then update
    let result = sqlx::query(
        "UPDATE brain_observations SET
            previous_versions = previous_versions || jsonb_build_array(
                jsonb_build_object('content', content, 'version', version, 'corrected_at', NOW()::text)
            ),
            content = $2,
            version = version + 1,
            last_accessed = NOW(),
            updated_at = NOW()
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

/// V0.8: Internal reflection helper used by `cuba_pre_compact` to build a
/// session-scoped summary before context compaction. Returns dense markdown
/// (~500 tokens) summarizing observations created in the given session.
///
/// Pure aggregation — does not modify state. Caller is responsible for
/// persisting it (typically into `brain_compaction_snapshots`).
pub async fn reflect(pool: &PgPool, session_id: uuid::Uuid) -> Result<String> {
    type Row = (String, i64);
    let by_type: Vec<Row> = sqlx::query_as(
        "SELECT observation_type, COUNT(*) FROM brain_observations
         WHERE session_id = $1
         GROUP BY observation_type
         ORDER BY COUNT(*) DESC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let recent: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT e.name, o.observation_type, o.content
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE o.session_id = $1 AND o.observation_type != 'superseded'
         ORDER BY o.created_at DESC
         LIMIT 12",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut md = String::with_capacity(2048);
    md.push_str("# Session reflection\n\n");
    if !by_type.is_empty() {
        md.push_str("## Counts by type\n");
        for (t, n) in &by_type {
            md.push_str(&format!("- {t}: {n}\n"));
        }
        md.push('\n');
    }
    if !recent.is_empty() {
        md.push_str("## Recent observations (newest first)\n");
        for (entity, ty, content) in &recent {
            let snippet = crate::handlers::zafra::safe_truncate(content, 160);
            md.push_str(&format!("- [{ty}] {entity}: {snippet}\n"));
        }
    }
    Ok(md)
}
