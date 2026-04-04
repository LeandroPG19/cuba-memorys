//! Handler: cuba_cronica — Observations CRUD + Episodic memory.
//!
//! FIX B1: Always fresh entity_id fetch before operations.
//! FIX B5: information_density uses unique word count, not total.
//! P3: Batch dedup in batch_add (1 query per batch, not N).
//! V5: Prediction Error Gating — classify by similarity.
//! V11: Compute and store embedding on insert — unblocks vector search in faro.
//!      Previously, embeddings were never written so WHERE embedding IS NOT NULL
//!      always returned 0 rows (vector search was dead code).
//! V4: Episodic memory (Tulving 1972) — episode_add/episode_list actions.
//!     Separate from semantic facts: specific events with actors, artifacts, time bounds.

use crate::cognitive::{density, prediction_error};
use crate::constants::{DEDUP_THRESHOLD, VALID_OBSERVATION_TYPES, VALID_SOURCES};
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let entity_name = args.get("entity_name").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "add" => add(pool, entity_name, &args).await,
        "delete" => delete_obs(pool, &args).await,
        "list" => list(pool, entity_name).await,
        "batch_add" => batch_add(pool, &args).await,
        "episode_add" => episode_add(pool, entity_name, &args).await,
        "episode_list" => episode_list(pool, entity_name).await,
        _ => anyhow::bail!(
            "Invalid action: {action}. Use add/delete/list/batch_add/episode_add/episode_list"
        ),
    }
}

/// Add a single observation to an entity (auto-creates entity if needed).
async fn add(pool: &PgPool, entity_name: &str, args: &Value) -> Result<Value> {
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required");
    }

    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if content.is_empty() || content.len() > 10_000 {
        anyhow::bail!("content must be 1-10000 characters");
    }

    let obs_type = args.get("observation_type").and_then(|v| v.as_str()).unwrap_or("fact");
    if !VALID_OBSERVATION_TYPES.contains(&obs_type) {
        anyhow::bail!("Invalid observation_type: {obs_type}");
    }

    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("agent");
    if !VALID_SOURCES.contains(&source) {
        anyhow::bail!("Invalid source: {source}");
    }

    // Auto-create entity if needed (FIX B1: always get fresh entity_id)
    let entity_id = ensure_entity(pool, entity_name).await?;

    // Dedup check (FIX B5: information_density with unique words)
    let density = information_density(content);
    let dedup = check_dedup(pool, entity_id, content).await?;

    match dedup {
        DedupResult::Duplicate(existing_preview) => {
            return Ok(serde_json::json!({
                "action": "add",
                "deduplicated": true,
                "existing_content": existing_preview,
                "message": "Near-duplicate detected. Observation NOT added."
            }));
        }
        DedupResult::Reinforce(obs_id) => {
            // V5: Very similar → reinforce existing (boost importance)
            sqlx::query(
                "UPDATE brain_observations SET
                    importance = LEAST(importance + 0.05, 1.0),
                    access_count = access_count + 1,
                    last_accessed = NOW()
                 WHERE id = $1"
            )
            .bind(obs_id)
            .execute(pool)
            .await?;

            return Ok(serde_json::json!({
                "action": "add",
                "prediction_error": "reinforce",
                "reinforced_id": obs_id.to_string(),
                "message": "Very similar observation exists. Reinforced existing instead."
            }));
        }
        DedupResult::Unique => {}
    }

    // Insert observation
    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_observations (entity_id, content, observation_type, source, importance)
         VALUES ($1, $2, $3, $4, $5) RETURNING id"
    )
    .bind(entity_id)
    .bind(content)
    .bind(obs_type)
    .bind(source)
    .bind(density.min(1.0))
    .fetch_one(pool)
    .await
    .context("failed to insert observation")?;

    tracing::info!(
        entity = %entity_name,
        obs_type = %obs_type,
        density = %density,
        "observation added"
    );

    // V11: Store embedding for vector search (non-blocking fire-and-forget).
    // Clones pool (cheap Arc clone) to move into spawned task.
    // embed() uses spawn_blocking internally (FIX B2) — safe to call from async context.
    let embed_pool = pool.clone();
    let obs_id_for_embed = row.0;
    let content_for_embed = content.to_string();
    tokio::spawn(async move {
        match crate::embeddings::onnx::embed_passage(&content_for_embed).await {
            Ok(emb) if !emb.iter().all(|&v| v == 0.0) => {
                let result = sqlx::query(
                    "UPDATE brain_observations SET embedding = $1::vector WHERE id = $2"
                )
                .bind(pgvector::Vector::from(emb))
                .bind(obs_id_for_embed)
                .execute(&embed_pool)
                .await;
                if let Err(e) = result {
                    tracing::warn!(obs_id = %obs_id_for_embed, error = %e, "failed to store embedding");
                }
            }
            _ => {} // no ONNX model loaded — skip silently
        }
    });

    // Overload warning — alert if entity has too many observations
    let obs_count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM brain_observations WHERE entity_id = $1 AND observation_type != 'superseded'"
    )
    .bind(entity_id)
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    let overload_warning = if obs_count.0 > 50 {
        Some(format!(
            "Entity '{}' has {} observations. Consider running cuba_zafra(action='summarize', entity_name='{}') to consolidate.",
            entity_name, obs_count.0, entity_name
        ))
    } else {
        None
    };

    Ok(serde_json::json!({
        "action": "add",
        "id": row.0.to_string(),
        "entity_name": entity_name,
        "observation_type": obs_type,
        "information_density": density,
        "overload_warning": overload_warning
    }))
}

/// Delete an observation by ID.
async fn delete_obs(pool: &PgPool, args: &Value) -> Result<Value> {
    let obs_id = args.get("observation_id").or_else(|| args.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let id: uuid::Uuid = obs_id.parse().context("invalid observation_id UUID")?;

    let result = sqlx::query("DELETE FROM brain_observations WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Observation not found");
    }

    Ok(serde_json::json!({
        "action": "delete",
        "deleted_id": obs_id
    }))
}

/// List observations for an entity.
async fn list(pool: &PgPool, entity_name: &str) -> Result<Value> {
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required");
    }

    let entity_id = get_entity_id(pool, entity_name).await?;

    let observations: Vec<(uuid::Uuid, String, String, f64, String, i32)> = sqlx::query_as(
        "SELECT id, content, observation_type, importance, source, access_count
         FROM brain_observations
         WHERE entity_id = $1 AND observation_type != 'superseded'
         ORDER BY importance DESC, created_at DESC"
    )
    .bind(entity_id)
    .fetch_all(pool)
    .await?;

    let obs_json: Vec<Value> = observations
        .iter()
        .map(|(id, content, obs_type, imp, source, ac)| {
            serde_json::json!({
                "id": id.to_string(),
                "content": content,
                "type": obs_type,
                "importance": imp,
                "source": source,
                "access_count": ac
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "list",
        "entity_name": entity_name,
        "observations": obs_json,
        "count": obs_json.len()
    }))
}

/// Batch add observations (P3: 1 dedup query per batch).
///
/// Uses explicit transaction — all-or-nothing atomicity.
async fn batch_add(pool: &PgPool, args: &Value) -> Result<Value> {
    let observations = args.get("observations")
        .and_then(|v| v.as_array())
        .context("'observations' array is required for batch_add")?;

    if observations.len() > 100 {
        anyhow::bail!("batch_add limit is 100 observations per call (got {})", observations.len());
    }

    // Pre-process: resolve entities and dedup checks outside transaction
    // (reads are safe to repeat on retry)
    struct PendingObs {
        entity_id: uuid::Uuid,
        content: String,
        obs_type: String,
        source: String,
        density: f64,
    }
    enum BatchAction {
        Insert(PendingObs),
        Reinforce(uuid::Uuid),
    }

    let mut actions: Vec<BatchAction> = Vec::new();
    let mut deduplicated = 0u32;

    for obs in observations {
        let entity_name = obs.get("entity_name").and_then(|v| v.as_str()).unwrap_or("");
        let content = obs.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let obs_type = obs.get("observation_type").and_then(|v| v.as_str()).unwrap_or("fact");
        let source = obs.get("source").and_then(|v| v.as_str()).unwrap_or("agent");

        if entity_name.is_empty() || content.is_empty() {
            continue;
        }

        let entity_id = ensure_entity(pool, entity_name).await?;
        let dedup = check_dedup(pool, entity_id, content).await?;

        match dedup {
            DedupResult::Duplicate(_) => {
                deduplicated += 1;
            }
            DedupResult::Reinforce(obs_id) => {
                actions.push(BatchAction::Reinforce(obs_id));
            }
            DedupResult::Unique => {
                let density = information_density(content);
                actions.push(BatchAction::Insert(PendingObs {
                    entity_id,
                    content: content.to_string(),
                    obs_type: obs_type.to_string(),
                    source: source.to_string(),
                    density: density.min(1.0),
                }));
            }
        }
    }

    // Execute all writes atomically in a single transaction
    let mut tx = pool.begin().await.context("failed to begin transaction")?;
    let mut added = 0u32;
    let mut reinforced = 0u32;
    // Collect (obs_id, content) for embedding storage after commit (V11)
    let mut inserted_for_embed: Vec<(uuid::Uuid, String)> = Vec::new();

    for action in &actions {
        match action {
            BatchAction::Insert(pending) => {
                let row: (uuid::Uuid,) = sqlx::query_as(
                    "INSERT INTO brain_observations (entity_id, content, observation_type, source, importance)
                     VALUES ($1, $2, $3, $4, $5) RETURNING id"
                )
                .bind(pending.entity_id)
                .bind(&pending.content)
                .bind(&pending.obs_type)
                .bind(&pending.source)
                .bind(pending.density)
                .fetch_one(&mut *tx)
                .await?;
                inserted_for_embed.push((row.0, pending.content.clone()));
                added += 1;
            }
            BatchAction::Reinforce(obs_id) => {
                sqlx::query(
                    "UPDATE brain_observations SET
                        importance = LEAST(importance + 0.05, 1.0),
                        access_count = access_count + 1,
                        last_accessed = NOW()
                     WHERE id = $1"
                )
                .bind(obs_id)
                .execute(&mut *tx)
                .await?;
                reinforced += 1;
            }
        }
    }

    tx.commit().await.context("failed to commit batch_add transaction")?;

    // V11: Store embeddings after commit (non-blocking, best-effort)
    if !inserted_for_embed.is_empty() {
        let embed_pool = pool.clone();
        tokio::spawn(async move {
            for (obs_id, content) in inserted_for_embed {
                if let Ok(emb) = crate::embeddings::onnx::embed_passage(&content).await
                    && !emb.iter().all(|&v| v == 0.0)
                {
                    let _ = sqlx::query(
                        "UPDATE brain_observations SET embedding = $1::vector WHERE id = $2"
                    )
                    .bind(pgvector::Vector::from(emb))
                    .bind(obs_id)
                    .execute(&embed_pool)
                    .await;
                }
            }
        });
    }

    Ok(serde_json::json!({
        "action": "batch_add",
        "added": added,
        "deduplicated": deduplicated,
        "reinforced": reinforced,
        "total_processed": added + deduplicated + reinforced
    }))
}

// ── Helpers ─────────────────────────────────────────────────────

/// Dedup result with Prediction Error Gating (V5).
enum DedupResult {
    Unique,
    Duplicate(String),
    Reinforce(uuid::Uuid),
}

/// Check for near-duplicates using pg_trgm similarity + adaptive PE gating (V5.1).
async fn check_dedup(pool: &PgPool, entity_id: uuid::Uuid, content: &str) -> Result<DedupResult> {
    let dupes: Vec<(uuid::Uuid, String, f64)> = sqlx::query_as(
        "SELECT id, content, similarity(content, $2)::float8 AS sim
         FROM brain_observations
         WHERE entity_id = $1 AND similarity(content, $2) > 0.3
         ORDER BY sim DESC LIMIT 5"
    )
    .bind(entity_id)
    .bind(content)
    .fetch_all(pool)
    .await?;

    if dupes.is_empty() {
        return Ok(DedupResult::Unique);
    }

    // V5.1: Fetch recent similarities for adaptive thresholds
    let recent_sims: Vec<(f64,)> = sqlx::query_as(
        "SELECT similarity(content, $2)::float8
         FROM brain_observations
         WHERE entity_id = $1 AND observation_type != 'superseded'
         ORDER BY created_at DESC LIMIT 20"
    )
    .bind(entity_id)
    .bind(content)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let recent: Vec<f64> = recent_sims.iter().map(|(s,)| *s).collect();

    for (obs_id, existing_content, sim) in &dupes {
        let sim = *sim;
        if sim > DEDUP_THRESHOLD {
            return Ok(DedupResult::Duplicate(super::zafra::safe_truncate(existing_content, 80).to_string()));
        }
        // V5.1: Adaptive PE gating (Friston, Nature 2023)
        let action = prediction_error::adaptive_gate(sim, &recent);
        match action {
            prediction_error::GatingAction::Reinforce => {
                return Ok(DedupResult::Reinforce(*obs_id));
            }
            prediction_error::GatingAction::Update | prediction_error::GatingAction::Create => {
                // Allow as new observation
            }
        }
    }

    Ok(DedupResult::Unique)
}

/// Ensure entity exists, creating if necessary. Returns entity_id.
async fn ensure_entity(pool: &PgPool, name: &str) -> Result<uuid::Uuid> {
    // Try to find existing
    let existing: Option<(uuid::Uuid,)> = sqlx::query_as(
        "SELECT id FROM brain_entities WHERE name = $1"
    )
    .bind(name)
    .fetch_optional(pool)
    .await?;

    if let Some((id,)) = existing {
        return Ok(id);
    }

    // Create new entity
    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_entities (name, entity_type)
         VALUES ($1, 'concept')
         ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
         RETURNING id"
    )
    .bind(name)
    .fetch_one(pool)
    .await?;

    tracing::info!(entity = %name, "auto-created entity for observation");
    Ok(row.0)
}

/// Get entity_id by name (strict — errors if not found).
async fn get_entity_id(pool: &PgPool, name: &str) -> Result<uuid::Uuid> {
    let row: Option<(uuid::Uuid,)> = sqlx::query_as(
        "SELECT id FROM brain_entities WHERE name = $1"
    )
    .bind(name)
    .fetch_optional(pool)
    .await?;

    row.map(|(id,)| id)
        .context(format!("Entity '{name}' not found"))
}

/// Add an episodic memory — a specific temporal event linked to an entity.
///
/// Episodes differ from observations (semantic facts):
/// - They represent specific events that happened at a point in time
/// - They have actors (who was involved) and artifacts (what was affected)
/// - They decay faster (power-law halflife ~3 days vs 30 days for facts)
async fn episode_add(pool: &PgPool, entity_name: &str, args: &Value) -> Result<Value> {
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required for episode_add");
    }

    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if content.is_empty() {
        anyhow::bail!("content is required for episode_add");
    }

    let actors: Vec<String> = args
        .get("actors")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let artifacts: Vec<String> = args
        .get("artifacts")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Auto-create entity if needed (same pattern as add)
    let entity_id = ensure_entity(pool, entity_name).await?;
    let density = information_density(content);

    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_episodes (entity_id, content, actors, artifacts, importance)
         VALUES ($1, $2, $3, $4, $5) RETURNING id",
    )
    .bind(entity_id)
    .bind(content)
    .bind(&actors)
    .bind(&artifacts)
    .bind(density.min(1.0))
    .fetch_one(pool)
    .await?;

    let ep_id = row.0;

    // Async embedding storage (same fire-and-forget pattern as observations)
    let embed_pool = pool.clone();
    let content_owned = content.to_string();
    tokio::spawn(async move {
        if let Ok(emb) = crate::embeddings::onnx::embed_passage(&content_owned).await
            && !emb.iter().all(|&v| v == 0.0)
        {
            let _ = sqlx::query(
                "UPDATE brain_episodes SET embedding = $1::vector WHERE id = $2",
            )
            .bind(pgvector::Vector::from(emb))
            .bind(ep_id)
            .execute(&embed_pool)
            .await;
        }
    });

    tracing::info!(
        entity = %entity_name,
        episode_id = %ep_id,
        actors = actors.len(),
        artifacts = artifacts.len(),
        "episode added"
    );

    Ok(serde_json::json!({
        "action": "episode_add",
        "id": ep_id.to_string(),
        "entity_name": entity_name,
        "actors": actors,
        "artifacts": artifacts,
        "importance": density.min(1.0)
    }))
}

/// List episodes for an entity, ordered by recency.
async fn episode_list(pool: &PgPool, entity_name: &str) -> Result<Value> {
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required for episode_list");
    }

    type EpisodeRow = (uuid::Uuid, String, Vec<String>, Vec<String>, f64, chrono::DateTime<chrono::Utc>);
    let episodes: Vec<EpisodeRow> = sqlx::query_as(
            "SELECT ep.id, ep.content, ep.actors, ep.artifacts,
                    ep.importance::float8, ep.started_at
             FROM brain_episodes ep
             JOIN brain_entities e ON ep.entity_id = e.id
             WHERE e.name = $1
             ORDER BY ep.started_at DESC
             LIMIT 50",
        )
        .bind(entity_name)
        .fetch_all(pool)
        .await?;

    let items: Vec<Value> = episodes
        .iter()
        .map(|(id, content, actors, artifacts, importance, started_at)| {
            serde_json::json!({
                "id": id.to_string(),
                "content": content,
                "actors": actors,
                "artifacts": artifacts,
                "importance": importance,
                "started_at": started_at.to_rfc3339()
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "episode_list",
        "entity_name": entity_name,
        "episodes": items,
        "count": items.len()
    }))
}

/// Delegate to cognitive::density module (DRY — eliminates inline duplication).
fn information_density(content: &str) -> f64 {
    density::information_density(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_density_diverse() {
        let d = information_density("rust is fast and safe for systems");
        assert!(d > 0.8, "diverse text should have high density: got {d}");
    }

    #[test]
    fn test_information_density_repetitive() {
        let d = information_density("hello hello hello hello hello");
        assert!(d < 0.3, "repetitive text should have low density: got {d}");
    }

    #[test]
    fn test_information_density_empty() {
        assert_eq!(information_density(""), 0.0);
    }
}
