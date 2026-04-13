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
    let entity_name = args
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match action {
        "add" => add(pool, entity_name, &args).await,
        "delete" => delete_obs(pool, &args).await,
        "list" => list(pool, entity_name).await,
        "batch_add" => batch_add(pool, &args).await,
        "episode_add" => episode_add(pool, entity_name, &args).await,
        "episode_list" => episode_list(pool, entity_name).await,
        "timeline" => timeline(pool, entity_name).await,
        _ => anyhow::bail!(
            "Invalid action: {action}. Use add/delete/list/batch_add/episode_add/episode_list/timeline"
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

    let obs_type = args
        .get("observation_type")
        .and_then(|v| v.as_str())
        .unwrap_or("fact");
    if !VALID_OBSERVATION_TYPES.contains(&obs_type) {
        anyhow::bail!("Invalid observation_type: {obs_type}");
    }

    let source = args
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("agent");
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
                 WHERE id = $1",
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

    // Insert observation with importance priors, session provenance, and auto-tags
    let importance = crate::constants::importance_prior(obs_type, density);
    let tags = extract_tags(content);

    // V0.6: Session provenance — link observation to active session
    let active_session_id: Option<uuid::Uuid> = sqlx::query_scalar(
        "SELECT id FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_observations (entity_id, content, observation_type, source, importance, session_id, tags)
         VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
    )
    .bind(entity_id)
    .bind(content)
    .bind(obs_type)
    .bind(source)
    .bind(importance)
    .bind(active_session_id)
    .bind(&tags)
    .fetch_one(pool)
    .await
    .context("failed to insert observation")?;

    tracing::info!(
        entity = %entity_name,
        obs_type = %obs_type,
        density = %density,
        importance = %importance,
        "observation added"
    );

    // V11: Store embedding for vector search (non-blocking fire-and-forget).
    // Clones pool (cheap Arc clone) to move into spawned task.
    // embed() uses spawn_blocking internally (FIX B2) — safe to call from async context.
    // V0.6: Contextual Retrieval — prepend [entity_type:entity_name] for +20% recall.
    let embed_pool = pool.clone();
    let obs_id_for_embed = row.0;
    let content_for_embed = content.to_string();
    let entity_name_for_embed = entity_name.to_string();
    tokio::spawn(async move {
        // Fetch entity type for contextual embedding
        let entity_type: String = sqlx::query_scalar(
            "SELECT entity_type FROM brain_entities WHERE name = $1",
        )
        .bind(&entity_name_for_embed)
        .fetch_optional(&embed_pool)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| "concept".to_string());

        // Guard: only store embeddings when the real ONNX model is loaded.
        // Hash-based fallback embeddings are not semantically valid — storing
        // them marked as 'multilingual-e5-small' would silently corrupt vector
        // search: future queries with the real model would compare ONNX vectors
        // against hash vectors, returning garbage similarity scores.
        if crate::embeddings::onnx::is_model_loaded() {
            match crate::embeddings::onnx::embed_passage_contextual(
                &content_for_embed,
                &entity_type,
                &entity_name_for_embed,
            )
            .await
            {
                Ok(emb) => {
                    let model = crate::embeddings::onnx::CURRENT_MODEL;
                    let result = sqlx::query(
                        "UPDATE brain_observations SET embedding = $1::vector, embedding_model = $2 WHERE id = $3",
                    )
                    .bind(pgvector::Vector::from(emb))
                    .bind(model)
                    .bind(obs_id_for_embed)
                    .execute(&embed_pool)
                    .await;
                    if let Err(e) = result {
                        tracing::warn!(obs_id = %obs_id_for_embed, error = %e, "failed to store embedding");
                    }
                }
                Err(e) => {
                    tracing::warn!(obs_id = %obs_id_for_embed, error = %e, "ONNX embed failed — skipping");
                }
            }
        }
    });

    // Overload warning + auto-consolidation trigger
    let obs_count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM brain_observations WHERE entity_id = $1 AND observation_type != 'superseded'"
    )
    .bind(entity_id)
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    let overload_warning = if obs_count.0 > 50 {
        // V0.6: Auto-consolidation — spawn merge+prune for overloaded entities
        let pool_clone = pool.clone();
        let eid = entity_id;
        tokio::spawn(async move {
            // Auto-merge >0.9 similarity within this entity
            let _ = sqlx::query(
                "WITH dupes AS (
                    SELECT a.id AS keep_id, b.id AS remove_id
                    FROM brain_observations a JOIN brain_observations b
                    ON a.entity_id = b.entity_id AND a.id < b.id
                    WHERE a.entity_id = $1
                      AND similarity(a.content, b.content) > 0.9
                      AND a.observation_type != 'superseded' AND b.observation_type != 'superseded'
                    LIMIT 10
                )
                UPDATE brain_observations SET observation_type = 'superseded'
                WHERE id IN (SELECT remove_id FROM dupes)",
            )
            .bind(eid)
            .execute(&pool_clone)
            .await;

            // Auto-prune <0.05 importance for this entity (protect decisions/lessons)
            let _ = sqlx::query(
                "DELETE FROM brain_observations WHERE entity_id = $1 AND importance < 0.05 AND observation_type NOT IN ('decision', 'lesson')",
            )
            .bind(eid)
            .execute(&pool_clone)
            .await;
        });

        Some(format!(
            "Entity '{}' has {} observations. Auto-consolidation triggered. Consider cuba_zafra(action='summarize') for further compression.",
            entity_name, obs_count.0
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
        "importance": importance,
        "tags": tags,
        "overload_warning": overload_warning
    }))
}

/// Delete an observation by ID.
async fn delete_obs(pool: &PgPool, args: &Value) -> Result<Value> {
    let obs_id = args
        .get("observation_id")
        .or_else(|| args.get("id"))
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
         ORDER BY importance DESC, created_at DESC",
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
    let observations = args
        .get("observations")
        .and_then(|v| v.as_array())
        .context("'observations' array is required for batch_add")?;

    if observations.len() > 100 {
        anyhow::bail!(
            "batch_add limit is 100 observations per call (got {})",
            observations.len()
        );
    }

    // Pre-process: resolve entities and dedup checks outside transaction
    // (reads are safe to repeat on retry)
    struct PendingObs {
        entity_id: uuid::Uuid,
        entity_name: String,
        entity_type: String,
        content: String,
        obs_type: String,
        source: String,
        importance: f64,
        tags: Vec<String>,
    }
    enum BatchAction {
        Insert(PendingObs),
        Reinforce(uuid::Uuid),
    }

    let mut actions: Vec<BatchAction> = Vec::new();
    let mut deduplicated = 0u32;

    for obs in observations {
        let entity_name = obs
            .get("entity_name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let content = obs.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let obs_type = obs
            .get("observation_type")
            .and_then(|v| v.as_str())
            .unwrap_or("fact");
        let source = obs
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("agent");

        if entity_name.is_empty() || content.is_empty() {
            continue;
        }

        let entity_id = ensure_entity(pool, entity_name).await?;
        // Fetch entity_type for contextual embedding
        let entity_type: String = sqlx::query_scalar(
            "SELECT entity_type FROM brain_entities WHERE id = $1",
        )
        .bind(entity_id)
        .fetch_one(pool)
        .await
        .unwrap_or_else(|_| "concept".to_string());

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
                let importance = crate::constants::importance_prior(obs_type, density);
                let tags = extract_tags(content);
                actions.push(BatchAction::Insert(PendingObs {
                    entity_id,
                    entity_name: entity_name.to_string(),
                    entity_type,
                    content: content.to_string(),
                    obs_type: obs_type.to_string(),
                    source: source.to_string(),
                    importance,
                    tags,
                }));
            }
        }
    }

    // V0.6: Fetch active session once for provenance
    let active_session_id: Option<uuid::Uuid> = sqlx::query_scalar(
        "SELECT id FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    // Execute all writes atomically in a single transaction
    let mut tx = pool.begin().await.context("failed to begin transaction")?;
    let mut added = 0u32;
    let mut reinforced = 0u32;
    // Collect (obs_id, content, entity_name, entity_type) for contextual embedding after commit
    let mut inserted_for_embed: Vec<(uuid::Uuid, String, String, String)> = Vec::new();

    for action in &actions {
        match action {
            BatchAction::Insert(pending) => {
                let row: (uuid::Uuid,) = sqlx::query_as(
                    "INSERT INTO brain_observations (entity_id, content, observation_type, source, importance, session_id, tags)
                     VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
                )
                .bind(pending.entity_id)
                .bind(&pending.content)
                .bind(&pending.obs_type)
                .bind(&pending.source)
                .bind(pending.importance)
                .bind(active_session_id)
                .bind(&pending.tags)
                .fetch_one(&mut *tx)
                .await?;
                inserted_for_embed.push((
                    row.0,
                    pending.content.clone(),
                    pending.entity_name.clone(),
                    pending.entity_type.clone(),
                ));
                added += 1;
            }
            BatchAction::Reinforce(obs_id) => {
                sqlx::query(
                    "UPDATE brain_observations SET
                        importance = LEAST(importance + 0.05, 1.0),
                        access_count = access_count + 1,
                        last_accessed = NOW()
                     WHERE id = $1",
                )
                .bind(obs_id)
                .execute(&mut *tx)
                .await?;
                reinforced += 1;
            }
        }
    }

    tx.commit()
        .await
        .context("failed to commit batch_add transaction")?;

    // V11+V0.6: Store contextual embeddings with model versioning after commit
    if !inserted_for_embed.is_empty() {
        let embed_pool = pool.clone();
        tokio::spawn(async move {
            for (obs_id, content, entity_name, entity_type) in inserted_for_embed {
                // Same guard as single-add: skip if no real ONNX model to avoid
                // storing hash embeddings that corrupt vector search.
                if crate::embeddings::onnx::is_model_loaded()
                    && let Ok(emb) = crate::embeddings::onnx::embed_passage_contextual(
                        &content,
                        &entity_type,
                        &entity_name,
                    )
                    .await
                {
                    let _ = sqlx::query(
                        "UPDATE brain_observations SET embedding = $1::vector, embedding_model = $2 WHERE id = $3",
                    )
                    .bind(pgvector::Vector::from(emb))
                    .bind(crate::embeddings::onnx::CURRENT_MODEL)
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

/// Chronological timeline of all observations + episodes for an entity,
/// including version history diffs.
async fn timeline(pool: &PgPool, entity_name: &str) -> Result<Value> {
    if entity_name.is_empty() {
        anyhow::bail!("entity_name is required for timeline");
    }

    type ObsRow = (
        uuid::Uuid,
        String,
        String,
        f64,
        i32,
        serde_json::Value,
        chrono::DateTime<chrono::Utc>,
    );
    let observations: Vec<ObsRow> = sqlx::query_as(
        "SELECT o.id, o.content, o.observation_type, o.importance::float8,
                o.version, o.previous_versions, o.created_at
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE e.name = $1 AND o.observation_type != 'superseded'
         ORDER BY o.created_at ASC
         LIMIT 100",
    )
    .bind(entity_name)
    .fetch_all(pool)
    .await?;

    type EpRow = (
        uuid::Uuid,
        String,
        Vec<String>,
        Vec<String>,
        f64,
        chrono::DateTime<chrono::Utc>,
    );
    let episodes: Vec<EpRow> = sqlx::query_as(
        "SELECT ep.id, ep.content, ep.actors, ep.artifacts,
                ep.importance::float8, ep.started_at
         FROM brain_episodes ep
         JOIN brain_entities e ON ep.entity_id = e.id
         WHERE e.name = $1
         ORDER BY ep.started_at ASC
         LIMIT 50",
    )
    .bind(entity_name)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut timeline_items: Vec<Value> = Vec::new();

    for (id, content, obs_type, importance, version, prev_versions, created_at) in &observations {
        let mut item = serde_json::json!({
            "type": "observation",
            "id": id.to_string(),
            "content": content,
            "observation_type": obs_type,
            "importance": importance,
            "version": version,
            "timestamp": created_at.to_rfc3339()
        });
        if *version > 1 {
            item["previous_versions"] = prev_versions.clone();
        }
        timeline_items.push(item);
    }

    for (id, content, actors, artifacts, importance, started_at) in &episodes {
        timeline_items.push(serde_json::json!({
            "type": "episode",
            "id": id.to_string(),
            "content": content,
            "actors": actors,
            "artifacts": artifacts,
            "importance": importance,
            "timestamp": started_at.to_rfc3339()
        }));
    }

    // Sort by timestamp (chronological)
    timeline_items.sort_by(|a, b| {
        let ts_a = a.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        let ts_b = b.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        ts_a.cmp(ts_b)
    });

    let count = timeline_items.len();
    Ok(serde_json::json!({
        "action": "timeline",
        "entity_name": entity_name,
        "items": timeline_items,
        "count": count
    }))
}

// ── Helpers ─────────────────────────────────────────────────────

/// Dedup result with Prediction Error Gating (V5).
enum DedupResult {
    Unique,
    Duplicate(String),
    Reinforce(uuid::Uuid),
}

/// Check for near-duplicates using pg_trgm similarity + adaptive PE gating (V5.1)
/// + V0.6 semantic dedup via embedding cosine similarity.
async fn check_dedup(pool: &PgPool, entity_id: uuid::Uuid, content: &str) -> Result<DedupResult> {
    let dupes: Vec<(uuid::Uuid, String, f64)> = sqlx::query_as(
        "SELECT id, content, similarity(content, $2)::float8 AS sim
         FROM brain_observations
         WHERE entity_id = $1 AND similarity(content, $2) > 0.3
         ORDER BY sim DESC LIMIT 5",
    )
    .bind(entity_id)
    .bind(content)
    .fetch_all(pool)
    .await?;

    if dupes.is_empty() {
        // V0.6: Even if no trigram match, check semantic dedup via embedding
        // This catches paraphrases that differ lexically but mean the same thing
        // Semantic dedup only makes sense with real ONNX embeddings. Hash fallback
        // vectors are not semantically comparable — querying them against stored
        // embeddings would give garbage similarity and produce false Reinforce results.
        if crate::embeddings::onnx::is_model_loaded()
            && let Ok(emb) = crate::embeddings::onnx::embed_passage(content).await
        {
            let semantic_match: Option<(uuid::Uuid, f64)> = sqlx::query_as(
                "SELECT id, (1.0 - (embedding <=> $1::vector))::float8 AS sim
                 FROM brain_observations
                 WHERE entity_id = $2 AND embedding IS NOT NULL AND observation_type != 'superseded'
                 ORDER BY embedding <=> $1::vector LIMIT 1",
            )
            .bind(pgvector::Vector::from(emb))
            .bind(entity_id)
            .fetch_optional(pool)
            .await?;

            if let Some((obs_id, sim)) = semantic_match
                && sim > 0.92
            {
                return Ok(DedupResult::Reinforce(obs_id));
            }
        }
        return Ok(DedupResult::Unique);
    }

    // V5.1: Fetch recent similarities for adaptive thresholds
    let recent_sims: Vec<(f64,)> = sqlx::query_as(
        "SELECT similarity(content, $2)::float8
         FROM brain_observations
         WHERE entity_id = $1 AND observation_type != 'superseded'
         ORDER BY created_at DESC LIMIT 20",
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
            return Ok(DedupResult::Duplicate(
                super::zafra::safe_truncate(existing_content, 80).to_string(),
            ));
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
    let existing: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
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
         RETURNING id",
    )
    .bind(name)
    .fetch_one(pool)
    .await?;

    tracing::info!(entity = %name, "auto-created entity for observation");
    Ok(row.0)
}

/// Get entity_id by name (strict — errors if not found).
async fn get_entity_id(pool: &PgPool, name: &str) -> Result<uuid::Uuid> {
    let row: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
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

    // V0.6: Session provenance for episodes
    let active_session_id: Option<uuid::Uuid> = sqlx::query_scalar(
        "SELECT id FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    let row: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_episodes (entity_id, content, actors, artifacts, importance, session_id)
         VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
    )
    .bind(entity_id)
    .bind(content)
    .bind(&actors)
    .bind(&artifacts)
    .bind(density.min(1.0))
    .bind(active_session_id)
    .fetch_one(pool)
    .await?;

    let ep_id = row.0;

    // Async embedding storage (same fire-and-forget pattern as observations)
    // V0.6: Contextual Retrieval for episodes too.
    let embed_pool = pool.clone();
    let content_owned = content.to_string();
    let entity_name_owned = entity_name.to_string();
    tokio::spawn(async move {
        let entity_type: String = sqlx::query_scalar(
            "SELECT entity_type FROM brain_entities WHERE name = $1",
        )
        .bind(&entity_name_owned)
        .fetch_optional(&embed_pool)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| "concept".to_string());

        // Same guard as observation storage: skip hash embeddings to avoid
        // corrupting the episode vector index.
        if crate::embeddings::onnx::is_model_loaded()
            && let Ok(emb) = crate::embeddings::onnx::embed_passage_contextual(
                &content_owned,
                &entity_type,
                &entity_name_owned,
            )
            .await
        {
            let model = crate::embeddings::onnx::CURRENT_MODEL;
            let _ = sqlx::query(
                "UPDATE brain_episodes SET embedding = $1::vector, embedding_model = $2 WHERE id = $3",
            )
            .bind(pgvector::Vector::from(emb))
            .bind(model)
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

    type EpisodeRow = (
        uuid::Uuid,
        String,
        Vec<String>,
        Vec<String>,
        f64,
        chrono::DateTime<chrono::Utc>,
    );
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

/// V0.6: Extract top-5 keywords from text (frequency-based, bilingual stopwords).
///
/// Simple TF extraction — no IDF since we don't have corpus stats.
/// Returns lowercase keywords sorted by descending frequency.
fn extract_tags(content: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "for", "and", "but", "or", "nor", "not", "no", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about", "above", "after", "again", "also", "as", "at",
        "before", "between", "by", "from", "how", "in", "into", "it", "its", "of", "on", "only",
        "out", "over", "own", "same", "that", "this", "these", "those", "through", "to", "under",
        "until", "up", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
        // Spanish
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con", "por",
        "para", "es", "son", "fue", "ser", "estar", "que", "se", "si", "como", "pero", "mas",
        "ya", "entre", "cuando", "todo", "esta", "desde", "su", "sus", "le", "les",
    ];

    let words: Vec<String> = content
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() > 2 && !STOPWORDS.contains(&&**w))
        .map(String::from)
        .collect();

    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for w in &words {
        *freq.entry(w.clone()).or_default() += 1;
    }

    let mut sorted: Vec<(String, usize)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.into_iter().take(5).map(|(w, _)| w).collect()
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
