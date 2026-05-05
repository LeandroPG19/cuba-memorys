//! Handler: cuba_zafra — Memory maintenance and consolidation.
//!
//! FIX R-001: safe_truncate() for UTF-8 char boundary safety.
//! FIX R-002: merge loop wrapped in transaction for atomicity.
//! V3: decay action now uses exponential decay on importance (replaces FSRS-6).
//!     importance * EXP(-0.693 * days / halflife) — simple, auditable, effective.

use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "decay" => {
            // V4: Stratified exponential decay — different halflife per observation_type.
            // fact/preference: 30d  |  error/solution: 14d  |  context/tool_usage: 7d
            // decision/lesson: ∞ (never decay — protected by WHERE clause).
            //
            // V0.9: Testing effect (Karpicke-Roediger Science 2008) — frequently
            // accessed observations decay slower:
            //   effective_halflife = base_halflife · (1 + ln(1 + access_count))
            // An observation accessed 50× has ≈4× longer effective halflife than
            // one accessed 0×. Implements retrieval-induced consolidation.
            //
            // Override with halflife_days to apply a global halflife (backward compat).
            let global_override = args.get("halflife_days").and_then(|v| v.as_f64());
            let result = if let Some(halflife) = global_override {
                sqlx::query(
                    "UPDATE brain_observations SET
                        importance = GREATEST(
                            importance * EXP(-0.693 * EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0
                                / ($1 * (1.0 + LN(1.0 + access_count::float8)))),
                            0.01
                        ),
                        updated_at = NOW()
                     WHERE observation_type NOT IN ('decision', 'lesson', 'superseded')
                       AND last_accessed < NOW() - INTERVAL '1 day'"
                )
                .bind(halflife)
                .execute(pool)
                .await?
            } else {
                sqlx::query(
                    "UPDATE brain_observations SET
                        importance = GREATEST(
                            importance * EXP(-0.693 * EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0 /
                                ((CASE observation_type
                                    WHEN 'fact'       THEN 30.0
                                    WHEN 'preference' THEN 30.0
                                    WHEN 'error'      THEN 14.0
                                    WHEN 'solution'   THEN 14.0
                                    WHEN 'context'    THEN  7.0
                                    WHEN 'tool_usage' THEN  7.0
                                    ELSE 30.0
                                END) * (1.0 + LN(1.0 + access_count::float8)))
                            ),
                            0.01
                        ),
                        updated_at = NOW()
                     WHERE observation_type NOT IN ('decision', 'lesson', 'superseded')
                       AND last_accessed < NOW() - INTERVAL '1 day'"
                )
                .execute(pool)
                .await?
            };
            Ok(serde_json::json!({
                "action": "decay",
                "decayed": result.rows_affected(),
                "stratification": {
                    "fact/preference": "30d base halflife",
                    "error/solution": "14d base halflife",
                    "context/tool_usage": "7d base halflife",
                    "decision/lesson": "never decay"
                },
                "formula": "importance * EXP(-0.693 * days / (base_halflife * (1 + ln(1 + access_count))))",
                "testing_effect": "Karpicke-Roediger 2008 — high-access obs decay slower"
            }))
        }
        "prune" => {
            let threshold = args
                .get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1);
            let result = sqlx::query("DELETE FROM brain_observations WHERE importance < $1 AND observation_type NOT IN ('decision', 'lesson')")
                .bind(threshold).execute(pool).await?;
            Ok(
                serde_json::json!({"action": "prune", "pruned": result.rows_affected(), "threshold": threshold}),
            )
        }
        "merge" => {
            let sim_threshold = args
                .get("similarity_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.8);
            // P2 FIX: Batch merge — find duplicates in one query, merge in batch
            let dupes: Vec<(uuid::Uuid, uuid::Uuid, f64)> = sqlx::query_as(
                "SELECT a.id, b.id, similarity(a.content, b.content)::float8 AS sim
                 FROM brain_observations a JOIN brain_observations b ON a.entity_id = b.entity_id AND a.id < b.id
                 WHERE similarity(a.content, b.content) > $1 AND a.observation_type != 'superseded' AND b.observation_type != 'superseded'
                 LIMIT 100"
            ).bind(sim_threshold).fetch_all(pool).await?;

            // FIX R-002: Atomic transaction — all-or-nothing merge
            let mut tx = pool
                .begin()
                .await
                .context("failed to begin merge transaction")?;
            let mut merged = 0u32;
            for (keep_id, remove_id, _) in &dupes {
                sqlx::query(
                    "UPDATE brain_observations SET observation_type = 'superseded' WHERE id = $1",
                )
                .bind(remove_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query("UPDATE brain_observations SET importance = LEAST(importance + 0.05, 1.0) WHERE id = $1").bind(keep_id).execute(&mut *tx).await?;
                merged += 1;
            }
            tx.commit()
                .await
                .context("failed to commit merge transaction")?;
            Ok(serde_json::json!({"action": "merge", "merged": merged, "threshold": sim_threshold}))
        }
        "stats" => {
            let entities: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_entities")
                .fetch_one(pool)
                .await?;
            let observations: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM brain_observations")
                .fetch_one(pool)
                .await?;
            let superseded: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM brain_observations WHERE observation_type = 'superseded'",
            )
            .fetch_one(pool)
            .await?;
            Ok(
                serde_json::json!({"action": "stats", "entities": entities.0, "observations": observations.0, "superseded": superseded.0, "active": observations.0 - superseded.0}),
            )
        }
        "pagerank" => {
            let ranked = crate::graph::pagerank::compute_and_store(pool).await?;
            Ok(serde_json::json!({"action": "pagerank", "updated": ranked}))
        }
        "summarize" => {
            let entity_name = args
                .get("entity_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let summary = args
                .get("compressed_summary")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if entity_name.is_empty() || summary.is_empty() {
                anyhow::bail!("entity_name and compressed_summary are required");
            }
            let entity_id: (uuid::Uuid,) =
                sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
                    .bind(entity_name)
                    .fetch_one(pool)
                    .await?;
            // Mark old observations as superseded
            let marked = sqlx::query("UPDATE brain_observations SET observation_type = 'superseded' WHERE entity_id = $1 AND observation_type != 'superseded'")
                .bind(entity_id.0).execute(pool).await?;
            // Insert new summary
            sqlx::query("INSERT INTO brain_observations (entity_id, content, observation_type, source) VALUES ($1, $2, 'fact', 'consolidation')")
                .bind(entity_id.0).bind(summary).execute(pool).await?;
            Ok(
                serde_json::json!({"action": "summarize", "entity": entity_name, "superseded": marked.rows_affected()}),
            )
        }
        "find_duplicates" => {
            let dupes: Vec<(String, String, f64)> = sqlx::query_as(
                "SELECT a.content, b.content, similarity(a.content, b.content)::float8 AS sim
                 FROM brain_observations a JOIN brain_observations b ON a.entity_id = b.entity_id AND a.id < b.id
                 WHERE similarity(a.content, b.content) > 0.7 AND a.observation_type != 'superseded' AND b.observation_type != 'superseded'
                 ORDER BY sim DESC LIMIT 20"
            ).fetch_all(pool).await?;
            // FIX R-001: safe_truncate prevents panic on multi-byte UTF-8
            let results: Vec<Value> = dupes
                .iter()
                .map(|(a, b, s)| {
                    serde_json::json!({
                        "content_a": safe_truncate(a, 100),
                        "content_b": safe_truncate(b, 100),
                        "similarity": s
                    })
                })
                .collect();
            Ok(
                serde_json::json!({"action": "find_duplicates", "duplicates": results, "count": results.len()}),
            )
        }
        "export" => {
            let entities: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as("SELECT id, name, entity_type, importance FROM brain_entities ORDER BY importance DESC LIMIT 500").fetch_all(pool).await?;
            let ent_json: Vec<Value> = entities.iter().map(|(id, n, t, i)| serde_json::json!({"id": id.to_string(), "name": n, "type": t, "importance": i})).collect();
            Ok(
                serde_json::json!({"action": "export", "entities": ent_json, "count": ent_json.len()}),
            )
        }
        "decay_episodes" => {
            // Power-law decay for episodic memories (Wixted 2004).
            // I(t) = 0.5 / (1 + c·t)^β — computed from creation time, NOT from current importance.
            //
            // FIX: Previous version used `importance / POWER(...)` which compounds multiplicatively
            // on each invocation (double-decay bug). This version computes the target importance
            // directly from the initial value (0.5) and age, making it IDEMPOTENT — calling it
            // twice produces the same result.
            //
            // Default: c=0.1, β=0.5 (Wixted & Ebbesen 1991 calibration).
            let c = args.get("c").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let beta = args.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let result = sqlx::query(
                "UPDATE brain_episodes SET
                    importance = GREATEST(
                        0.5 / POWER(
                            1.0 + $1 * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0,
                            $2
                        ),
                        0.01
                    )
                 WHERE created_at < NOW() - INTERVAL '1 hour'",
            )
            .bind(c)
            .bind(beta)
            .execute(pool)
            .await?;
            Ok(serde_json::json!({
                "action": "decay_episodes",
                "decayed": result.rows_affected(),
                "formula": "0.5 / (1 + c·t)^β  (Wixted 2004, idempotent from initial=0.5)",
                "c": c,
                "beta": beta
            }))
        }
        "reembed" => {
            // Re-encode all observations using the current ONNX model (embed_passage prefix).
            // Run this after switching embedding models to ensure consistency.

            // Require a real ONNX model — reembed with hash fallback embeddings
            // would replace valid ONNX vectors with semantically meaningless hashes.
            if !crate::embeddings::onnx::is_model_loaded() {
                return Ok(serde_json::json!({
                    "action": "reembed",
                    "updated": 0,
                    "error": "ONNX model not loaded — set ONNX_MODEL_PATH to enable reembed"
                }));
            }

            let batch_size = args
                .get("batch_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(500);
            let current_model = crate::embeddings::onnx::CURRENT_MODEL;
            // V0.6: Only re-encode observations with stale/missing model or embedding
            let obs: Vec<(uuid::Uuid, String)> = sqlx::query_as(
                "SELECT id, content FROM brain_observations
                 WHERE observation_type != 'superseded'
                   AND (embedding_model != $2 OR embedding_model IS NULL OR embedding IS NULL)
                 ORDER BY importance DESC
                 LIMIT $1",
            )
            .bind(batch_size)
            .bind(current_model)
            .fetch_all(pool)
            .await?;

            let total = obs.len();
            let mut updated = 0u32;

            // V0.9.2: progress notifications. Token derived from action +
            // batch size — clients use it to correlate intermediate progress
            // with the final tools/call response.
            let progress_token = args
                .get("_meta")
                .and_then(|m| m.get("progressToken"))
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_else(|| format!("zafra_reembed_{}", batch_size));
            let progress_step = (total / 20).max(1); // ~5% increments

            for (i, (obs_id, content)) in obs.into_iter().enumerate() {
                match crate::embeddings::onnx::embed_passage(&content).await {
                    Ok(emb) => {
                        if sqlx::query(
                            "UPDATE brain_observations SET embedding = $1::vector, embedding_model = $2 WHERE id = $3",
                        )
                        .bind(pgvector::Vector::from(emb))
                        .bind(current_model)
                        .bind(obs_id)
                        .execute(pool)
                        .await
                        .is_ok()
                        {
                            updated += 1;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(obs_id = %obs_id, error = %e, "reembed: ONNX failed for observation");
                    }
                }
                if total > 50 && (i + 1) % progress_step == 0 {
                    crate::protocol::notify_progress(
                        &progress_token,
                        (i + 1) as f64,
                        Some(total as f64),
                        Some(&format!("re-embedded {}/{}", i + 1, total)),
                    );
                }
            }

            Ok(serde_json::json!({
                "action": "reembed",
                "total_fetched": total,
                "updated": updated,
                "model": "multilingual-e5-small (passage: prefix)",
                "note": "Run after switching embedding models to ensure vector search consistency"
            }))
        }
        _ => anyhow::bail!("Invalid action: {action}"),
    }
}

/// FIX R-001: Truncate string to max_bytes at a valid UTF-8 char boundary.
///
/// Rust `&str[..n]` panics if `n` is not a char boundary (e.g., slicing
/// mid-emoji or mid-accent). This function walks backwards to find the
/// nearest valid boundary. O(1) amortized — max 4 bytes backtrack (max
/// UTF-8 char width).
pub fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Walk backwards to find a valid char boundary (max 4 steps)
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
