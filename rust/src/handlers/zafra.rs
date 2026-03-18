//! Handler: cuba_zafra — Memory maintenance and consolidation.
//!
//! FIX R-001: safe_truncate() for UTF-8 char boundary safety.
//! FIX R-002: merge loop wrapped in transaction for atomicity.

use crate::cognitive::fsrs;
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "decay" => {
            let decayed = fsrs::batch_decay(pool, &[]).await?;
            Ok(serde_json::json!({"action": "decay", "decayed": decayed}))
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
        "backfill" => {
            // Backfill missing Dual-Strength columns with defaults
            let result = sqlx::query("UPDATE brain_observations SET storage_strength = 0.5, retrieval_strength = 1.0 WHERE storage_strength IS NULL")
                .execute(pool).await?;
            Ok(serde_json::json!({"action": "backfill", "updated": result.rows_affected()}))
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
