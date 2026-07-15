use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

async fn refresh_ood_cache(pool: &PgPool, project_id: Option<uuid::Uuid>) -> Result<()> {
    use crate::search::ood::{MIN_SAMPLES_FOR_OOD, OodStats};
    let raw: Vec<(pgvector::Vector,)> = sqlx::query_as(
        "SELECT embedding FROM brain_observations
         WHERE embedding IS NOT NULL AND observation_type != 'superseded'
           AND ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY id LIMIT 5000",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?;
    if raw.len() < MIN_SAMPLES_FOR_OOD {
        return Ok(());
    }
    let embeddings: Vec<Vec<f32>> = raw.into_iter().map(|(v,)| v.to_vec()).collect();
    if let Some(stats) = OodStats::fit(&embeddings) {
        crate::search::ood_cache::store(project_id, stats);
    }
    Ok(())
}

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");

    match action {
        "decay" => {
            let global_override = args.get("halflife_days").and_then(|v| v.as_f64());
            let result = if let Some(halflife) = global_override {
                sqlx::query(
                    "UPDATE brain_observations SET
                        importance = GREATEST(
                            importance * EXP(-0.693
                                * EXTRACT(EPOCH FROM (NOW() - GREATEST(last_accessed, last_decayed_at))) / 86400.0
                                / ($1 * (1.0 + LN(1.0 + access_count::float8)))),
                            0.01
                        ),
                        last_decayed_at = NOW(),
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
                            importance * EXP(-0.693
                                * EXTRACT(EPOCH FROM (NOW() - GREATEST(last_accessed, last_decayed_at))) / 86400.0
                                / ((CASE observation_type
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
                        last_decayed_at = NOW(),
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
            let dupes: Vec<(uuid::Uuid, uuid::Uuid, f64)> = sqlx::query_as(
                "SELECT a.id, b.id, similarity(a.content, b.content)::float8 AS sim
                 FROM brain_observations a JOIN brain_observations b ON a.entity_id = b.entity_id AND a.id < b.id
                 WHERE similarity(a.content, b.content) > $1 AND a.observation_type != 'superseded' AND b.observation_type != 'superseded'
                 LIMIT 100"
            ).bind(sim_threshold).fetch_all(pool).await?;

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
            let project_id = crate::project::current_project_id(pool).await?;
            refresh_ood_cache(pool, project_id).await.ok();
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
            let energy = crate::graph::energy::refresh_energy_scores(pool)
                .await
                .unwrap_or(0);
            Ok(
                serde_json::json!({"action": "pagerank", "updated": ranked, "energy_refreshed": energy}),
            )
        }
        "communities" => {
            let (communities, nodes) = crate::graph::community::detect_and_persist(pool).await?;
            Ok(serde_json::json!({
                "action": "communities",
                "communities": communities.len(),
                "nodes_tagged": nodes,
                "algorithm": "leiden_v1"
            }))
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
            let marked = sqlx::query("UPDATE brain_observations SET observation_type = 'superseded' WHERE entity_id = $1 AND observation_type != 'superseded'")
                .bind(entity_id.0).execute(pool).await?;
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
            let current_model = crate::embeddings::onnx::current_model();
            let obs: Vec<(uuid::Uuid, String)> = sqlx::query_as(
                "SELECT id, content FROM brain_observations
                 WHERE observation_type != 'superseded'
                   AND (embedding_model != $2 OR embedding_model IS NULL OR embedding IS NULL)
                 ORDER BY importance DESC
                 LIMIT $1",
            )
            .bind(batch_size)
            .bind(&current_model)
            .fetch_all(pool)
            .await?;

            let total = obs.len();
            let mut updated = 0u32;

            let progress_token = args
                .get("_meta")
                .and_then(|m| m.get("progressToken"))
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_else(|| format!("zafra_reembed_{}", batch_size));
            let progress_step = (total / 20).max(1);

            for (i, (obs_id, content)) in obs.into_iter().enumerate() {
                match crate::embeddings::onnx::embed_passage(&content).await {
                    Ok(emb) => {
                        if sqlx::query(
                            "UPDATE brain_observations SET embedding = $1::vector, embedding_model = $2 WHERE id = $3",
                        )
                        .bind(pgvector::Vector::from(emb))
                        .bind(&current_model)
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
                "model": current_model,
                "dim": crate::embeddings::onnx::embedding_dim(),
                "note": "Run after switching embedding models to ensure vector search consistency"
            }))
        }
        _ => anyhow::bail!("Invalid action: {action}"),
    }
}

pub fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
