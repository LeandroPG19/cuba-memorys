//! Handler: cuba_faro — Hybrid search with RRF fusion.
//!
//! FIX B2: embed() runs in spawn_blocking (not blocking event loop).
//! §A: Weighted RRF Entropy Routing — dynamic weight based on query entropy.
//! V8: Graceful degradation — if vector search fails, fallback to text.
//! VF2: Testing Effect — search matches update access tracking.
//! V4-RRF: k=60 constant (Cormack 2009) — removed adaptive instability.
//! V10: importance integrated into SQL score (score*0.7 + importance*0.3).
//!      Activates the cognitive pipeline — PageRank/Hebbian/decay now affect ranking.

use crate::cognitive::dual_strength;
use crate::search::confidence as grounding;
use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;
use std::collections::HashMap;

const DEFAULT_LIMIT: i64 = 10;
const MAX_LIMIT: i64 = 50;
const DEFAULT_MAX_TOKENS: i64 = 5000;
const GRAPHRAG_TOP_K: usize = 3;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("hybrid");
    let scope = args.get("scope").and_then(|v| v.as_str()).unwrap_or("all");
    let limit = args.get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_LIMIT)
        .min(MAX_LIMIT);

    let max_tokens = args.get("max_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_MAX_TOKENS);

    match mode {
        "hybrid" => hybrid_search(pool, query, scope, limit, max_tokens).await,
        "verify" => verify_claim(pool, query).await,
        _ => anyhow::bail!("Invalid mode: {mode}. Use hybrid/verify"),
    }
}

/// §A V2: Weighted RRF Entropy Routing — 3 ranges (Elastic 2025).
async fn hybrid_search(pool: &PgPool, query: &str, scope: &str, limit: i64, max_tokens: i64) -> Result<Value> {
    // §A V2: 3-range entropy routing (keyword / mixed / semantic)
    let query_entropy = crate::search::rrf::query_entropy(query);
    let (text_weight, vector_weight) = entropy_weights(query_entropy);

    // Run text search (always available — V8 graceful degradation base)
    let text_results = text_search(pool, query, scope, limit * 2).await?;

    // Run vector search (may fail gracefully — V8)
    let vector_results = vector_search(pool, query, scope, limit * 2).await;

    // V4: Fixed k=60 (Cormack 2009 consensus — eliminates non-monotonic instability)
    let rrf_k: f64 = 60.0;

    // RRF Fusion with entropy-weighted scores and adaptive k
    let mut fused_scores: HashMap<String, (f64, Value)> = HashMap::new();

    // Add text results with RRF rank score
    for (rank, result) in text_results.iter().enumerate() {
        let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let rrf_score = text_weight / (rrf_k + rank as f64 + 1.0);
        fused_scores.insert(id, (rrf_score, result.clone()));
    }

    // Add vector results (V8: only if available)
    if let Ok(vec_results) = vector_results {
        for (rank, result) in vec_results.iter().enumerate() {
            let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let rrf_score = vector_weight / (rrf_k + rank as f64 + 1.0);
            fused_scores
                .entry(id.clone())
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert((rrf_score, result.clone()));
        }
    }

    // Sort by fused score
    let mut results: Vec<(String, f64, Value)> = fused_scores
        .into_iter()
        .map(|(id, (score, result))| (id, score, result))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit as usize);

    // VF2: Testing Effect — update access tracking on matched observations
    let matched_obs_ids: Vec<uuid::Uuid> = results.iter()
        .filter_map(|(_, _, r)| {
            r.get("id").and_then(|v| v.as_str())
                .and_then(|s| s.parse::<uuid::Uuid>().ok())
        })
        .collect();

    if !matched_obs_ids.is_empty()
        && let Err(e) = dual_strength::on_search_match(pool, &matched_obs_ids).await {
            tracing::warn!(error = %e, "failed to apply Testing Effect boost");
        }

    // Session awareness: check active session and boost matching results
    let session_boost = get_session_goals(pool).await.unwrap_or_default();
    if !session_boost.is_empty() {
        for (_, score, result) in &mut results {
            if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
                let content_lower = content.to_lowercase();
                for goal in &session_boost {
                    if content_lower.contains(&goal.to_lowercase()) {
                        *score *= 1.3; // 30% boost for session-relevant results
                        break;
                    }
                }
            }
        }
        // Re-sort after session boost
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // GraphRAG enrichment — add degree-1 neighbors for top-K results (V9)
    let graphrag_context = enrich_graphrag(pool, &results, GRAPHRAG_TOP_K).await;

    // Token-budget truncation
    let results_json: Vec<Value> = results
        .iter()
        .map(|(_, score, result)| {
            let mut r = result.clone();
            if let Some(obj) = r.as_object_mut() {
                obj.insert("fused_score".to_string(), serde_json::json!(score));
            }
            r
        })
        .collect();

    // Token budget enforcement: truncate content per result instead of dropping entire results.
    // This gives Claude more results with shorter content (better for grounding).
    // Estimate: 1 token ≈ 4 chars.
    let mut token_budget = max_tokens;
    let mut budget_results: Vec<Value> = Vec::with_capacity(results_json.len());
    for mut r in results_json {
        let content_len = r.get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.len() as i64 / 4)
            .unwrap_or(20);
        if token_budget <= 0 {
            break;
        }
        if content_len > token_budget {
            // Truncate content to fit remaining budget instead of dropping
            let max_chars = (token_budget * 4) as usize;
            let truncated: Option<String> = r.get("content")
                .and_then(|v| v.as_str())
                .map(|s| crate::handlers::zafra::safe_truncate(s, max_chars).to_string());
            if let (Some(obj), Some(t)) = (r.as_object_mut(), truncated) {
                obj.insert("content".to_string(), serde_json::json!(t));
            }
            budget_results.push(r);
            break;
        }
        token_budget -= content_len;
        budget_results.push(r);
    }

    Ok(serde_json::json!({
        "mode": "hybrid",
        "query": query,
        "results": budget_results,
        "count": budget_results.len(),
        "graphrag_context": graphrag_context
    }))
}

/// Verify a claim against stored knowledge with source diversity scoring.
async fn verify_claim(pool: &PgPool, claim: &str) -> Result<Value> {
    let evidence: Vec<(String, f64, String)> = sqlx::query_as(
        "SELECT content, similarity(content, $1)::float8 AS sim, observation_type
         FROM brain_observations
         WHERE similarity(content, $1) > 0.3
           AND observation_type != 'superseded'
         ORDER BY sim DESC
         LIMIT 10"
    )
    .bind(claim)
    .fetch_all(pool)
    .await?;

    let similarities: Vec<f64> = evidence.iter().map(|(_, s, _)| *s).collect();
    let sources: Vec<&str> = evidence.iter().map(|(_, _, t)| t.as_str()).collect();

    let (confidence, level) = grounding::compute_grounding(&similarities, &sources);

    let evidence_json: Vec<Value> = evidence
        .iter()
        .map(|(content, sim, obs_type)| {
            serde_json::json!({
                "content": content,
                "similarity": sim,
                "type": obs_type
            })
        })
        .collect();

    Ok(serde_json::json!({
        "mode": "verify",
        "claim": claim,
        "confidence": confidence,
        "grounding": level,
        "evidence": evidence_json,
        "evidence_count": evidence_json.len(),
        "source_diversity": sources.len()
    }))
}

/// Execute a parameterized text search query and map results to JSON via closure.
async fn search_table<T, F>(
    pool: &PgPool,
    sql: &str,
    query: &str,
    limit: i64,
    mapper: F,
) -> Result<Vec<Value>>
where
    T: for<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> + Send + Unpin,
    F: Fn(T) -> Value,
{
    let rows: Vec<T> = sqlx::query_as(sql)
        .bind(query)
        .bind(limit)
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(mapper).collect())
}

/// Text search using PostgreSQL full-text + trigram similarity.
async fn text_search(pool: &PgPool, query: &str, scope: &str, limit: i64) -> Result<Vec<Value>> {
    let mut results = Vec::new();

    if scope == "all" || scope == "entities" {
        let entities = search_table::<(uuid::Uuid, String, String, f64, f64), _>(
            pool,
            // V10: importance*0.3 activates the cognitive pipeline in ranking.
            // PageRank, Hebbian boosts, and decay now directly affect which entities surface.
            "SELECT id, name, entity_type, importance::float8,
                    (  (ts_rank(search_vector, plainto_tsquery('simple', $1))
                      + similarity(name, $1)) * 0.7
                     + importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_entities
             WHERE search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(name, $1) > 0.3
             ORDER BY score DESC
             LIMIT $2",
            query,
            limit,
            |(id, name, entity_type, importance, score)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "entity",
                    "name": name,
                    "entity_type": entity_type,
                    "importance": importance,
                    "score": score
                })
            },
        ).await?;
        results.extend(entities);
    }

    if scope == "all" || scope == "observations" {
        let observations = search_table::<(uuid::Uuid, String, String, String, f64, f64), _>(
            pool,
            // V10: importance*0.3 — same as entities query, makes importance real.
            "SELECT o.id, e.name, o.content, o.observation_type, o.importance::float8,
                    (  (ts_rank(o.search_vector, plainto_tsquery('simple', $1))
                      + similarity(o.content, $1)) * 0.7
                     + o.importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE (o.search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(o.content, $1) > 0.3)
               AND o.observation_type != 'superseded'
             ORDER BY score DESC
             LIMIT $2",
            query,
            limit,
            |(id, entity_name, content, obs_type, importance, score)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "observation",
                    "entity_name": entity_name,
                    "content": content,
                    "observation_type": obs_type,
                    "importance": importance,
                    "score": score
                })
            },
        ).await?;
        results.extend(observations);
    }

    if scope == "all" || scope == "errors" {
        let errors = search_table::<(uuid::Uuid, String, String, bool, f64), _>(
            pool,
            "SELECT id, error_type, error_message, resolved,
                    (ts_rank(search_vector, plainto_tsquery('simple', $1)) +
                    similarity(error_message, $1))::float8 AS score
             FROM brain_errors
             WHERE search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(error_message, $1) > 0.3
             ORDER BY score DESC
             LIMIT $2",
            query,
            limit,
            |(id, error_type, error_message, resolved, score)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "error",
                    "error_type": error_type,
                    "error_message": error_message,
                    "resolved": resolved,
                    "score": score
                })
            },
        ).await?;
        results.extend(errors);
    }

    Ok(results)
}

/// Vector search via pgvector cosine similarity.
/// V8: Returns Result — callers can gracefully degrade on failure.
async fn vector_search(pool: &PgPool, query: &str, _scope: &str, limit: i64) -> Result<Vec<Value>> {
    // Compute embedding via ONNX (or hash fallback).
    // embed() already runs in spawn_blocking internally (FIX B2).
    let embedding = match crate::embeddings::onnx::embed(query).await {
        Ok(emb) => emb,
        Err(_) => return Ok(vec![]), // V8: graceful degradation
    };

    // Skip if embedding is all zeros (no model loaded)
    if embedding.iter().all(|&v| v == 0.0) {
        return Ok(vec![]); // V8: graceful degradation
    }

    // If we had real embeddings:
    let observations: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as(
        "SELECT o.id, e.name, o.content,
                1.0 - (o.embedding <=> $1::vector) AS cosine_sim
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE o.embedding IS NOT NULL
           AND o.observation_type != 'superseded'
         ORDER BY o.embedding <=> $1::vector
         LIMIT $2"
    )
    .bind(pgvector::Vector::from(embedding))
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let results: Vec<Value> = observations
        .iter()
        .map(|(id, entity_name, content, sim)| {
            serde_json::json!({
                "id": id.to_string(),
                "type": "observation",
                "entity_name": entity_name,
                "content": content,
                "cosine_similarity": sim
            })
        })
        .collect();

    Ok(results)
}

// ── Utility Functions ───────────────────────────────────────────

/// §A V2: 3-range entropy routing (Elastic Search Labs, 2025).
///
/// | Entropy | Query Type | text | vector |
/// |---------|------------|------|--------|
/// | < 2.0   | Keyword    | 0.7  | 0.3    |
/// | 2.0-3.5 | Mixed      | 0.5  | 0.5    |
/// | > 3.5   | Semantic   | 0.3  | 0.7    |
fn entropy_weights(entropy: f64) -> (f64, f64) {
    if entropy < 2.0 {
        (0.7, 0.3) // Keyword-heavy: prefer text match
    } else if entropy <= 3.5 {
        (0.5, 0.5) // Balanced: equal weight
    } else {
        (0.3, 0.7) // Semantic-heavy: prefer vector search
    }
}

// V3 adaptive_rrf_k REMOVED — k=60 constant per Gemini Deep Research audit 2026-03-14.
// Rationale: dynamic sqrt-based k introduced non-monotonic ranking instabilities
// and violated determinism. Cormack et al. 2009, Azure AI Search, Elasticsearch
// all converge on k=60 as empirically optimal.

/// Get active session goals for session-aware boosting.
async fn get_session_goals(pool: &PgPool) -> Result<Vec<String>> {
    let row: Option<(serde_json::Value,)> = sqlx::query_as(
        "SELECT goals FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
    )
    .fetch_optional(pool)
    .await?;

    if let Some((goals,)) = row {
        let goals: Vec<String> = serde_json::from_value(goals).unwrap_or_default();
        Ok(goals)
    } else {
        Ok(vec![])
    }
}

/// V9: GraphRAG enrichment — fetch degree-1 neighbors for top-K results.
///
/// For each top result that has an entity name, we query its related entities
/// to provide graph context. This helps the AI understand the broader
/// knowledge structure around search matches.
async fn enrich_graphrag(
    pool: &PgPool,
    results: &[(String, f64, Value)],
    top_k: usize,
) -> Value {
    let mut context: Vec<Value> = Vec::new();

    for (_, _, result) in results.iter().take(top_k) {
        let entity_name = result.get("entity_name")
            .or_else(|| result.get("name"))
            .and_then(|v| v.as_str());

        if let Some(name) = entity_name {
            // N+1 fix: single CTE resolves entity id once, no subselect repetition.
            let neighbors: Vec<(String, String, f64)> = match sqlx::query_as(
                "WITH src AS (SELECT id FROM brain_entities WHERE name = $1 LIMIT 1)
                 SELECT e.name, r.relation_type, e.importance::float8
                 FROM brain_relations r
                 JOIN src ON r.from_entity = src.id OR r.to_entity = src.id
                 JOIN brain_entities e ON e.id = CASE
                     WHEN r.from_entity = src.id THEN r.to_entity
                     ELSE r.from_entity
                 END
                 ORDER BY r.strength DESC
                 LIMIT 5"
            )
            .bind(name)
            .fetch_all(pool)
            .await {
                Ok(n) => n,
                Err(_) => continue,
            };

            if !neighbors.is_empty() {
                let neighbor_list: Vec<Value> = neighbors.iter().map(|(n, rel, imp)| {
                    serde_json::json!({"name": n, "relation": rel, "importance": imp})
                }).collect();

                context.push(serde_json::json!({
                    "entity": name,
                    "neighbors": neighbor_list
                }));
            }
        }
    }

    serde_json::json!(context)
}
