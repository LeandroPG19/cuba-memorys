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

/// V0.6: Score breakdown — tracks individual RRF components per result.
struct FusedResult {
    text_score: f64,
    vector_score: f64,
    session_boosted: bool,
    total: f64,
    data: Value,
}

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    let mode = args
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("hybrid");
    let scope = args.get("scope").and_then(|v| v.as_str()).unwrap_or("all");
    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_LIMIT)
        .min(MAX_LIMIT);

    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_MAX_TOKENS);

    // V0.6: Compact format — abbreviated keys, ~35% token savings
    let format = args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("verbose");

    // V0.6: Tag filter
    let tag_filter = args.get("tags").and_then(|v| v.as_str());

    // Temporal filters (ISO8601 strings)
    let before = args.get("before").and_then(|v| v.as_str());
    let after = args.get("after").and_then(|v| v.as_str());
    let time_bounds = parse_time_bounds(before, after);

    let search_opts = SearchOpts {
        scope,
        limit,
        max_tokens,
        time_bounds,
        format,
        tag_filter,
    };

    match mode {
        "hybrid" => hybrid_search(pool, query, &search_opts).await,
        "verify" => verify_claim(pool, query).await,
        _ => anyhow::bail!("Invalid mode: {mode}. Use hybrid/verify"),
    }
}

/// Parsed temporal bounds for search filtering.
struct TimeBounds {
    after: chrono::DateTime<chrono::Utc>,
    before: chrono::DateTime<chrono::Utc>,
}

/// Parse optional before/after ISO8601 strings into DateTime bounds.
fn parse_time_bounds(before: Option<&str>, after: Option<&str>) -> TimeBounds {
    let after_ts = after
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&chrono::Utc))
        .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).unwrap());
    let before_ts = before
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&chrono::Utc))
        .unwrap_or_else(|| chrono::Utc::now() + chrono::Duration::days(365));
    TimeBounds {
        after: after_ts,
        before: before_ts,
    }
}

/// Search options for hybrid search.
struct SearchOpts<'a> {
    scope: &'a str,
    limit: i64,
    max_tokens: i64,
    time_bounds: TimeBounds,
    format: &'a str,
    tag_filter: Option<&'a str>,
}

/// §A V2: Weighted RRF Entropy Routing — 3 ranges (Elastic 2025).
async fn hybrid_search(pool: &PgPool, query: &str, opts: &SearchOpts<'_>) -> Result<Value> {
    // §A V2: 3-range entropy routing (keyword / mixed / semantic)
    let query_entropy = crate::search::rrf::query_entropy(query);
    let (text_weight, vector_weight) = entropy_weights(query_entropy);

    // Run text search (always available — V8 graceful degradation base)
    let mut text_results =
        text_search(pool, query, opts.scope, opts.limit * 2, &opts.time_bounds).await?;

    // V0.6: Tag filter — if specified, run additional tag-based query and merge
    if let Some(tag) = opts.tag_filter {
        let tagged_obs: Vec<(uuid::Uuid, String, String, String, f64, f64)> = sqlx::query_as(
            "SELECT o.id, e.name, o.content, o.observation_type, o.importance::float8,
                    (o.importance::float8 * 0.8 + 0.2)::float8 AS score
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE $1 = ANY(o.tags)
               AND o.observation_type != 'superseded'
             ORDER BY o.importance DESC
             LIMIT $2",
        )
        .bind(tag)
        .bind(opts.limit)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        for (id, entity_name, content, obs_type, importance, score) in tagged_obs {
            // Avoid duplicates
            let id_str = id.to_string();
            if !text_results
                .iter()
                .any(|r| r.get("id").and_then(|v| v.as_str()) == Some(&id_str))
            {
                text_results.push(serde_json::json!({
                    "id": id_str,
                    "type": "observation",
                    "entity_name": entity_name,
                    "content": content,
                    "observation_type": obs_type,
                    "importance": importance,
                    "score": score,
                    "matched_tag": tag
                }));
            }
        }
    }

    // Run vector search (may fail gracefully — V8)
    let vector_results =
        vector_search(pool, query, opts.scope, opts.limit * 2, &opts.time_bounds).await;

    // V4: Fixed k=60 (Cormack 2009 consensus — eliminates non-monotonic instability)
    // Use the canonical constant from rrf.rs to avoid drift if the value ever changes.
    let rrf_k = crate::search::rrf::RRF_K;

    let mut fused_scores: HashMap<String, FusedResult> = HashMap::new();

    // Add text results with RRF rank score
    for (rank, result) in text_results.iter().enumerate() {
        let id = result
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let rrf_score = text_weight / (rrf_k + rank as f64 + 1.0);
        fused_scores.insert(
            id,
            FusedResult {
                text_score: rrf_score,
                vector_score: 0.0,
                session_boosted: false,
                total: rrf_score,
                data: result.clone(),
            },
        );
    }

    // Add vector results (V8: only if available)
    if let Ok(vec_results) = vector_results {
        for (rank, result) in vec_results.iter().enumerate() {
            let id = result
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let rrf_score = vector_weight / (rrf_k + rank as f64 + 1.0);
            fused_scores
                .entry(id.clone())
                .and_modify(|fr| {
                    fr.vector_score = rrf_score;
                    fr.total += rrf_score;
                })
                .or_insert(FusedResult {
                    text_score: 0.0,
                    vector_score: rrf_score,
                    session_boosted: false,
                    total: rrf_score,
                    data: result.clone(),
                });
        }
    }

    // Sort by fused score
    let mut results: Vec<(String, FusedResult)> = fused_scores.into_iter().collect();
    results.sort_by(|a, b| {
        b.1.total
            .partial_cmp(&a.1.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(opts.limit as usize);

    // VF2: Testing Effect — update access tracking on matched observations
    let matched_obs_ids: Vec<uuid::Uuid> = results
        .iter()
        .filter_map(|(_, fr)| {
            fr.data
                .get("id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<uuid::Uuid>().ok())
        })
        .collect();

    if !matched_obs_ids.is_empty()
        && let Err(e) = dual_strength::on_search_match(pool, &matched_obs_ids).await
    {
        tracing::warn!(error = %e, "failed to apply Testing Effect boost");
    }

    // Session awareness: check active session and boost matching results.
    //
    // V0.7 (Mejora 5): Word-level overlap instead of substring match.
    // Previous `.contains()` caused false positives: goal "rust" matched
    // "frustrated", "entrusted", "robust". Now uses tokenized word sets
    // with proportional boost (bag-of-words model, Salton 1971).
    let session_boost = get_session_goals(pool).await.unwrap_or_default();
    if !session_boost.is_empty() {
        for (_, fr) in &mut results {
            if let Some(content) = fr.data.get("content").and_then(|v| v.as_str()) {
                let content_words: std::collections::HashSet<String> = content
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| w.len() > 1)
                    .map(String::from)
                    .collect();
                for goal in &session_boost {
                    let goal_words: std::collections::HashSet<String> = goal
                        .to_lowercase()
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|w| w.len() > 1)
                        .map(String::from)
                        .collect();
                    let overlap = content_words.intersection(&goal_words).count();
                    if overlap > 0 {
                        let match_ratio = overlap as f64 / goal_words.len().max(1) as f64;
                        fr.total *= 1.0 + 0.3 * match_ratio; // Up to 1.3x at full overlap
                        fr.session_boosted = true;
                        break;
                    }
                }
            }
        }
        // Re-sort after session boost
        results.sort_by(|a, b| {
            b.1.total
                .partial_cmp(&a.1.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // GraphRAG enrichment — add degree-1 neighbors for top-K results (V9)
    let graphrag_context = enrich_graphrag(pool, &results, GRAPHRAG_TOP_K).await;

    // V0.6: Score breakdown — include component scores in each result
    let results_json: Vec<Value> = results
        .iter()
        .map(|(_, fr)| {
            let mut r = fr.data.clone();
            if let Some(obj) = r.as_object_mut() {
                obj.insert("fused_score".to_string(), serde_json::json!(fr.total));
                obj.insert("text_score".to_string(), serde_json::json!(fr.text_score));
                obj.insert(
                    "vector_score".to_string(),
                    serde_json::json!(fr.vector_score),
                );
                obj.insert(
                    "session_boosted".to_string(),
                    serde_json::json!(fr.session_boosted),
                );
            }
            r
        })
        .collect();

    // Token budget enforcement: truncate content per result instead of dropping entire results.
    // This gives Claude more results with shorter content (better for grounding).
    // Estimate: 1 token ≈ 4 chars.
    let mut token_budget = opts.max_tokens;
    let mut budget_results: Vec<Value> = Vec::with_capacity(results_json.len());
    for mut r in results_json {
        let content_len = r
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.len() as i64 / 4)
            .unwrap_or(20);
        if token_budget <= 0 {
            break;
        }
        if content_len > token_budget {
            // Truncate content to fit remaining budget instead of dropping
            let max_chars = (token_budget * 4) as usize;
            let truncated: Option<String> = r
                .get("content")
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

    // V0.6: Compact format — abbreviated keys for ~35% token savings
    let final_results: Vec<Value> = if opts.format == "compact" {
        budget_results.iter().map(compact_result).collect()
    } else {
        budget_results
    };

    Ok(serde_json::json!({
        "mode": "hybrid",
        "query": query,
        "results": final_results,
        "count": final_results.len(),
        "graphrag_context": graphrag_context
    }))
}

/// V0.6: Transform a result into compact format with abbreviated keys.
fn compact_result(r: &Value) -> Value {
    let content = r
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| crate::handlers::zafra::safe_truncate(s, 200));
    serde_json::json!({
        "e": r.get("entity_name").or_else(|| r.get("name")),
        "c": content,
        "s": r.get("fused_score"),
        "t": r.get("type").or_else(|| r.get("observation_type")),
        "i": r.get("importance")
    })
}

/// Verify a claim against stored knowledge with source diversity scoring.
///
/// V0.7: Hybrid verification — combines trigram + embedding search (Mejora 2).
///       Previously only used pg_trgm, missing all paraphrase matches.
///       Also eliminates double trigram scan (Mejora 9): entity_name is now
///       returned from the first query instead of running a second scan.
async fn verify_claim(pool: &PgPool, claim: &str) -> Result<Value> {
    use std::collections::HashMap;

    // 1. Trigram evidence (with entity_name — Mejora 9: eliminates second scan)
    let trigram_evidence: Vec<(uuid::Uuid, String, f64, String, String)> = sqlx::query_as(
        "SELECT o.id, o.content, similarity(o.content, $1)::float8 AS sim,
                o.observation_type, e.name AS entity_name
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE similarity(o.content, $1) > 0.3
           AND o.observation_type != 'superseded'
         ORDER BY sim DESC
         LIMIT 10",
    )
    .bind(claim)
    .fetch_all(pool)
    .await?;

    // 2. Semantic evidence via embeddings (Mejora 2: catches paraphrase matches).
    //
    // Guard: only run when a real ONNX model is loaded. Hash-based fallback embeddings
    // stored in the DB are not semantically meaningful, so skipping avoids false evidence.
    //
    // Threshold: cosine distance < 0.8 (sim > 0.2). Without a threshold, top-10 nearest
    // regardless of distance pulls in unrelated observations that lower avg_sim and degrade
    // the confidence score. Distance 0.8 corresponds to sim ≈ 0.2 — a weak but real signal.
    let semantic_evidence: Vec<(uuid::Uuid, String, f64, String, String)> =
        if crate::embeddings::onnx::is_model_loaded() {
            match crate::embeddings::onnx::embed(claim).await {
                Ok(emb) => {
                    sqlx::query_as(
                        "SELECT o.id, o.content,
                                (1.0 - (o.embedding <=> $1::vector))::float8 AS sim,
                                o.observation_type, e.name AS entity_name
                         FROM brain_observations o
                         JOIN brain_entities e ON o.entity_id = e.id
                         WHERE o.embedding IS NOT NULL
                           AND o.observation_type != 'superseded'
                           AND (o.embedding <=> $1::vector) < 0.8
                         ORDER BY o.embedding <=> $1::vector
                         LIMIT 10",
                    )
                    .bind(pgvector::Vector::from(emb))
                    .fetch_all(pool)
                    .await
                    .unwrap_or_default()
                }
                Err(e) => {
                    tracing::warn!(error = %e, "embedding failed during verify — skipping semantic evidence");
                    vec![]
                }
            }
        } else {
            vec![] // V8: graceful degradation if no ONNX model
        };

    // 3. Merge by ID, take max similarity per observation (Robertson 1977 fusion)
    let mut merged: HashMap<uuid::Uuid, (String, f64, String, String)> = HashMap::new();
    for (id, content, sim, obs_type, entity_name) in
        trigram_evidence.iter().chain(semantic_evidence.iter())
    {
        merged
            .entry(*id)
            .and_modify(|(_, existing_sim, _, _)| *existing_sim = existing_sim.max(*sim))
            .or_insert((
                content.clone(),
                *sim,
                obs_type.clone(),
                entity_name.clone(),
            ));
    }

    // Sort by similarity descending
    let mut evidence_list: Vec<(uuid::Uuid, String, f64, String, String)> = merged
        .into_iter()
        .map(|(id, (content, sim, obs_type, entity_name))| {
            (id, content, sim, obs_type, entity_name)
        })
        .collect();
    evidence_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    evidence_list.truncate(10);

    let similarities: Vec<f64> = evidence_list.iter().map(|(_, _, s, _, _)| *s).collect();
    let sources: Vec<&str> = evidence_list.iter().map(|(_, _, _, t, _)| t.as_str()).collect();

    // Mejora 9: top_entity from merged results (no second query)
    let top_entity: Option<String> = evidence_list.first().map(|(_, _, _, _, e)| e.clone());

    // Count total observations for entity-relative coverage (Mejora 7 integration)
    let total_obs: Option<usize> = if let Some(ref entity_name) = top_entity {
        sqlx::query_as::<_, (i64,)>(
            "SELECT COUNT(*) FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE e.name = $1 AND o.observation_type != 'superseded'",
        )
        .bind(entity_name)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|(c,)| c as usize)
    } else {
        None
    };

    let (confidence, level) = grounding::compute_grounding(&similarities, &sources, total_obs);

    // Log verify prediction to brain_verify_log
    sqlx::query(
        "INSERT INTO brain_verify_log (claim, entity_name, confidence, grounding_level)
         VALUES ($1, $2, $3, $4)",
    )
    .bind(claim)
    .bind(top_entity.as_deref())
    .bind(confidence)
    .bind(level)
    .execute(pool)
    .await
    .ok(); // Non-fatal

    // Fetch historical calibration for this grounding level
    let calibration: Option<(f64,)> = sqlx::query_as(
        "SELECT (COUNT(*) FILTER (WHERE outcome = 'correct') + 1)::float8 /
                (COUNT(*) FILTER (WHERE outcome = 'correct') + COUNT(*) FILTER (WHERE outcome = 'incorrect') + 2)::float8
         FROM brain_verify_log
         WHERE grounding_level = $1 AND outcome != 'pending'"
    )
    .bind(level)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    let evidence_json: Vec<Value> = evidence_list
        .iter()
        .map(|(_, content, sim, obs_type, entity_name)| {
            serde_json::json!({
                "content": content,
                "similarity": sim,
                "type": obs_type,
                "entity": entity_name
            })
        })
        .collect();

    let mut response = serde_json::json!({
        "mode": "verify",
        "claim": claim,
        "confidence": confidence,
        "grounding": level,
        "evidence": evidence_json,
        "evidence_count": evidence_json.len(),
        "source_diversity": sources.len()
    });

    if let Some((cal,)) = calibration {
        response["calibrated_accuracy"] = serde_json::json!(cal);
    }

    Ok(response)
}

/// Text search using PostgreSQL full-text + trigram similarity.
/// Temporal filtering: $3=after, $4=before applied to created_at on each table.
async fn text_search(
    pool: &PgPool,
    query: &str,
    scope: &str,
    limit: i64,
    tb: &TimeBounds,
) -> Result<Vec<Value>> {
    let mut results = Vec::new();

    if scope == "all" || scope == "entities" {
        let rows: Vec<(uuid::Uuid, String, String, f64, f64)> = sqlx::query_as(
            "SELECT id, name, entity_type, importance::float8,
                    (  (ts_rank(search_vector, plainto_tsquery('simple', $1))
                      + similarity(name, $1)) * 0.7
                     + importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_entities
             WHERE (search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(name, $1) > 0.3)
               AND created_at >= $3 AND created_at <= $4
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .fetch_all(pool)
        .await?;
        results.extend(
            rows.into_iter()
                .map(|(id, name, entity_type, importance, score)| {
                    serde_json::json!({
                        "id": id.to_string(),
                        "type": "entity",
                        "name": name,
                        "entity_type": entity_type,
                        "importance": importance,
                        "score": score
                    })
                }),
        );
    }

    if scope == "all" || scope == "observations" {
        let rows: Vec<(uuid::Uuid, String, String, String, f64, f64)> = sqlx::query_as(
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
               AND o.created_at >= $3 AND o.created_at <= $4
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .fetch_all(pool)
        .await?;
        results.extend(rows.into_iter().map(
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
        ));
    }

    if scope == "all" || scope == "errors" {
        let rows: Vec<(uuid::Uuid, String, String, bool, f64)> = sqlx::query_as(
            "SELECT id, error_type, error_message, resolved,
                    (ts_rank(search_vector, plainto_tsquery('simple', $1)) +
                    similarity(error_message, $1))::float8 AS score
             FROM brain_errors
             WHERE (search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(error_message, $1) > 0.3)
               AND created_at >= $3 AND created_at <= $4
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .fetch_all(pool)
        .await?;
        results.extend(
            rows.into_iter()
                .map(|(id, error_type, error_message, resolved, score)| {
                    serde_json::json!({
                        "id": id.to_string(),
                        "type": "error",
                        "error_type": error_type,
                        "error_message": error_message,
                        "resolved": resolved,
                        "score": score
                    })
                }),
        );
    }

    // Search episodes (always included in "all" scope — separate memory system)
    if scope == "all" {
        let rows: Vec<(uuid::Uuid, String, String, f64, f64)> = sqlx::query_as(
            "SELECT ep.id, e.name, ep.content, ep.importance::float8,
                    (  (ts_rank(ep.search_vector, plainto_tsquery('simple', $1))
                      + similarity(ep.content, $1)) * 0.7
                     + ep.importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_episodes ep
             JOIN brain_entities e ON ep.entity_id = e.id
             WHERE (ep.search_vector @@ plainto_tsquery('simple', $1)
                OR similarity(ep.content, $1) > 0.3)
               AND ep.created_at >= $3 AND ep.created_at <= $4
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .fetch_all(pool)
        .await
        .unwrap_or_default(); // Non-fatal if table doesn't exist
        results.extend(
            rows.into_iter()
                .map(|(id, entity_name, content, importance, score)| {
                    serde_json::json!({
                        "id": id.to_string(),
                        "type": "episode",
                        "entity_name": entity_name,
                        "content": content,
                        "importance": importance,
                        "score": score
                    })
                }),
        );
    }

    Ok(results)
}

/// Vector search via pgvector cosine similarity.
/// V8: Returns Result — callers can gracefully degrade on failure.
async fn vector_search(
    pool: &PgPool,
    query: &str,
    _scope: &str,
    limit: i64,
    tb: &TimeBounds,
) -> Result<Vec<Value>> {
    // Gate on real ONNX model — hash fallback embeddings are not semantically
    // meaningful and querying DB with them against NULL embeddings returns empty
    // anyway. Early return avoids the semaphore acquisition and embed call.
    if !crate::embeddings::onnx::is_model_loaded() {
        return Ok(vec![]); // V8: graceful degradation
    }

    let embedding = match crate::embeddings::onnx::embed(query).await {
        Ok(emb) => emb,
        Err(e) => {
            tracing::warn!(error = %e, "ONNX embed failed in vector_search — degrading");
            return Ok(vec![]);
        }
    };

    // Vector search with temporal filtering on observations
    let observations: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as(
        "SELECT o.id, e.name, o.content,
                1.0 - (o.embedding <=> $1::vector) AS cosine_sim
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE o.embedding IS NOT NULL
           AND o.observation_type != 'superseded'
           AND o.created_at >= $3 AND o.created_at <= $4
         ORDER BY o.embedding <=> $1::vector
         LIMIT $2",
    )
    .bind(pgvector::Vector::from(embedding.clone()))
    .bind(limit)
    .bind(tb.after)
    .bind(tb.before)
    .fetch_all(pool)
    .await?;

    let mut results: Vec<Value> = observations
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

    // Also search episodes with temporal filtering
    let episodes: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as(
        "SELECT ep.id, e.name, ep.content,
                1.0 - (ep.embedding <=> $1::vector) AS cosine_sim
         FROM brain_episodes ep
         JOIN brain_entities e ON ep.entity_id = e.id
         WHERE ep.embedding IS NOT NULL
           AND ep.created_at >= $3 AND ep.created_at <= $4
         ORDER BY ep.embedding <=> $1::vector
         LIMIT $2",
    )
    .bind(pgvector::Vector::from(embedding))
    .bind(limit)
    .bind(tb.after)
    .bind(tb.before)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    results.extend(episodes.iter().map(|(id, entity_name, content, sim)| {
        serde_json::json!({
            "id": id.to_string(),
            "type": "episode",
            "entity_name": entity_name,
            "content": content,
            "cosine_similarity": sim
        })
    }));

    Ok(results)
}

// ── Utility Functions ───────────────────────────────────────────

/// §A V3: Smooth sigmoid entropy routing (Mejora 4 — Jaynes 1957).
///
/// Replaces V2 step function which had 40% relative jumps at thresholds 2.0/3.5.
/// The logistic sigmoid is the maximum-entropy smooth monotone transition
/// between two asymptotes (text-heavy → vector-heavy).
///
/// | Entropy | text_w | vector_w |
/// |---------|--------|----------|
/// | 0.0     | ~0.70  | ~0.30    |
/// | 2.75    | 0.50   | 0.50     | (midpoint)
/// | 5.0+    | ~0.30  | ~0.70    |
fn entropy_weights(entropy: f64) -> (f64, f64) {
    let midpoint = 2.75; // Center of original [2.0, 3.5] range
    let k = 2.0; // Steepness: 10%-90% transition spans ~2.2 units
    let t = 1.0 / (1.0 + (-k * (entropy - midpoint)).exp()); // sigmoid ∈ [0, 1]
    let text_w = 0.7 - 0.4 * t; // 0.7 → 0.3
    let vector_w = 0.3 + 0.4 * t; // 0.3 → 0.7
    (text_w, vector_w)
}

// V3 adaptive_rrf_k REMOVED — k=60 constant per Gemini Deep Research audit 2026-03-14.
// Rationale: dynamic sqrt-based k introduced non-monotonic ranking instabilities
// and violated determinism. Cormack et al. 2009, Azure AI Search, Elasticsearch
// all converge on k=60 as empirically optimal.

/// Get active session goals for session-aware boosting.
async fn get_session_goals(pool: &PgPool) -> Result<Vec<String>> {
    let row: Option<(serde_json::Value,)> = sqlx::query_as(
        "SELECT goals FROM brain_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
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
    results: &[(String, FusedResult)],
    top_k: usize,
) -> Value {
    let mut context: Vec<Value> = Vec::new();

    for (_, fr) in results.iter().take(top_k) {
        let entity_name = fr
            .data
            .get("entity_name")
            .or_else(|| fr.data.get("name"))
            .and_then(|v: &Value| v.as_str());

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
                 LIMIT 5",
            )
            .bind(name)
            .fetch_all(pool)
            .await
            {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// entropy_weights must always sum to 1.0 regardless of input.
    #[test]
    fn test_entropy_weights_sum_to_one() {
        for &e in &[0.0f64, 1.0, 2.0, 2.75, 3.5, 5.0, 10.0] {
            let (tw, vw) = entropy_weights(e);
            assert!(
                (tw + vw - 1.0).abs() < 1e-12,
                "weights must sum to 1.0 at entropy={e}: got {tw}+{vw}={:.15}",
                tw + vw
            );
        }
    }

    /// text_w must be strictly decreasing and vector_w strictly increasing
    /// — smooth sigmoid replaces the V2 step function with monotone output.
    #[test]
    fn test_entropy_weights_monotone() {
        let entropies = [0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0];
        let weights: Vec<(f64, f64)> = entropies.iter().map(|&e| entropy_weights(e)).collect();
        for i in 1..weights.len() {
            let (tw_prev, vw_prev) = weights[i - 1];
            let (tw_curr, vw_curr) = weights[i];
            assert!(
                tw_curr < tw_prev,
                "text_w must decrease: at e={} got {tw_curr} >= prev {tw_prev}",
                entropies[i]
            );
            assert!(
                vw_curr > vw_prev,
                "vector_w must increase: at e={} got {vw_curr} <= prev {vw_prev}",
                entropies[i]
            );
        }
    }

    /// Verify asymptotes: low entropy → text-heavy, high entropy → vector-heavy.
    #[test]
    fn test_entropy_weights_asymptotes() {
        let (tw_low, vw_low) = entropy_weights(0.0);
        assert!(tw_low > 0.68, "low entropy should be text-heavy: got text_w={tw_low}");
        assert!(vw_low < 0.32, "low entropy should minimize vector_w: got {vw_low}");

        let (tw_high, vw_high) = entropy_weights(10.0);
        assert!(tw_high < 0.32, "high entropy should minimize text_w: got {tw_high}");
        assert!(vw_high > 0.68, "high entropy should be vector-heavy: got {vw_high}");

        // Midpoint (entropy = 2.75): exact 50/50 split
        let (tw_mid, vw_mid) = entropy_weights(2.75);
        assert!((tw_mid - 0.5).abs() < 1e-10, "midpoint should give text_w=0.5: got {tw_mid}");
        assert!((vw_mid - 0.5).abs() < 1e-10, "midpoint should give vector_w=0.5: got {vw_mid}");
    }

    /// V2 step function had a 40% jump at entropy=2.0. Verify V3 sigmoid
    /// transition is smooth: Δweight < 0.05 per 0.1 entropy unit around threshold.
    #[test]
    fn test_entropy_weights_no_discontinuity() {
        // Check around the old V2 thresholds (2.0 and 3.5)
        for &threshold in &[2.0f64, 3.5] {
            let (tw_before, _) = entropy_weights(threshold - 0.1);
            let (tw_after, _) = entropy_weights(threshold + 0.1);
            let jump = (tw_before - tw_after).abs();
            assert!(
                jump < 0.05,
                "sigmoid should be smooth: jump of {jump:.4} at threshold {threshold} (V2 had ~0.20)"
            );
        }
    }
}
