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

#[derive(Clone)]
struct FusedResult {
    text_score: f64,
    vector_score: f64,
    bm25_score: f64,
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

    let format = args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("compact");

    let tag_filter = args.get("tags").and_then(|v| v.as_str());

    let track_access = args
        .get("track_access")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let associative = args
        .get("associative")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let before = args.get("before").and_then(|v| v.as_str());
    let after = args.get("after").and_then(|v| v.as_str());
    let time_bounds = parse_time_bounds(before, after);

    let diversify = args
        .get("diversify")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mmr_lambda = args
        .get("mmr_lambda")
        .and_then(|v| v.as_f64())
        .unwrap_or(crate::search::mmr::DEFAULT_LAMBDA);

    let abstain_ood = args
        .get("abstain_ood")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let ood_threshold = args.get("ood_threshold").and_then(|v| v.as_f64());

    let enable_bm25 = args
        .get("enable_bm25")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let enable_rerank = args
        .get("rerank")
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| {
            crate::mode::active().rerank_default() && crate::search::rerank::enabled()
        });

    let project_id = crate::project::current_project_id(pool).await?;

    if abstain_ood
        && mode == "hybrid"
        && let Some(answer) = check_ood(pool, query, ood_threshold, project_id).await
    {
        return Ok(answer);
    }

    let search_opts = SearchOpts {
        scope,
        limit,
        max_tokens,
        time_bounds,
        format,
        tag_filter,
        project_id,
        diversify,
        mmr_lambda,
        enable_bm25,
        enable_rerank,
        track_access,
        associative,
    };

    match mode {
        "hybrid" => hybrid_search(pool, query, &search_opts).await,
        "verify" => verify_claim(pool, query, project_id).await,
        _ => anyhow::bail!("Invalid mode: {mode}. Use hybrid/verify"),
    }
}

struct TimeBounds {
    after: chrono::DateTime<chrono::Utc>,
    before: chrono::DateTime<chrono::Utc>,
}

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

struct SearchOpts<'a> {
    scope: &'a str,
    limit: i64,
    max_tokens: i64,
    time_bounds: TimeBounds,
    format: &'a str,
    tag_filter: Option<&'a str>,
    project_id: Option<uuid::Uuid>,
    diversify: bool,
    mmr_lambda: f64,
    enable_bm25: bool,
    enable_rerank: bool,
    track_access: bool,
    associative: bool,
}

async fn hybrid_search(pool: &PgPool, query: &str, opts: &SearchOpts<'_>) -> Result<Value> {
    let query_entropy = crate::search::rrf::query_entropy(query);
    let (text_weight, vector_weight) = entropy_weights(query_entropy);
    let bm25_weight = text_weight;

    let mut text_results = text_search(
        pool,
        query,
        opts.scope,
        opts.limit * 2,
        &opts.time_bounds,
        opts.project_id,
    )
    .await?;

    if let Some(tag) = opts.tag_filter {
        let tagged_obs: Vec<(uuid::Uuid, String, String, String, f64, f64)> = sqlx::query_as(
            "SELECT o.id, e.name, o.content, o.observation_type, o.importance::float8,
                    (o.importance::float8 * 0.8 + 0.2)::float8 AS score
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE $1 = ANY(o.tags)
               AND o.observation_type != 'superseded'
               AND ($3::uuid IS NULL OR o.project_id = $3 OR o.project_id IS NULL)
             ORDER BY o.importance DESC
             LIMIT $2",
        )
        .bind(tag)
        .bind(opts.limit)
        .bind(opts.project_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        for (id, entity_name, content, obs_type, importance, score) in tagged_obs {
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

    let vector_results = vector_search(
        pool,
        query,
        opts.scope,
        opts.limit * 2,
        &opts.time_bounds,
        opts.project_id,
    )
    .await;

    let rrf_k = crate::search::rrf::RRF_K;

    let mut fused_scores: HashMap<String, FusedResult> = HashMap::new();

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
                bm25_score: 0.0,
                session_boosted: false,
                total: rrf_score,
                data: result.clone(),
            },
        );
    }

    let vector_failed = match &vector_results {
        Ok(_) => false,
        Err(e) => {
            tracing::error!(
                error = %e,
                "VECTOR SEARCH FAILED — hybrid retrieval degraded to lexical only. \
                 Run `cuba-memorys doctor`: this is usually the embedding dimension \
                 disagreeing with the column."
            );
            true
        }
    };

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
                    bm25_score: 0.0,
                    session_boosted: false,
                    total: rrf_score,
                    data: result.clone(),
                });
        }
    }

    if opts.enable_bm25 {
        let bm25_results = crate::search::bm25::bm25_search(
            pool,
            query,
            opts.scope,
            opts.limit * 2,
            opts.project_id,
        )
        .await
        .unwrap_or_default();
        for (rank, result) in bm25_results.iter().enumerate() {
            let id = result
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let rrf_score = bm25_weight / (rrf_k + rank as f64 + 1.0);
            fused_scores
                .entry(id.clone())
                .and_modify(|fr| {
                    fr.bm25_score = rrf_score;
                    fr.total += rrf_score;
                })
                .or_insert(FusedResult {
                    text_score: 0.0,
                    vector_score: 0.0,
                    bm25_score: rrf_score,
                    session_boosted: false,
                    total: rrf_score,
                    data: result.clone(),
                });
        }
    }

    if opts.associative {
        associative_expand(pool, query, opts.project_id, &mut fused_scores).await;
    }

    let mut results: Vec<(String, FusedResult)> = fused_scores.into_iter().collect();
    results.sort_by(|a, b| {
        b.1.total
            .partial_cmp(&a.1.total)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    const FUSION_DEDUP: f64 = 0.85;
    let mut deduped: Vec<(String, FusedResult)> = Vec::with_capacity(results.len());
    for (id, fr) in results {
        let content = fr
            .data
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let duplicate = deduped.iter().any(|(_, existing)| {
            let other = existing
                .data
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            crate::search::rrf::content_overlap(content, other) > FUSION_DEDUP
        });
        if !duplicate {
            deduped.push((id, fr));
        }
    }
    results = deduped;

    let pool_size = if opts.enable_rerank {
        50.min(results.len())
    } else if opts.diversify {
        (opts.limit as usize * 5).min(results.len())
    } else {
        (opts.limit as usize).min(results.len())
    };
    results.truncate(pool_size);

    let mut reranker_failed = false;
    if opts.enable_rerank && results.len() > 1 {
        let contents: Vec<&str> = results
            .iter()
            .map(|(_, fr)| {
                fr.data
                    .get("content")
                    .or_else(|| fr.data.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            })
            .collect();
        let have_model = crate::search::rerank::enabled();

        let rerank_budget = std::time::Duration::from_secs(
            std::env::var("CUBA_RERANK_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(20),
        );
        let rerank_result = match tokio::time::timeout(
            rerank_budget,
            crate::search::rerank::rerank(query, &contents),
        )
        .await
        {
            Ok(inner) => inner,
            Err(_) => {
                tracing::warn!(
                    secs = rerank_budget.as_secs(),
                    "reranker excedió su presupuesto — se devuelve el ranking RRF"
                );
                Err(anyhow::anyhow!("reranker timeout"))
            }
        };
        match rerank_result {
            Ok(reranked) => {
                let original = results.clone();
                results = reranked
                    .into_iter()
                    .filter_map(|(idx, score)| {
                        let mut entry = original.get(idx).cloned()?;
                        if have_model {
                            entry.1.total = score;
                        } else {
                            entry.1.total += score * 0.0001;
                        }
                        Some(entry)
                    })
                    .collect();
            }
            Err(e) => {
                tracing::error!(
                    error = %format!("{e:#}"),
                    "RERANK FAILED — el ranking se devuelve SIN reordenar. El modelo se \
                     cargó y se gastó el tiempo de inferencia, pero sus scores se \
                     descartaron: los resultados son los de RRF."
                );
                reranker_failed = true;
            }
        }
    }

    let matched_obs_ids: Vec<uuid::Uuid> = results
        .iter()
        .filter_map(|(_, fr)| {
            fr.data
                .get("id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<uuid::Uuid>().ok())
        })
        .collect();

    if opts.track_access
        && !matched_obs_ids.is_empty()
        && let Err(e) = dual_strength::on_search_match(pool, &matched_obs_ids).await
    {
        tracing::warn!(error = %e, "failed to apply Testing Effect boost");
    }

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
                        fr.total *= 1.0 + 0.3 * match_ratio;
                        fr.session_boosted = true;
                        break;
                    }
                }
            }
        }
        results.sort_by(|a, b| {
            b.1.total
                .partial_cmp(&a.1.total)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
    }

    if opts.diversify && results.len() > 1 {
        let n = results.len();
        let relevance: Vec<f64> = results.iter().map(|(_, fr)| fr.total).collect();
        let token_sets: Vec<std::collections::HashSet<String>> = results
            .iter()
            .map(|(_, fr)| {
                fr.data
                    .get("content")
                    .or_else(|| fr.data.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        s.to_lowercase()
                            .split(|c: char| !c.is_alphanumeric())
                            .filter(|w| w.len() > 2)
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
        let mut pairwise = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            pairwise[i][i] = 1.0;
            for j in (i + 1)..n {
                let inter = token_sets[i].intersection(&token_sets[j]).count();
                let union = token_sets[i].union(&token_sets[j]).count();
                let jaccard = if union == 0 {
                    0.0
                } else {
                    inter as f64 / union as f64
                };
                pairwise[i][j] = jaccard;
                pairwise[j][i] = jaccard;
            }
        }
        let picks = crate::search::mmr::mmr_select(
            &relevance,
            &pairwise,
            opts.mmr_lambda,
            opts.limit as usize,
        );
        let reordered: Vec<(String, FusedResult)> = picks
            .into_iter()
            .filter_map(|i| results.get(i).cloned())
            .collect();
        results = reordered;
    } else {
        results.truncate(opts.limit as usize);
    }

    let graphrag_context = enrich_graphrag(pool, &results, GRAPHRAG_TOP_K).await;

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
                obj.insert("bm25_score".to_string(), serde_json::json!(fr.bm25_score));
                obj.insert(
                    "session_boosted".to_string(),
                    serde_json::json!(fr.session_boosted),
                );
            }
            r
        })
        .collect();

    use crate::search::budget::{count_tokens, truncate_to_budget};

    let shaped: Vec<Value> = if opts.format == "compact" {
        results_json.iter().map(compact_result).collect()
    } else {
        results_json
    };
    let text_key = if opts.format == "compact" {
        "c"
    } else {
        "content"
    };

    let mut token_budget = opts.max_tokens;
    let mut final_results: Vec<Value> = Vec::with_capacity(shaped.len());
    for mut r in shaped {
        if token_budget <= 0 {
            break;
        }
        let row_tokens = count_tokens(&r.to_string()) as i64;
        if row_tokens > token_budget {
            let truncated: Option<String> = r
                .get(text_key)
                .and_then(|v| v.as_str())
                .map(|s| truncate_to_budget(s, token_budget.max(0) as usize));
            if let (Some(obj), Some(t)) = (r.as_object_mut(), truncated) {
                obj.insert(text_key.to_string(), serde_json::json!(t));
            }
            final_results.push(r);
            break;
        }
        token_budget -= row_tokens;
        final_results.push(r);
    }

    let mut response = serde_json::json!({
        "mode": "hybrid",
        "query": query,
        "results": final_results,
        "count": final_results.len(),
        "graphrag_context": graphrag_context
    });

    if vector_failed {
        response["degraded"] = serde_json::json!(true);
        response["degraded_reason"] = serde_json::json!(
            "La búsqueda vectorial falló: estos resultados son SOLO léxicos y el recall \
             está degradado. Causa habitual: la dimensión del embedding no coincide con la \
             de la columna. Diagnosticá con `cuba-memorys doctor`."
        );
    }

    if reranker_failed {
        response["reranker_degraded"] = serde_json::json!(true);
        response["reranker_degraded_reason"] = serde_json::json!(
            "Pediste rerank y el cross-encoder falló: estos resultados vienen SIN reordenar, \
             tal cual los dejó RRF. Se pagó el tiempo de inferencia y no se aplicó nada. \
             Mirá los logs (nivel ERROR) para la causa."
        );
    }

    Ok(response)
}

fn compact_chars() -> usize {
    std::env::var("CUBA_COMPACT_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1200)
}

fn compact_result(r: &Value) -> Value {
    let content = r
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| crate::handlers::zafra::safe_truncate(s, compact_chars()));
    serde_json::json!({
        "id": r.get("id"),
        "e": r.get("entity_name").or_else(|| r.get("name")),
        "c": content,
        "s": r.get("fused_score"),
        "t": r.get("type").or_else(|| r.get("observation_type")),
        "i": r.get("importance")
    })
}

async fn verify_claim(pool: &PgPool, claim: &str, project_id: Option<uuid::Uuid>) -> Result<Value> {
    use std::collections::HashMap;

    sqlx::query("SET LOCAL hnsw.ef_search = 200")
        .execute(pool)
        .await
        .ok();

    let trigram_evidence: Vec<(uuid::Uuid, String, f64, String, String)> = sqlx::query_as(
        "SELECT o.id, o.content, similarity(o.content, $1)::float8 AS sim,
                o.observation_type, e.name AS entity_name
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE similarity(o.content, $1) > 0.3
           AND o.observation_type != 'superseded'
           AND ($2::uuid IS NULL OR o.project_id = $2 OR o.project_id IS NULL)
         ORDER BY sim DESC
         LIMIT 10",
    )
    .bind(claim)
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let semantic_evidence: Vec<(uuid::Uuid, String, f64, String, String)> =
        if crate::embeddings::onnx::is_model_loaded() {
            match crate::embeddings::onnx::embed(claim).await {
                Ok(emb) => sqlx::query_as(
                    "SELECT o.id, o.content,
                                (1.0 - (o.embedding <=> $1::vector))::float8 AS sim,
                                o.observation_type, e.name AS entity_name
                         FROM brain_observations o
                         JOIN brain_entities e ON o.entity_id = e.id
                         WHERE o.embedding IS NOT NULL
                           AND o.observation_type != 'superseded'
                           AND (o.embedding <=> $1::vector) < 0.8
                           AND ($2::uuid IS NULL OR o.project_id = $2 OR o.project_id IS NULL)
                         ORDER BY o.embedding <=> $1::vector
                         LIMIT 10",
                )
                .bind(pgvector::Vector::from(emb))
                .bind(project_id)
                .fetch_all(pool)
                .await
                .unwrap_or_default(),
                Err(e) => {
                    tracing::warn!(error = %e, "embedding failed during verify — skipping semantic evidence");
                    vec![]
                }
            }
        } else {
            vec![]
        };

    let mut merged: HashMap<uuid::Uuid, (String, f64, String, String)> = HashMap::new();
    for (id, content, sim, obs_type, entity_name) in
        trigram_evidence.iter().chain(semantic_evidence.iter())
    {
        merged
            .entry(*id)
            .and_modify(|(_, existing_sim, _, _)| *existing_sim = existing_sim.max(*sim))
            .or_insert((content.clone(), *sim, obs_type.clone(), entity_name.clone()));
    }

    let mut evidence_list: Vec<(uuid::Uuid, String, f64, String, String)> = merged
        .into_iter()
        .map(|(id, (content, sim, obs_type, entity_name))| {
            (id, content, sim, obs_type, entity_name)
        })
        .collect();
    evidence_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    evidence_list.retain(|(_, _, sim, _, _)| *sim >= grounding::MIN_EVIDENCE_SIMILARITY);
    evidence_list.truncate(10);

    let top_entity: Option<String> = evidence_list.first().map(|(_, _, _, _, e)| e.clone());

    let judge = crate::cognitive::judge::resolve_judge();
    let max_judged = crate::cognitive::judge::default_max_pairs();
    let to_judge: Vec<(usize, &String, f64)> = evidence_list
        .iter()
        .take(max_judged)
        .enumerate()
        .map(|(i, (_, content, sim, _, _))| (i, content, *sim))
        .collect();

    let judgments = futures::future::join_all(to_judge.iter().map(|(i, content, sim)| {
        let judge = &judge;
        async move { (*i, *sim, judge.judge_claim(claim, content).await) }
    }))
    .await;

    let mut judged: Vec<grounding::JudgedEvidence> = Vec::with_capacity(judgments.len());
    let mut verdicts: HashMap<usize, Value> = HashMap::with_capacity(judgments.len());

    for (i, sim, result) in judgments {
        let judgment = match result {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    backend = judge.backend_name(),
                    "judge unavailable for this evidence — it will not count as support"
                );
                continue;
            }
        };
        verdicts.insert(
            i,
            serde_json::json!({
                "verdict": judgment.verdict,
                "reason": judgment.reason,
            }),
        );
        judged.push(grounding::JudgedEvidence {
            similarity: sim,
            verdict: judgment.verdict,
            judge_confidence: judgment.confidence,
        });
    }

    let (confidence, level) = grounding::compute_grounding_judged(&judged);

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
    .ok();

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
        .enumerate()
        .map(|(i, (_, content, sim, obs_type, entity_name))| {
            let mut e = serde_json::json!({
                "content": content,
                "similarity": sim,
                "type": obs_type,
                "entity": entity_name
            });
            if let Some(v) = verdicts.get(&i) {
                e["verdict"] = v["verdict"].clone();
                e["reason"] = v["reason"].clone();
            }
            e
        })
        .collect();

    let interpretation = match level {
        "contradicted" => "The stored evidence CONTRADICTS this claim.",
        "verified" => "The stored evidence supports this claim.",
        "partial" | "weak" => "Partial support — some evidence backs this claim, not decisively.",
        _ if judged.is_empty() && evidence_list.is_empty() => {
            "Nothing in memory relates to this claim. No grounding — treat as unverified."
        }
        _ => {
            "Memory holds related material, but none of it asserts this claim. \
             No grounding — being on-topic is not support."
        }
    };

    let mut response = serde_json::json!({
        "mode": "verify",
        "claim": claim,
        "confidence": confidence,
        "grounding": level,
        "interpretation": interpretation,
        "evidence": evidence_json,
        "evidence_count": evidence_json.len(),
        "judged_by": judge.backend_name(),
        "verdicts_reached": judged.len()
    });

    if let Some((cal,)) = calibration {
        response["calibrated_accuracy"] = serde_json::json!(cal);
    }

    Ok(response)
}

async fn text_search(
    pool: &PgPool,
    query: &str,
    scope: &str,
    limit: i64,
    tb: &TimeBounds,
    project_id: Option<uuid::Uuid>,
) -> Result<Vec<Value>> {
    let mut results = Vec::new();

    if scope == "all" || scope == "entities" {
        let rows: Vec<(uuid::Uuid, String, String, f64, f64)> = sqlx::query_as(
            "SELECT id, name, entity_type, importance::float8,
                    (  (ts_rank(search_vector, cuba_or_tsquery($1))
                      + similarity(name, $1)) * 0.7
                     + importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_entities
             WHERE (search_vector @@ cuba_or_tsquery($1)
                OR similarity(name, $1) > 0.3)
               AND created_at >= $3 AND created_at <= $4
               AND ($5::uuid IS NULL OR project_id = $5 OR project_id IS NULL)
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .bind(project_id)
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
                    (  (ts_rank(o.search_vector, cuba_or_tsquery($1))
                      + similarity(o.content, $1)) * 0.7
                     + o.importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE (o.search_vector @@ cuba_or_tsquery($1)
                OR similarity(o.content, $1) > 0.3)
               AND o.observation_type != 'superseded'
               AND o.created_at >= $3 AND o.created_at <= $4
               AND ($5::uuid IS NULL OR o.project_id = $5 OR o.project_id IS NULL)
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .bind(project_id)
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
                    (ts_rank(search_vector, cuba_or_tsquery($1)) +
                    similarity(error_message, $1))::float8 AS score
             FROM brain_errors
             WHERE (search_vector @@ cuba_or_tsquery($1)
                OR similarity(error_message, $1) > 0.3)
               AND created_at >= $3 AND created_at <= $4
               AND ($5::uuid IS NULL OR project_id = $5 OR project_id IS NULL)
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .bind(project_id)
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

    if scope == "all" {
        let rows: Vec<(uuid::Uuid, String, String, f64, f64)> = sqlx::query_as(
            "SELECT ep.id, e.name, ep.content, ep.importance::float8,
                    (  (ts_rank(ep.search_vector, cuba_or_tsquery($1))
                      + similarity(ep.content, $1)) * 0.7
                     + ep.importance::float8 * 0.3
                    )::float8 AS score
             FROM brain_episodes ep
             JOIN brain_entities e ON ep.entity_id = e.id
             WHERE (ep.search_vector @@ cuba_or_tsquery($1)
                OR similarity(ep.content, $1) > 0.3)
               AND ep.created_at >= $3 AND ep.created_at <= $4
               AND ($5::uuid IS NULL OR ep.project_id = $5 OR ep.project_id IS NULL)
             ORDER BY score DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(tb.after)
        .bind(tb.before)
        .bind(project_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
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

async fn vector_search(
    pool: &PgPool,
    query: &str,
    _scope: &str,
    limit: i64,
    tb: &TimeBounds,
    project_id: Option<uuid::Uuid>,
) -> Result<Vec<Value>> {
    if !crate::embeddings::onnx::is_model_loaded() {
        return Ok(vec![]);
    }

    let embedding = match crate::embeddings::onnx::embed(query).await {
        Ok(emb) => emb,
        Err(e) => {
            tracing::warn!(error = %e, "ONNX embed failed in vector_search — degrading");
            return Ok(vec![]);
        }
    };

    let observations: Vec<(uuid::Uuid, String, String, f64, f64)> = sqlx::query_as(
        "SELECT o.id, e.name, o.content, o.importance::float8,
                1.0 - (o.embedding <=> $1::vector) AS cosine_sim
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE o.embedding IS NOT NULL
           AND o.observation_type != 'superseded'
           AND o.created_at >= $3 AND o.created_at <= $4
           AND ($5::uuid IS NULL OR o.project_id = $5 OR o.project_id IS NULL)
         ORDER BY o.embedding <=> $1::vector
         LIMIT $2",
    )
    .bind(pgvector::Vector::from(embedding.clone()))
    .bind(limit)
    .bind(tb.after)
    .bind(tb.before)
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let mut results: Vec<Value> = observations
        .iter()
        .map(|(id, entity_name, content, importance, sim)| {
            serde_json::json!({
                "id": id.to_string(),
                "type": "observation",
                "entity_name": entity_name,
                "content": content,
                "importance": importance,
                "cosine_similarity": sim
            })
        })
        .collect();

    let episodes: Vec<(uuid::Uuid, String, String, f64)> = sqlx::query_as(
        "SELECT ep.id, e.name, ep.content,
                1.0 - (ep.embedding <=> $1::vector) AS cosine_sim
         FROM brain_episodes ep
         JOIN brain_entities e ON ep.entity_id = e.id
         WHERE ep.embedding IS NOT NULL
           AND ep.created_at >= $3 AND ep.created_at <= $4
           AND ($5::uuid IS NULL OR ep.project_id = $5 OR ep.project_id IS NULL)
         ORDER BY ep.embedding <=> $1::vector
         LIMIT $2",
    )
    .bind(pgvector::Vector::from(embedding))
    .bind(limit)
    .bind(tb.after)
    .bind(tb.before)
    .bind(project_id)
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

fn entropy_weights(entropy: f64) -> (f64, f64) {
    let midpoint = 2.75;
    let k = 2.0;
    let t = 1.0 / (1.0 + (-k * (entropy - midpoint)).exp());
    let text_w = 0.7 - 0.4 * t;
    let vector_w = 0.3 + 0.4 * t;
    (text_w, vector_w)
}

async fn associative_expand(
    pool: &PgPool,
    query: &str,
    project_id: Option<uuid::Uuid>,
    fused: &mut HashMap<String, FusedResult>,
) {
    const SEED_K: i64 = 5;
    const EXPAND_ENTITIES: usize = 8;
    const OBS_PER_ENTITY: i64 = 2;
    const MAX_HOPS: usize = 2;
    const ASSOC_WEIGHT: f64 = 0.02;

    let seeds: Vec<(uuid::Uuid,)> = match sqlx::query_as(
        "SELECT id FROM brain_entities
         WHERE (search_vector @@ cuba_or_tsquery($1) OR similarity(name, $1) > 0.3)
           AND ($2::uuid IS NULL OR project_id = $2 OR project_id IS NULL)
         ORDER BY importance DESC
         LIMIT $3",
    )
    .bind(query)
    .bind(project_id)
    .bind(SEED_K)
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!(error = %e, "associative: seed query failed — skipping");
            return;
        }
    };
    if seeds.is_empty() {
        return;
    }
    let seed_ids: Vec<uuid::Uuid> = seeds.iter().map(|(id,)| *id).collect();
    let seed_set: std::collections::HashSet<uuid::Uuid> = seed_ids.iter().copied().collect();

    let activated =
        match crate::graph::activation::spread_from_entities(pool, &seed_ids, MAX_HOPS).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(error = %e, "associative: activation spread failed — skipping");
                return;
            }
        };

    for (entity_id, activation) in activated
        .into_iter()
        .filter(|(id, _)| !seed_set.contains(id))
        .take(EXPAND_ENTITIES)
    {
        let obs: Vec<(uuid::Uuid, String, String, String, f64)> = match sqlx::query_as(
            "SELECT o.id, e.name, o.content, o.observation_type, o.importance::float8
             FROM brain_observations o
             JOIN brain_entities e ON e.id = o.entity_id
             WHERE o.entity_id = $1
               AND o.observation_type != 'superseded'
               AND ($2::uuid IS NULL OR o.project_id = $2 OR o.project_id IS NULL)
             ORDER BY o.importance DESC
             LIMIT $3",
        )
        .bind(entity_id)
        .bind(project_id)
        .bind(OBS_PER_ENTITY)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                tracing::warn!(error = %e, "associative: obs fetch failed — skipping entity");
                continue;
            }
        };

        for (id, entity_name, content, obs_type, importance) in obs {
            let id_str = id.to_string();
            if fused.contains_key(&id_str) {
                continue;
            }
            let assoc_score = ASSOC_WEIGHT * activation as f64 * importance;
            fused.insert(
                id_str.clone(),
                FusedResult {
                    text_score: 0.0,
                    vector_score: 0.0,
                    bm25_score: 0.0,
                    session_boosted: false,
                    total: assoc_score,
                    data: serde_json::json!({
                        "id": id_str,
                        "type": "observation",
                        "entity_name": entity_name,
                        "content": content,
                        "observation_type": obs_type,
                        "importance": importance,
                        "score": assoc_score,
                        "associative": true
                    }),
                },
            );
        }
    }
}

async fn get_session_goals(pool: &PgPool) -> Result<Vec<String>> {
    let Some(sid) = crate::session::session_id() else {
        return Ok(Vec::new());
    };
    let row: Option<(serde_json::Value,)> =
        sqlx::query_as("SELECT goals FROM brain_sessions WHERE id = $1 AND ended_at IS NULL")
            .bind(sid)
            .fetch_optional(pool)
            .await?;

    if let Some((goals,)) = row {
        let goals: Vec<String> = serde_json::from_value(goals).unwrap_or_default();
        Ok(goals)
    } else {
        Ok(vec![])
    }
}

async fn enrich_graphrag(pool: &PgPool, results: &[(String, FusedResult)], top_k: usize) -> Value {
    let mut context: Vec<Value> = Vec::new();

    for (_, fr) in results.iter().take(top_k) {
        let entity_name = fr
            .data
            .get("entity_name")
            .or_else(|| fr.data.get("name"))
            .and_then(|v: &Value| v.as_str());

        if let Some(name) = entity_name {
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

fn env_threshold() -> Option<f64> {
    std::env::var("CUBA_OOD_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|t| t.is_finite() && *t > 0.0)
}

async fn calibrated_threshold(pool: &PgPool, dim: usize) -> Option<f64> {
    static CACHE: tokio::sync::OnceCell<Option<f64>> = tokio::sync::OnceCell::const_new();
    *CACHE
        .get_or_init(|| async { crate::search::calibrate::load_ood_threshold(pool, dim).await })
        .await
}

fn ood_abstain_json(query: &str, threshold: f64, dist: f64) -> Option<Value> {
    if dist <= threshold {
        return None;
    }
    Some(serde_json::json!({
        "mode": "hybrid",
        "query": query,
        "results": [],
        "count": 0,
        "ood": true,
        "mahalanobis_distance": dist,
        "ood_threshold": threshold,
        "abstain_reason": format!(
            "Query is out of distribution (distance {:.2} > threshold {:.2}). \
             No relevant memory found — consider rephrasing or adding the topic explicitly.",
            dist, threshold
        ),
        "graphrag_context": []
    }))
}

async fn check_ood(
    pool: &PgPool,
    query: &str,
    threshold: Option<f64>,
    project_id: Option<uuid::Uuid>,
) -> Option<Value> {
    use crate::search::ood::{MIN_SAMPLES_FOR_OOD, OodStats, default_threshold};

    if !crate::embeddings::onnx::is_model_loaded() {
        return None;
    }
    let query_emb = crate::embeddings::onnx::embed_passage(query).await.ok()?;
    let tau = match threshold.or_else(env_threshold) {
        Some(t) => t,
        None => calibrated_threshold(pool, query_emb.len())
            .await
            .unwrap_or_else(|| default_threshold(query_emb.len())),
    };

    if let Some(stats) = crate::search::ood_cache::get(project_id)
        && let Some(dist) = stats.mahalanobis(&query_emb)
    {
        return ood_abstain_json(query, tau, dist);
    }

    let raw: Vec<(pgvector::Vector,)> = sqlx::query_as(
        "SELECT embedding FROM brain_observations
         WHERE embedding IS NOT NULL AND observation_type != 'superseded'
           AND ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY id
         LIMIT 5000",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await
    .ok()?;

    if raw.len() < MIN_SAMPLES_FOR_OOD {
        return None;
    }
    let embeddings: Vec<Vec<f32>> = raw.into_iter().map(|(v,)| v.to_vec()).collect();
    let stats = OodStats::fit(&embeddings)?;
    crate::search::ood_cache::store(project_id, stats.clone());
    let dist = stats.mahalanobis(&query_emb)?;
    ood_abstain_json(query, tau, dist)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_entropy_weights_asymptotes() {
        let (tw_low, vw_low) = entropy_weights(0.0);
        assert!(
            tw_low > 0.68,
            "low entropy should be text-heavy: got text_w={tw_low}"
        );
        assert!(
            vw_low < 0.32,
            "low entropy should minimize vector_w: got {vw_low}"
        );

        let (tw_high, vw_high) = entropy_weights(10.0);
        assert!(
            tw_high < 0.32,
            "high entropy should minimize text_w: got {tw_high}"
        );
        assert!(
            vw_high > 0.68,
            "high entropy should be vector-heavy: got {vw_high}"
        );

        let (tw_mid, vw_mid) = entropy_weights(2.75);
        assert!(
            (tw_mid - 0.5).abs() < 1e-10,
            "midpoint should give text_w=0.5: got {tw_mid}"
        );
        assert!(
            (vw_mid - 0.5).abs() < 1e-10,
            "midpoint should give vector_w=0.5: got {vw_mid}"
        );
    }

    #[test]
    fn test_entropy_weights_no_discontinuity() {
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
