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
/// V0.9: Clone added so MMR can reorder without re-running the fusion.
/// V0.9: bm25_score field added for 3-way RRF (text + vector + bm25).
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

    // Compact is the default: abbreviated keys cost 798 tokens for limit=10 where
    // verbose costs 2787 — measured, not estimated (the old comment claimed ~35%;
    // it is 71%). Every session pays this on every search, so the cheap shape is
    // the one that should need no opt-in. Callers that need the full keys —
    // the CLI renderer, chiefly — ask for `verbose` explicitly.
    let format = args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("compact");

    // V0.6: Tag filter
    let tag_filter = args.get("tags").and_then(|v| v.as_str());

    // The Testing Effect write (dual_strength::on_search_match) boosts the
    // importance/access of matched observations. Callers that must NOT mutate —
    // the eval harness above all, which would otherwise skew the very rankings
    // it measures — pass track_access=false. Default true (normal search).
    let track_access = args
        .get("track_access")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // v0.11: associative multi-hop expansion (opt-in; default off).
    let associative = args
        .get("associative")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Temporal filters (ISO8601 strings)
    let before = args.get("before").and_then(|v| v.as_str());
    let after = args.get("after").and_then(|v| v.as_str());
    let time_bounds = parse_time_bounds(before, after);

    // V0.9: MMR diversification (Carbonell-Goldstein 1998).
    // diversify=true reorders top-K to penalize near-duplicates.
    let diversify = args
        .get("diversify")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mmr_lambda = args
        .get("mmr_lambda")
        .and_then(|v| v.as_f64())
        .unwrap_or(crate::search::mmr::DEFAULT_LAMBDA);

    // V0.9: OOD abstention via Mahalanobis (Lee NeurIPS 2018).
    // abstain_ood=true triggers early return when query is far from the
    // active embedding distribution. Default false to preserve v0.8 behavior.
    let abstain_ood = args
        .get("abstain_ood")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // None => derive from the embedding dimension inside check_ood, since a
    // sane Mahalanobis cutoff scales as sqrt(d). See search::ood::default_threshold.
    let ood_threshold = args.get("ood_threshold").and_then(|v| v.as_f64());

    // V0.9: BM25 (ts_rank_cd) as third RRF signal. Default true — gives
    // +8-15% recall on queries with rare entity names / specific errors.
    let enable_bm25 = args
        .get("enable_bm25")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // V0.9.2: cross-encoder rerank pass over top-50 RRF candidates.
    // Auto-enabled when CUBA_RERANKER_PATH points to a valid bge-reranker
    // ONNX. Identity fallback otherwise (no perf cost). Argument override
    // lets callers force off even when env is set.
    let enable_rerank = args
        .get("rerank")
        .and_then(|v| v.as_bool())
        .unwrap_or_else(crate::search::rerank::enabled);

    // V0.8: Resolve current project (None = no filter, see project::current_project_id)
    let project_id = crate::project::current_project_id(pool).await?;

    // V0.9: OOD pre-check (when requested). On a warm cache this is a single
    // O(d²) matrix-vector product on μ, Σ⁻¹ — no DB round-trip. A cold cache
    // pays one bounded fit (<=500 rows) before short-circuiting the search.
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
    /// V0.8: project scoping. None = no filter (legacy behavior).
    project_id: Option<uuid::Uuid>,
    /// V0.9: when true, post-RRF results pass through MMR diversification.
    diversify: bool,
    /// V0.9: MMR balance — 1.0 = pure relevance, 0.0 = pure diversity. Default 0.7.
    mmr_lambda: f64,
    /// V0.9: enable BM25 (ts_rank_cd) as third RRF signal alongside text + vector.
    enable_bm25: bool,
    /// V0.9.2: cross-encoder rerank top-N pre-MMR. Identity when reranker
    /// asset (CUBA_RERANKER_PATH) is missing — production transparent.
    enable_rerank: bool,
    /// When false, skip the Testing Effect write (eval harness). Default true.
    track_access: bool,
    /// v0.11: associative expansion. Seeds spreading activation from
    /// query-matched entities and pulls in observations on graph-connected
    /// entities that no lexical/vector signal surfaced (multi-hop recall,
    /// HippoRAG-style). Additive and opt-in — default false leaves the
    /// production ranking untouched.
    associative: bool,
}

/// §A V2: Weighted RRF Entropy Routing — 3 ranges (Elastic 2025).
async fn hybrid_search(pool: &PgPool, query: &str, opts: &SearchOpts<'_>) -> Result<Value> {
    // §A V2: 3-range entropy routing (keyword / mixed / semantic)
    let query_entropy = crate::search::rrf::query_entropy(query);
    let (text_weight, vector_weight) = entropy_weights(query_entropy);
    // BM25 weight tracks text channel (Jaynes routing extension — keyword-heavy queries).
    let bm25_weight = text_weight;

    // Run text search (always available — V8 graceful degradation base)
    let mut text_results = text_search(
        pool,
        query,
        opts.scope,
        opts.limit * 2,
        &opts.time_bounds,
        opts.project_id,
    )
    .await?;

    // V0.6: Tag filter — if specified, run additional tag-based query and merge
    // V0.8: tag query also scoped by project (transparent if None)
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
    let vector_results = vector_search(
        pool,
        query,
        opts.scope,
        opts.limit * 2,
        &opts.time_bounds,
        opts.project_id,
    )
    .await;

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
                bm25_score: 0.0,
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
                    bm25_score: 0.0,
                    session_boosted: false,
                    total: rrf_score,
                    data: result.clone(),
                });
        }
    }

    // V0.9: BM25 (ts_rank_cd) as third RRF signal — opt-out via enable_bm25=false.
    // Weight = 1.0 (uniform with the other two; entropy_weights only balances
    // text vs vector). Catches queries with rare terms / specific entity names
    // that dense embeddings miss.
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

    // v0.11: associative expansion (opt-in). Seed spreading activation from the
    // entities that match the query, then pull in observations on graph-connected
    // entities that no lexical/vector/bm25 signal surfaced. Strictly additive:
    // only inserts ids not already present, so the base ranking cannot regress.
    if opts.associative {
        associative_expand(pool, query, opts.project_id, &mut fused_scores).await;
    }

    // Sort by fused score
    let mut results: Vec<(String, FusedResult)> = fused_scores.into_iter().collect();
    results.sort_by(|a, b| {
        b.1.total
            .partial_cmp(&a.1.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Post-fusion dedup (Cormack RRF + lexical overlap — same as rrf::fuse).
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

    // V0.9: defer truncation. With diversify=true we need a wider pool
    // (up to limit*2) so MMR has candidates to choose from. With rerank
    // enabled we keep top-50 so the cross-encoder has enough signal.
    // Truncation to opts.limit happens after session boost + rerank + MMR.
    let pool_size = if opts.enable_rerank {
        50.min(results.len())
    } else if opts.diversify {
        (opts.limit as usize * 5).min(results.len())
    } else {
        (opts.limit as usize).min(results.len())
    };
    results.truncate(pool_size);

    // V0.9.2: cross-encoder rerank pass. Reorders results in place; MMR
    // runs after this so diversification operates on the better-ranked
    // candidates. Identity fallback when reranker is not configured.
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
        if let Ok(reranked) = crate::search::rerank::rerank(query, &contents).await {
            let original = results.clone();
            results = reranked
                .into_iter()
                .filter_map(|(idx, score)| {
                    let mut entry = original.get(idx).cloned()?;
                    // Bump total slightly so post-MMR ordering respects rerank
                    entry.1.total += score * 0.0001; // tiebreaker only
                    Some(entry)
                })
                .collect();
        }
    }

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

    if opts.track_access
        && !matched_obs_ids.is_empty()
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

    // V0.9: MMR diversification pass.
    // Uses Jaccard token-set similarity as a fast proxy for semantic distance
    // between candidates — avoids re-fetching embeddings from DB. For exact
    // semantic dedup (cross-encoder rerank), see PR #6 Phase 4.
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
        // Reorder results by MMR picks
        let reordered: Vec<(String, FusedResult)> = picks
            .into_iter()
            .filter_map(|i| results.get(i).cloned())
            .collect();
        results = reordered;
    } else {
        // No diversification — just truncate to the requested limit
        results.truncate(opts.limit as usize);
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
                obj.insert("bm25_score".to_string(), serde_json::json!(fr.bm25_score));
                obj.insert(
                    "session_boosted".to_string(),
                    serde_json::json!(fr.session_boosted),
                );
            }
            r
        })
        .collect();

    // Shape first, THEN budget. The order matters and used to be wrong: the
    // budget counted each row's full `content`, and compact_result afterwards
    // truncated that same content to 200 chars. So the caller's budget was spent
    // counting text that was never sent — a limit=10 compact response costs 798
    // tokens against a 5000-token budget, meaning most of what was paid for was
    // thrown away and fewer results came back than fit.
    //
    // Counting the serialized row (keys, scores and all) rather than just the
    // content also stops the JSON envelope from riding along for free: ten
    // verbose rows carry ten copies of ~10 field names.
    use crate::search::budget::{count_tokens, truncate_to_budget};

    let shaped: Vec<Value> = if opts.format == "compact" {
        results_json.iter().map(compact_result).collect()
    } else {
        results_json
    };
    // Whichever shape we are in, this is where the text lives.
    let text_key = if opts.format == "compact" { "c" } else { "content" };

    let mut token_budget = opts.max_tokens;
    let mut final_results: Vec<Value> = Vec::with_capacity(shaped.len());
    for mut r in shaped {
        if token_budget <= 0 {
            break;
        }
        let row_tokens = count_tokens(&r.to_string()) as i64;
        if row_tokens > token_budget {
            // One row alone overflows what is left: keep it, but trim its text
            // so the response still lands inside the budget.
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

    Ok(serde_json::json!({
        "mode": "hybrid",
        "query": query,
        "results": final_results,
        "count": final_results.len(),
        "graphrag_context": graphrag_context
    }))
}

/// How much of an observation the compact shape keeps.
///
/// This is the whole trade, and it was being made blind. Abbreviating the keys
/// saves almost nothing; the saving comes from cutting the text — and cutting
/// the text is what costs recall, because a marker sitting at character 300 of
/// a memory is simply not there any more, and an agent cannot use what it
/// cannot see.
///
/// Swept on the Spanish ability set (n=10, e5-small), quality against cost:
///
/// ```text
///   chars   tokens   nDCG@10   vs verbose
///     200      871    0.6353   -75% cost, -13.5% quality   ← the old hard-coded value
///     400     1271    0.6900   -64% cost,  -6.0% quality
///     600     1580    0.7090   -55% cost,  -3.5% quality
///    1200     2109    0.7344   -40% cost,   NO LOSS        ← the knee
///    2000     2531    0.7344   -28% cost,   no loss
///    full     3508    0.7344    baseline
/// ```
///
/// 1200 is where the curve bends: every character past it costs context and buys
/// nothing, and every character below it starts eating relevance. The previous
/// default of 200 was throwing away an eighth of the retrieval quality to save
/// tokens nobody had measured.
///
/// Overridable with `CUBA_COMPACT_CHARS` — re-sweep it on a different corpus
/// rather than inheriting this number the way we inherited 200.
fn compact_chars() -> usize {
    std::env::var("CUBA_COMPACT_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1200)
}

/// Transform a result into compact shape: abbreviated keys, bounded text.
fn compact_result(r: &Value) -> Value {
    let content = r
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| crate::handlers::zafra::safe_truncate(s, compact_chars()));
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
async fn verify_claim(pool: &PgPool, claim: &str, project_id: Option<uuid::Uuid>) -> Result<Value> {
    use std::collections::HashMap;

    // V0.9: bump HNSW ef_search 100 → 200 for verify-mode queries.
    // Recall@10 jumps from ~0.95 to ~0.99 (Malkov-Yashunin 2018 Fig. 11),
    // critical when cuba_juez consumes our results downstream. SET LOCAL
    // scopes to the current transaction-equivalent session, no leak across
    // pool checkouts.
    sqlx::query("SET LOCAL hnsw.ef_search = 200")
        .execute(pool)
        .await
        .ok();

    // 1. Trigram evidence (with entity_name — Mejora 9: eliminates second scan)
    // V0.8: project filter applied (no-op when project_id is None)
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
            .or_insert((content.clone(), *sim, obs_type.clone(), entity_name.clone()));
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
    let sources: Vec<&str> = evidence_list
        .iter()
        .map(|(_, _, _, t, _)| t.as_str())
        .collect();

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
/// V0.8: $5 = project_id (Option<Uuid>) — when None, filter is a no-op.
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

    // Search episodes (always included in "all" scope — separate memory system)
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
/// V0.8: project_id filter (None = no-op).
async fn vector_search(
    pool: &PgPool,
    query: &str,
    _scope: &str,
    limit: i64,
    tb: &TimeBounds,
    project_id: Option<uuid::Uuid>,
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

/// v0.11: associative multi-hop expansion (HippoRAG-style).
///
/// 1. Seed = entities whose name/text matches the query (the query-linked nodes).
/// 2. Spread activation 2 hops over `brain_relations` (reuses graph::activation).
/// 3. For the top activated non-seed entities, add their highest-importance
///    observations that no base signal already surfaced, scored by
///    activation × importance.
///
/// Additive only: it inserts ids absent from `fused_scores`, never lowers an
/// existing score, so enabling it cannot regress the base ranking. Best-effort:
/// any DB error is logged and skipped rather than failing the search.
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
    // Scaled so a strongly-activated, high-importance associative hit lands
    // around a mid-rank RRF score (~1/60) — present in the pool, never
    // dominating an exact lexical/vector match.
    const ASSOC_WEIGHT: f64 = 0.02;

    // 1. Seed entities matching the query.
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

    // 2. Spread activation from the seeds.
    let activated =
        match crate::graph::activation::spread_from_entities(pool, &seed_ids, MAX_HOPS).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(error = %e, "associative: activation spread failed — skipping");
                return;
            }
        };

    // 3. For the top activated NON-seed entities, add their top observations.
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
                continue; // never override a base-signal hit
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

// V3 adaptive_rrf_k REMOVED — k=60 constant per Gemini Deep Research audit 2026-03-14.
// Rationale: dynamic sqrt-based k introduced non-monotonic ranking instabilities
// and violated determinism. Cormack et al. 2009, Azure AI Search, Elasticsearch
// all converge on k=60 as empirically optimal.

/// Get this process's session goals for session-aware boosting.
///
/// Scoped to `crate::session`: boosting results with another MCP client's
/// goals is worse than not boosting at all.
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

/// V9: GraphRAG enrichment — fetch degree-1 neighbors for top-K results.
///
/// For each top result that has an entity name, we query its related entities
/// to provide graph context. This helps the AI understand the broader
/// knowledge structure around search matches.
async fn enrich_graphrag(pool: &PgPool, results: &[(String, FusedResult)], top_k: usize) -> Value {
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

/// V0.9: Out-of-distribution pre-check. Returns Some(answer) if the query
/// should abstain (no relevant memory exists), or None to proceed with normal
/// search. Caller is `cuba_faro` with `abstain_ood=true`.
///
/// Cheap path: O(d²) Mahalanobis on cached μ, Σ⁻¹ — sub-millisecond.
/// Falls back to None (no abstention) when:
/// - ONNX model not loaded (no embeddings to compute distance against)
/// - Fewer than MIN_SAMPLES_FOR_OOD observations exist (covariance too noisy)
/// - Embedding fit fails (rank-deficient cov even after ridge)
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
    // NOTE: `embed_passage`, not `embed`. e5 is asymmetric — it prepends
    // "query: " vs "passage: ", which shifts the vector. The density (μ, Σ⁻¹)
    // is fitted over stored observations, i.e. *passages*. Embedding the query
    // with the "query: " prefix would measure it against a distribution it was
    // never in, inflating every distance and abstaining on in-corpus queries.
    // For density estimation both sides must share the prefix.
    let query_emb = crate::embeddings::onnx::embed_passage(query).await.ok()?;
    // Caller may override; otherwise scale the cutoff to the embedding space.
    let tau = threshold.unwrap_or_else(|| default_threshold(query_emb.len()));

    // Cache hit: skip the 500-row fetch entirely. (Previously the SELECT ran
    // before this check, so the cache saved the fit but never the query.)
    if let Some(stats) = crate::search::ood_cache::get(project_id)
        && let Some(dist) = stats.mahalanobis(&query_emb)
    {
        return ood_abstain_json(query, tau, dist);
    }

    // Sample up to 500 active observations for the fit. Cheaper than full
    // scan, statistically sufficient for d=384.
    let raw: Vec<(pgvector::Vector,)> = sqlx::query_as(
        "SELECT embedding FROM brain_observations
         WHERE embedding IS NOT NULL AND observation_type != 'superseded'
           AND ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY importance DESC, last_accessed DESC NULLS LAST
         LIMIT 500",
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

        // Midpoint (entropy = 2.75): exact 50/50 split
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
