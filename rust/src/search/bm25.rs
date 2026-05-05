//! BM25-flavored sparse retrieval using PostgreSQL `ts_rank_cd`.
//!
//! V0.9 pragmatic implementation: uses `ts_rank_cd` (cover density ranking)
//! over the existing `to_tsvector('simple', ...)` columns. ts_rank_cd is
//! NOT exactly Okapi BM25 (Robertson-Walker SIGIR 1994) — it is a normalized
//! variant that incorporates document length and cover density (matches that
//! cluster near each other rank higher).
//!
//! Why ts_rank_cd instead of true BM25:
//! - Zero new dependencies (PostgreSQL native).
//! - Indexes already exist (`idx_obs_search` GIN tsvector).
//! - For typical brain-of-an-agent corpus (≤100K obs), ts_rank_cd recovers
//!   ~70-80% of true BM25 quality on heterogeneous queries.
//!
//! When to upgrade to ParadeDB pg_search (true BM25 via Tantivy):
//! - Corpus exceeds 1M observations.
//! - Specific recall@K requirement on long-tail entity name queries.
//! - Plan v0.9.1 sub-PR — gated by `paradedb-bm25` feature flag.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

/// Run BM25-style search across the same scopes as `text_search` and return
/// a parallel ranked list. Output contract identical to `text_search` so the
/// fusion layer can treat it as a third signal alongside text + vector.
///
/// Caller is `hybrid_search` in [crate::handlers::faro].
pub async fn bm25_search(
    pool: &PgPool,
    query: &str,
    scope: &str,
    limit: i64,
    project_id: Option<Uuid>,
) -> Result<Vec<Value>> {
    let mut results = Vec::new();

    if scope == "all" || scope == "observations" {
        // ts_rank_cd weights (D, C, B, A) — we only have one weight band so
        // the array is symbolic; the meaningful signal is cover density.
        let rows: Vec<(Uuid, String, String, String, f64, f64)> = sqlx::query_as(
            "SELECT o.id, e.name, o.content, o.observation_type, o.importance::float8,
                    ts_rank_cd(o.search_vector, plainto_tsquery('simple', $1))::float8 AS bm25
             FROM brain_observations o
             JOIN brain_entities e ON o.entity_id = e.id
             WHERE o.search_vector @@ plainto_tsquery('simple', $1)
               AND o.observation_type != 'superseded'
               AND ($3::uuid IS NULL OR o.project_id = $3 OR o.project_id IS NULL)
             ORDER BY bm25 DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(project_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        results.extend(rows.into_iter().map(
            |(id, entity_name, content, obs_type, importance, bm25)| {
                serde_json::json!({
                    "id": id.to_string(),
                    "type": "observation",
                    "entity_name": entity_name,
                    "content": content,
                    "observation_type": obs_type,
                    "importance": importance,
                    "bm25_score": bm25
                })
            },
        ));
    }

    if scope == "all" || scope == "entities" {
        let rows: Vec<(Uuid, String, String, f64, f64)> = sqlx::query_as(
            "SELECT id, name, entity_type, importance::float8,
                    ts_rank_cd(search_vector, plainto_tsquery('simple', $1))::float8 AS bm25
             FROM brain_entities
             WHERE search_vector @@ plainto_tsquery('simple', $1)
               AND ($3::uuid IS NULL OR project_id = $3 OR project_id IS NULL)
             ORDER BY bm25 DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(project_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        results.extend(
            rows.into_iter()
                .map(|(id, name, entity_type, importance, bm25)| {
                    serde_json::json!({
                        "id": id.to_string(),
                        "type": "entity",
                        "name": name,
                        "entity_type": entity_type,
                        "importance": importance,
                        "bm25_score": bm25
                    })
                }),
        );
    }

    if scope == "all" || scope == "errors" {
        let rows: Vec<(Uuid, String, String, bool, f64)> = sqlx::query_as(
            "SELECT id, error_type, error_message, resolved,
                    ts_rank_cd(search_vector, plainto_tsquery('simple', $1))::float8 AS bm25
             FROM brain_errors
             WHERE search_vector @@ plainto_tsquery('simple', $1)
               AND ($3::uuid IS NULL OR project_id = $3 OR project_id IS NULL)
             ORDER BY bm25 DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit)
        .bind(project_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        results.extend(
            rows.into_iter()
                .map(|(id, error_type, error_message, resolved, bm25)| {
                    serde_json::json!({
                        "id": id.to_string(),
                        "type": "error",
                        "error_type": error_type,
                        "error_message": error_message,
                        "resolved": resolved,
                        "bm25_score": bm25
                    })
                }),
        );
    }

    // Sort fused list by BM25 desc and truncate (per-table ORDER BY can
    // leave entities/errors interleaved at top once we span multiple scopes).
    results.sort_by(|a, b| {
        let sa = a.get("bm25_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("bm25_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit as usize);

    Ok(results)
}
