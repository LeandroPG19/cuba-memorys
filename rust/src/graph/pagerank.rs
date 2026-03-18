//! §B: PageRank with NF-IDF Hub Dampening.
//!
//! P1 FIX: Batch UPDATE with unnest() — 1 query instead of N individual UPDATEs.
//! §B: NF-IDF Hub Dampening: w = strength / ln(1 + degree).

use anyhow::Result;
use sqlx::PgPool;
use std::collections::HashMap;

const DAMPING: f64 = 0.85;
const ITERATIONS: usize = 20;
const CONVERGENCE_THRESHOLD: f64 = 1e-6;

/// Compute PageRank from a list of edges.
///
/// Extracts the core algorithm from database I/O to improve maintainability
/// and testability.
pub fn compute_pagerank(edges: &[(uuid::Uuid, uuid::Uuid, f64)]) -> (Vec<uuid::Uuid>, Vec<f64>) {
    if edges.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Build adjacency: node_id → (outgoing nodes with weights)
    let mut nodes: HashMap<uuid::Uuid, usize> = HashMap::new();
    let mut node_list: Vec<uuid::Uuid> = Vec::new();

    for (from, to, _) in edges {
        for id in [from, to] {
            if !nodes.contains_key(id) {
                let idx = node_list.len();
                nodes.insert(*id, idx);
                node_list.push(*id);
            }
        }
    }

    let n = node_list.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Build adjacency lists
    let mut outgoing: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    let mut out_weight_sum: Vec<f64> = vec![0.0; n];

    for (from, to, weight) in edges {
        let from_idx = nodes[from];
        let to_idx = nodes[to];
        outgoing[from_idx].push((to_idx, *weight));
        out_weight_sum[from_idx] += weight;
    }

    // Power iteration
    let init_rank = 1.0 / n as f64;
    let mut ranks: Vec<f64> = vec![init_rank; n];
    let mut new_ranks: Vec<f64> = vec![0.0; n];

    for _iter in 0..ITERATIONS {
        // Teleportation base
        new_ranks.fill((1.0 - DAMPING) / n as f64);

        // Distribute rank through edges
        for i in 0..n {
            if out_weight_sum[i] > 0.0 {
                for &(j, weight) in &outgoing[i] {
                    new_ranks[j] += DAMPING * ranks[i] * weight / out_weight_sum[i];
                }
            } else {
                // Dangling node: distribute equally
                let share = DAMPING * ranks[i] / n as f64;
                for r in new_ranks.iter_mut() {
                    *r += share;
                }
            }
        }

        // Convergence check
        let delta: f64 = ranks
            .iter()
            .zip(new_ranks.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        std::mem::swap(&mut ranks, &mut new_ranks);

        if delta < CONVERGENCE_THRESHOLD {
            tracing::info!(iteration = _iter, delta = %delta, "PageRank converged");
            break;
        }
    }

    (node_list, ranks)
}

/// Compute PageRank and store results (batch UPDATE — P1 fix).
///
/// Returns number of entities updated.
pub async fn compute_and_store(pool: &PgPool) -> Result<usize> {
    // Fetch all relations with NF-IDF hub dampening (§B)
    let edges: Vec<(uuid::Uuid, uuid::Uuid, f64)> = sqlx::query_as(
        r#"
        SELECT r.from_entity, r.to_entity,
               r.strength / LN(1.0 + COALESCE(deg.degree, 1)) AS dampened_weight
        FROM brain_relations r
        LEFT JOIN (
            SELECT from_entity, COUNT(*) AS degree
            FROM brain_relations
            GROUP BY from_entity
        ) deg ON r.from_entity = deg.from_entity
        "#,
    )
    .fetch_all(pool)
    .await?;

    let (ids, scores) = compute_pagerank(&edges);
    let n = ids.len();

    if n == 0 {
        return Ok(0);
    }

    // P1 FIX: Batch UPDATE with unnest() — 1 query instead of N
    sqlx::query(
        r#"
        UPDATE brain_entities AS e
        SET importance = v.rank,
            updated_at = NOW()
        FROM (SELECT UNNEST($1::uuid[]) AS id, UNNEST($2::float8[]) AS rank) AS v
        WHERE e.id = v.id
        "#,
    )
    .bind(&ids)
    .bind(&scores)
    .execute(pool)
    .await?;

    tracing::info!(entities = n, "PageRank updated (batch P1)");
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_convergence_constants() {
        assert!(DAMPING > 0.0 && DAMPING < 1.0);
        assert!(ITERATIONS > 0);
        assert!(CONVERGENCE_THRESHOLD > 0.0);
    }

    #[test]
    fn test_compute_pagerank_empty() {
        let (ids, scores) = compute_pagerank(&[]);
        assert!(ids.is_empty());
        assert!(scores.is_empty());
    }

    #[test]
    fn test_compute_pagerank_simple() {
        let node1 = uuid::Uuid::new_v4();
        let node2 = uuid::Uuid::new_v4();
        let node3 = uuid::Uuid::new_v4();

        // 1 -> 2, 2 -> 3, 3 -> 1 (cycle)
        let edges = vec![
            (node1, node2, 1.0),
            (node2, node3, 1.0),
            (node3, node1, 1.0),
        ];

        let (ids, scores) = compute_pagerank(&edges);
        assert_eq!(ids.len(), 3);
        assert_eq!(scores.len(), 3);

        // In a symmetric cycle, they should all have roughly equal PageRank
        let expected = 1.0 / 3.0;
        for score in scores {
            assert!((score - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_compute_pagerank_dangling() {
        let node1 = uuid::Uuid::new_v4();
        let node2 = uuid::Uuid::new_v4();

        // 1 -> 2, 2 has no outgoing (dangling)
        let edges = vec![(node1, node2, 1.0)];

        let (ids, scores) = compute_pagerank(&edges);
        assert_eq!(ids.len(), 2);
        assert_eq!(scores.len(), 2);

        let sum: f64 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5); // Should still sum to 1
    }
}
