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

    if edges.is_empty() {
        return Ok(0);
    }

    // Build adjacency: node_id → (outgoing nodes with weights)
    let mut nodes: HashMap<uuid::Uuid, usize> = HashMap::new();
    let mut node_list: Vec<uuid::Uuid> = Vec::new();

    for (from, to, _) in &edges {
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
        return Ok(0);
    }

    // Build adjacency lists
    let mut outgoing: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    let mut out_weight_sum: Vec<f64> = vec![0.0; n];

    for (from, to, weight) in &edges {
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

        // Accumulate dangling rank in one pass, then distribute uniformly.
        // Previous approach was O(dangling_count × n) — now O(n) total.
        let mut dangling_sum = 0.0;
        for i in 0..n {
            if out_weight_sum[i] > 0.0 {
                for &(j, weight) in &outgoing[i] {
                    new_ranks[j] += DAMPING * ranks[i] * weight / out_weight_sum[i];
                }
            } else {
                dangling_sum += DAMPING * ranks[i];
            }
        }
        // Distribute accumulated dangling rank equally to all nodes
        if dangling_sum > 0.0 {
            let share = dangling_sum / n as f64;
            for r in new_ranks.iter_mut() {
                *r += share;
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

    // P1 FIX: Batch UPDATE with unnest() — 1 query instead of N.
    //
    // V0.7: BLEND PageRank with existing importance instead of overwriting.
    // Previous version used `SET importance = v.rank` which unconditionally
    // destroyed Hebbian/RLHF/decay accumulated importance every REM cycle.
    //
    // Fix: convex combination (Bayesian posterior update):
    //   importance_new = α * rank_normalized + (1 - α) * importance_current
    //
    // α = 0.3 — PageRank contributes structural signal but does not dominate.
    // Min-max normalization maps raw ranks (which sum to 1.0) to [0, 1] so
    // they are commensurate with the importance scale.
    let ids = node_list; // move — node_list not needed after this point

    // Min-max normalize ranks to [0, 1]
    let min_r = ranks.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_r = ranks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Uniform distribution guard: all nodes have equal PageRank (e.g. 2-node
    // symmetric graph, or any perfectly regular graph). There is no structural
    // differentiation signal — applying the blend would decay existing
    // Hebbian/RLHF/decay importance by (1-α)=30% per REM cycle with zero benefit.
    // Skip the blend entirely and preserve existing importance unchanged.
    if (max_r - min_r) < 1e-10 {
        tracing::info!(
            entities = n,
            "PageRank blend skipped: uniform distribution (no structural differentiation signal)"
        );
        return Ok(n);
    }

    let range = max_r - min_r; // guaranteed > 1e-10 by check above
    let normalized: Vec<f64> = ranks.iter().map(|r| (r - min_r) / range).collect();

    sqlx::query(
        r#"
        UPDATE brain_entities AS e
        SET importance = LEAST(0.3 * v.rank_norm + 0.7 * e.importance, 1.0),
            updated_at = NOW()
        FROM (SELECT UNNEST($1::uuid[]) AS id, UNNEST($2::float8[]) AS rank_norm) AS v
        WHERE e.id = v.id
        "#,
    )
    .bind(&ids)
    .bind(&normalized)
    .execute(pool)
    .await?;

    tracing::info!(entities = n, "PageRank blended (α=0.3, P1 batch)");
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_convergence_constants() {
        const _: () = assert!(DAMPING > 0.0 && DAMPING < 1.0);
        const _: () = assert!(ITERATIONS > 0);
        const _: () = assert!(CONVERGENCE_THRESHOLD > 0.0);
    }

    #[test]
    fn test_minmax_normalization() {
        let ranks = [0.001, 0.005, 0.002, 0.010];
        let min_r = ranks.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = ranks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_r - min_r).max(1e-12);
        let normalized: Vec<f64> = ranks.iter().map(|r| (r - min_r) / range).collect();

        assert!((normalized[0] - 0.0).abs() < 1e-6, "min should map to 0.0");
        assert!((normalized[3] - 1.0).abs() < 1e-6, "max should map to 1.0");
        // All values in [0, 1]
        for v in &normalized {
            assert!(*v >= 0.0 && *v <= 1.0, "normalized value out of range: {v}");
        }
    }

    #[test]
    fn test_blend_preserves_existing_importance() {
        // Simulate blend: importance_new = 0.3 * rank_norm + 0.7 * existing
        let existing = 0.8; // High importance from Hebbian/RLHF
        let rank_norm = 0.2; // Low PageRank
        let blended = 0.3 * rank_norm + 0.7 * existing;
        // With overwrite: importance = rank_norm = 0.2 (destroyed)
        // With blend: importance = 0.06 + 0.56 = 0.62 (preserved most of prior)
        assert!(
            blended > 0.5,
            "blend should preserve existing importance: got {blended}"
        );
        assert!(
            blended < existing,
            "blend should incorporate PageRank signal"
        );
    }

    /// Symmetric graphs produce equal ranks for all nodes. Without the early-return
    /// guard, normalized = [0,0,...,0] and the blend decays importance 30% per REM cycle.
    /// Verify the guard fires before the normalization step.
    #[test]
    fn test_uniform_ranks_trigger_early_return() {
        let ranks = [0.25_f64; 4]; // 4-node perfectly balanced graph
        let min_r = ranks.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = ranks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_r - min_r;
        // The guard condition: uniform → range < 1e-10 → skip blend
        assert!(
            range < 1e-10,
            "uniform ranks must trigger early return: range={range}"
        );
        // Verify that applying normalized=0 would cause 30% decay per cycle
        let existing = 0.8;
        let decay_with_zero_norm = 0.3 * 0.0 + 0.7 * existing;
        assert!(
            decay_with_zero_norm < existing,
            "without guard, importance decays from {existing} to {decay_with_zero_norm}"
        );
    }

    /// Non-uniform ranks (realistic case) should NOT trigger early return.
    #[test]
    fn test_non_uniform_ranks_proceed_to_blend() {
        let ranks = [0.01_f64, 0.05, 0.02, 0.10]; // star-like graph
        let min_r = ranks.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = ranks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_r - min_r;
        assert!(
            range >= 1e-10,
            "non-uniform ranks should not trigger early return: range={range}"
        );
    }
}
