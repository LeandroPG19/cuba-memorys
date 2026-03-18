//! §C: Neighbor Diffusion — 1-hop importance spreading.
//!
//! Collins & Loftus (1975) spreading activation, simplified to single-hop
//! weighted diffusion. Runs in REM sleep daemon as background consolidation.
//!
//! NOTE: This is NOT a full Random Walk with Restart (which iterates to
//! convergence like PageRank). For topology-aware importance, see
//! `graph::pagerank::compute_and_store`.

use anyhow::Result;
use sqlx::PgPool;

/// 1-hop weighted diffusion during REM consolidation.
///
/// For each high-importance entity (seed), diffuse a fraction of its
/// importance to direct neighbors, weighted by relation strength.
///
/// - α=0.85: restart probability (fraction kept by seed)
/// - (1−α)=0.15: fraction diffused to neighbors
pub async fn neighbor_diffusion(pool: &PgPool) -> Result<()> {
    // Get seed entities (high importance)
    let seeds: Vec<(uuid::Uuid, f64)> = sqlx::query_as(
        "SELECT id, importance FROM brain_entities
         WHERE importance > 0.5
         ORDER BY importance DESC
         LIMIT 20",
    )
    .fetch_all(pool)
    .await?;

    if seeds.is_empty() {
        return Ok(());
    }

    // For each seed, diffuse importance to direct neighbors
    for (seed_id, seed_importance) in &seeds {
        // Get direct neighbors
        let neighbors: Vec<(uuid::Uuid, f64)> = sqlx::query_as(
            "SELECT CASE WHEN from_entity = $1 THEN to_entity ELSE from_entity END,
                    strength
             FROM brain_relations
             WHERE from_entity = $1 OR to_entity = $1",
        )
        .bind(seed_id)
        .fetch_all(pool)
        .await?;

        if neighbors.is_empty() {
            continue;
        }

        // Diffuse a fraction of seed importance to neighbors
        let alpha = 0.85;
        let boost_per_neighbor = seed_importance * (1.0 - alpha) / neighbors.len() as f64;

        for (neighbor_id, strength) in &neighbors {
            let weighted_boost = boost_per_neighbor * strength;
            sqlx::query(
                "UPDATE brain_entities SET
                    importance = LEAST(importance + $1, 1.0),
                    updated_at = NOW()
                 WHERE id = $2",
            )
            .bind(weighted_boost)
            .bind(neighbor_id)
            .execute(pool)
            .await?;
        }
    }

    tracing::info!(seeds = seeds.len(), "neighbor diffusion completed");
    Ok(())
}
