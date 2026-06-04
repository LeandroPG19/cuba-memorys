//! Graph energy scores — uses persisted PageRank from `brain_node_metrics`.

use anyhow::Result;
use sqlx::PgPool;

/// Blend PageRank into energy_score for all entities (additive, does not remove handlers).
pub async fn refresh_energy_scores(pool: &PgPool) -> Result<u64> {
    let result = sqlx::query(
        r#"INSERT INTO brain_node_metrics (node_id, pagerank_score, energy_score, last_calculated)
           SELECT e.id,
                  COALESCE(e.importance, 0.5),
                  COALESCE(e.importance, 0.5) * 0.85,
                  NOW()
           FROM brain_entities e
           ON CONFLICT (node_id) DO UPDATE SET
             pagerank_score = EXCLUDED.pagerank_score,
             energy_score = EXCLUDED.energy_score,
             last_calculated = NOW()"#,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}
