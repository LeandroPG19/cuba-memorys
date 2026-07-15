use anyhow::Result;
use sqlx::PgPool;

pub async fn refresh_energy_scores(pool: &PgPool) -> Result<u64> {
    let result = sqlx::query(
        r#"INSERT INTO brain_node_metrics (node_id, pagerank_score, energy_score, last_calculated)
           SELECT e.id,
                  COALESCE(m.pagerank_score, e.importance::float8, 0.5),
                  LEAST(
                    COALESCE(m.pagerank_score, e.importance::float8, 0.5) * 0.75
                    + COALESCE(m.betweenness_centrality, 0.0) * 0.25,
                    1.0
                  ),
                  NOW()
           FROM brain_entities e
           LEFT JOIN brain_node_metrics m ON m.node_id = e.id
           ON CONFLICT (node_id) DO UPDATE SET
             pagerank_score = COALESCE(EXCLUDED.pagerank_score, brain_node_metrics.pagerank_score),
             energy_score = EXCLUDED.energy_score,
             last_calculated = NOW()"#,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}
