use anyhow::Result;
use sqlx::PgPool;

use crate::constants::{BCM_THROTTLE_SCALE, HEBBIAN_ACCESS_BOOST};

const BCM_THETA_MIN: f64 = 10.0;

const BCM_EMA_ALPHA: f64 = 0.15;

const HEBBIAN_TAU_SECS: f64 = 600.0;

pub async fn boost_on_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_entities SET
            bcm_theta = GREATEST($5, (1.0 - $6) * COALESCE(bcm_theta, $5) + $6 * access_count::float8),
            importance = LEAST(
                importance + $1
                  * GREATEST(0.1, 1.0 - (access_count::float8 / GREATEST(COALESCE(bcm_theta, $2), access_count::float8)) * $3)
                  * (1.0 - EXP(-LEAST(EXTRACT(EPOCH FROM (NOW() - updated_at)), 86400.0) / $7)),
                1.0
            ),
            access_count = access_count + 1,
            updated_at = NOW()
         WHERE id = $4"
    )
    .bind(HEBBIAN_ACCESS_BOOST)
    .bind(BCM_THETA_MIN)
    .bind(BCM_THROTTLE_SCALE)
    .bind(entity_id)
    .bind(BCM_THETA_MIN)
    .bind(BCM_EMA_ALPHA)
    .bind(HEBBIAN_TAU_SECS)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn boost_neighbors(pool: &PgPool, entity_id: uuid::Uuid) -> Result<usize> {
    let result = sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(importance + $1 * 0.5 * sub.max_strength, 1.0),
            updated_at = NOW()
         FROM (
             SELECT
                 CASE WHEN from_entity = $2 THEN to_entity ELSE from_entity END AS neighbor_id,
                 MAX(strength) AS max_strength
             FROM brain_relations
             WHERE from_entity = $2 OR to_entity = $2
             GROUP BY CASE WHEN from_entity = $2 THEN to_entity ELSE from_entity END
         ) sub
         WHERE brain_entities.id = sub.neighbor_id",
    )
    .bind(HEBBIAN_ACCESS_BOOST)
    .bind(entity_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}
