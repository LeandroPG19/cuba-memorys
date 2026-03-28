//! Dual-Strength Model — access tracking (V3).
//!
//! Only last_accessed and access_count are maintained — these feed the
//! exponential decay formula in zafra "decay" and drive the importance
//! ORDER BY in faro search.

use anyhow::Result;
use sqlx::PgPool;

/// Update last_accessed + access_count on entity read.
pub async fn on_entity_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_observations SET
            last_accessed = NOW(),
            access_count = access_count + 1
         WHERE entity_id = $1
           AND observation_type != 'superseded'"
    )
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Update last_accessed on search match.
///
/// Resets the exponential decay clock for matched observations.
pub async fn on_search_match(pool: &PgPool, observation_ids: &[uuid::Uuid]) -> Result<()> {
    if observation_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "UPDATE brain_observations SET
            last_accessed = NOW(),
            access_count = access_count + 1
         WHERE id = ANY($1)"
    )
    .bind(observation_ids)
    .execute(pool)
    .await?;
    Ok(())
}
