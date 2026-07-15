use anyhow::Result;
use sqlx::PgPool;

pub async fn on_entity_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_observations SET
            last_accessed = NOW(),
            access_count = access_count + 1
         WHERE entity_id = $1
           AND observation_type != 'superseded'",
    )
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn on_search_match(pool: &PgPool, observation_ids: &[uuid::Uuid]) -> Result<()> {
    if observation_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "UPDATE brain_observations SET
            last_accessed = NOW(),
            access_count = access_count + 1
         WHERE id = ANY($1)",
    )
    .bind(observation_ids)
    .execute(pool)
    .await?;
    Ok(())
}
