use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let entity_name = args.get("entity_name").and_then(|v| v.as_str());
    let observation_id = args.get("observation_id").and_then(|v| v.as_str());

    match action {
        "positive" => positive(pool, entity_name, observation_id).await,
        "negative" => negative(pool, entity_name, observation_id).await,
        "correct" => correct(pool, observation_id, &args).await,
        "promote" => set_trust(pool, observation_id, crate::core::trust::TRUSTED).await,
        "quarantine" => set_trust(pool, observation_id, crate::core::trust::QUARANTINED).await,
        "pending" => list_quarantined(pool, &args).await,
        _ => anyhow::bail!(
            "Invalid action: {action}. Use positive/negative/correct/promote/quarantine/pending"
        ),
    }
}

async fn set_trust(pool: &PgPool, observation_id: Option<&str>, trust: &str) -> Result<Value> {
    let obs_id_str = observation_id.context("observation_id is required")?;
    let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
    let project_id = crate::project::current_project_id(pool).await?;
    if !crate::project::observation_in_scope(pool, obs_id, project_id).await? {
        anyhow::bail!("observation not in current project scope");
    }

    let result =
        sqlx::query("UPDATE brain_observations SET trust = $1, updated_at = NOW() WHERE id = $2")
            .bind(trust)
            .bind(obs_id)
            .execute(pool)
            .await
            .context("updating observation trust")?;

    if result.rows_affected() == 0 {
        anyhow::bail!("observation not found: {obs_id}");
    }

    let event = if trust == crate::core::trust::TRUSTED {
        "memory.promote"
    } else {
        "memory.quarantine"
    };
    crate::handlers::archivo::handle(
        pool,
        serde_json::json!({
            "action": "append",
            "event_action": event,
            "payload": { "observation_id": obs_id.to_string(), "trust": trust }
        }),
    )
    .await
    .ok();

    Ok(serde_json::json!({
        "action": if trust == crate::core::trust::TRUSTED { "promote" } else { "quarantine" },
        "observation_id": obs_id.to_string(),
        "trust": trust,
        "retrievable": trust == crate::core::trust::TRUSTED,
    }))
}

async fn list_quarantined(pool: &PgPool, args: &Value) -> Result<Value> {
    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .clamp(1, 200);
    let project_id = crate::project::current_project_id(pool).await?;

    let rows: Vec<(
        uuid::Uuid,
        String,
        String,
        String,
        chrono::DateTime<chrono::Utc>,
    )> = sqlx::query_as(
        "SELECT o.id, e.name, o.content, o.source, o.created_at
             FROM brain_observations o
             JOIN brain_entities e ON e.id = o.entity_id
             WHERE o.trust = 'quarantined'
               AND ($1::uuid IS NULL OR o.project_id = $1 OR o.project_id IS NULL)
             ORDER BY o.created_at DESC
             LIMIT $2",
    )
    .bind(project_id)
    .bind(limit)
    .fetch_all(pool)
    .await
    .context("listing quarantined observations")?;

    let items: Vec<Value> = rows
        .iter()
        .map(|(id, entity, content, source, created)| {
            serde_json::json!({
                "observation_id": id.to_string(),
                "entity": entity,
                "content": content,
                "source": source,
                "created_at": created.to_rfc3339(),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "pending",
        "count": items.len(),
        "quarantined": items,
        "note": "these are withheld from cuba_faro until promoted with action=promote",
    }))
}

async fn positive(
    pool: &PgPool,
    entity_name: Option<&str>,
    observation_id: Option<&str>,
) -> Result<Value> {
    let mut boosted = 0u32;
    let project_id = crate::project::current_project_id(pool).await?;

    if let Some(obs_id_str) = observation_id {
        let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
        if !crate::project::observation_in_scope(pool, obs_id, project_id).await? {
            anyhow::bail!("observation not in current project scope");
        }
        let result = sqlx::query(
            "UPDATE brain_observations SET
                importance = LEAST(
                    importance + (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * (1.0 - importance),
                    1.0
                ),
                access_count = access_count + 1,
                last_accessed = NOW()
             WHERE id = $1",
        )
        .bind(obs_id)
        .execute(pool)
        .await?;
        boosted += result.rows_affected() as u32;
    }

    if let Some(name) = entity_name {
        let result = sqlx::query(
            "UPDATE brain_entities SET
                importance = LEAST(
                    importance + (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * (1.0 - importance),
                    1.0
                ),
                access_count = access_count + 1,
                updated_at = NOW()
             WHERE name = $1
               AND ($2::uuid IS NULL OR project_id = $2 OR project_id IS NULL)",
        )
        .bind(name)
        .bind(project_id)
        .execute(pool)
        .await?;
        boosted += result.rows_affected() as u32;
    }

    Ok(serde_json::json!({
        "action": "positive",
        "boosted_count": boosted,
        "rule": "oja_positive_robbins_monro"
    }))
}

async fn negative(
    pool: &PgPool,
    entity_name: Option<&str>,
    observation_id: Option<&str>,
) -> Result<Value> {
    let mut decreased = 0u32;
    let project_id = crate::project::current_project_id(pool).await?;

    if let Some(obs_id_str) = observation_id {
        let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
        if !crate::project::observation_in_scope(pool, obs_id, project_id).await? {
            anyhow::bail!("observation not in current project scope");
        }
        let result = sqlx::query(
            "UPDATE brain_observations SET
                importance = GREATEST(
                    importance - (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * importance,
                    0.0
                ),
                last_accessed = NOW()
             WHERE id = $1",
        )
        .bind(obs_id)
        .execute(pool)
        .await?;
        decreased += result.rows_affected() as u32;
    }

    if let Some(name) = entity_name {
        let result = sqlx::query(
            "UPDATE brain_entities SET
                importance = GREATEST(
                    importance - (0.05 / SQRT(1.0 + access_count::float8 / 100.0)) * importance,
                    0.0
                ),
                updated_at = NOW()
             WHERE name = $1
               AND ($2::uuid IS NULL OR project_id = $2 OR project_id IS NULL)",
        )
        .bind(name)
        .bind(project_id)
        .execute(pool)
        .await?;
        decreased += result.rows_affected() as u32;
    }

    Ok(serde_json::json!({
        "action": "negative",
        "decreased_count": decreased,
        "rule": "oja_negative_robbins_monro"
    }))
}

async fn correct(pool: &PgPool, observation_id: Option<&str>, args: &Value) -> Result<Value> {
    let obs_id_str = observation_id.context("observation_id required for correct")?;
    let obs_id: uuid::Uuid = obs_id_str.parse().context("invalid observation_id")?;
    let correction = args
        .get("correction")
        .and_then(|v| v.as_str())
        .context("correction text is required")?;

    crate::redact::refuse_secrets(args, "correction", correction)?;

    let project_id = crate::project::current_project_id(pool).await?;
    if !crate::project::observation_in_scope(pool, obs_id, project_id).await? {
        anyhow::bail!("observation not in current project scope");
    }

    let result = sqlx::query(
        "UPDATE brain_observations SET
            previous_versions = previous_versions || jsonb_build_array(
                jsonb_build_object('content', content, 'version', version, 'corrected_at', NOW()::text)
            ),
            content = $2,
            version = version + 1,
            last_accessed = NOW(),
            updated_at = NOW()
         WHERE id = $1"
    )
    .bind(obs_id)
    .bind(correction)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        anyhow::bail!("Observation not found");
    }

    Ok(serde_json::json!({
        "action": "correct",
        "observation_id": obs_id_str,
        "new_content": correction,
        "versioned": true
    }))
}

pub async fn reflect(pool: &PgPool, session_id: uuid::Uuid) -> Result<String> {
    type Row = (String, i64);
    let by_type: Vec<Row> = sqlx::query_as(
        "SELECT observation_type, COUNT(*) FROM brain_observations
         WHERE session_id = $1
         GROUP BY observation_type
         ORDER BY COUNT(*) DESC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let recent: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT e.name, o.observation_type, o.content
         FROM brain_observations o
         JOIN brain_entities e ON o.entity_id = e.id
         WHERE o.session_id = $1 AND o.observation_type != 'superseded'
         ORDER BY o.created_at DESC
         LIMIT 12",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut md = String::with_capacity(2048);
    md.push_str("# Session reflection\n\n");
    if !by_type.is_empty() {
        md.push_str("## Counts by type\n");
        for (t, n) in &by_type {
            md.push_str(&format!("- {t}: {n}\n"));
        }
        md.push('\n');
    }
    if !recent.is_empty() {
        md.push_str("## Recent observations (newest first)\n");
        for (entity, ty, content) in &recent {
            let snippet = crate::handlers::zafra::safe_truncate(content, 160);
            md.push_str(&format!("- [{ty}] {entity}: {snippet}\n"));
        }
    }
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_that_cannot_connect() -> PgPool {
        sqlx::postgres::PgPoolOptions::new()
            .acquire_timeout(std::time::Duration::from_millis(250))
            .connect_lazy("postgres://eco-test:unused@127.0.0.1:63999/does-not-exist")
            .expect("connect_lazy only parses the URL, it does not dial the network")
    }

    #[tokio::test]
    async fn correcting_an_observation_cannot_smuggle_in_what_writing_it_would_have_refused() {
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
        let pool = pool_that_cannot_connect();

        let args = serde_json::json!({
            "action": "correct",
            "observation_id": uuid::Uuid::nil().to_string(),
            "correction": "en realidad la cabecera era ghp_abcdefghijklmnop"
        });

        let Err(failure) = handle(&pool, args).await else {
            panic!(
                "correct answered Ok on a pool that cannot connect: nothing but the secret gate \
                 could have answered, and the gate must refuse"
            );
        };

        let chain = format!("{failure:#}");
        assert!(
            chain.contains("github token") && chain.contains("correction"),
            "correct overwrites the content of an already-stored observation, so an ungated \
             correction is a way to put into the graph exactly what cuba_cronica add refuses to \
             accept — a hole underneath an existing guard. Got: {chain}"
        );
        assert!(
            !chain.contains("ghp_abcdefghijklmnop"),
            "the refusal repeated the secret, and refusals get logged: {chain}"
        );
    }
}
