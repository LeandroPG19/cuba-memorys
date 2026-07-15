use anyhow::Result;
use sqlx::PgPool;
use std::env;
use uuid::Uuid;

use crate::constants::KILL_SWITCH_ENV;

pub fn filter_disabled() -> bool {
    env::var(KILL_SWITCH_ENV)
        .ok()
        .is_some_and(|v| v.eq_ignore_ascii_case("off"))
}

pub fn project_scope_clause(project_param: &str) -> String {
    format!("({project_param}::uuid IS NULL OR project_id = {project_param} OR project_id IS NULL)")
}

async fn propagate_rls_guc(pool: &PgPool, value: &str) -> Result<()> {
    sqlx::query("SELECT set_config('app.current_project', $1, false)")
        .bind(value)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn current_project_id(pool: &PgPool) -> Result<Option<Uuid>> {
    if filter_disabled() {
        propagate_rls_guc(pool, "*").await.ok();
        return Ok(None);
    }
    let pid = crate::session::project_id();
    let value = pid.as_ref().map(|u| u.to_string()).unwrap_or_default();
    propagate_rls_guc(pool, &value).await.ok();
    Ok(pid)
}

pub async fn resolve_project_name(pool: &PgPool, name: &str) -> Result<Option<Uuid>> {
    let row: Option<(Uuid,)> = sqlx::query_as("SELECT id FROM brain_projects WHERE name = $1")
        .bind(name)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|(id,)| id))
}

pub async fn upsert_project(pool: &PgPool, name: &str) -> Result<Uuid> {
    let row: (Uuid,) = sqlx::query_as(
        "INSERT INTO brain_projects (name) VALUES ($1)
         ON CONFLICT (name) DO UPDATE SET last_active_at = NOW()
         RETURNING id",
    )
    .bind(name)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

pub async fn observation_in_scope(
    pool: &PgPool,
    observation_id: Uuid,
    project_id: Option<Uuid>,
) -> Result<bool> {
    if filter_disabled() {
        return Ok(true);
    }
    let Some(pid) = project_id else {
        return Ok(true);
    };
    let row: Option<(i32,)> = sqlx::query_as(
        "SELECT 1 FROM brain_observations
         WHERE id = $1 AND (project_id = $2 OR project_id IS NULL)
         LIMIT 1",
    )
    .bind(observation_id)
    .bind(pid)
    .fetch_optional(pool)
    .await?;
    Ok(row.is_some())
}
