//! Project scoping helpers (v0.8).

use anyhow::Result;
use sqlx::PgPool;
use std::env;
use uuid::Uuid;

use crate::constants::KILL_SWITCH_ENV;

/// Returns true when the kill-switch env var disables all project filtering.
pub fn filter_disabled() -> bool {
    env::var(KILL_SWITCH_ENV)
        .ok()
        .is_some_and(|v| v.eq_ignore_ascii_case("off"))
}

/// SQL fragment: restrict rows to active project + global NULL rows.
/// Bind `$n` to `Option<Uuid>` from [`current_project_id`].
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

/// Resolve the active project for the session **this process** opened.
///
/// Previously this ran `SELECT ... WHERE ended_at IS NULL ORDER BY started_at
/// DESC LIMIT 1`, which is a global question: with several MCP clients sharing
/// one database, the newest session anywhere won, and observations were tagged
/// with someone else's project. See [`crate::session`].
///
/// No session in this process => `None` => unscoped, which is the safe answer.
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

/// Resolve a project name to its UUID. Returns None if not found.
pub async fn resolve_project_name(pool: &PgPool, name: &str) -> Result<Option<Uuid>> {
    let row: Option<(Uuid,)> = sqlx::query_as("SELECT id FROM brain_projects WHERE name = $1")
        .bind(name)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|(id,)| id))
}

/// Resolve a project name to its UUID, creating it if missing.
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

/// Verify an observation belongs to the current project scope (IDOR guard).
pub async fn observation_in_scope(
    pool: &PgPool,
    observation_id: Uuid,
    project_id: Option<Uuid>,
) -> Result<bool> {
    if filter_disabled() {
        return Ok(true);
    }
    // let-else instead of is_none() + unwrap(): the compiler now enforces what a
    // comment used to promise.
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
