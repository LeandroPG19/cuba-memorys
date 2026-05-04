//! Project scoping helpers (v0.8).
//!
//! Resolves the active project for the current MCP session and exposes
//! filter clauses that handlers can splice into their SQL.
//!
//! ## Resolution order
//! 1. If `CUBA_PROJECT_FILTER=off` → no filter (admin/debug kill-switch).
//! 2. Else: query the most recent active session (`ended_at IS NULL`) and use
//!    its `project_id`. If no active session or session has NULL project →
//!    no filter (returns rows from all projects + global NULL rows).
//!
//! ## Backward compatibility
//! Legacy rows have `project_id IS NULL` (= global). They remain visible from
//! every project scope. Behavior of v0.7 callers (no `cuba_jornada start`
//! with `project`) is preserved exactly.

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

/// Resolve the active project for the most recent open session.
///
/// Returns:
/// - `Ok(None)` when filter is disabled, no active session exists, or the
///   session has no project assigned. Callers MUST treat None as "no filter".
/// - `Ok(Some(uuid))` when the active session is bound to a project.
///
/// V0.9: also propagates the resolved project (or sentinel `*` when filter
/// is disabled) into PostgreSQL session via `SET LOCAL app.current_project`,
/// which feeds the RLS policies installed by migration 0017. The `SET LOCAL`
/// is best-effort and silently ignored if the migration has not run (older
/// DBs predate RLS).
pub async fn current_project_id(pool: &PgPool) -> Result<Option<Uuid>> {
    if filter_disabled() {
        // Tell RLS to bypass too
        sqlx::query("SET LOCAL app.current_project = '*'")
            .execute(pool)
            .await
            .ok();
        return Ok(None);
    }
    let row: Option<(Option<Uuid>,)> = sqlx::query_as(
        "SELECT project_id FROM brain_sessions
         WHERE ended_at IS NULL
         ORDER BY started_at DESC
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await?;
    let pid = row.and_then(|(p,)| p);
    // Propagate to RLS context. Empty string = no filter (v0.7/v0.8 back-compat).
    let value = pid
        .as_ref()
        .map(|u| u.to_string())
        .unwrap_or_default();
    let stmt = format!(
        "SET LOCAL app.current_project = '{}'",
        value.replace('\'', "''")
    );
    sqlx::query(&stmt).execute(pool).await.ok();
    Ok(pid)
}

/// Resolve a project name to its UUID. Returns None if not found.
pub async fn resolve_project_name(pool: &PgPool, name: &str) -> Result<Option<Uuid>> {
    let row: Option<(Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_projects WHERE name = $1")
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
