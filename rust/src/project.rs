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

pub fn rls_scope() -> String {
    if filter_disabled() {
        return "*".to_string();
    }
    crate::session::project_id()
        .map(|u| u.to_string())
        .unwrap_or_default()
}

pub async fn current_project_id(_pool: &PgPool) -> Result<Option<Uuid>> {
    if filter_disabled() {
        return Ok(None);
    }
    Ok(crate::session::project_id())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_scope_the_pool_stamps_is_the_active_project() {
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
        crate::session::clear();
        assert_eq!(
            rls_scope(),
            "",
            "with no session the scope is empty, which the tenant_isolation policy reads \
             as unfiltered — the WHERE clause in each handler is what narrows it there"
        );

        let project = Uuid::new_v4();
        crate::session::set(Uuid::new_v4(), Some(project));
        assert_eq!(
            rls_scope(),
            project.to_string(),
            "before_acquire stamps this onto every connection the pool hands out; if it \
             disagreed with what the handler binds, RLS would clamp to a different project"
        );

        crate::session::clear();
    }

    #[tokio::test]
    async fn the_kill_switch_widens_the_scope_instead_of_emptying_it() {
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
        crate::session::set(Uuid::new_v4(), Some(Uuid::new_v4()));
        unsafe { std::env::set_var(KILL_SWITCH_ENV, "off") };

        assert_eq!(
            rls_scope(),
            "*",
            "the kill switch has to say `*` explicitly: an empty string means the same \
             thing to the policy today, but only `*` survives a policy that stops \
             treating empty as unfiltered"
        );

        unsafe { std::env::remove_var(KILL_SWITCH_ENV) };
        crate::session::clear();
    }
}
