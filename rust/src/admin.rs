use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

type SessionRow = (
    uuid::Uuid,
    Option<String>,
    chrono::DateTime<chrono::Utc>,
    Option<String>,
);

type PeerRow = (
    String,
    String,
    Option<chrono::DateTime<chrono::Utc>>,
    Option<String>,
    Option<String>,
);

type FailureRow = (
    uuid::Uuid,
    String,
    Option<String>,
    String,
    i32,
    chrono::DateTime<chrono::Utc>,
);

type ConflictRow = (uuid::Uuid, uuid::Uuid, String, String, Option<String>);

pub const METHODS: [&str; 4] = [
    "admin/status",
    "admin/clients",
    "admin/traffic",
    "admin/problems",
];

pub fn is_admin_method(method: &str) -> bool {
    METHODS.contains(&method)
}

pub async fn handle(
    pool: &PgPool,
    method: &str,
    uptime_secs: u64,
    connected: Vec<Value>,
) -> Result<Value> {
    match method {
        "admin/status" => status(pool, uptime_secs).await,
        "admin/clients" => clients(pool, connected).await,
        "admin/traffic" => traffic(pool).await,
        "admin/problems" => problems(pool).await,
        _ => anyhow::bail!("unknown admin method: {method}"),
    }
}

async fn status(pool: &PgPool, uptime_secs: u64) -> Result<Value> {
    let url = std::env::var("DATABASE_URL").unwrap_or_default();
    let checks = crate::doctor::run_checks(pool, &url).await;

    let migrations: i64 = sqlx::query_scalar("SELECT count(*) FROM _sqlx_migrations")
        .fetch_one(pool)
        .await
        .unwrap_or(0);
    let size: Option<String> =
        sqlx::query_scalar("SELECT pg_size_pretty(pg_database_size(current_database()))")
            .fetch_optional(pool)
            .await
            .ok()
            .flatten();
    let node: Option<uuid::Uuid> = crate::db::node_id(pool).await.ok();

    let counts: (i64, i64, i64, i64) = sqlx::query_as(
        "SELECT (SELECT count(*) FROM brain_observations)::bigint,
                (SELECT count(*) FROM brain_entities)::bigint,
                (SELECT count(*) FROM brain_facts)::bigint,
                (SELECT count(*) FROM brain_relations)::bigint",
    )
    .fetch_one(pool)
    .await
    .unwrap_or((0, 0, 0, 0));

    Ok(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": uptime_secs,
        "address": crate::http::bind_addr(),
        "node_id": node,
        "node_name": std::env::var("CUBA_NODE_NAME").ok().filter(|v| !v.trim().is_empty()),
        "database": crate::doctor::redact_url(&url),
        "database_size": size,
        "migrations_applied": migrations,
        "embedding_dim": crate::embeddings::onnx::embedding_dim(),
        "tools": crate::constants::tools_for_profile().len(),
        "corpus": {
            "observations": counts.0,
            "entities": counts.1,
            "facts": counts.2,
            "relations": counts.3,
        },
        "checks": checks,
    }))
}

async fn clients(pool: &PgPool, connected: Vec<Value>) -> Result<Value> {
    let sessions: Vec<SessionRow> = sqlx::query_as(
        "SELECT s.id, s.session_name, s.started_at, p.name
             FROM brain_sessions s LEFT JOIN brain_projects p ON p.id = s.project_id
             WHERE s.ended_at IS NULL ORDER BY s.started_at DESC LIMIT 20",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let peers: Vec<PeerRow> = sqlx::query_as(
        "SELECT name, url, last_synced_at, last_manifest_hash, last_error
         FROM brain_sync_peers ORDER BY name",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    Ok(serde_json::json!({
        "connected": connected,
        "sessions": sessions
            .into_iter()
            .map(|(id, name, started, project)| serde_json::json!({
                "id": id, "name": name, "started_at": started, "project": project
            }))
            .collect::<Vec<_>>(),
        "peers": peers
            .into_iter()
            .map(|(name, url, at, hash, err)| serde_json::json!({
                "name": name, "url": url, "last_synced_at": at,
                "last_manifest_hash": hash, "last_error": err
            }))
            .collect::<Vec<_>>(),
    }))
}

async fn traffic(pool: &PgPool) -> Result<Value> {
    let failures: Vec<FailureRow> = sqlx::query_as(
        "SELECT id, tool, client, error, elapsed_ms, created_at
         FROM brain_handler_failures ORDER BY created_at DESC LIMIT 50",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    Ok(serde_json::json!({
        "recent": crate::observability::recent_calls(200),
        "totals": crate::observability::tool_totals(),
        "ring_capacity": crate::observability::ring_capacity(),
        "failures": failures
            .into_iter()
            .map(|(id, tool, client, error, ms, at)| serde_json::json!({
                "id": id, "tool": tool, "client": client,
                "error": error, "elapsed_ms": ms, "at": at
            }))
            .collect::<Vec<_>>(),
        "note": "recent calls live in a ring in the daemon and are lost when it restarts; \
                 only failures are stored. Recording every read would make this table grow \
                 faster than the memory it watches.",
    }))
}

async fn problems(pool: &PgPool) -> Result<Value> {
    let url = std::env::var("DATABASE_URL").unwrap_or_default();
    let failing: Vec<crate::doctor::Check> = crate::doctor::run_checks(pool, &url)
        .await
        .into_iter()
        .filter(|c| c.status != crate::doctor::Status::Ok)
        .collect();

    let conflicts: Vec<ConflictRow> = sqlx::query_as(
        "SELECT id, observation_id, local_content, incoming_content, incoming_origin_node
         FROM brain_sync_conflicts WHERE resolved_at IS NULL
         ORDER BY detected_at DESC LIMIT 50",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let quarantined: i64 = sqlx::query_scalar(
        "SELECT (SELECT count(*) FROM brain_observations WHERE trust = 'quarantined')
              + (SELECT count(*) FROM brain_episodes WHERE trust = 'quarantined')
              + (SELECT count(*) FROM brain_errors WHERE trust = 'quarantined')",
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let unresolved_errors: i64 =
        sqlx::query_scalar("SELECT count(*) FROM brain_errors WHERE NOT resolved")
            .fetch_one(pool)
            .await
            .unwrap_or(0);

    Ok(serde_json::json!({
        "checks": failing,
        "conflicts": conflicts
            .into_iter()
            .map(|(id, obs, ours, theirs, node)| serde_json::json!({
                "id": id, "observation_id": obs, "ours": ours,
                "theirs": theirs, "their_node": node
            }))
            .collect::<Vec<_>>(),
        "peer_notices": crate::handlers::sync::pending_notices(pool).await.unwrap_or_default(),
        "quarantined": quarantined,
        "unresolved_errors": unresolved_errors,
    }))
}
