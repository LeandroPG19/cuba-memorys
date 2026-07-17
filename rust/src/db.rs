use anyhow::{Context, Result};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use std::str::FromStr;
use std::time::Duration;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

fn connect_options(database_url: &str) -> Result<PgConnectOptions> {
    Ok(PgConnectOptions::from_str(database_url)
        .context("invalid DATABASE_URL")?
        .log_statements(tracing::log::LevelFilter::Debug)
        .log_slow_statements(tracing::log::LevelFilter::Warn, Duration::from_secs(1)))
}

/// Shared by both pools. `after_connect` is the part that matters: every
/// connection must land in UTC, or exponential decay and the REM cycle
/// silently drift.
fn pool_options() -> PgPoolOptions {
    let node_name = std::env::var("CUBA_NODE_NAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_default();

    PgPoolOptions::new()
        .max_connections(10)
        .acquire_timeout(Duration::from_secs(5))
        .idle_timeout(Duration::from_secs(600))
        .max_lifetime(Duration::from_secs(1800))
        .after_connect(move |conn, _meta| {
            let node = node_name.clone();
            Box::pin(async move {
                sqlx::query("SET timezone TO 'UTC'")
                    .execute(&mut *conn)
                    .await?;
                sqlx::query("SET hnsw.ef_search = 100")
                    .execute(&mut *conn)
                    .await
                    .ok();
                sqlx::query("SELECT set_config('app.current_project', '', false)")
                    .execute(&mut *conn)
                    .await
                    .ok();
                sqlx::query("SELECT set_config('cuba.node_name', $1, false)")
                    .bind(&node)
                    .execute(&mut *conn)
                    .await
                    .ok();
                Ok(())
            })
        })
        // `project::current_project_id` sets `app.current_project` with
        // `set_config(..., false)` (session-scoped, not transaction-local)
        // through a bare `.execute(pool)` call, so the value it leaves behind
        // outlives that single logical request on whichever physical
        // connection the pool happened to hand it. Without this hook a
        // connection released back to the idle queue keeps carrying a
        // previous request's (or previous project's) GUC value into
        // whatever unrelated request acquires it next. Resetting here on
        // every release closes that gap: a reused connection always rejoins
        // the pool at the same '' default `after_connect` establishes for
        // brand-new connections, instead of leaking a stale project id.
        .after_release(|conn, _meta| {
            Box::pin(async move {
                sqlx::query("SELECT set_config('app.current_project', '', false)")
                    .execute(&mut *conn)
                    .await
                    .ok();
                Ok(true)
            })
        })
}

pub async fn create_pool(database_url: &str) -> Result<PgPool> {
    let pool = pool_options()
        .min_connections(1)
        .connect_with(connect_options(database_url)?)
        .await
        .context("failed to connect to PostgreSQL")?;

    tracing::info!("connected to PostgreSQL");

    init_schema(&pool).await?;

    Ok(pool)
}

/// A pool that has not connected to anything yet.
///
/// `connect_lazy_with` cannot fail: it hands back a pool whose *first query*
/// is what reaches PostgreSQL — and what reports it unreachable. That is the
/// difference between an MCP server that cannot serve a tool call and one
/// that never speaks the protocol at all.
///
/// A `database_url` that `PgConnectOptions::from_str` itself rejects (missing
/// scheme, stray characters, ...) gets the same treatment instead of
/// propagating that parse error: falling back to `PgConnectOptions::new()`
/// keeps this function infallible, so a malformed URL also fails at first
/// query instead of taking the process down before it can speak MCP.
///
/// `min_connections` is deliberately NOT set here. A lazy pool that insists on
/// keeping one connection open would spend the whole session retrying a
/// database that is not there.
///
/// Migrations are not run: `init_schema` needs a live connection. If
/// PostgreSQL shows up later, the schema is whatever the last successful
/// startup left.
pub fn create_lazy_pool(database_url: &str) -> PgPool {
    let options = connect_options(database_url).unwrap_or_else(|_| {
        PgConnectOptions::new()
            .log_statements(tracing::log::LevelFilter::Debug)
            .log_slow_statements(tracing::log::LevelFilter::Warn, Duration::from_secs(1))
    });
    pool_options().connect_lazy_with(options)
}

pub async fn init_schema(pool: &PgPool) -> Result<()> {
    let skip = std::env::var("CUBA_SKIP_MIGRATIONS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false);

    if skip {
        let applied: Option<(i64,)> =
            sqlx::query_as("SELECT MAX(version) FROM _sqlx_migrations WHERE success = TRUE")
                .fetch_optional(pool)
                .await
                .context(
                    "CUBA_SKIP_MIGRATIONS is set but _sqlx_migrations is unreadable — \
             run migrations once as an admin role before starting the app",
                )?;
        match applied.map(|(v,)| v) {
            Some(v) => tracing::warn!(
                latest_migration = v,
                "CUBA_SKIP_MIGRATIONS active — skipping migrator (non-superuser runtime)"
            ),
            None => anyhow::bail!(
                "CUBA_SKIP_MIGRATIONS is set but no migrations are applied — \
                 initialize the database with an admin role first"
            ),
        }
    } else {
        MIGRATOR
            .run(pool)
            .await
            .context("failed to run sqlx migrations")?;

        tracing::info!("sqlx migrations applied");
    }

    sqlx::query("SET timezone TO 'UTC'")
        .execute(pool)
        .await
        .context("failed to set timezone to UTC")?;

    tracing::info!("schema initialized (timezone=UTC)");

    let pgvector_check: Option<(String,)> =
        sqlx::query_as("SELECT extname::text FROM pg_extension WHERE extname = 'vector'")
            .fetch_optional(pool)
            .await?;

    if pgvector_check.is_some() {
        tracing::info!("pgvector extension detected");
        sqlx::query("SET hnsw.ef_search = 100")
            .execute(pool)
            .await
            .ok();
    } else {
        tracing::warn!("pgvector extension NOT found — vector search disabled");
    }

    Ok(())
}

pub async fn assert_embedding_dim(pool: &PgPool) -> Result<()> {
    if !crate::embeddings::onnx::is_model_loaded() {
        return Ok(());
    }
    let runtime_dim = crate::embeddings::onnx::embedding_dim();
    let expected = format!("vector({runtime_dim})");

    let columns: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT c.relname::text, a.attname::text, format_type(a.atttypid, a.atttypmod)::text
         FROM pg_attribute a
         JOIN pg_class c ON c.oid = a.attrelid
         JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public'
           AND c.relkind = 'r'
           AND a.attnum > 0 AND NOT a.attisdropped
           AND format_type(a.atttypid, a.atttypmod) LIKE 'vector(%'
         ORDER BY c.relname",
    )
    .fetch_all(pool)
    .await
    .context("reading the vector column types")?;

    if columns.is_empty() {
        return Ok(());
    }

    let mismatched: Vec<String> = columns
        .iter()
        .filter(|(_, _, ty)| ty != &expected)
        .map(|(t, c, ty)| format!("  {t}.{c} es {ty}"))
        .collect();

    if !mismatched.is_empty() {
        anyhow::bail!(
            "el modelo de embeddings produce {expected}, pero estas columnas no coinciden:\n\
             {}\n\n\
             El servidor NO arranca así: las escrituras a esas tablas fallarían, y la búsqueda\n\
             vectorial devolvería resultados solo léxicos sin avisar de nada.\n\n\
             Si cambiaste de modelo:  scripts/migrate-embedding-dim.sh {runtime_dim}  y después  cuba-memorys reembed\n\
             Si no querías cambiarlo: revisá CUBA_EMBEDDING_DIM y ONNX_MODEL_PATH en la config del cliente MCP.",
            mismatched.join("\n")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrator_loaded() {
        let count = MIGRATOR.iter().count();
        assert!(
            count >= 25,
            "expected at least 25 migrations (0001-0025), got {count}"
        );
    }

    #[test]
    fn migrations_in_order() {
        let versions: Vec<i64> = MIGRATOR.iter().map(|m| m.version).collect();
        let mut sorted = versions.clone();
        sorted.sort();
        assert_eq!(versions, sorted, "migrations must be in sorted order");
    }

    #[tokio::test]
    async fn create_lazy_pool_survives_a_malformed_database_url() {
        // A syntactically invalid DATABASE_URL must not make this function
        // fail: that would propagate out of run_mcp() and exit the process
        // before it ever speaks the MCP protocol — the exact bug this
        // fallback exists to avoid, just triggered by a bad string instead
        // of an unreachable host.
        for bad_url in ["not a url", "", "://nope", "🦀🦀🦀"] {
            let pool = create_lazy_pool(bad_url);
            assert_eq!(
                pool.size(),
                0,
                "a lazy pool must not have connected to anything yet for input {bad_url:?}"
            );
        }
    }

    #[tokio::test]
    async fn released_connection_does_not_leak_app_current_project() {
        let Ok(url) = std::env::var("DATABASE_URL") else {
            eprintln!(
                "skipping released_connection_does_not_leak_app_current_project: DATABASE_URL not set"
            );
            return;
        };

        // max_connections(1) forces the second query below to reuse the
        // exact physical connection the first query released, making this
        // deterministic instead of racing the pool for which connection it
        // hands back.
        let pool = pool_options()
            .max_connections(1)
            .connect_with(connect_options(&url).expect("valid DATABASE_URL"))
            .await
            .expect("connect to test database");

        sqlx::query(
            "SELECT set_config('app.current_project', 'deadbeef-0000-0000-0000-000000000000', false)",
        )
        .execute(&pool)
        .await
        .expect("set_config on connection #1");

        let (leaked,): (String,) =
            sqlx::query_as("SELECT current_setting('app.current_project', true)")
                .fetch_one(&pool)
                .await
                .expect("read back app.current_project on the reused connection");

        assert_eq!(
            leaked, "",
            "after_release must reset app.current_project so a reused connection \
             does not carry a stale project id into the next request"
        );
    }
}
