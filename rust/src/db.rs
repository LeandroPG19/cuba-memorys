//! Database abstraction layer — sqlx PgPool + sqlx-migrate + pgvector.
//!
//! V0.9: migrated from ad-hoc idempotent CREATE IF NOT EXISTS to versioned
//! `sqlx::migrate!()` against `rust/migrations/`. All migrations are still
//! idempotent (DO $$ ... IF NOT EXISTS ... END $$) so legacy DBs (v0.7/v0.8)
//! that already have schema applied remain compatible:
//!
//! - Fresh DB: `_sqlx_migrations` table is empty → all migrations run in order (0001→0025).
//! - Legacy v0.8 DB: `_sqlx_migrations` is empty BUT schema exists → migrations
//!   re-run safely because every block checks `information_schema` first.
//!
//! After this PR, v0.9.x adds new migrations as files `00{14,15,...}_<name>.up.sql`
//! without touching this file again.

use anyhow::{Context, Result};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use std::str::FromStr;
use std::time::Duration;

/// Compile-time embedded migrations from `rust/migrations/`.
/// sqlx-macros walks the directory and bakes the SQL into the binary,
/// so the deployed artifact does not need filesystem access.
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Create and initialize the database connection pool.
pub async fn create_pool(database_url: &str) -> Result<PgPool> {
    let connect_options = PgConnectOptions::from_str(database_url)
        .context("invalid DATABASE_URL")?
        .log_statements(tracing::log::LevelFilter::Debug)
        .log_slow_statements(tracing::log::LevelFilter::Warn, Duration::from_secs(1));

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .min_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .idle_timeout(Duration::from_secs(600))
        .max_lifetime(Duration::from_secs(1800))
        .after_connect(|conn, _meta| {
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
                Ok(())
            })
        })
        .connect_with(connect_options)
        .await
        .context("failed to connect to PostgreSQL")?;

    tracing::info!("connected to PostgreSQL");

    init_schema(&pool).await?;

    Ok(pool)
}

/// Initialize database schema via sqlx-migrate.
///
/// V0.9 change: replaces ~10 hand-rolled `sqlx::raw_sql(MIGRATION_X)` calls
/// with a single `MIGRATOR.run(pool)` against versioned files.
///
/// Behavior:
/// 1. Run all pending migrations (order: 0001 → 0013) from `rust/migrations/`.
///    Each is idempotent so legacy DBs are not affected.
/// 2. Force `SET timezone TO 'UTC'` for exponential decay + REM consistency (FIX R-004).
/// 3. Probe pgvector extension and configure ef_search if present.
pub async fn init_schema(pool: &PgPool) -> Result<()> {
    // Bug 0.7: when the app runs as a NON-superuser (NOSUPERUSER, no CREATE on
    // schema public), it must NOT run migrations — DDL is a deploy-time task for
    // an admin role (cuba). `CUBA_SKIP_MIGRATIONS=1` skips the migrator and just
    // asserts the schema was already migrated, so RLS/audit stay enforced at
    // runtime without granting the app DDL privileges.
    let skip = std::env::var("CUBA_SKIP_MIGRATIONS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false);

    if skip {
        // Read-only sanity check: the schema must already be present. Fail loudly
        // (never silently) if an admin has not applied migrations first.
        let applied: Option<(i64,)> = sqlx::query_as(
            "SELECT MAX(version) FROM _sqlx_migrations WHERE success = TRUE",
        )
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

    // FIX R-004: Force UTC timezone for exponential decay + REM consistency
    sqlx::query("SET timezone TO 'UTC'")
        .execute(pool)
        .await
        .context("failed to set timezone to UTC")?;

    tracing::info!("schema initialized (timezone=UTC)");

    // Check pgvector extension
    let pgvector_check: Option<(String,)> =
        sqlx::query_as("SELECT extname::text FROM pg_extension WHERE extname = 'vector'")
            .fetch_optional(pool)
            .await?;

    if pgvector_check.is_some() {
        tracing::info!("pgvector extension detected");
        // P4: Set ef_search=100 for better recall (default 40, pgvector benchmarks 2025)
        sqlx::query("SET hnsw.ef_search = 100")
            .execute(pool)
            .await
            .ok();
    } else {
        tracing::warn!("pgvector extension NOT found — vector search disabled");
    }

    Ok(())
}

/// Refuse to serve with a model whose dimension disagrees with the column.
///
/// The server used to start happily in this state. It would answer every query,
/// return ten confident rows, and be silently wrong: pgvector rejects a
/// comparison between a 384-d query and a 1024-d column, the vector branch fails,
/// and the hybrid search collapses into a lexical one. Nothing in any response
/// said so — which is the same failure mode as the original "dead vector branch"
/// bug, arriving by a different road.
///
/// A server that cannot do its job must say so at startup, not degrade quietly
/// for weeks. Only enforced when a real ONNX model is loaded: with no model there
/// is no vector branch to break, and a lexical-only server is a legitimate (if
/// diminished) thing to be.
pub async fn assert_embedding_dim(pool: &PgPool) -> Result<()> {
    if !crate::embeddings::onnx::is_model_loaded() {
        return Ok(());
    }
    let runtime_dim = crate::embeddings::onnx::embedding_dim();

    let column: Option<String> = sqlx::query_scalar(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute
         WHERE attrelid = 'brain_observations'::regclass AND attname = 'embedding'",
    )
    .fetch_optional(pool)
    .await
    .context("reading the embedding column type")?;

    // No column yet (fresh database, migrations about to run) — nothing to check.
    let Some(column) = column else {
        return Ok(());
    };

    let expected = format!("vector({runtime_dim})");
    if column != expected {
        anyhow::bail!(
            "el modelo de embeddings produce {expected} pero la columna es {column}.\n\
             El servidor NO arranca así: cada búsqueda vectorial fallaría y la búsqueda \n\
             híbrida devolvería resultados solo léxicos sin avisar de nada.\n\n\
             Si cambiaste de modelo:  scripts/migrate-embedding-dim.sh {runtime_dim}  y después  cuba-memorys reembed\n\
             Si no querías cambiarlo: revisá CUBA_EMBEDDING_DIM y ONNX_MODEL_PATH en la config del cliente MCP."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check that the migrator exists and resolved the directory.
    /// If migrations are missing, sqlx::migrate! macro fails to compile.
    #[test]
    fn migrator_loaded() {
        let count = MIGRATOR.iter().count();
        assert!(
            count >= 25,
            "expected at least 25 migrations (0001-0025), got {count}"
        );
    }

    /// Verify migrations are applied in monotonic version order.
    #[test]
    fn migrations_in_order() {
        let versions: Vec<i64> = MIGRATOR.iter().map(|m| m.version).collect();
        let mut sorted = versions.clone();
        sorted.sort();
        assert_eq!(versions, sorted, "migrations must be in sorted order");
    }
}
