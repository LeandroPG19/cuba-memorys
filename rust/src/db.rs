//! Database abstraction layer — sqlx PgPool + schema init + pgvector.
//!
//! FIX V3: Built-in connection retry via sqlx pool options.
//! FIX B4: Schema init includes Dual-Strength columns.

use anyhow::{Context, Result};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use std::str::FromStr;
use std::time::Duration;

/// Schema SQL embedded at compile time — identical to Python + Dual-Strength cols.
const SCHEMA_SQL: &str = include_str!("schema.sql");

/// Dual-Strength migration — adds storage_strength + retrieval_strength columns.
const DUAL_STRENGTH_MIGRATION: &str = r#"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_observations'
        AND column_name = 'storage_strength'
    ) THEN
        ALTER TABLE brain_observations
            ADD COLUMN storage_strength FLOAT DEFAULT 0.5,
            ADD COLUMN retrieval_strength FLOAT DEFAULT 1.0;
    END IF;
END $$;
"#;

/// BCM θ_M EMA migration — adds bcm_theta column for persistent sliding threshold.
/// V3: Deep Research 2026-03-14.
const BCM_THETA_MIGRATION: &str = r#"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_entities'
        AND column_name = 'bcm_theta'
    ) THEN
        ALTER TABLE brain_entities
            ADD COLUMN bcm_theta FLOAT DEFAULT 10.0;
    END IF;
END $$;
"#;

/// Create and initialize the database connection pool.
///
/// Args:
///     database_url: PostgreSQL connection string.
///
/// Returns:
///     Configured PgPool with schema initialized.
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
        .max_lifetime(Duration::from_secs(1800)) // pool_recycle equivalent
        .connect_with(connect_options)
        .await
        .context("failed to connect to PostgreSQL")?;

    tracing::info!("connected to PostgreSQL");

    // Initialize schema
    init_schema(&pool).await?;

    Ok(pool)
}

/// Initialize database schema — extensions, tables, indexes, migrations.
async fn init_schema(pool: &PgPool) -> Result<()> {
    // Execute base schema (CREATE IF NOT EXISTS — idempotent)
    sqlx::raw_sql(SCHEMA_SQL)
        .execute(pool)
        .await
        .context("failed to initialize schema")?;

    // FIX R-004: Force UTC timezone for FSRS decay + REM consistency
    sqlx::query("SET timezone TO 'UTC'")
        .execute(pool)
        .await
        .context("failed to set timezone to UTC")?;

    tracing::info!("schema initialized (timezone=UTC)");

    // Apply Dual-Strength migration (idempotent)
    sqlx::raw_sql(DUAL_STRENGTH_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply Dual-Strength migration")?;

    tracing::info!("Dual-Strength columns verified");

    // Apply BCM theta migration (idempotent) — V3 Deep Research
    sqlx::raw_sql(BCM_THETA_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply BCM theta migration")?;

    tracing::info!("BCM theta column verified");

    // Check pgvector extension
    let pgvector_check: Option<(String,)> = sqlx::query_as(
        "SELECT extname::text FROM pg_extension WHERE extname = 'vector'"
    )
    .fetch_optional(pool)
    .await?;

    if pgvector_check.is_some() {
        tracing::info!("pgvector extension detected");
        // P4: Set ef_search=100 for better recall (default 40, pgvector benchmarks 2025)
        sqlx::query("SET hnsw.ef_search = 100")
            .execute(pool)
            .await
            .ok(); // Non-fatal if setting not supported
    } else {
        tracing::warn!("pgvector extension NOT found — vector search disabled");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_sql_is_not_empty() {
        assert!(!SCHEMA_SQL.is_empty());
        assert!(SCHEMA_SQL.contains("brain_entities"));
        assert!(SCHEMA_SQL.contains("brain_observations"));
        assert!(SCHEMA_SQL.contains("brain_relations"));
    }
}
