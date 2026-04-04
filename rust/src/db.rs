//! Database abstraction layer — sqlx PgPool + schema init + pgvector.
//!
//! FIX V3: Built-in connection retry via sqlx pool options.

use anyhow::{Context, Result};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use std::str::FromStr;
use std::time::Duration;

/// Schema SQL embedded at compile time.
const SCHEMA_SQL: &str = include_str!("schema.sql");

/// FIX-OBS-001: Add updated_at to brain_observations.
/// The column was missing from schema but referenced in decay/eco/REM queries,
/// causing all those operations to fail silently.
const OBS_UPDATED_AT_MIGRATION: &str = r#"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'brain_observations' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE brain_observations ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;
"#;

/// Episodic memory migration — creates brain_episodes table if missing.
/// Safe to run on existing DBs (IF NOT EXISTS throughout).
const EPISODES_MIGRATION: &str = r#"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_episodes'
    ) THEN
        CREATE TABLE brain_episodes (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            entity_id UUID NOT NULL REFERENCES brain_entities(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            ended_at TIMESTAMPTZ,
            actors TEXT[] DEFAULT '{}',
            artifacts TEXT[] DEFAULT '{}',
            importance FLOAT DEFAULT 0.5
                CHECK (importance >= 0.0 AND importance <= 1.0),
            embedding vector(384),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', content)
            ) STORED
        );
        CREATE INDEX idx_episodes_entity ON brain_episodes(entity_id);
        CREATE INDEX idx_episodes_search ON brain_episodes USING GIN(search_vector);
        CREATE INDEX idx_episodes_trgm   ON brain_episodes USING GIN(content gin_trgm_ops);
        CREATE INDEX idx_episodes_time   ON brain_episodes(started_at DESC);
        -- HNSW vector index for episode semantic search
        CREATE INDEX idx_episodes_embedding_hnsw
            ON brain_episodes USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 128);
    END IF;
END $$;
"#;

/// Prospective memory triggers migration (cuba_centinela).
const TRIGGERS_MIGRATION: &str = r#"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_triggers'
    ) THEN
        CREATE TABLE brain_triggers (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            entity_pattern TEXT NOT NULL,
            condition_type TEXT NOT NULL
                CHECK (condition_type IN ('on_access', 'on_session_start', 'on_error_match')),
            message TEXT NOT NULL,
            observation_id UUID REFERENCES brain_observations(id) ON DELETE SET NULL,
            active BOOLEAN DEFAULT TRUE,
            fire_count INT DEFAULT 0,
            max_fires INT DEFAULT 1,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
        CREATE INDEX idx_triggers_active ON brain_triggers(active) WHERE active = TRUE;
        CREATE INDEX idx_triggers_pattern ON brain_triggers USING GIN(entity_pattern gin_trgm_ops);
    END IF;
END $$;
"#;

/// Bayesian calibration verify log migration (cuba_calibrar).
const VERIFY_LOG_MIGRATION: &str = r#"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'brain_verify_log'
    ) THEN
        CREATE TABLE brain_verify_log (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            claim TEXT NOT NULL,
            entity_name TEXT,
            confidence FLOAT NOT NULL,
            grounding_level TEXT NOT NULL,
            outcome TEXT DEFAULT 'pending'
                CHECK (outcome IN ('pending', 'correct', 'incorrect', 'unknown')),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_verify_log_entity ON brain_verify_log(entity_name);
        CREATE INDEX idx_verify_log_outcome ON brain_verify_log(outcome);
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

    // FIX R-004: Force UTC timezone for exponential decay + REM consistency
    sqlx::query("SET timezone TO 'UTC'")
        .execute(pool)
        .await
        .context("failed to set timezone to UTC")?;

    tracing::info!("schema initialized (timezone=UTC)");

    // Episodic memory table (idempotent)
    sqlx::raw_sql(EPISODES_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply episodes migration")?;

    tracing::info!("brain_episodes table verified");

    // FIX-OBS-001: Add updated_at to brain_observations (idempotent)
    sqlx::raw_sql(OBS_UPDATED_AT_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply obs updated_at migration")?;

    tracing::info!("brain_observations.updated_at verified");

    // Apply BCM theta migration (idempotent) — V3 Deep Research
    sqlx::raw_sql(BCM_THETA_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply BCM theta migration")?;

    tracing::info!("BCM theta column verified");

    // Triggers table (cuba_centinela) — idempotent
    sqlx::raw_sql(TRIGGERS_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply triggers migration")?;

    tracing::info!("brain_triggers table verified");

    // Verify log table (cuba_calibrar) — idempotent
    sqlx::raw_sql(VERIFY_LOG_MIGRATION)
        .execute(pool)
        .await
        .context("failed to apply verify_log migration")?;

    tracing::info!("brain_verify_log table verified");

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
