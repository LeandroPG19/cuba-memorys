use anyhow::{Context, Result};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use std::str::FromStr;
use std::time::Duration;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

pub async fn create_pool(database_url: &str) -> Result<PgPool> {
    let connect_options = PgConnectOptions::from_str(database_url)
        .context("invalid DATABASE_URL")?
        .log_statements(tracing::log::LevelFilter::Debug)
        .log_slow_statements(tracing::log::LevelFilter::Warn, Duration::from_secs(1));

    let node_name = std::env::var("CUBA_NODE_NAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_default();

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .min_connections(1)
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
        .connect_with(connect_options)
        .await
        .context("failed to connect to PostgreSQL")?;

    tracing::info!("connected to PostgreSQL");

    init_schema(&pool).await?;

    Ok(pool)
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
}
