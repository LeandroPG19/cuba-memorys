//! `cuba-memorys doctor` — read-only health check.
//!
//! Every one of the four critical v0.10 bugs was **silent**: the server kept
//! answering, just wrong. Recall collapsed to zero because `plainto_tsquery`
//! ANDs its terms; the vector branch returned an empty vec whenever the ONNX
//! model was missing; the OOD threshold was inverted and flagged the entire
//! corpus as out-of-distribution; and the session/project id was resolved
//! globally across MCP processes. Nothing in any response said "degraded".
//!
//! This module is the answer: one command that asserts the invariants those
//! bugs violated, and exits non-zero when the system is lying to you. It only
//! ever issues `SELECT`s — running it can never change what the brain knows.

use std::time::Instant;

use anyhow::{Context, Result};
use sqlx::{PgPool, Row};

use crate::embeddings::onnx;

/// A healthy schema has this many `brain_*` tables (verified against the
/// known-good backup: 23).
const EXPECTED_TABLES: i64 = 23;

/// Extensions the retrieval stack cannot work without.
const REQUIRED_EXTENSIONS: [&str; 2] = ["vector", "pg_trgm"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Invariant holds.
    Ok,
    /// Degraded but serving: worth knowing, not worth failing CI.
    Warn,
    /// Broken: retrieval is wrong, or will be.
    Fail,
}

impl Status {
    fn glyph(self) -> &'static str {
        match self {
            Status::Ok => "  ok  ",
            Status::Warn => " warn ",
            Status::Fail => " FAIL ",
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
    /// What to actually do about it. Only set when something is wrong.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl Check {
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self { name: name.into(), status: Status::Ok, detail: detail.into(), hint: None }
    }
    fn warn(name: &str, detail: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Warn,
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }
    fn fail(name: &str, detail: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Fail,
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }
}

/// Strip the password out of a Postgres URL so the report is safe to paste.
pub fn redact_url(url: &str) -> String {
    // postgresql://user:password@host:port/db → postgresql://user:***@host:port/db
    let Some((scheme, rest)) = url.split_once("://") else {
        return "<unparseable>".to_string();
    };
    let Some((creds, host)) = rest.split_once('@') else {
        return format!("{scheme}://{rest}");
    };
    let user = creds.split_once(':').map_or(creds, |(u, _)| u);
    format!("{scheme}://{user}:***@{host}")
}

/// Parse the declared dimension out of a pgvector column type, e.g. `vector(384)`.
fn parse_vector_dim(col_type: &str) -> Option<i64> {
    col_type
        .trim()
        .strip_prefix("vector(")?
        .strip_suffix(')')?
        .parse()
        .ok()
}

/// Run every check. Read-only: issues `SELECT`s and nothing else.
pub async fn run_checks(pool: &PgPool, url: &str) -> Vec<Check> {
    let mut checks = Vec::new();

    checks.push(Check::ok("database_url", redact_url(url)));

    // --- Connectivity -------------------------------------------------------
    let t0 = Instant::now();
    match sqlx::query("SELECT 1").fetch_one(pool).await {
        Ok(_) => checks.push(Check::ok(
            "connection",
            format!("responde en {} ms", t0.elapsed().as_millis()),
        )),
        Err(e) => {
            checks.push(Check::fail(
                "connection",
                format!("no se pudo consultar: {e}"),
                "verificá que el contenedor de Postgres esté arriba y que DATABASE_URL apunte a él",
            ));
            return checks; // nothing below can run
        }
    }

    // --- Extensions ---------------------------------------------------------
    match sqlx::query("SELECT extname::text FROM pg_extension")
        .fetch_all(pool)
        .await
    {
        Ok(rows) => {
            let have: Vec<String> = rows.iter().filter_map(|r| r.try_get(0).ok()).collect();
            let missing: Vec<&str> = REQUIRED_EXTENSIONS
                .iter()
                .filter(|e| !have.iter().any(|h| h == *e))
                .copied()
                .collect();
            if missing.is_empty() {
                checks.push(Check::ok("extensions", have.join(", ")));
            } else {
                checks.push(Check::fail(
                    "extensions",
                    format!("faltan: {}", missing.join(", ")),
                    "CREATE EXTENSION vector; CREATE EXTENSION pg_trgm;",
                ));
            }
        }
        Err(e) => checks.push(Check::warn("extensions", format!("no se pudo leer: {e}"), "revisar permisos")),
    }

    // --- Migrations ---------------------------------------------------------
    match sqlx::query(
        "SELECT count(*)::bigint AS total, count(*) FILTER (WHERE NOT success)::bigint AS dirty
         FROM _sqlx_migrations",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let total: i64 = row.try_get("total").unwrap_or(0);
            let dirty: i64 = row.try_get("dirty").unwrap_or(0);
            if dirty > 0 {
                checks.push(Check::fail(
                    "migrations",
                    format!("{total} aplicadas, {dirty} en estado dirty"),
                    "una migración quedó a medias: restaurá desde backup antes de seguir",
                ));
            } else {
                checks.push(Check::ok("migrations", format!("{total} aplicadas, ninguna dirty")));
            }
        }
        Err(e) => checks.push(Check::fail(
            "migrations",
            format!("no existe _sqlx_migrations: {e}"),
            "la base no está inicializada — arrancá el servidor una vez para que migre",
        )),
    }

    // --- Schema shape -------------------------------------------------------
    match sqlx::query(
        "SELECT count(*)::bigint FROM information_schema.tables
         WHERE table_schema='public' AND table_name LIKE 'brain%'",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let n: i64 = row.try_get(0).unwrap_or(0);
            if n >= EXPECTED_TABLES {
                checks.push(Check::ok("schema", format!("{n} tablas brain_*")));
            } else {
                checks.push(Check::fail(
                    "schema",
                    format!("{n} tablas brain_* (se esperaban {EXPECTED_TABLES})"),
                    "esquema incompleto — corré las migraciones",
                ));
            }
        }
        Err(e) => checks.push(Check::warn("schema", format!("no se pudo contar: {e}"), "revisar permisos")),
    }

    // --- Bug #1: the OR-tsquery function (silent zero recall without it) -----
    match sqlx::query("SELECT count(*)::bigint FROM pg_proc WHERE proname='cuba_or_tsquery'")
        .fetch_one(pool)
        .await
    {
        Ok(row) => {
            let n: i64 = row.try_get(0).unwrap_or(0);
            if n > 0 {
                checks.push(Check::ok("recall/or_tsquery", "presente (migración 0026)"));
            } else {
                checks.push(Check::fail(
                    "recall/or_tsquery",
                    "falta la función cuba_or_tsquery",
                    "sin ella plainto_tsquery une los términos con AND: una query descriptiva \
                     larga devuelve CERO resultados, en silencio. Aplicá la migración 0026.",
                ));
            }
        }
        Err(e) => checks.push(Check::warn("recall/or_tsquery", format!("no verificable: {e}"), "revisar permisos")),
    }

    // --- Bug #2: the vector branch dies silently without a model ------------
    let model = onnx::current_model();
    if onnx::is_model_loaded() {
        checks.push(Check::ok("onnx_model", format!("cargado ({model})")));
    } else {
        let path = std::env::var("ONNX_MODEL_PATH").unwrap_or_else(|_| "<no seteado>".into());
        checks.push(Check::fail(
            "onnx_model",
            format!("NO cargado — ONNX_MODEL_PATH={path}"),
            "la búsqueda vectorial devuelve vacío en silencio y el retrieval queda solo léxico. \
             Seteá ONNX_MODEL_PATH y ORT_DYLIB_PATH en el entorno del proceso MCP (no solo en tu shell).",
        ));
    }

    // --- Dimension agreement: model vs column (breaks everything if off) ----
    let runtime_dim = onnx::embedding_dim() as i64;
    match sqlx::query(
        "SELECT format_type(atttypid, atttypmod)::text FROM pg_attribute
         WHERE attrelid='brain_observations'::regclass AND attname='embedding'",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let col_type: String = row.try_get(0).unwrap_or_default();
            match parse_vector_dim(&col_type) {
                Some(col_dim) if col_dim == runtime_dim => checks.push(Check::ok(
                    "embedding_dim",
                    format!("runtime {runtime_dim}-d == columna {col_type}"),
                )),
                Some(col_dim) => checks.push(Check::fail(
                    "embedding_dim",
                    format!("runtime {runtime_dim}-d ≠ columna vector({col_dim})"),
                    "el modelo y la columna no coinciden: todo insert vectorial falla o se corrompe. \
                     Alineá CUBA_EMBEDDING_DIM con la columna, o migrá con scripts/migrate-embedding-dim.sh",
                )),
                None => checks.push(Check::warn(
                    "embedding_dim",
                    format!("tipo de columna inesperado: {col_type}"),
                    "se esperaba vector(N)",
                )),
            }
        }
        Err(e) => checks.push(Check::warn("embedding_dim", format!("no verificable: {e}"), "revisar esquema")),
    }

    // --- Coverage: observations with no vector are invisible to dense search -
    match sqlx::query(
        "SELECT count(*)::bigint AS total,
                count(*) FILTER (WHERE embedding IS NULL)::bigint AS missing,
                count(*) FILTER (WHERE embedding IS NOT NULL AND embedding_model IS DISTINCT FROM $1)::bigint AS stale
         FROM brain_observations",
    )
    .bind(&model)
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let total: i64 = row.try_get("total").unwrap_or(0);
            let missing: i64 = row.try_get("missing").unwrap_or(0);
            let stale: i64 = row.try_get("stale").unwrap_or(0);
            if missing == 0 && stale == 0 {
                checks.push(Check::ok(
                    "embedding_coverage",
                    format!("{total} observaciones, todas con vector del modelo actual"),
                ));
            } else {
                checks.push(Check::warn(
                    "embedding_coverage",
                    format!("{total} observaciones: {missing} sin vector, {stale} de otro modelo"),
                    "cuba_zafra action=reembed",
                ));
            }
        }
        Err(e) => checks.push(Check::warn("embedding_coverage", format!("no verificable: {e}"), "revisar esquema")),
    }

    // --- Bitemporal invariant (migration 0029 enforces it; check it holds) --
    match sqlx::query(
        "SELECT count(*)::bigint FROM brain_facts
         WHERE valid_to IS NOT NULL AND valid_to <= valid_from",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let bad: i64 = row.try_get(0).unwrap_or(0);
            if bad == 0 {
                checks.push(Check::ok("bitemporal", "sin intervalos invertidos"));
            } else {
                checks.push(Check::fail(
                    "bitemporal",
                    format!("{bad} hechos con valid_to <= valid_from"),
                    "el histórico temporal miente: las consultas 'as of' devolverán resultados imposibles",
                ));
            }
        }
        Err(e) => checks.push(Check::warn("bitemporal", format!("no verificable: {e}"), "revisar esquema")),
    }

    // --- The decay anchor (without it, REM re-applies decay 10-80×) ---------
    match sqlx::query(
        "SELECT count(*)::bigint FROM information_schema.columns
         WHERE table_name='brain_observations' AND column_name='last_decayed_at'",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let n: i64 = row.try_get(0).unwrap_or(0);
            if n > 0 {
                checks.push(Check::ok("decay_anchor", "last_decayed_at presente (migración 0028)"));
            } else {
                checks.push(Check::fail(
                    "decay_anchor",
                    "falta la columna last_decayed_at",
                    "sin ancla, el ciclo REM re-aplica el decaimiento cada 4 h y la importancia \
                     se hunde 10-80×. Aplicá la migración 0028.",
                ));
            }
        }
        Err(e) => checks.push(Check::warn("decay_anchor", format!("no verificable: {e}"), "revisar esquema")),
    }

    // --- Bug 0.7: superuser makes RLS and the audit log decorative ----------
    match sqlx::query(
        "SELECT current_user::text AS usr,
                COALESCE((SELECT rolsuper FROM pg_roles WHERE rolname = current_user), false) AS super",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let usr: String = row.try_get("usr").unwrap_or_default();
            let is_super: bool = row.try_get("super").unwrap_or(false);
            if is_super {
                checks.push(Check::warn(
                    "runtime_role",
                    format!("la app corre como '{usr}' (SUPERUSER)"),
                    "un superuser ignora RLS y puede alterar el audit_log: el aislamiento por \
                     proyecto y la trazabilidad append-only son decorativos. Usá el rol cuba_app \
                     (scripts/create-app-role.sql).",
                ));
            } else {
                checks.push(Check::ok("runtime_role", format!("'{usr}' sin superuser — RLS y audit efectivos")));
            }
        }
        Err(e) => checks.push(Check::warn("runtime_role", format!("no verificable: {e}"), "revisar permisos")),
    }

    // --- Corpus size (context, not a verdict) -------------------------------
    match sqlx::query(
        "SELECT (SELECT count(*) FROM brain_observations)::bigint AS obs,
                (SELECT count(*) FROM brain_entities)::bigint AS ent,
                (SELECT count(*) FROM brain_facts)::bigint AS facts,
                pg_size_pretty(pg_database_size(current_database()))::text AS size",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let obs: i64 = row.try_get("obs").unwrap_or(0);
            let ent: i64 = row.try_get("ent").unwrap_or(0);
            let facts: i64 = row.try_get("facts").unwrap_or(0);
            let size: String = row.try_get("size").unwrap_or_default();
            checks.push(Check::ok(
                "corpus",
                format!("{obs} observaciones · {ent} entidades · {facts} hechos · {size}"),
            ));
        }
        Err(e) => checks.push(Check::warn("corpus", format!("no verificable: {e}"), "revisar esquema")),
    }

    checks
}

/// CLI entry point. Exits non-zero when any check fails.
pub async fn run_cli(args: &[String]) -> Result<()> {
    let json = args.iter().any(|a| a == "--json");
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: cuba-memorys doctor [--json]\n\n\
             Read-only health check: asserts the invariants whose violation is silent\n\
             (zero recall, dead vector branch, dimension drift, inert RLS, decay anchor).\n\
             Exits 1 if any check fails."
        );
        return Ok(());
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for doctor")?;

    let checks = run_checks(&pool, &url).await;
    let failed = checks.iter().filter(|c| c.status == Status::Fail).count();
    let warned = checks.iter().filter(|c| c.status == Status::Warn).count();

    if json {
        println!(
            "{}",
            serde_json::json!({
                "healthy": failed == 0,
                "failed": failed,
                "warnings": warned,
                "checks": checks,
            })
        );
    } else {
        println!("cuba-memorys doctor — v{}\n", env!("CARGO_PKG_VERSION"));
        for c in &checks {
            println!("[{}] {:<20} {}", c.status.glyph(), c.name, c.detail);
            if let Some(h) = &c.hint {
                println!("{:<8} {:<20} → {}", "", "", h);
            }
        }
        println!();
        if failed == 0 && warned == 0 {
            println!("Todo sano: {} chequeos, 0 fallos.", checks.len());
        } else {
            println!(
                "{} chequeos · {} fallo(s) · {} aviso(s).",
                checks.len(),
                failed,
                warned
            );
        }
    }

    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_the_password() {
        assert_eq!(
            redact_url("postgresql://cuba:memorys2026@127.0.0.1:5491/brain"),
            "postgresql://cuba:***@127.0.0.1:5491/brain"
        );
    }

    #[test]
    fn redacts_url_without_credentials() {
        assert_eq!(
            redact_url("postgresql://localhost:5432/brain"),
            "postgresql://localhost:5432/brain"
        );
    }

    #[test]
    fn parses_the_declared_vector_dimension() {
        assert_eq!(parse_vector_dim("vector(384)"), Some(384));
        assert_eq!(parse_vector_dim("vector(1024)"), Some(1024));
        assert_eq!(parse_vector_dim("text"), None);
    }
}
