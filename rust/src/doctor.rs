use std::time::Instant;

use anyhow::{Context, Result};
use sqlx::{PgPool, Row};

use crate::embeddings::onnx;

const EXPECTED_TABLES: i64 = 23;

const REQUIRED_EXTENSIONS: [&str; 2] = ["vector", "pg_trgm"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Warn,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl Check {
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Ok,
            detail: detail.into(),
            hint: None,
        }
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

pub fn redact_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return "<unparseable>".to_string();
    };
    let Some((creds, host)) = rest.split_once('@') else {
        return format!("{scheme}://{rest}");
    };
    let user = creds.split_once(':').map_or(creds, |(u, _)| u);
    format!("{scheme}://{user}:***@{host}")
}

fn parse_vector_dim(col_type: &str) -> Option<i64> {
    col_type
        .trim()
        .strip_prefix("vector(")?
        .strip_suffix(')')?
        .parse()
        .ok()
}

fn stale_processes() -> Vec<u32> {
    let mut stale = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return stale;
    };

    for entry in entries.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let exe_link = entry.path().join("exe");
        let Ok(exe) = std::fs::read_link(&exe_link) else {
            continue;
        };

        let raw = exe.to_string_lossy();
        let (path, deleted) = match raw.strip_suffix(" (deleted)") {
            Some(p) => (std::path::PathBuf::from(p), true),
            None => (exe.clone(), false),
        };
        if path.file_name().and_then(|n| n.to_str()) != Some("cuba-memorys") {
            continue;
        }

        if deleted {
            stale.push(pid);
            continue;
        }
        let exe = path;

        let (Ok(bin_meta), Ok(proc_meta)) =
            (std::fs::metadata(&exe), std::fs::metadata(entry.path()))
        else {
            continue;
        };
        let (Ok(built), Ok(started)) = (bin_meta.modified(), proc_meta.modified()) else {
            continue;
        };
        if built > started {
            stale.push(pid);
        }
    }
    stale
}

fn latest_published_version() -> Option<String> {
    let out = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--max-time",
            "5",
            "https://registry.npmjs.org/cuba-memorys/latest",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    json.get("version")?.as_str().map(String::from)
}

pub async fn run_checks(pool: &PgPool, url: &str) -> Vec<Check> {
    let mut checks = Vec::new();

    let mode = crate::mode::active();
    checks.push(Check::ok(
        "mode",
        format!("{} — {}", mode.as_str(), mode.describe()),
    ));

    checks.push(Check::ok("database_url", redact_url(url)));

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
            return checks;
        }
    }

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
        Err(e) => checks.push(Check::warn(
            "extensions",
            format!("no se pudo leer: {e}"),
            "revisar permisos",
        )),
    }

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
                checks.push(Check::ok(
                    "migrations",
                    format!("{total} aplicadas, ninguna dirty"),
                ));
            }
        }
        Err(e) => checks.push(Check::fail(
            "migrations",
            format!("no existe _sqlx_migrations: {e}"),
            "la base no está inicializada — arrancá el servidor una vez para que migre",
        )),
    }

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
        Err(e) => checks.push(Check::warn(
            "schema",
            format!("no se pudo contar: {e}"),
            "revisar permisos",
        )),
    }

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
        Err(e) => checks.push(Check::warn(
            "recall/or_tsquery",
            format!("no verificable: {e}"),
            "revisar permisos",
        )),
    }

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

    if crate::search::rerank::enabled() {
        checks.push(Check::ok(
            "reranker",
            "cargado (bge-reranker-v2-m3) — reordena los candidatos por cross-encoder",
        ));
    } else {
        checks.push(Check::warn(
            "reranker",
            "no cargado — el ranking se devuelve tal cual (RRF, sin reordenar)",
            "es opcional, pero si querías reordenar y no pusiste el modelo, no está \
             pasando nada. Instalalo: cuba-memorys models reranker",
        ));
    }

    if crate::cognitive::nli::available() {
        if crate::cognitive::nli::enabled() {
            checks.push(Check::ok(
                "nli_entailment",
                "cargado (mDeBERTa-v3-xnli) — verify decide en ~50 ms, sin LLM",
            ));
        } else {
            checks.push(Check::fail(
                "nli_entailment",
                "hay un modelo NLI en disco pero NO carga",
                "verify se cae al juez LLM (~20 s por afirmación) o al heurístico, que \
                 no decide nada. Suele ser libonnxruntime.so: comprobá ORT_DYLIB_PATH.",
            ));
        }
    } else {
        checks.push(Check::warn(
            "nli_entailment",
            "sin modelo NLI local",
            "`cuba_faro mode=verify` depende de un LLM (~20 s por afirmación), y sin CLI \
             ni sampling degrada a `unknown`. Instalalo: cuba-memorys models nli",
        ));
    }

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
        Err(e) => checks.push(Check::warn(
            "embedding_dim",
            format!("no verificable: {e}"),
            "revisar esquema",
        )),
    }

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
        Err(e) => checks.push(Check::warn(
            "bitemporal",
            format!("no verificable: {e}"),
            "revisar esquema",
        )),
    }

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
                checks.push(Check::ok(
                    "decay_anchor",
                    "last_decayed_at presente (migración 0028)",
                ));
            } else {
                checks.push(Check::fail(
                    "decay_anchor",
                    "falta la columna last_decayed_at",
                    "sin ancla, el ciclo REM re-aplica el decaimiento cada 4 h y la importancia \
                     se hunde 10-80×. Aplicá la migración 0028.",
                ));
            }
        }
        Err(e) => checks.push(Check::warn(
            "decay_anchor",
            format!("no verificable: {e}"),
            "revisar esquema",
        )),
    }

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
        Err(e) => checks.push(Check::warn(
            "corpus",
            format!("no verificable: {e}"),
            "revisar esquema",
        )),
    }

    match sqlx::query(
        "SELECT count(*)::bigint AS isolated, (SELECT count(*) FROM brain_entities)::bigint AS total
         FROM brain_entities e
         WHERE NOT EXISTS (SELECT 1 FROM brain_relations r
                           WHERE r.from_entity = e.id OR r.to_entity = e.id)",
    )
    .fetch_one(pool)
    .await
    {
        Ok(row) => {
            let isolated: i64 = row.try_get("isolated").unwrap_or(0);
            let total: i64 = row.try_get("total").unwrap_or(0);
            let pct = if total > 0 { isolated * 100 / total } else { 0 };
            if pct >= 50 {
                checks.push(Check::warn(
                    "graph_connectivity",
                    format!("{isolated} de {total} entidades ({pct}%) no tienen ninguna relación"),
                    "el retrieval asociativo multi-hop y PageRank no las alcanzan: para el grafo \
                     no existen. cuba_reflexion las lista; cuba_puente las conecta.",
                ));
            } else {
                checks.push(Check::ok(
                    "graph_connectivity",
                    format!("{isolated} de {total} entidades aisladas ({pct}%)"),
                ));
            }
        }
        Err(e) => checks.push(Check::warn(
            "graph_connectivity",
            format!("no verificable: {e}"),
            "revisar esquema",
        )),
    }

    let stale = stale_processes();
    if stale.is_empty() {
        checks.push(Check::ok(
            "binary_freshness",
            "ningún proceso MCP corre un binario obsoleto",
        ));
    } else {
        let pids: Vec<String> = stale.iter().map(u32::to_string).collect();
        checks.push(Check::warn(
            "binary_freshness",
            format!(
                "{} proceso(s) MCP corren un binario más viejo que el de disco (pid {})",
                stale.len(),
                pids.join(", ")
            ),
            "recompilaste pero el cliente MCP sigue sirviendo la imagen vieja en memoria: \
             reiniciá el cliente para que tome el binario nuevo",
        ));
    }

    checks
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let json = args.iter().any(|a| a == "--json");
    let check_updates = args.iter().any(|a| a == "--check-updates");
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: cuba-memorys doctor [--json] [--check-updates]\n\n\
             Read-only health check: asserts the invariants whose violation is silent\n\
             (zero recall, dead vector branch, dimension drift, inert RLS, decay anchor,\n\
             stale binary in a live MCP process). Exits 1 if any check fails.\n\n\
             --check-updates  compara con la última versión publicada en npm. Nunca actualiza\n\
                              solo, y no corre salvo que lo pidas: es la única red que toca."
        );
        return Ok(());
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for doctor")?;

    let mut checks = run_checks(&pool, &url).await;

    if check_updates {
        let current = env!("CARGO_PKG_VERSION");
        match latest_published_version() {
            Some(latest) if latest != current => checks.push(Check::warn(
                "version",
                format!("corriendo v{current}; la última publicada es v{latest}"),
                "actualizá cuando quieras — doctor nunca lo hace por vos",
            )),
            Some(latest) => checks.push(Check::ok("version", format!("v{latest}, al día"))),
            None => checks.push(Check::warn(
                "version",
                "no se pudo consultar el registro de npm".to_string(),
                "sin red, o curl no está disponible — no afecta a nada más",
            )),
        }
    }
    let checks = checks;
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
    fn a_replaced_binary_is_recognized_despite_the_deleted_suffix() {
        let raw = "/home/x/rust/target/release/cuba-memorys (deleted)";
        let (path, deleted) = match raw.strip_suffix(" (deleted)") {
            Some(p) => (std::path::PathBuf::from(p), true),
            None => (std::path::PathBuf::from(raw), false),
        };
        assert!(deleted);
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some("cuba-memorys")
        );

        let live = "/home/x/rust/target/release/cuba-memorys";
        assert!(live.strip_suffix(" (deleted)").is_none());
    }

    #[test]
    fn parses_the_declared_vector_dimension() {
        assert_eq!(parse_vector_dim("vector(384)"), Some(384));
        assert_eq!(parse_vector_dim("vector(1024)"), Some(1024));
        assert_eq!(parse_vector_dim("text"), None);
    }
}
