use anyhow::{Context, Result};
use sqlx::PgPool;

use crate::codegraph::{self, EdgeKind, Symbol, SymbolKind};

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut path_arg: Option<String> = None;
    let mut langs: Vec<String> = Vec::new();
    let mut dry_run = false;
    let mut json = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--path" | "-p" => path_arg = it.next().cloned(),
            "--lang" => {
                if let Some(v) = it.next() {
                    langs = v.split(',').map(|s| s.trim().to_string()).collect();
                }
            }
            "--dry-run" => dry_run = true,
            "--json" => json = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys codegraph build [--path DIR] [--lang rust,python] [--dry-run] [--json]\n\n\
                     Parses source with tree-sitter (deterministic, no LLM, nothing leaves\n\
                     this process) and folds it into the SAME knowledge graph cuba_puente\n\
                     and cuba_faro already use: functions/structs/classes become\n\
                     brain_entities (entity_type='code_symbol'), resolved calls and use/import\n\
                     statements become brain_relations with provenance='extracted'.\n\n\
                     A call only becomes an edge when its callee name matches exactly one\n\
                     symbol in the parsed batch — ambiguous names are dropped, not guessed.\n\n\
                     --dry-run prints counts without writing anything."
                );
                return Ok(());
            }
            "build" => {}
            other => anyhow::bail!("unknown codegraph flag: {other} (try --help)"),
        }
    }

    let root = codegraph::resolve_path(path_arg.as_deref());
    let extensions = codegraph::default_extensions_for(&langs);

    let result = codegraph::extract_dir(&root, &extensions)
        .with_context(|| format!("scanning {}", root.display()))?;
    let call_edges = codegraph::resolve_call_edges(&result.symbols);

    if dry_run {
        let report = serde_json::json!({
            "action": "codegraph_build",
            "dry_run": true,
            "files_parsed": result.files_parsed,
            "files_skipped": result.files_skipped,
            "symbols_found": result.symbols.len(),
            "call_edges_resolved": call_edges.len(),
            "import_statements": result.imports.iter().map(|m| m.paths.len()).sum::<usize>(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for codegraph build")?;

    let project_id = crate::project::current_project_id(&pool).await?;

    let mut entities_written = 0u32;
    for symbol in &result.symbols {
        upsert_symbol(&pool, symbol, project_id).await?;
        entities_written += 1;
    }

    let mut edges_written = 0u32;
    for edge in &call_edges {
        if upsert_edge(&pool, &edge.from, &edge.to, edge.kind, project_id).await? {
            edges_written += 1;
        }
    }

    let mut import_edges_written = 0u32;
    for module in &result.imports {
        let from = format!("{}::<module>", module.file);
        upsert_placeholder_entity(&pool, &from, "module", project_id).await?;
        for path in &module.paths {
            upsert_placeholder_entity(&pool, path, "external_dependency", project_id).await?;
            if upsert_edge(&pool, &from, path, EdgeKind::Imports, project_id).await? {
                import_edges_written += 1;
            }
        }
    }

    let report = serde_json::json!({
        "action": "codegraph_build",
        "dry_run": false,
        "files_parsed": result.files_parsed,
        "files_skipped": result.files_skipped,
        "symbols_written": entities_written,
        "call_edges_written": edges_written,
        "import_edges_attempted": import_edges_written,
    });

    if json {
        println!("{report}");
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}

async fn upsert_symbol(
    pool: &PgPool,
    symbol: &Symbol,
    project_id: Option<uuid::Uuid>,
) -> Result<()> {
    let entity_id: (uuid::Uuid,) = sqlx::query_as(
        "INSERT INTO brain_entities (name, entity_type, project_id)
         VALUES ($1, 'code_symbol', $2)
         ON CONFLICT (name) DO UPDATE SET entity_type = 'code_symbol'
         RETURNING id",
    )
    .bind(&symbol.qualified_name)
    .bind(project_id)
    .fetch_one(pool)
    .await?;

    let kind_label = match symbol.kind {
        SymbolKind::Function => "function",
        SymbolKind::Struct => "struct",
        SymbolKind::Class => "class",
        SymbolKind::Module => "module",
    };
    let content = format!(
        "{} `{}` in {}:{}-{}\n{}",
        kind_label,
        symbol.simple_name,
        symbol.file,
        symbol.line_start,
        symbol.line_end,
        symbol.signature
    );

    sqlx::query(
        "INSERT INTO brain_observations (entity_id, content, observation_type, source, project_id)
         SELECT $1, $2, 'context', 'agent', $3
         WHERE NOT EXISTS (
             SELECT 1 FROM brain_observations
             WHERE entity_id = $1 AND content = $2
         )",
    )
    .bind(entity_id.0)
    .bind(&content)
    .bind(project_id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Import targets are raw paths (`crate::db::create_pool`, `os.path`) that
/// almost never match a parsed symbol's qualified name — resolving them
/// properly needs full module-path resolution, which this extractor doesn't
/// attempt. Instead each unique path becomes its own lightweight placeholder
/// entity, so "this file depends on X" is still a real edge in the graph even
/// when X is an external crate or stdlib module with no parsed symbols of its
/// own.
async fn upsert_placeholder_entity(
    pool: &PgPool,
    name: &str,
    entity_type: &str,
    project_id: Option<uuid::Uuid>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO brain_entities (name, entity_type, project_id)
         VALUES ($1, $2, $3)
         ON CONFLICT (name) DO NOTHING",
    )
    .bind(name)
    .bind(entity_type)
    .bind(project_id)
    .execute(pool)
    .await?;
    Ok(())
}

async fn upsert_edge(
    pool: &PgPool,
    from_name: &str,
    to_name: &str,
    kind: EdgeKind,
    project_id: Option<uuid::Uuid>,
) -> Result<bool> {
    let from_id: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
            .bind(from_name)
            .fetch_optional(pool)
            .await?;
    let Some((from_id,)) = from_id else {
        return Ok(false);
    };

    let to_id: Option<(uuid::Uuid,)> =
        sqlx::query_as("SELECT id FROM brain_entities WHERE name = $1")
            .bind(to_name)
            .fetch_optional(pool)
            .await?;
    let Some((to_id,)) = to_id else {
        return Ok(false);
    };

    let result = sqlx::query(
        "INSERT INTO brain_relations (from_entity, to_entity, relation_type, project_id, provenance)
         VALUES ($1, $2, $3, $4, 'extracted')
         ON CONFLICT (from_entity, to_entity, relation_type)
         DO UPDATE SET strength = LEAST(brain_relations.strength + 0.1, 1.0), last_traversed = NOW()",
    )
    .bind(from_id)
    .bind(to_id)
    .bind(kind.as_relation_type())
    .bind(project_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}
