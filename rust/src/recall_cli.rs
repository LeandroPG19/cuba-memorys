//! `cuba-memorys recall` — the context an agent should already have.
//!
//! The CLAUDE.md in this setup says: *"MANDATORY session start → cuba_jornada
//! then cuba_faro"*. That is an instruction to a model, and a model can forget
//! it. The same file lists the failure as an anti-pattern — AP2, Context Amnesia
//! — which is an admission that the instruction is not reliable enough to be
//! trusted, written by the person who wrote the instruction.
//!
//! A hook does not forget. This command prints what the agent should know before
//! it does anything, so a `SessionStart` hook can inject it deterministically.
//! The difference is between asking and knowing.
//!
//! Read-only, and bounded: it must never dump the whole brain into the first
//! turn of a conversation.

use anyhow::{Context, Result};
use sqlx::{PgPool, Row};

/// Token budget for the injected block. Whatever this costs is paid on EVERY
/// session, so it has to stay small enough to be worth it — a recall that eats
/// 10k tokens of context to save an agent one search is a bad trade.
const DEFAULT_BUDGET: usize = 900;

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut project: Option<String> = None;
    let mut budget = DEFAULT_BUDGET;
    let mut quiet = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--project" | "-p" => project = it.next().cloned(),
            "--max-tokens" => {
                budget = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--max-tokens needs an integer")?
            }
            // For hooks: say nothing at all when there is nothing to say, rather
            // than printing a header over an empty result.
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys recall [--project NAME] [--max-tokens 900] [--quiet]\n\n\
                     Prints the memory an agent should start a session with: the last session,\n\
                     unresolved errors, recent decisions, and the entities that matter here.\n\n\
                     Meant for a SessionStart hook — a hook cannot forget to call it, which is\n\
                     the whole point. Install it with: cuba-memorys setup hook"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown recall flag: {other} (try --help)"),
        }
    }

    // Infer the project from the working directory when not told. This is what
    // makes the hook zero-config: the same hook line works in every repo.
    let project = project.or_else(|| {
        std::env::current_dir().ok().and_then(|d| {
            d.file_name()
                .and_then(|n| n.to_str())
                .map(std::string::ToString::to_string)
        })
    });

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for recall")?;

    let text = build(&pool, project.as_deref(), budget).await?;
    if text.trim().is_empty() {
        if !quiet {
            println!("(cuba-memorys: sin memoria previa de este proyecto)");
        }
        return Ok(());
    }
    println!("{text}");
    Ok(())
}

async fn build(pool: &PgPool, project: Option<&str>, budget: usize) -> Result<String> {
    use crate::search::budget::count_tokens;

    /// Append a section only while it fits. Sections are added in the order an
    /// agent most regrets not having them, so hitting the budget drops the least
    /// important one rather than truncating mid-sentence.
    fn push(out: &mut String, spent: &mut usize, budget: usize, section: &str) {
        let cost = count_tokens(section);
        if *spent + cost <= budget {
            out.push_str(section);
            *spent += cost;
        }
    }

    let mut out = String::new();
    let mut spent = 0usize;

    out.push_str("## Memoria de cuba-memorys\n");

    // 1. Where we left off. Nothing else is as useful as this in the first turn.
    if let Ok(Some(row)) = sqlx::query(
        "SELECT name, summary, outcome, ended_at::date::text AS d
         FROM brain_sessions
         WHERE summary IS NOT NULL AND ended_at IS NOT NULL
         ORDER BY ended_at DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        let name: String = row.try_get("name").unwrap_or_default();
        let summary: String = row.try_get("summary").unwrap_or_default();
        let outcome: String = row.try_get("outcome").unwrap_or_default();
        let d: String = row.try_get("d").unwrap_or_default();
        let summary = crate::handlers::zafra::safe_truncate(&summary, 400);
        push(
            &mut out,
            &mut spent,
            budget,
            &format!("\n### Última sesión ({d}, {outcome})\n**{name}**\n{summary}\n"),
        );
    }

    // 2. Unresolved errors — the ones an agent is about to repeat.
    if let Ok(rows) = sqlx::query(
        // Test fixtures are not lessons. Injecting "TestError: Project A error"
        // into every session costs context and teaches nothing.
        "SELECT error_type, error_message FROM brain_errors
         WHERE resolved = false
           AND error_type NOT ILIKE '%test%'
           AND error_message NOT ILIKE '%test error%'
           AND error_message NOT ILIKE '%smoke test%'
         ORDER BY created_at DESC LIMIT 4",
    )
    .fetch_all(pool)
    .await
        && !rows.is_empty()
    {
        let mut s = String::from("\n### Errores SIN resolver\n");
        for r in &rows {
            let t: String = r.try_get("error_type").unwrap_or_default();
            let m: String = r.try_get("error_message").unwrap_or_default();
            s.push_str(&format!(
                "- **{t}**: {}\n",
                crate::handlers::zafra::safe_truncate(&m, 120)
            ));
        }
        push(&mut out, &mut spent, budget, &s);
    }

    // 3. Decisions already taken here — so they are not re-litigated.
    if let Ok(rows) = sqlx::query(
        "SELECT e.name, o.content
         FROM brain_observations o
         JOIN brain_entities e ON e.id = o.entity_id
         WHERE o.observation_type = 'decision'
         ORDER BY o.importance DESC, o.created_at DESC LIMIT 4",
    )
    .fetch_all(pool)
    .await
        && !rows.is_empty()
    {
        let mut s = String::from("\n### Decisiones ya tomadas\n");
        for r in &rows {
            let n: String = r.try_get("name").unwrap_or_default();
            let c: String = r.try_get("content").unwrap_or_default();
            s.push_str(&format!(
                "- [{n}] {}\n",
                crate::handlers::zafra::safe_truncate(&c, 130)
            ));
        }
        push(&mut out, &mut spent, budget, &s);
    }

    // 4. How things are DONE here. This is the section that stops an agent
    // rediscovering the test command for the twentieth time — and it is ranked by
    // what has actually worked, not by what has been read the most.
    if let Ok(rows) = sqlx::query(
        "SELECT name, trigger_context, success_count, failure_count
         FROM brain_procedures
         ORDER BY (success_count::float8 / GREATEST(success_count + failure_count, 1)) DESC,
                  success_count DESC
         LIMIT 5",
    )
    .fetch_all(pool)
    .await
        && !rows.is_empty()
    {
        let mut s = String::from("\n### Procedimientos conocidos\n");
        for r in &rows {
            let n: String = r.try_get("name").unwrap_or_default();
            let t: String = r.try_get("trigger_context").unwrap_or_default();
            let ok: i32 = r.try_get("success_count").unwrap_or(0);
            let ko: i32 = r.try_get("failure_count").unwrap_or(0);
            let record = if ok + ko == 0 {
                "sin probar".to_string()
            } else {
                format!("{ok}/{} ok", ok + ko)
            };
            s.push_str(&format!("- **{n}** ({record})"));
            if !t.is_empty() {
                s.push_str(&format!(" — {t}"));
            }
            s.push('\n');
        }
        s.push_str("_Traelos con `cuba_receta action=get`._\n");
        push(&mut out, &mut spent, budget, &s);
    }

    // 5. What this project is about, if we could name it.
    if let Some(p) = project
        && let Ok(rows) = sqlx::query(
            "SELECT e.name, e.entity_type
             FROM brain_entities e
             WHERE e.name ILIKE '%' || $1 || '%'
             ORDER BY e.importance DESC LIMIT 5",
        )
        .bind(p)
        .fetch_all(pool)
        .await
        && !rows.is_empty()
    {
        let names: Vec<String> = rows
            .iter()
            .filter_map(|r| r.try_get::<String, _>("name").ok())
            .collect();
        push(
            &mut out,
            &mut spent,
            budget,
            &format!("\n### Entidades de «{p}»\n{}\n", names.join(", ")),
        );
    }

    push(
        &mut out,
        &mut spent,
        budget,
        "\n_Buscá más con `cuba_faro`. Antes de proponer un enfoque, consultá `cuba_expediente` \
         con `proposed_action` para no repetir un error ya cometido._\n",
    );

    // Header only? Then there was nothing worth saying.
    if spent == 0 {
        return Ok(String::new());
    }
    Ok(out)
}
