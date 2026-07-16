use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

fn print_help() {
    println!(
        "cuba-memorys {version} — knowledge-graph memory server (MCP)

USAGE:
  cuba-memorys                  run the MCP server on stdio (how an MCP client launches it)
  cuba-memorys <command> [args]

THE BRAIN, WITHOUT AN LLM IN BETWEEN:
  search <query>    hybrid search (use --format verbose for the score breakdown)
  save <content>    write an observation
  delete <id>       remove an observation
  export            dump the graph (json | markdown)
  dashboard         what is in there, at a glance

OPERATIONS:
  doctor            health check: schema, embedding dim, config coherence, stale processes
  recall            the session-start context injection (wire it with `setup hook`)
  reembed           recompute every embedding — after changing model or dimension
  calibrate         recompute the abstention threshold from the live corpus
  link              auto-link entities by NPMI co-occurrence
  dedupe            find entities that are the same thing under different names
  skills <dir>      export procedures as Claude Code skills
  eval              retrieval benchmark (nDCG@10, MRR, recall) — read-only
  sync              git-friendly export/import of the graph (export|import|diff|status)
  hook install      wire git so sync export/import run on commit/checkout automatically
  codegraph build   parse source (tree-sitter, rust|python) into brain_entities/relations
  setup             wire this server into your MCP clients; `setup check` audits them

  -h, --help        this
  -V, --version     print the version and exit — touches no database

DATABASE_URL points at the brain. `doctor` will tell you if anything is off.
Docs: https://github.com/LeandroPG19/cuba-memorys",
        version = env!("CARGO_PKG_VERSION")
    );
}

async fn drain_background_tasks() {
    let lost = cuba_memorys::tasks::drain(DRAIN_TIMEOUT).await;
    if lost > 0 {
        tracing::error!(
            lost,
            "background tasks did not finish before shutdown — some embeddings \
             were not persisted; recover with cuba_zafra action=reembed"
        );
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cuba_memorys=info".parse().unwrap()),
        )
        .init();

    let argv: Vec<String> = std::env::args().collect();
    match argv.get(1).map(String::as_str) {
        Some("eval") => {
            if let Err(e) = cuba_memorys::eval::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "eval failed");
                eprintln!("eval error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("skills") => {
            if let Err(e) = cuba_memorys::skills_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "skills failed");
                eprintln!("skills error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("reembed") => {
            if let Err(e) = cuba_memorys::reembed_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "reembed failed");
                eprintln!("reembed error: {e:#}");
                std::process::exit(1);
            }
            drain_background_tasks().await;
            return;
        }
        Some("recall") => {
            if let Err(e) = cuba_memorys::recall_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "recall failed");
                eprintln!("recall error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("models") => {
            if let Err(e) = cuba_memorys::models_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "models failed");
                eprintln!("models: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("link") => {
            if let Err(e) = cuba_memorys::link_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "link failed");
                eprintln!("link error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("calibrate") => {
            if let Err(e) = cuba_memorys::calibrate_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "calibrate failed");
                eprintln!("calibrate error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("dedupe") => {
            if let Err(e) = cuba_memorys::dedupe_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "dedupe failed");
                eprintln!("dedupe error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("sync") => {
            if let Err(e) = cuba_memorys::sync_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "sync failed");
                eprintln!("sync error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("codegraph") => {
            if let Err(e) = cuba_memorys::codegraph_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "codegraph failed");
                eprintln!("codegraph error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("hook") => {
            if let Err(e) = cuba_memorys::hooks_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "hook failed");
                eprintln!("hook error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("secure") => {
            if let Err(e) = cuba_memorys::secure_cli::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "secure failed");
                eprintln!("secure: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("doctor") => {
            if let Err(e) = cuba_memorys::doctor::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "doctor failed");
                eprintln!("doctor error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some(cmd @ ("search" | "save" | "delete" | "export" | "dashboard")) => {
            let rest = &argv[2..];
            let result = match cmd {
                "search" => cuba_memorys::cli::run_search(rest).await,
                "save" => cuba_memorys::cli::run_save(rest).await,
                "delete" => cuba_memorys::cli::run_delete(rest).await,
                "dashboard" => cuba_memorys::dashboard::run_cli(rest).await,
                _ => cuba_memorys::export::run_cli(rest).await,
            };
            if let Err(e) = result {
                tracing::error!(error = %format!("{e:#}"), command = cmd, "command failed");
                eprintln!("{cmd}: {e:#}");
                std::process::exit(1);
            }
            drain_background_tasks().await;
            return;
        }
        Some("setup") => {
            if let Err(e) = cuba_memorys::setup_agent::run_cli(&argv[2..]) {
                tracing::error!(error = %format!("{e:#}"), "setup failed");
                eprintln!("setup: {e:#}");
                std::process::exit(1);
            }
            return;
        }

        Some("--version" | "-V" | "version") => {
            println!("cuba-memorys {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some("--help" | "-h" | "help") => {
            print_help();
            return;
        }

        Some(unknown) => {
            eprintln!("cuba-memorys: unknown command '{unknown}'\n");
            print_help();
            std::process::exit(2);
        }

        None => {}
    }

    tracing::info!(version = env!("CARGO_PKG_VERSION"), "cuba-memorys starting");

    {
        let url = cuba_memorys::setup::resolve_database_url().await;
        match cuba_memorys::db::create_pool(&url).await {
            Ok(pool) => {
                if let Err(e) = cuba_memorys::db::assert_embedding_dim(&pool).await {
                    tracing::error!(error = %format!("{e:#}"), "arranque abortado");
                    eprintln!("\ncuba-memorys NO puede arrancar:\n\n{e:#}\n");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "no se pudo verificar la dimensión del embedding al arrancar");
            }
        }
    }

    if let Err(e) = cuba_memorys::observability::init() {
        tracing::warn!(error = %e, "observability init failed — continuing without /metrics");
    }

    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler")
                .recv()
                .await;
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => tracing::info!("SIGINT received"),
            _ = terminate => tracing::info!("SIGTERM received"),
        }
    };

    tokio::select! {
        result = cuba_memorys::protocol::run_mcp() => {
            if let Err(e) = result {
                tracing::error!(error = %e, "MCP protocol error");
                drain_background_tasks().await;
                std::process::exit(1);
            }
        }
        _ = shutdown => {
            tracing::info!("shutting down gracefully");
        }
    }

    drain_background_tasks().await;
}
