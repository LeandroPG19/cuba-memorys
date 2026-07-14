//! Cuba-Memorys MCP Server
//!
//! Knowledge Graph MCP server: Hebbian/BCM learning, RRF hybrid search, episodic
//! and procedural memory, contradiction detection, Bayesian calibration,
//! bitemporal facts, contextual retrieval, and autonomous REM sleep
//! consolidation.
//!
//! No version, no tool count. Both live in exactly one place each —
//! `CARGO_PKG_VERSION` and `constants::tools_for_profile()` — and both were
//! wrong here for four releases running (v0.6.0, 19 tools; then 25, when there
//! were 28). A number copied into a docstring is a number that will drift, and
//! a comment that lies is worse than no comment. Ask `tools/list`.

use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// How long shutdown waits for detached writes (embeddings) to land.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// The command surface, printed on `--help` and on an unrecognised argument.
///
/// Goes to stdout only from paths that return immediately — stdout is the MCP
/// protocol channel once the server is up, and a stray byte there desyncs the
/// client's JSON-RPC framing.
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
  setup             wire this server into your MCP clients; `setup check` audits them

  -h, --help        this
  -V, --version     print the version and exit — touches no database

DATABASE_URL points at the brain. `doctor` will tell you if anything is off.
Docs: https://github.com/LeandroPG19/cuba-memorys",
        version = env!("CARGO_PKG_VERSION")
    );
}

/// Wait for tracked background tasks. Must run before any `process::exit`,
/// which skips destructors and would abort them mid-write.
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
    // Structured JSON logging to stderr (MCP uses stdout for protocol)
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cuba_memorys=info".parse().unwrap()),
        )
        .init();

    // Subcommand dispatch. `eval` runs the read-only retrieval benchmark and
    // `doctor` the read-only health check; both exit without ever reaching the
    // MCP server. Anything else falls through to the stdio server. Kept before
    // the server setup so no subcommand touches stdout's protocol channel.
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
        Some("doctor") => {
            if let Err(e) = cuba_memorys::doctor::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "doctor failed");
                eprintln!("doctor error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        // The human surface: query and write the brain without an LLM in between.
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
        // Client wiring. Touches no database, so it runs even when the brain is down.
        Some("setup") => {
            if let Err(e) = cuba_memorys::setup_agent::run_cli(&argv[2..]) {
                tracing::error!(error = %format!("{e:#}"), "setup failed");
                eprintln!("setup: {e:#}");
                std::process::exit(1);
            }
            return;
        }

        // `--version` must be inert. It fell through to the server before, which
        // means asking this binary what version it was CONNECTED TO A DATABASE AND
        // RAN MIGRATIONS — the one command a person types precisely because they do
        // not yet trust what they have installed. It returns before any of that now.
        Some("--version" | "-V" | "version") => {
            println!("cuba-memorys {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some("--help" | "-h" | "help") => {
            print_help();
            return;
        }

        // An argument we do not recognise is a mistake, not a server launch.
        //
        // The old catch-all sent `doctro`, `--verison`, and every other typo straight
        // into the MCP server, where it sat silent on a stdio socket nobody was
        // speaking to — looking, to the person who typed it, exactly like a hang.
        // The server is what you get with NO arguments; that is how MCP clients
        // launch it, and it is the only way to get it.
        Some(unknown) => {
            eprintln!("cuba-memorys: unknown command '{unknown}'\n");
            print_help();
            std::process::exit(2);
        }

        None => {}
    }

    tracing::info!(version = env!("CARGO_PKG_VERSION"), "cuba-memorys starting");

    // Fail loudly, at startup, rather than silently on every query. A dimension
    // mismatch turns hybrid search into lexical search with no visible symptom.
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
                // Not fatal here: the existing startup path has its own retry and
                // setup logic, and duplicating that decision would be worse.
                tracing::warn!(error = %e, "no se pudo verificar la dimensión del embedding al arrancar");
            }
        }
    }

    // V0.9: optional Prometheus /metrics endpoint (no-op without `observability` feature).
    if let Err(e) = cuba_memorys::observability::init() {
        tracing::warn!(error = %e, "observability init failed — continuing without /metrics");
    }

    // Graceful shutdown on SIGTERM/SIGINT
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

    // Run MCP protocol with graceful shutdown
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

    // The select! above exits as soon as run_mcp() returns — which is what
    // happens on every normal session end, when the MCP client closes stdin.
    // Without this drain the runtime is dropped here and every in-flight
    // embedding write is aborted silently.
    drain_background_tasks().await;
}
