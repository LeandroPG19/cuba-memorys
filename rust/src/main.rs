//! Cuba-Memorys MCP Server
//!
//! Knowledge Graph MCP server: 25 tools, Hebbian/BCM learning, RRF hybrid
//! search, episodic memory, contradiction detection, Bayesian calibration,
//! bitemporal facts, contextual retrieval, and autonomous REM sleep
//! consolidation. Version comes from `CARGO_PKG_VERSION` — do not restate it
//! here; this docstring spent four releases claiming v0.6.0 / 19 tools.

use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// How long shutdown waits for detached writes (embeddings) to land.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

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
        Some("doctor") => {
            if let Err(e) = cuba_memorys::doctor::run_cli(&argv[2..]).await {
                tracing::error!(error = %format!("{e:#}"), "doctor failed");
                eprintln!("doctor error: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }

    tracing::info!(version = env!("CARGO_PKG_VERSION"), "cuba-memorys starting");

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
