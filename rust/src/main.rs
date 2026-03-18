//! Cuba-Memorys MCP Server — Rust Rewrite v2.0.0
//!
//! Knowledge Graph MCP server with FSRS-6, Dual-Strength Model,
//! Hebbian learning, hybrid search, and REM sleep consolidation.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

    tracing::info!(version = env!("CARGO_PKG_VERSION"), "cuba-memorys starting");

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
                std::process::exit(1);
            }
        }
        _ = shutdown => {
            tracing::info!("shutting down gracefully");
        }
    }
}
