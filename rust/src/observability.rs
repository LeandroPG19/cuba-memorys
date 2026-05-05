//! V0.9: OpenTelemetry-compatible metrics + Prometheus exporter.
//!
//! Feature-gated under `observability` so the default binary stays lean.
//! When enabled, exposes a `/metrics` endpoint on `CUBA_METRICS_PORT`
//! (default 9090, bind `127.0.0.1`).
//!
//! Metrics published (RED + USE pattern):
//! - `cuba_handler_duration_seconds{tool}` — histogram of dispatch latency
//! - `cuba_handler_calls_total{tool,outcome}` — counter
//! - `cuba_judge_calls_total{backend,verdict}` — LLM-judge invocations
//! - `cuba_judge_timeout_total{backend}` — judge subprocess timeouts
//! - `cuba_embedding_cache_hits_total` / `cuba_embedding_cache_miss_total`
//!
//! Naming follows OpenMetrics conventions: snake_case, suffix `_total` for
//! counters, `_seconds` for time, no metric prefix outside `cuba_`.

#[cfg(feature = "observability")]
use anyhow::Result;
#[cfg(feature = "observability")]
use std::net::SocketAddr;

/// Initialize the Prometheus exporter on `CUBA_METRICS_PORT`. Idempotent.
/// No-op when feature `observability` is disabled.
#[cfg(feature = "observability")]
pub fn init() -> Result<()> {
    let port: u16 = std::env::var("CUBA_METRICS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9090);
    let bind: std::net::IpAddr = std::env::var("CUBA_METRICS_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = SocketAddr::new(bind, port);

    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()
        .map_err(|e| anyhow::anyhow!("Prometheus exporter failed to install: {e}"))?;

    tracing::info!(addr = %addr, "Prometheus /metrics endpoint live");
    // Pre-register histograms so they appear with zero counts even before traffic
    metrics::describe_histogram!(
        "cuba_handler_duration_seconds",
        "MCP handler dispatch latency"
    );
    metrics::describe_counter!(
        "cuba_handler_calls_total",
        "Total MCP handler dispatches by tool and outcome"
    );
    metrics::describe_counter!(
        "cuba_judge_calls_total",
        "LLM-judge invocations by backend and verdict"
    );
    metrics::describe_counter!(
        "cuba_judge_timeout_total",
        "LLM-judge subprocess timeouts by backend"
    );
    Ok(())
}

#[cfg(not(feature = "observability"))]
pub fn init() -> anyhow::Result<()> {
    Ok(())
}

/// Record a handler dispatch outcome. No-op without `observability`.
#[inline]
pub fn record_handler(tool: &str, outcome: &str, elapsed_secs: f64) {
    #[cfg(feature = "observability")]
    {
        metrics::histogram!("cuba_handler_duration_seconds", "tool" => tool.to_string())
            .record(elapsed_secs);
        metrics::counter!(
            "cuba_handler_calls_total",
            "tool" => tool.to_string(),
            "outcome" => outcome.to_string(),
        )
        .increment(1);
    }
    #[cfg(not(feature = "observability"))]
    {
        let _ = (tool, outcome, elapsed_secs);
    }
}

/// Record a judge backend invocation.
#[inline]
pub fn record_judge(backend: &str, verdict: &str) {
    #[cfg(feature = "observability")]
    {
        metrics::counter!(
            "cuba_judge_calls_total",
            "backend" => backend.to_string(),
            "verdict" => verdict.to_string(),
        )
        .increment(1);
    }
    #[cfg(not(feature = "observability"))]
    {
        let _ = (backend, verdict);
    }
}

/// Record a judge timeout.
#[inline]
pub fn record_judge_timeout(backend: &str) {
    #[cfg(feature = "observability")]
    {
        metrics::counter!(
            "cuba_judge_timeout_total",
            "backend" => backend.to_string(),
        )
        .increment(1);
    }
    #[cfg(not(feature = "observability"))]
    {
        let _ = backend;
    }
}
