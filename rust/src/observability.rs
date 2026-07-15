#[cfg(feature = "observability")]
use anyhow::Result;
#[cfg(feature = "observability")]
use std::net::SocketAddr;

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
