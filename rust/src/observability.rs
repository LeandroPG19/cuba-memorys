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

const RING_CAPACITY: usize = 500;

#[derive(Clone, serde::Serialize)]
pub struct Call {
    pub tool: String,
    pub client: Option<String>,
    pub outcome: &'static str,
    pub elapsed_ms: u64,
    pub at: chrono::DateTime<chrono::Utc>,
}

#[derive(Default, Clone, serde::Serialize)]
pub struct ToolTotals {
    pub calls: u64,
    pub failures: u64,
    pub slowest_ms: u64,
    pub total_ms: u64,
}

struct Traffic {
    recent: std::collections::VecDeque<Call>,
    totals: std::collections::BTreeMap<String, ToolTotals>,
}

static TRAFFIC: std::sync::LazyLock<std::sync::Mutex<Traffic>> = std::sync::LazyLock::new(|| {
    std::sync::Mutex::new(Traffic {
        recent: std::collections::VecDeque::with_capacity(RING_CAPACITY),
        totals: std::collections::BTreeMap::new(),
    })
});

pub fn recent_calls(limit: usize) -> Vec<Call> {
    let Ok(traffic) = TRAFFIC.lock() else {
        return Vec::new();
    };
    traffic.recent.iter().rev().take(limit).cloned().collect()
}

pub fn tool_totals() -> std::collections::BTreeMap<String, ToolTotals> {
    TRAFFIC.lock().map(|t| t.totals.clone()).unwrap_or_default()
}

pub fn ring_capacity() -> usize {
    RING_CAPACITY
}

#[inline]
pub fn record_handler(tool: &str, outcome: &'static str, elapsed_secs: f64) {
    let elapsed_ms = (elapsed_secs * 1000.0).round() as u64;
    if let Ok(mut traffic) = TRAFFIC.lock() {
        if traffic.recent.len() == RING_CAPACITY {
            traffic.recent.pop_front();
        }
        traffic.recent.push_back(Call {
            tool: tool.to_string(),
            client: crate::session::current_client(),
            outcome,
            elapsed_ms,
            at: chrono::Utc::now(),
        });
        let totals = traffic.totals.entry(tool.to_string()).or_default();
        totals.calls += 1;
        totals.total_ms += elapsed_ms;
        totals.slowest_ms = totals.slowest_ms.max(elapsed_ms);
        if outcome != "ok" {
            totals.failures += 1;
        }
    }

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
