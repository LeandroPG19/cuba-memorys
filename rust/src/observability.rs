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
}
