//! Tracked background tasks with drain-on-shutdown.
//!
//! # Why this exists
//!
//! Embedding generation runs off the request path so `cuba_cronica` can answer
//! without waiting on ONNX inference. It used to do that with a bare
//! `tokio::spawn`, dropping the `JoinHandle`. Two consequences, both silent:
//!
//! 1. When `run_mcp()` returned (the MCP client closes stdin at session end),
//!    `main` fell out of its `tokio::select!` and the runtime was dropped —
//!    aborting every in-flight task. Any observation whose embedding was still
//!    being computed kept `embedding = NULL` forever, with no log line.
//! 2. A panic inside the task was swallowed by the discarded `JoinHandle`.
//!
//! Tasks registered here are awaited by [`drain`] during shutdown, bounded by a
//! timeout so a wedged task cannot hang the process.

use std::future::Future;
use std::sync::LazyLock;
use std::time::Duration;

use tokio_util::task::TaskTracker;

static TRACKER: LazyLock<TaskTracker> = LazyLock::new(TaskTracker::new);

/// Spawn a background task that shutdown will wait for.
///
/// Use this instead of `tokio::spawn` for any work that must survive the
/// response it was detached from — durability-relevant writes, above all.
pub fn spawn<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    TRACKER.spawn(future);
}

/// Number of tasks still running.
pub fn inflight() -> usize {
    TRACKER.len()
}

/// Close the tracker and wait for in-flight tasks, up to `timeout`.
///
/// Returns the number of tasks that did **not** finish in time. Zero means a
/// clean drain. Callers should log a non-zero result: it is the only signal
/// that background writes were lost.
pub async fn drain(timeout: Duration) -> usize {
    TRACKER.close();
    if TRACKER.is_empty() {
        return 0;
    }
    tracing::debug!(inflight = TRACKER.len(), "draining background tasks");
    match tokio::time::timeout(timeout, TRACKER.wait()).await {
        Ok(()) => 0,
        Err(_) => TRACKER.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn drain_waits_for_inflight_work() {
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..8 {
            let c = counter.clone();
            spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                c.fetch_add(1, Ordering::SeqCst);
            });
        }
        let lost = drain(Duration::from_secs(5)).await;
        assert_eq!(lost, 0, "no task should be lost");
        assert_eq!(
            counter.load(Ordering::SeqCst),
            8,
            "every task must run to completion before drain returns"
        );
    }
}
