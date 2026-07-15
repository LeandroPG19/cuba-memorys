use std::future::Future;
use std::sync::LazyLock;
use std::time::Duration;

use tokio_util::task::TaskTracker;

static TRACKER: LazyLock<TaskTracker> = LazyLock::new(TaskTracker::new);

pub fn spawn<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    TRACKER.spawn(future);
}

pub fn inflight() -> usize {
    TRACKER.len()
}

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
