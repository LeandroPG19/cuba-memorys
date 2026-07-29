use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};

use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_id: Uuid,
    pub project_id: Option<Uuid>,
}

/// The single-client session: one process serving one stdio client, or a CLI
/// subcommand. This is the original behaviour and stays the default.
static ACTIVE: RwLock<Option<ActiveSession>> = RwLock::new(None);

/// The daemon's session table, keyed by client. One process now serves several
/// unrelated clients, so `jornada start` in one editor window must not become
/// the active session of another.
static PER_CLIENT: LazyLock<RwLock<HashMap<String, ActiveSession>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static DAEMON: AtomicBool = AtomicBool::new(false);

tokio::task_local! {
    static CLIENT: String;
}

/// Switches session state from "one global" to "one per client". Called once by
/// the HTTP transport before it serves anything.
pub fn enable_daemon_mode() {
    DAEMON.store(true, Ordering::Relaxed);
}

pub fn daemon_mode() -> bool {
    DAEMON.load(Ordering::Relaxed)
}

/// The client this task is answering, if any. Absent in CLI subcommands, in
/// background tasks spawned off a request, and in the REM daemon.
pub fn current_client() -> Option<String> {
    CLIENT.try_with(|c| c.clone()).ok()
}

/// Runs `fut` attributed to `key`. Every session read inside resolves against
/// that client's row instead of the global one.
pub async fn with_client<F, R>(key: String, fut: F) -> R
where
    F: std::future::Future<Output = R>,
{
    CLIENT.scope(key, fut).await
}

/// Drops a client's session state. The daemon outlives its clients, so without
/// this the table grows for as long as the process runs.
pub fn forget_client(key: &str) {
    if let Ok(mut guard) = PER_CLIENT.write() {
        guard.remove(key);
    }
}

/// Every session open in the daemon right now. The REM cycle protects recent
/// work from decay and, shared, it has to protect every client's — not just
/// whichever one happened to write last.
pub fn all_active() -> Vec<ActiveSession> {
    PER_CLIENT
        .read()
        .map(|g| g.values().copied().collect())
        .unwrap_or_default()
}

pub fn set(session_id: Uuid, project_id: Option<Uuid>) {
    let value = ActiveSession {
        session_id,
        project_id,
    };
    match current_client() {
        Some(key) => {
            if let Ok(mut guard) = PER_CLIENT.write() {
                guard.insert(key, value);
            }
        }
        None => {
            if let Ok(mut guard) = ACTIVE.write() {
                *guard = Some(value);
            }
        }
    }
}

pub fn clear() {
    match current_client() {
        Some(key) => {
            if let Ok(mut guard) = PER_CLIENT.write() {
                guard.remove(&key);
            }
        }
        None => {
            if let Ok(mut guard) = ACTIVE.write() {
                *guard = None;
            }
        }
    }
}

pub fn get() -> Option<ActiveSession> {
    if let Some(key) = current_client() {
        return PER_CLIENT.read().ok().and_then(|g| g.get(&key).copied());
    }
    // No client in scope. Under the daemon that means a background task or the
    // REM cycle, and answering with some other client's session would silently
    // attribute one window's writes to another — say nothing instead.
    if daemon_mode() {
        return None;
    }
    ACTIVE.read().ok().and_then(|g| *g)
}

pub fn project_id() -> Option<Uuid> {
    get().and_then(|s| s.project_id)
}

pub fn session_id() -> Option<Uuid> {
    get().map(|s| s.session_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_clear_roundtrip() {
        let sid = Uuid::new_v4();
        let pid = Uuid::new_v4();
        clear();
        assert_eq!(get(), None, "starts empty");

        set(sid, Some(pid));
        assert_eq!(session_id(), Some(sid));
        assert_eq!(project_id(), Some(pid));

        set(sid, None);
        assert_eq!(project_id(), None);

        clear();
        assert_eq!(get(), None);
        assert_eq!(project_id(), None);
    }

    #[tokio::test]
    async fn clients_do_not_see_each_others_sessions() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        with_client("editor-a".to_string(), async {
            set(a, None);
            assert_eq!(session_id(), Some(a));
        })
        .await;

        with_client("editor-b".to_string(), async {
            // b starts blind to a's session, and its own write stays its own.
            assert_eq!(session_id(), None, "b must not inherit a's session");
            set(b, None);
            assert_eq!(session_id(), Some(b));
        })
        .await;

        with_client("editor-a".to_string(), async {
            assert_eq!(session_id(), Some(a), "a's session survived b's write");
        })
        .await;

        forget_client("editor-a");
        forget_client("editor-b");
    }

    #[tokio::test]
    async fn daemon_hides_the_global_session_from_unscoped_tasks() {
        let sid = Uuid::new_v4();
        set(sid, None); // no client in scope -> global slot
        assert_eq!(
            session_id(),
            Some(sid),
            "global still readable in stdio mode"
        );

        enable_daemon_mode();
        assert_eq!(
            session_id(),
            None,
            "a background task must not adopt a stray global session"
        );

        DAEMON.store(false, Ordering::Relaxed);
        clear();
    }

    #[tokio::test]
    async fn forget_client_drops_the_row() {
        let sid = Uuid::new_v4();
        with_client("ephemeral".to_string(), async {
            set(sid, None);
            assert_eq!(session_id(), Some(sid));
        })
        .await;

        forget_client("ephemeral");

        with_client("ephemeral".to_string(), async {
            assert_eq!(session_id(), None, "row is gone after forget_client");
        })
        .await;
    }
}
