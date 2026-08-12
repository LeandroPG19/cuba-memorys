use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};

use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_id: Uuid,
    pub project_id: Option<Uuid>,
}

static ACTIVE: RwLock<Option<ActiveSession>> = RwLock::new(None);

static PER_CLIENT: LazyLock<RwLock<HashMap<String, ActiveSession>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static DAEMON: AtomicBool = AtomicBool::new(false);

tokio::task_local! {
    static CLIENT: String;
    static SCOPE: Scope;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Full,
    Peer,
}

pub const PEER_VERBS: [(&str, &str); 2] = [("cuba_sync", "status"), ("cuba_sync", "pull")];

pub fn current_scope() -> Scope {
    SCOPE.try_with(|s| *s).unwrap_or(Scope::Full)
}

pub async fn with_scope<F, R>(scope: Scope, fut: F) -> R
where
    F: std::future::Future<Output = R>,
{
    SCOPE.scope(scope, fut).await
}

#[cfg(test)]
pub static GLOBAL_STATE_GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub fn enable_daemon_mode() {
    DAEMON.store(true, Ordering::Relaxed);
}

pub fn daemon_mode() -> bool {
    DAEMON.load(Ordering::Relaxed)
}

pub fn current_client() -> Option<String> {
    CLIENT.try_with(|c| c.clone()).ok()
}

pub async fn with_client<F, R>(key: String, fut: F) -> R
where
    F: std::future::Future<Output = R>,
{
    CLIENT.scope(key, fut).await
}

pub fn forget_client(key: &str) {
    if let Ok(mut guard) = PER_CLIENT.write() {
        guard.remove(key);
    }
}

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

    #[tokio::test]
    async fn set_get_clear_roundtrip() {
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
        let sid = Uuid::new_v4();
        let pid = Uuid::new_v4();
        clear();
        assert_eq!(
            get(),
            None,
            "otro test escribió el ACTIVE global entre el clear y esta lectura: sin el cerrojo \
             de arriba esta suite falla ~5% de las veces y nadie se cree su verde"
        );

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
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        with_client("editor-a".to_string(), async {
            set(a, None);
            assert_eq!(session_id(), Some(a));
        })
        .await;

        with_client("editor-b".to_string(), async {
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
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
        let sid = Uuid::new_v4();
        set(sid, None);
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
        let _one_at_a_time = crate::session::GLOBAL_STATE_GUARD.lock().await;
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
