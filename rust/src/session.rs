//! The session owned by *this* server process.
//!
//! # Why this exists
//!
//! `brain_sessions` is shared by every `cuba-memorys` process pointed at the
//! same database — one per MCP client. Resolving "the active session" with
//!
//! ```sql
//! SELECT ... FROM brain_sessions WHERE ended_at IS NULL
//! ORDER BY started_at DESC LIMIT 1
//! ```
//!
//! asks a global question, so the answer was simply *whoever opened a session
//! last*. Three concrete failures observed on a live database holding 51
//! simultaneously-open sessions:
//!
//! - `project::current_project_id` tagged new observations with another
//!   client's project, silently breaking project isolation.
//! - `faro::get_session_goals` boosted search results using another project's
//!   goals.
//! - `jornada end` closed a session belonging to a different process.
//!
//! A session is per-connection state, so it lives in the process that opened
//! it. When this process has no session, handlers fall back to the unscoped
//! (global) view rather than borrowing someone else's.

use std::sync::RwLock;

use uuid::Uuid;

/// The session this process opened via `cuba_jornada start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_id: Uuid,
    pub project_id: Option<Uuid>,
}

static ACTIVE: RwLock<Option<ActiveSession>> = RwLock::new(None);

/// Record the session opened by this process. Replaces any previous one.
pub fn set(session_id: Uuid, project_id: Option<Uuid>) {
    if let Ok(mut guard) = ACTIVE.write() {
        *guard = Some(ActiveSession {
            session_id,
            project_id,
        });
    }
}

/// Forget this process's session (called on `cuba_jornada end`).
pub fn clear() {
    if let Ok(mut guard) = ACTIVE.write() {
        *guard = None;
    }
}

/// The session this process opened, if any.
///
/// Returns a copy — never hold the lock across an `.await`.
pub fn get() -> Option<ActiveSession> {
    ACTIVE.read().ok().and_then(|g| *g)
}

/// Convenience: the project bound to this process's session.
pub fn project_id() -> Option<Uuid> {
    get().and_then(|s| s.project_id)
}

/// Convenience: this process's session id.
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

        // A session without a project scopes globally, not to a stale project.
        set(sid, None);
        assert_eq!(project_id(), None);

        clear();
        assert_eq!(get(), None);
        assert_eq!(project_id(), None);
    }
}
