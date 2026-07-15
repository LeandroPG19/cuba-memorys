use std::sync::RwLock;

use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_id: Uuid,
    pub project_id: Option<Uuid>,
}

static ACTIVE: RwLock<Option<ActiveSession>> = RwLock::new(None);

pub fn set(session_id: Uuid, project_id: Option<Uuid>) {
    if let Ok(mut guard) = ACTIVE.write() {
        *guard = Some(ActiveSession {
            session_id,
            project_id,
        });
    }
}

pub fn clear() {
    if let Ok(mut guard) = ACTIVE.write() {
        *guard = None;
    }
}

pub fn get() -> Option<ActiveSession> {
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
}
