//! Global OOD statistics cache — refreshed by `cuba_zafra`, read by `cuba_faro`.

use std::sync::RwLock;

use uuid::Uuid;

use super::ood::OodStats;

static OOD_CACHE: RwLock<Option<(Option<Uuid>, OodStats)>> = RwLock::new(None);

pub fn store(project_id: Option<Uuid>, stats: OodStats) {
    if let Ok(mut guard) = OOD_CACHE.write() {
        *guard = Some((project_id, stats));
    }
}

pub fn get(project_id: Option<Uuid>) -> Option<OodStats> {
    OOD_CACHE.read().ok().and_then(|g| {
        g.as_ref()
            .filter(|(p, _)| *p == project_id)
            .map(|(_, s)| s.clone())
    })
}

pub fn clear() {
    if let Ok(mut guard) = OOD_CACHE.write() {
        *guard = None;
    }
}
