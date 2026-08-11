use std::sync::{Arc, RwLock};

use uuid::Uuid;

use super::ood::OodStats;

static OOD_CACHE: RwLock<Option<(Option<Uuid>, Arc<OodStats>)>> = RwLock::new(None);

pub fn store(project_id: Option<Uuid>, stats: Arc<OodStats>) {
    if let Ok(mut guard) = OOD_CACHE.write() {
        *guard = Some((project_id, stats));
    }
}

pub fn get(project_id: Option<Uuid>) -> Option<Arc<OodStats>> {
    OOD_CACHE.read().ok().and_then(|g| {
        g.as_ref()
            .filter(|(p, _)| *p == project_id)
            .map(|(_, s)| Arc::clone(s))
    })
}

pub fn clear() {
    if let Ok(mut guard) = OOD_CACHE.write() {
        *guard = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn dummy_stats() -> OodStats {
        OodStats {
            mean: DVector::zeros(4),
            inverse_covariance: DMatrix::identity(4, 4),
            n_samples: 10,
            shrinkage: 0.1,
        }
    }

    #[test]
    fn get_returns_the_same_allocation_stored_not_a_clone() {
        let _guard = TEST_LOCK.lock().unwrap();
        let project_id = Some(Uuid::new_v4());
        let stats = Arc::new(dummy_stats());

        store(project_id, Arc::clone(&stats));

        let first = get(project_id).expect("stats were just stored");
        let second = get(project_id).expect("stats were just stored");

        assert!(
            Arc::ptr_eq(&first, &second),
            "two get() calls must point at the same allocation, not one clone each"
        );
        assert!(Arc::ptr_eq(&first, &stats));

        clear();
    }
}
