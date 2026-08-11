use std::sync::{Arc, Mutex, OnceLock};

use uuid::Uuid;

use crate::constants::CACHE_TTL_SECS;
use crate::search::cache::TtlLruCache;

use super::ood::OodStats;

const MAX_CACHED_PROJECTS: usize = 8;
const STATS_TTL_SECS: u64 = CACHE_TTL_SECS;

static OOD_CACHE: OnceLock<Mutex<TtlLruCache<Arc<OodStats>>>> = OnceLock::new();

fn cache() -> &'static Mutex<TtlLruCache<Arc<OodStats>>> {
    OOD_CACHE.get_or_init(|| {
        Mutex::new(TtlLruCache::with_config(
            MAX_CACHED_PROJECTS,
            STATS_TTL_SECS,
        ))
    })
}

fn key(project_id: Option<Uuid>) -> String {
    match project_id {
        Some(id) => id.to_string(),
        None => "global".to_string(),
    }
}

pub fn store(project_id: Option<Uuid>, stats: Arc<OodStats>) {
    if let Ok(mut guard) = cache().lock() {
        guard.put(key(project_id), stats);
    }
}

pub fn get(project_id: Option<Uuid>) -> Option<Arc<OodStats>> {
    cache().lock().ok()?.get(&key(project_id))
}

pub fn clear() {
    if let Ok(mut guard) = cache().lock() {
        guard.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use std::time::Duration;

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
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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

    #[test]
    fn two_projects_queried_alternately_both_stay_cached() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear();
        let first = Some(Uuid::new_v4());
        let second = Some(Uuid::new_v4());

        store(first, Arc::new(dummy_stats()));
        store(second, Arc::new(dummy_stats()));

        for round in 0..3 {
            assert!(
                get(first).is_some(),
                "round {round}: project {first:?} was evicted by a query on another project. \
                 A single-slot cache refits 1024x1024 covariance from scratch (11,4 s measured) \
                 every time two projects alternate, and the real corpus has 40 of them"
            );
            assert!(
                get(second).is_some(),
                "round {round}: project {second:?} was evicted by a query on another project"
            );
        }

        clear();
    }

    #[test]
    fn the_global_scope_does_not_share_its_slot_with_a_project() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear();
        let project_id = Some(Uuid::new_v4());
        let global = Arc::new(dummy_stats());
        let scoped = Arc::new(dummy_stats());

        store(None, Arc::clone(&global));
        store(project_id, Arc::clone(&scoped));

        assert!(
            get(None).is_some_and(|cached| Arc::ptr_eq(&cached, &global)),
            "storing a project evicted or overwrote the unscoped stats: the two are fitted over \
             different row sets, and serving one in place of the other flips abstention verdicts"
        );
        assert!(
            get(project_id).is_some_and(|cached| Arc::ptr_eq(&cached, &scoped)),
            "the project stats were lost or replaced by the unscoped ones"
        );

        clear();
    }

    #[test]
    fn the_cache_never_holds_more_projects_than_its_cap() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear();
        let ids: Vec<Option<Uuid>> = (0..MAX_CACHED_PROJECTS + 4)
            .map(|_| Some(Uuid::new_v4()))
            .collect();

        for id in &ids {
            store(*id, Arc::new(dummy_stats()));
        }

        assert!(
            get(ids[0]).is_none(),
            "the oldest project survived {} insertions into a cache capped at \
             {MAX_CACHED_PROJECTS}. Each entry holds a d*d f64 inverse covariance — 8 MiB at \
             d=1024 — so an unbounded map is gigabytes of resident memory",
            ids.len()
        );
        assert!(
            get(ids[ids.len() - 1]).is_some(),
            "the most recent insertion must survive: eviction has to drop the least recently \
             used entry, not the newest one"
        );

        clear();
    }

    #[test]
    fn an_entry_older_than_the_ttl_is_never_served() {
        let mut expiring: TtlLruCache<Arc<OodStats>> =
            TtlLruCache::with_config(MAX_CACHED_PROJECTS, 0);
        expiring.put(key(Some(Uuid::new_v4())), Arc::new(dummy_stats()));
        let expired_key = key(None);
        expiring.put(expired_key.clone(), Arc::new(dummy_stats()));

        assert!(
            expiring.get(&expired_key).is_none(),
            "an entry past its TTL was served. Without expiry the OOD statistics stay frozen \
             forever — clear() has no caller in the tree, so the TTL is the only invalidation"
        );

        let production_ttl = Duration::from_secs(STATS_TTL_SECS);
        assert!(
            production_ttl > Duration::ZERO && production_ttl <= Duration::from_secs(3600),
            "the production cache must expire within the hour, got {production_ttl:?}. A TTL of \
             zero refits on every query (11,4 s each); an effectively infinite one serves stats \
             fitted before 10.000 observations were written"
        );
    }
}
