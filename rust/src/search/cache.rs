//! LRU Cache with TTL — replaces Python dict FIFO.
//!
//! FIX B6: Uses `lru` crate for O(1) real LRU eviction (not FIFO).
//! V7: TTL-based expiry to prevent stale embeddings.

use crate::constants::{CACHE_MAX_ENTRIES, CACHE_TTL_SECS};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Cached entry with TTL.
struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
}

/// LRU cache with TTL support.
pub struct TtlLruCache<V> {
    inner: LruCache<String, CacheEntry<V>>,
    ttl: Duration,
}

impl<V: Clone> Default for TtlLruCache<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone> TtlLruCache<V> {
    /// Create new cache with configured max size and TTL.
    pub fn new() -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(CACHE_MAX_ENTRIES).unwrap_or(NonZeroUsize::MIN)),
            ttl: Duration::from_secs(CACHE_TTL_SECS),
        }
    }

    /// Create cache with custom capacity and TTL.
    pub fn with_config(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::MIN)),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Get value from cache if present and not expired.
    ///
    /// Uses `peek()` first to check TTL without promoting to MRU.
    /// Only promotes (via `get()`) if the entry is still live — prevents
    /// expired entries from evicting valid ones during the LRU check.
    pub fn get(&mut self, key: &str) -> Option<V> {
        let dominated = self
            .inner
            .peek(key)
            .is_some_and(|entry| entry.inserted_at.elapsed() >= self.ttl);
        if dominated {
            self.inner.pop(key);
            return None;
        }
        self.inner.get(key).map(|entry| entry.value.clone())
    }

    /// Insert value into cache.
    pub fn put(&mut self, key: String, value: V) {
        self.inner.put(
            key,
            CacheEntry {
                value,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (usize, usize) {
        (self.inner.len(), self.inner.cap().get())
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Evict expired entries proactively.
    pub fn evict_expired(&mut self) {
        let keys_to_remove: Vec<String> = self
            .inner
            .iter()
            .filter(|(_, entry)| entry.inserted_at.elapsed() >= self.ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            self.inner.pop(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache: TtlLruCache<Vec<f32>> = TtlLruCache::with_config(3, 60);
        cache.put("key1".into(), vec![1.0, 2.0]);
        assert!(cache.get("key1").is_some());
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache: TtlLruCache<String> = TtlLruCache::with_config(2, 60);
        cache.put("a".into(), "alpha".into());
        cache.put("b".into(), "beta".into());
        cache.put("c".into(), "gamma".into()); // Should evict "a"
        assert!(cache.get("a").is_none(), "LRU should have evicted 'a'");
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: TtlLruCache<i32> = TtlLruCache::with_config(10, 60);
        cache.put("x".into(), 1);
        cache.put("y".into(), 2);
        let (len, cap) = cache.stats();
        assert_eq!(len, 2);
        assert_eq!(cap, 10);
    }
}
