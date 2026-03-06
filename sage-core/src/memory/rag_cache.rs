//! FIFO + TTL cache for File Search query results.
//! Evicts oldest-inserted entry at capacity (not LRU — no access-time refresh).
//! Stores results as raw bytes (Arrow IPC or msgpack).

use dashmap::DashMap;
use pyo3::prelude::*;
use std::time::{Duration, Instant};

struct CacheEntry {
    data: Vec<u8>,
    inserted_at: Instant,
}

#[pyclass]
pub struct RagCache {
    cache: DashMap<u64, CacheEntry>,
    max_entries: usize,
    ttl: Duration,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl RagCache {
    #[new]
    #[pyo3(signature = (max_entries=1000, ttl_seconds=3600))]
    pub fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: DashMap::new(),
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Store a query result.
    pub fn put(&self, query_hash: u64, data: Vec<u8>) {
        // Evict oldest if at capacity
        if self.cache.len() >= self.max_entries {
            if let Some(oldest_key) = self.find_oldest() {
                self.cache.remove(&oldest_key);
            }
        }
        self.cache.insert(
            query_hash,
            CacheEntry {
                data,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Retrieve a cached result. Returns None on miss or TTL expiry.
    pub fn get(&self, query_hash: u64) -> Option<Vec<u8>> {
        if let Some(entry) = self.cache.get(&query_hash) {
            if entry.inserted_at.elapsed() < self.ttl {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(entry.data.clone());
            }
            // Expired — remove it
            drop(entry);
            self.cache.remove(&query_hash);
        }
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    /// Cache stats: (hits, misses, entries)
    pub fn stats(&self) -> (u64, u64, usize) {
        (
            self.hits.load(std::sync::atomic::Ordering::Relaxed),
            self.misses.load(std::sync::atomic::Ordering::Relaxed),
            self.cache.len(),
        )
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.cache.clear();
    }
}

impl RagCache {
    fn find_oldest(&self) -> Option<u64> {
        self.cache
            .iter()
            .min_by_key(|entry| entry.value().inserted_at)
            .map(|entry| *entry.key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_and_get() {
        let cache = RagCache::new(10, 3600);
        cache.put(42, vec![1, 2, 3]);
        assert_eq!(cache.get(42), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_miss() {
        let cache = RagCache::new(10, 3600);
        assert_eq!(cache.get(99), None);
        let (_, misses, _) = cache.stats();
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let cache = RagCache::new(2, 3600);
        cache.put(1, vec![10]);
        cache.put(2, vec![20]);
        cache.put(3, vec![30]); // Should evict oldest (1)
        assert_eq!(cache.get(1), None);
        assert!(cache.get(2).is_some() || cache.get(3).is_some());
    }

    #[test]
    fn test_ttl_expiry() {
        let cache = RagCache::new(10, 0); // TTL = 0 seconds
        cache.put(1, vec![1]);
        // Immediately expired
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert_eq!(cache.get(1), None);
    }

    #[test]
    fn test_stats() {
        let cache = RagCache::new(10, 3600);
        cache.put(1, vec![1]);
        cache.get(1); // hit
        cache.get(2); // miss
        let (hits, misses, entries) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(entries, 1);
    }
}
