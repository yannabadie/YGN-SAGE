import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.memory.remote_rag import RagCacheFallback, get_rag_cache


def test_fallback_cache_put_get():
    cache = RagCacheFallback(max_entries=10, ttl_seconds=3600)
    cache.put(42, b"hello world")
    assert cache.get(42) == b"hello world"


def test_fallback_cache_miss():
    cache = RagCacheFallback()
    assert cache.get(999) is None


def test_fallback_cache_eviction():
    cache = RagCacheFallback(max_entries=2, ttl_seconds=3600)
    cache.put(1, b"a")
    cache.put(2, b"b")
    cache.put(3, b"c")  # evicts oldest (key 1)
    assert cache.get(1) is None  # evicted
    assert cache.get(3) == b"c"


def test_fallback_cache_ttl():
    cache = RagCacheFallback(max_entries=10, ttl_seconds=0)
    cache.put(1, b"data")
    import time
    time.sleep(0.01)
    assert cache.get(1) is None  # Expired


def test_get_rag_cache_returns_something():
    cache = get_rag_cache()
    assert cache is not None
    assert hasattr(cache, "put")
    assert hasattr(cache, "get")
