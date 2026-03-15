"""Tests for provider discovery caching."""
from __future__ import annotations
import json
import time
from pathlib import Path
import pytest


class TestDiscoveryCache:
    def test_write_and_read_cache(self, tmp_path):
        from sage.providers.connector import _write_cache, _read_cache
        models = [{"id": "gemini-2.5-flash", "provider": "google"}]
        _write_cache("google", models, cache_dir=tmp_path)
        cached = _read_cache("google", cache_dir=tmp_path)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["id"] == "gemini-2.5-flash"

    def test_cache_expires_after_ttl(self, tmp_path):
        from sage.providers.connector import _write_cache, _read_cache
        models = [{"id": "test-model", "provider": "test"}]
        _write_cache("test", models, cache_dir=tmp_path)
        cache_file = tmp_path / "test_models.json"
        data = json.loads(cache_file.read_text())
        data["timestamp"] = time.time() - 90000  # 25 hours ago
        cache_file.write_text(json.dumps(data))
        cached = _read_cache("test", cache_dir=tmp_path)
        assert cached is None

    def test_missing_cache_returns_none(self, tmp_path):
        from sage.providers.connector import _read_cache
        cached = _read_cache("nonexistent", cache_dir=tmp_path)
        assert cached is None

    def test_corrupt_cache_returns_none(self, tmp_path):
        cache_file = tmp_path / "bad_models.json"
        cache_file.write_text("not json{{{")
        from sage.providers.connector import _read_cache
        cached = _read_cache("bad", cache_dir=tmp_path)
        assert cached is None
