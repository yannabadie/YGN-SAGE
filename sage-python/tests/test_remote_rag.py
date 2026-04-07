import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.memory.remote_rag import ExoCortex


def test_exocortex_init_without_key(monkeypatch):
    """ExoCortex initializes gracefully without API key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    exo = ExoCortex()
    assert not exo.is_available


def test_exocortex_store_name_from_env(monkeypatch):
    """ExoCortex reads store name from SAGE_EXOCORTEX_STORE env var."""
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "projects/123/fileSearchStores/test-store")
    exo = ExoCortex()
    assert exo.store_name == "projects/123/fileSearchStores/test-store"


def test_exocortex_get_tool_returns_none_when_unavailable(monkeypatch):
    """get_tool returns None when no store configured."""
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    exo = ExoCortex()
    assert exo.get_file_search_tool() is None


def test_google_provider_accepts_file_search_stores():
    """GoogleProvider.generate() accepts file_search_store_names param."""
    from sage.llm.google import GoogleProvider
    import inspect
    sig = inspect.signature(GoogleProvider.generate)
    assert "file_search_store_names" in sig.parameters
