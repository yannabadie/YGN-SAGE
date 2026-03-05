import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.memory.remote_rag import ExoCortex


def test_exocortex_init_without_key():
    """ExoCortex initializes gracefully without API key."""
    import os
    saved = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        exo = ExoCortex()
        assert not exo.is_available
    finally:
        if saved:
            os.environ["GOOGLE_API_KEY"] = saved


def test_exocortex_store_name_from_env():
    """ExoCortex reads store name from SAGE_EXOCORTEX_STORE env var."""
    import os
    os.environ["SAGE_EXOCORTEX_STORE"] = "projects/123/fileSearchStores/test-store"
    try:
        exo = ExoCortex()
        assert exo.store_name == "projects/123/fileSearchStores/test-store"
    finally:
        del os.environ["SAGE_EXOCORTEX_STORE"]


def test_exocortex_get_tool_returns_none_when_unavailable():
    """get_tool returns None when no store configured."""
    import os
    saved = os.environ.pop("SAGE_EXOCORTEX_STORE", None)
    saved_key = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        exo = ExoCortex()
        assert exo.get_file_search_tool() is None
    finally:
        if saved:
            os.environ["SAGE_EXOCORTEX_STORE"] = saved
        if saved_key:
            os.environ["GOOGLE_API_KEY"] = saved_key


def test_google_provider_accepts_file_search_stores():
    """GoogleProvider.generate() accepts file_search_store_names param."""
    from sage.llm.google import GoogleProvider
    import inspect
    sig = inspect.signature(GoogleProvider.generate)
    assert "file_search_store_names" in sig.parameters
