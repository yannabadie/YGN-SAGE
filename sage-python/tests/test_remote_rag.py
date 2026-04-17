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


def test_upload_signature_has_timeout_param():
    """Regression for 2026-04-17 hang: upload() must expose a polling
    timeout. Pre-fix the polling loop on Google's index operation had no
    deadline — a single slow upload would hang the whole pipeline.
    """
    import inspect
    sig = inspect.signature(ExoCortex.upload)
    assert "timeout_s" in sig.parameters
    # Default must be sane (not infinity, not absurdly small)
    assert 30.0 <= sig.parameters["timeout_s"].default <= 600.0


@pytest.mark.asyncio
async def test_upload_raises_timeout_when_operation_never_done(monkeypatch):
    """When Google's operation polling never returns done=True, upload()
    must raise TimeoutError within the budget so callers can move on
    instead of hanging the pipeline.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "fileSearchStores/test")
    exo = ExoCortex()

    # Stub the genai client. The fake operation is never done.
    class _NeverDoneOperation:
        done = False

    class _FakeOperations:
        def get(self, op):  # noqa: ARG002
            return _NeverDoneOperation()

    class _FakeStores:
        def upload_to_file_search_store(self, **_kw):
            return _NeverDoneOperation()

    class _FakeClient:
        file_search_stores = _FakeStores()
        operations = _FakeOperations()

    # Replace the genai.Client constructor so the inner _upload sees our fake.
    import google.genai as _genai
    monkeypatch.setattr(_genai, "Client", lambda **_kw: _FakeClient())
    # Bypass the SSL patcher (it inspects real client internals).
    import sage.llm._ssl as _ssl
    monkeypatch.setattr(_ssl, "patch_genai_ssl", lambda _c: None)

    with pytest.raises(TimeoutError):
        await exo.upload("/tmp/fake.pdf", "fake", timeout_s=2.0)
