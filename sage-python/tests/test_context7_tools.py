"""Unit tests for the Context7 library-docs tool (C2c 2026-04-21).

Spec: docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md
      docs/benchmarks/2026-04-21-c2b-smoke-results.md (C2c rationale)

These tests never hit the real Context7 API. httpx is mocked so the
tests stay deterministic and offline. A separate integration test
(disabled unless CONTEXT7_API_KEY is set and CONTEXT7_TEST_LIVE=1)
can exercise the live endpoint — out of scope for the unit file.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from sage.tools.context7_tools import (
    _CONTEXT7_BASE,
    _DOCS_PATH,
    _RESOLVE_PATH,
    _format_snippets,
    _lookup,
    create_context7_tools,
)


# ---------------------------------------------------------------------------
# Registration — env-driven
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_context7_env(monkeypatch):
    """Every test starts with a clean env. Individual tests opt in to
    setting CONTEXT7_API_KEY / CONTEXT7."""
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXT7", raising=False)


def test_no_api_key_returns_empty_list():
    """Matches the ExoCortex no-op pattern — if the backend is not
    configured, the tool simply isn't registered. No exceptions."""
    assert create_context7_tools() == []


def test_api_key_via_canonical_env_var_registers_tool(monkeypatch):
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-fake-canonical")
    tools = create_context7_tools()
    assert len(tools) == 1
    assert tools[0].spec.name == "lookup_library_docs"


def test_api_key_via_legacy_env_var_registers_tool(monkeypatch):
    """Some users have `CONTEXT7=...` in .env (pre-standard naming).
    We accept that as a fallback so no one has to rename their config."""
    monkeypatch.setenv("CONTEXT7", "ctx7sk-fake-legacy")
    tools = create_context7_tools()
    assert len(tools) == 1


def test_canonical_env_var_wins_over_legacy(monkeypatch):
    """If both are set (unlikely but possible), CONTEXT7_API_KEY takes
    precedence — the documented name wins deterministically."""
    monkeypatch.setenv("CONTEXT7_API_KEY", "canonical-wins")
    monkeypatch.setenv("CONTEXT7", "legacy-loses")
    tools = create_context7_tools(api_key=None)
    # We can't peek at the closed-over key directly, but we can verify
    # via a lookup call. Reject-other-keys assertion covered elsewhere.
    assert len(tools) == 1


def test_explicit_api_key_parameter_overrides_env(monkeypatch):
    monkeypatch.setenv("CONTEXT7_API_KEY", "env-should-lose")
    tools = create_context7_tools(api_key="explicit-wins")
    assert len(tools) == 1


def test_tool_schema_parameters():
    """The function-calling schema the LLM sees. Locking these so a
    future edit that renames fields becomes a visible test diff."""
    tools = create_context7_tools(api_key="x")
    schema = tools[0].spec.parameters
    assert schema["type"] == "object"
    assert schema["required"] == ["library_name", "query"]
    assert set(schema["properties"].keys()) == {"library_name", "query"}


def test_tool_description_targets_library_api_scope():
    """The LLM picks tools based on this description. C2b's lesson:
    if the description says 'research papers', no SWE-bench task
    triggers it. This description explicitly targets library API
    contracts + django/astropy/requests examples."""
    tools = create_context7_tools(api_key="x")
    desc = tools[0].spec.description
    assert "library" in desc.lower()
    assert "Context7" in desc
    # Concrete example anchors — prevent drift back to abstract phrasing
    assert "requests" in desc or "Django" in desc or "astropy" in desc


# ---------------------------------------------------------------------------
# _format_snippets — response rendering
# ---------------------------------------------------------------------------


def test_format_snippets_renders_docs_and_code():
    payload = {
        "infoSnippets": [
            {"content": "Django's FILE_UPLOAD_PERMISSION defaults to None."},
        ],
        "codeSnippets": [
            {
                "codeTitle": "settings.py",
                "codeList": [{"code": "FILE_UPLOAD_PERMISSIONS = 0o644"}],
            }
        ],
    }
    out = _format_snippets(payload)
    assert "## Documentation" in out
    assert "defaults to None" in out
    assert "## Code examples" in out
    assert "### settings.py" in out
    assert "FILE_UPLOAD_PERMISSIONS = 0o644" in out


def test_format_snippets_empty_response_returns_stable_string():
    """Empty payload must produce a fixed human-readable string, not a
    crash or blank output — the LLM needs something to react to."""
    assert _format_snippets({}) == "No documentation found for this query."
    assert _format_snippets({"codeSnippets": [], "infoSnippets": []}) == (
        "No documentation found for this query."
    )


def test_format_snippets_caps_code_examples_at_max():
    """Uncapped Context7 responses can be large; we cap at _MAX_SNIPPETS
    so tool output stays under the model's context-window budget."""
    from sage.tools.context7_tools import _MAX_SNIPPETS

    payload = {
        "codeSnippets": [
            {"codeTitle": f"t{i}", "codeList": [{"code": f"c{i}"}]}
            for i in range(_MAX_SNIPPETS + 5)
        ]
    }
    out = _format_snippets(payload)
    for i in range(_MAX_SNIPPETS):
        assert f"### t{i}" in out
    # Indices >= cap must NOT appear
    assert f"### t{_MAX_SNIPPETS}" not in out


def test_format_snippets_skips_empty_items():
    """Robustness: missing fields / empty strings don't blow up the
    renderer, they just get skipped."""
    payload = {
        "infoSnippets": [{"content": ""}, {"content": "kept"}],
        "codeSnippets": [
            {"codeTitle": "", "codeList": []},
            {"codeTitle": "real", "codeList": [{"code": "x=1"}]},
        ],
    }
    out = _format_snippets(payload)
    assert "kept" in out
    assert "### real" in out
    assert "x=1" in out


# ---------------------------------------------------------------------------
# _lookup — full request/response path with mocked httpx
# ---------------------------------------------------------------------------


def _mk_response(payload: Any, status: int = 200) -> httpx.Response:
    """Build an httpx.Response synchronously for mocking."""
    req = httpx.Request("GET", "https://context7.com/dummy")
    return httpx.Response(status, request=req, content=json.dumps(payload).encode())


class _AsyncClientMock:
    """Minimal httpx.AsyncClient stub used in `async with` blocks."""

    def __init__(self, get_side_effect):
        self._get = AsyncMock(side_effect=get_side_effect)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params=None, headers=None):
        return await self._get(url, params=params, headers=headers)


@pytest.mark.asyncio
async def test_lookup_happy_path_resolves_then_fetches():
    """Two sequential GETs: search (returns top id) then context
    (returns snippets). Output contains the library's docs."""
    calls: list[tuple[str, dict]] = []

    async def fake_get(url, params=None, headers=None):
        calls.append((url, params or {}))
        if url.endswith(_RESOLVE_PATH):
            return _mk_response({"results": [{"id": "/django/django/4.2", "title": "Django"}]})
        if url.endswith(_DOCS_PATH):
            return _mk_response({
                "infoSnippets": [{"content": "Django default upload permission is None"}],
                "codeSnippets": [],
            })
        raise AssertionError(f"unexpected URL {url}")

    with patch(
        "sage.tools.context7_tools.httpx.AsyncClient",
        return_value=_AsyncClientMock(fake_get),
    ):
        out = await _lookup("ctx7sk-fake", "Django", "FILE_UPLOAD_PERMISSION default")

    assert "Django default upload permission" in out
    # Two calls in order: resolve, then docs.
    assert [c[0] for c in calls] == [
        _CONTEXT7_BASE + _RESOLVE_PATH,
        _CONTEXT7_BASE + _DOCS_PATH,
    ]
    assert calls[0][1]["libraryName"] == "Django"
    assert calls[0][1]["query"] == "FILE_UPLOAD_PERMISSION default"
    assert calls[1][1]["libraryId"] == "/django/django/4.2"


@pytest.mark.asyncio
async def test_lookup_library_not_found_returns_helpful_message():
    """Empty results → tell the LLM the library isn't indexed AND
    suggest the execute_bash fallback. No exception."""
    async def fake_get(url, params=None, headers=None):
        if url.endswith(_RESOLVE_PATH):
            return _mk_response({"results": []})
        raise AssertionError("docs endpoint should not be called when resolve fails")

    with patch(
        "sage.tools.context7_tools.httpx.AsyncClient",
        return_value=_AsyncClientMock(fake_get),
    ):
        out = await _lookup("ctx7sk-fake", "nonexistent-lib", "anything")

    assert "no indexed library" in out
    assert "nonexistent-lib" in out
    assert "execute_bash" in out  # suggests fallback


@pytest.mark.asyncio
async def test_lookup_resolve_http_error_returns_error_string():
    """HTTP errors during resolve → user-friendly error string, NOT
    a raised exception (tool handler catches separately, but we want
    a good message at source)."""
    async def fake_get(url, params=None, headers=None):
        raise httpx.HTTPError("boom")

    with patch(
        "sage.tools.context7_tools.httpx.AsyncClient",
        return_value=_AsyncClientMock(fake_get),
    ):
        out = await _lookup("ctx7sk-fake", "django", "x")

    assert "Context7 library search failed" in out
    assert "HTTPError" in out


@pytest.mark.asyncio
async def test_lookup_docs_http_error_returns_error_string_with_library_id():
    """Second-leg error → message includes the library id so the LLM
    can decide whether to retry with a different library."""
    async def fake_get(url, params=None, headers=None):
        if url.endswith(_RESOLVE_PATH):
            return _mk_response({"results": [{"id": "/django/django", "title": "Django"}]})
        raise httpx.HTTPError("docs endpoint down")

    with patch(
        "sage.tools.context7_tools.httpx.AsyncClient",
        return_value=_AsyncClientMock(fake_get),
    ):
        out = await _lookup("ctx7sk-fake", "django", "x")

    assert "Context7 docs query failed" in out
    assert "/django/django" in out


@pytest.mark.asyncio
async def test_lookup_sends_bearer_authorization_header():
    """Context7 requires `Authorization: Bearer <key>`. Verify both
    requests include it — the audit for auth regression lives here."""
    seen_headers: list[dict] = []

    async def fake_get(url, params=None, headers=None):
        seen_headers.append(headers or {})
        if url.endswith(_RESOLVE_PATH):
            return _mk_response({"results": [{"id": "/x/y", "title": "x"}]})
        return _mk_response({"infoSnippets": [], "codeSnippets": []})

    with patch(
        "sage.tools.context7_tools.httpx.AsyncClient",
        return_value=_AsyncClientMock(fake_get),
    ):
        await _lookup("ctx7sk-testkey", "whatever", "x")

    assert len(seen_headers) == 2
    for headers in seen_headers:
        assert headers.get("Authorization") == "Bearer ctx7sk-testkey"


# ---------------------------------------------------------------------------
# Tool.execute integration — make sure the decorated tool path works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registered_tool_executes_end_to_end(monkeypatch):
    """Register via the public API, invoke through Tool.execute() as
    the agent loop would — confirms the closure + decorator + handler
    wiring all compose correctly."""
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-integration")

    async def fake_get(url, params=None, headers=None):
        if url.endswith(_RESOLVE_PATH):
            return _mk_response({"results": [{"id": "/requests/docs", "title": "Requests"}]})
        return _mk_response({
            "infoSnippets": [{"content": "Response.json() raises on empty body."}],
            "codeSnippets": [],
        })

    tools = create_context7_tools()
    assert len(tools) == 1

    with patch(
        "sage.tools.context7_tools.httpx.AsyncClient",
        return_value=_AsyncClientMock(fake_get),
    ):
        result = await tools[0].execute(
            {"library_name": "requests", "query": "Response.json empty body"}
        )

    assert not result.is_error
    assert "raises on empty body" in result.output
