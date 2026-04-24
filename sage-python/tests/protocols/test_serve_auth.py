"""AUDIT3 / A19 — MCP/A2A gateway auth helpers."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from sage.protocols import auth as auth_mod
from sage.protocols.auth import (
    DEFAULT_BIND_HOST,
    PUBLIC_BIND_HOST,
    bearer_token_from_env,
    constant_time_compare,
    is_public_bind,
    require_bearer_middleware,
    resolve_bind_host,
    warn_insecure_bind,
)


# -------- bind host resolution --------


def test_bind_defaults_to_localhost(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SAGE_PROTOCOL_BIND_ALL", raising=False)
    assert resolve_bind_host() == DEFAULT_BIND_HOST
    assert is_public_bind() is False


def test_bind_all_env_switches_to_public(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_PROTOCOL_BIND_ALL", "1")
    assert resolve_bind_host() == PUBLIC_BIND_HOST
    assert is_public_bind() is True


def test_explicit_override_wins_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_PROTOCOL_BIND_ALL", "1")
    assert resolve_bind_host("10.0.0.1") == "10.0.0.1"


# -------- bearer token --------


def test_bearer_token_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SAGE_PROTOCOL_BEARER_TOKEN", raising=False)
    assert bearer_token_from_env() is None


def test_bearer_token_trims_whitespace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_PROTOCOL_BEARER_TOKEN", "  secret123  ")
    assert bearer_token_from_env() == "secret123"


# -------- timing-safe compare --------


def test_constant_time_compare_equal():
    assert constant_time_compare("abc", "abc") is True


def test_constant_time_compare_different():
    assert constant_time_compare("abc", "xyz") is False


def test_constant_time_compare_uses_hmac(monkeypatch: pytest.MonkeyPatch):
    with patch("sage.protocols.auth.hmac.compare_digest") as mocked:
        mocked.return_value = True
        constant_time_compare("a", "b")
        mocked.assert_called_once()


# -------- insecure-bind warning --------


def test_warn_insecure_bind_fires_on_public_without_token(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.delenv("SAGE_PROTOCOL_BEARER_TOKEN", raising=False)
    auth_mod._reset_warn_flag_for_tests()
    with caplog.at_level(logging.WARNING, logger="sage.protocols.auth"):
        warn_insecure_bind(PUBLIC_BIND_HOST)
    assert any("UNSAFE for public exposure" in r.message for r in caplog.records)


def test_warn_insecure_bind_silent_on_localhost(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.delenv("SAGE_PROTOCOL_BEARER_TOKEN", raising=False)
    auth_mod._reset_warn_flag_for_tests()
    with caplog.at_level(logging.WARNING, logger="sage.protocols.auth"):
        warn_insecure_bind(DEFAULT_BIND_HOST)
    assert not any("UNSAFE" in r.message for r in caplog.records)


def test_warn_insecure_bind_silent_when_token_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("SAGE_PROTOCOL_BEARER_TOKEN", "t")
    auth_mod._reset_warn_flag_for_tests()
    with caplog.at_level(logging.WARNING, logger="sage.protocols.auth"):
        warn_insecure_bind(PUBLIC_BIND_HOST)
    assert not any("UNSAFE" in r.message for r in caplog.records)


def test_warn_insecure_bind_fires_only_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.delenv("SAGE_PROTOCOL_BEARER_TOKEN", raising=False)
    auth_mod._reset_warn_flag_for_tests()
    with caplog.at_level(logging.WARNING, logger="sage.protocols.auth"):
        warn_insecure_bind(PUBLIC_BIND_HOST)
        warn_insecure_bind(PUBLIC_BIND_HOST)
    unsafe_lines = [r for r in caplog.records if "UNSAFE" in r.message]
    assert len(unsafe_lines) == 1


# -------- middleware --------


def _make_request(authorization: str | None) -> SimpleNamespace:
    headers = {"authorization": authorization} if authorization is not None else {}
    return SimpleNamespace(
        url=SimpleNamespace(path="/resources/list"),
        method="POST",
        headers=headers,
        client=SimpleNamespace(host="1.2.3.4"),
    )


@pytest.mark.asyncio
async def test_middleware_no_token_is_open(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SAGE_PROTOCOL_BEARER_TOKEN", raising=False)
    call_next = AsyncMock(return_value="ok")
    mw = require_bearer_middleware()
    assert await mw(_make_request(None), call_next) == "ok"


@pytest.mark.asyncio
async def test_middleware_correct_bearer_passes():
    call_next = AsyncMock(return_value="ok")
    mw = require_bearer_middleware(token="abc")
    assert await mw(_make_request("Bearer abc"), call_next) == "ok"


@pytest.mark.asyncio
async def test_middleware_wrong_bearer_returns_401():
    call_next = AsyncMock(return_value="ok")
    mw = require_bearer_middleware(token="abc")
    response = await mw(_make_request("Bearer wrong"), call_next)
    # Starlette available in the test env -> JSONResponse with 401
    assert getattr(response, "status_code", None) == 401
    call_next.assert_not_called()


@pytest.mark.asyncio
async def test_middleware_missing_header_returns_401():
    call_next = AsyncMock(return_value="ok")
    mw = require_bearer_middleware(token="abc")
    response = await mw(_make_request(None), call_next)
    assert getattr(response, "status_code", None) == 401


@pytest.mark.asyncio
async def test_middleware_uses_env_token_when_param_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAGE_PROTOCOL_BEARER_TOKEN", "envtok")
    call_next = AsyncMock(return_value="ok")
    mw = require_bearer_middleware()
    assert await mw(_make_request("Bearer envtok"), call_next) == "ok"


# -------- A19 wiring: verify middleware INSTALLED on A2A app --------


def _a2a_has_bearer_middleware(app) -> bool:
    """Inspect Starlette's user_middleware stack for BaseHTTPMiddleware."""
    from starlette.middleware.base import BaseHTTPMiddleware
    return any(mw.cls is BaseHTTPMiddleware for mw in app.user_middleware)


def test_a2a_app_installs_middleware_when_token_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_PROTOCOL_BEARER_TOKEN", "tok")
    pytest.importorskip("a2a")
    from sage.protocols.a2a_server import create_a2a_app
    app = create_a2a_app()
    assert _a2a_has_bearer_middleware(app), (
        "A19 regression: A2A app built without bearer-token middleware despite "
        "SAGE_PROTOCOL_BEARER_TOKEN being set"
    )


def test_a2a_app_no_middleware_when_token_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SAGE_PROTOCOL_BEARER_TOKEN", raising=False)
    pytest.importorskip("a2a")
    from sage.protocols.a2a_server import create_a2a_app
    app = create_a2a_app()
    assert not _a2a_has_bearer_middleware(app), (
        "A2A app should not install auth middleware when no token configured"
    )
