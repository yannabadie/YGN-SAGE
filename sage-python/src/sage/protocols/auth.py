"""Minimum-viable MCP/A2A gateway authentication (AUDIT.md §6 / A19).

Scope:
- Localhost-by-default bind address (was already the default in
  `serve.py`; this module hoists the policy so other call sites can
  share it).
- `SAGE_PROTOCOL_BIND_ALL=1` opt-in to bind 0.0.0.0 (public exposure).
- Startup SECURITY WARN when bound to 0.0.0.0 without a bearer token.
- Bearer-token check with constant-time compare (`hmac.compare_digest`).
- Audit-log helper for every resource/tool access.

Not scope (follow-up):
- OAuth2 / JWT / PKCE — MCP 2025-11-25 spec recommends OAuth2 for HTTP
  transports; this module only covers shared-secret bearer-tokens.
- Role-based access control / capability scopes — only auth present
  today is "is the bearer token valid?".
- mTLS / transport-layer auth — terminate at the HTTP layer only.

The middleware functions return frameworks-compatible callables
(Starlette/FastAPI) but are NOT yet wired into `a2a_server.py` or
`mcp_server.py`. Wiring is the second phase of A19.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any, Awaitable, Callable

__all__ = [
    "DEFAULT_BIND_HOST",
    "PUBLIC_BIND_HOST",
    "audit_log_access",
    "bearer_token_from_env",
    "constant_time_compare",
    "is_public_bind",
    "require_bearer_middleware",
    "resolve_bind_host",
    "warn_insecure_bind",
]

_log = logging.getLogger(__name__)

DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_BIND_HOST = "0.0.0.0"

# Only warn once per process about insecure bind to keep logs quiet.
_INSECURE_BIND_WARNED = False


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def is_public_bind() -> bool:
    """True when SAGE_PROTOCOL_BIND_ALL=1 is set."""
    return _env_bool("SAGE_PROTOCOL_BIND_ALL")


def resolve_bind_host(override: str | None = None) -> str:
    """Decide the bind host.

    Explicit `override` wins; otherwise uses SAGE_PROTOCOL_BIND_ALL env
    (public 0.0.0.0 when set, localhost otherwise).
    """
    if override is not None:
        return override
    return PUBLIC_BIND_HOST if is_public_bind() else DEFAULT_BIND_HOST


def bearer_token_from_env() -> str | None:
    """Return the server-side bearer token or None when unset."""
    token = os.environ.get("SAGE_PROTOCOL_BEARER_TOKEN", "").strip()
    return token or None


def constant_time_compare(a: str, b: str) -> bool:
    """Timing-attack-safe string equality."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def warn_insecure_bind(host: str) -> None:
    """Emit a one-shot WARN when binding publicly without a bearer token."""
    global _INSECURE_BIND_WARNED
    if _INSECURE_BIND_WARNED:
        return
    if host == PUBLIC_BIND_HOST and bearer_token_from_env() is None:
        _log.warning(
            "SECURITY: SAGE protocol server bound to %s without "
            "SAGE_PROTOCOL_BEARER_TOKEN — this is UNSAFE for public exposure. "
            "Set SAGE_PROTOCOL_BEARER_TOKEN to require Bearer auth on every request.",
            host,
        )
        _INSECURE_BIND_WARNED = True


def _reset_warn_flag_for_tests() -> None:
    """Test-only helper to re-arm `warn_insecure_bind`."""
    global _INSECURE_BIND_WARNED
    _INSECURE_BIND_WARNED = False


def audit_log_access(path: str, method: str, client_ip: str | None = None) -> None:
    """Emit a timestamped audit line for an incoming request.

    Callers pass what they have; `client_ip` may be None when the
    framework doesn't expose it.
    """
    _log.info(
        "protocol_access method=%s path=%s client=%s",
        method,
        path,
        client_ip or "unknown",
    )


def require_bearer_middleware(
    token: str | None = None,
) -> Callable[[Any, Callable[[Any], Awaitable[Any]]], Awaitable[Any]]:
    """Return a Starlette/FastAPI-compatible middleware.

    When the resolved token is set, every request must include
    `Authorization: Bearer <token>`. Constant-time compared. When no
    token is configured, the middleware short-circuits (no auth) —
    combine with `warn_insecure_bind` to surface that state at startup.
    """
    expected = token if token is not None else bearer_token_from_env()

    async def _middleware(
        request: Any, call_next: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        # Attempt best-effort audit log; don't fail the request on
        # unknown attributes when used with a non-standard framework.
        try:
            path = getattr(getattr(request, "url", None), "path", "?")
            method = getattr(request, "method", "?")
            client_ip: str | None = None
            client = getattr(request, "client", None)
            if client is not None:
                client_ip = getattr(client, "host", None)
            audit_log_access(path, method, client_ip)
        except Exception:  # noqa: BLE001
            pass

        if expected is None:
            return await call_next(request)

        auth_header = ""
        try:
            auth_header = request.headers.get("authorization", "")
        except Exception:  # noqa: BLE001
            auth_header = ""
        if not auth_header.lower().startswith("bearer "):
            return _unauthorized_response()
        supplied = auth_header[len("Bearer ") :].strip()
        if not constant_time_compare(supplied, expected):
            return _unauthorized_response()
        return await call_next(request)

    return _middleware


def _unauthorized_response() -> Any:
    """Build a minimal 401 response.

    Import Starlette lazily so that importing this module doesn't force
    a hard dependency on the HTTP framework when callers only want the
    helper functions (bearer_token_from_env, resolve_bind_host, etc.).
    """
    try:
        from starlette.responses import JSONResponse  # type: ignore[import-not-found]

        return JSONResponse(
            {"error": "unauthorized", "detail": "valid Bearer token required"},
            status_code=401,
        )
    except ImportError:  # pragma: no cover — framework missing is a deploy bug
        return ("unauthorized", 401)
