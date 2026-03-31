"""SSL verification — auto-detects corporate proxy certificates.

Tries standard SSL first. If self-signed certificate errors occur,
automatically falls back to verify=False. No manual configuration needed.
"""
from __future__ import annotations

import logging
import os
import ssl

_log = logging.getLogger(__name__)
_SSL_VERIFIED: bool | None = None  # None = not yet tested


def ssl_verify() -> bool:
    """Return whether SSL verification should be enabled.

    Auto-detects: tries a real connection on first call.
    If corporate proxy with self-signed cert is detected, returns False permanently.
    Override: SAGE_SSL_VERIFY=false forces bypass, SAGE_SSL_VERIFY=true forces verify.
    """
    global _SSL_VERIFIED

    # Explicit override
    env = os.environ.get("SAGE_SSL_VERIFY", "").lower()
    if env == "false":
        return False
    if env == "true":
        return True

    # Auto-detect on first call
    if _SSL_VERIFIED is None:
        _SSL_VERIFIED = _probe_ssl()
        if not _SSL_VERIFIED:
            _log.info("SSL auto-detect: self-signed certificate detected, bypassing verification")

    return _SSL_VERIFIED


def _probe_ssl() -> bool:
    """Probe if SSL verification works with a known good host."""
    try:
        import socket
        ctx = ssl.create_default_context()
        with socket.create_connection(("www.google.com", 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname="www.google.com") as ssock:
                _ = ssock.version()
        return True
    except (ssl.SSLCertVerificationError, ssl.SSLError):
        return False
    except Exception:
        return True  # Network error, not SSL — keep verify=True


def patch_genai_ssl(client) -> None:
    """Patch a genai.Client for SSL bypass (sync httpx + async aiohttp).

    The google-genai SDK lazily creates HTTP clients using _ensure_*_ssl_ctx()
    factory methods. Patching the client objects alone is insufficient because
    the SDK recreates them. We must override the factories.
    """
    if ssl_verify():
        return

    no_verify_ctx = ssl.create_default_context()
    no_verify_ctx.check_hostname = False
    no_verify_ctx.verify_mode = ssl.CERT_NONE
    ac = client._api_client

    # Override SSL context factory methods (the SDK calls these lazily)
    if hasattr(ac, '_ensure_aiohttp_ssl_ctx'):
        ac._ensure_aiohttp_ssl_ctx = lambda: no_verify_ctx
    if hasattr(ac, '_ensure_httpx_ssl_ctx'):
        ac._ensure_httpx_ssl_ctx = lambda: False

    # Also patch pre-created clients
    try:
        import httpx
        ac._httpx_client = httpx.Client(verify=False, timeout=60)
        ac._async_httpx_client = httpx.AsyncClient(verify=False, timeout=60)
    except Exception:
        _log.debug("Failed to patch genai httpx client for SSL bypass", exc_info=True)

    try:
        import aiohttp
        connector = aiohttp.TCPConnector(ssl=no_verify_ctx)
        ac._aiohttp_session = aiohttp.ClientSession(connector=connector)
    except Exception:
        _log.debug("Failed to patch genai aiohttp for SSL bypass", exc_info=True)
