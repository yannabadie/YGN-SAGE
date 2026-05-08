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
    """Patch a genai.Client to use the CA bundle when standard SSL fails.

    Directive #3: NEVER use verify=False when ca-bundle.pem is available.
    Prefer the CA bundle from REQUESTS_CA_BUNDLE or SSL_CERT_FILE.
    Only fall back to verify=False when no bundle is configured.
    """
    if ssl_verify():
        return

    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get(
        "SSL_CERT_FILE"
    )
    # Directive #3 / REVIEW5: NEVER silently fall back to verify=False.
    # When no CA bundle is configured, keep SSL enabled (True) and let
    # the caller diagnose the issue.  Explicit bypass requires
    # SAGE_SSL_VERIFY=false.
    verify_setting: str | bool = ca_bundle if ca_bundle else True

    try:
        import httpx
        client._api_client._httpx_client = httpx.Client(
            verify=verify_setting, timeout=60,
        )
        if hasattr(client._api_client, "_async_httpx_client"):
            client._api_client._async_httpx_client = httpx.AsyncClient(
                verify=verify_setting, timeout=60,
            )
    except Exception:
        _log.debug("Failed to patch genai httpx client for SSL", exc_info=True)

    # google-genai async uses aiohttp internally — patch SSL context
    try:
        import aiohttp
        ssl_ctx = ssl.create_default_context()
        if isinstance(verify_setting, str):
            ssl_ctx.load_verify_locations(cafile=verify_setting)
        # REVIEW5: no silent verify=False fallback.
        # Default SSL context stays enabled (verify_mode=CERT_REQUIRED).
        # Explicit bypass only via SAGE_SSL_VERIFY=false.
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        if hasattr(client._api_client, "_async_client"):
            client._api_client._async_client = aiohttp.ClientSession(
                connector=connector,
            )
        import google.genai._api_client as _gc
        if hasattr(_gc, "_DEFAULT_CONNECTOR"):
            _gc._DEFAULT_CONNECTOR = connector
    except Exception:
        _log.debug("Failed to patch genai aiohttp for SSL", exc_info=True)
