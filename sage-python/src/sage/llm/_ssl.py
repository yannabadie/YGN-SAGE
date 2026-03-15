"""SSL verification helper — controlled by SAGE_SSL_VERIFY env var."""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def ssl_verify() -> bool:
    """Return False only if SAGE_SSL_VERIFY=false is explicitly set."""
    return os.environ.get("SAGE_SSL_VERIFY", "true").lower() != "false"


def patch_genai_ssl(client) -> None:
    """Patch a ``genai.Client`` to use the configured SSL verification setting.

    When ``ssl_verify()`` returns True (default), this is a no-op.
    When ``SAGE_SSL_VERIFY=false``, patches the client to skip verification.
    """
    if not ssl_verify():
        try:
            import httpx
            client._api_client._httpx_client = httpx.Client(verify=False, timeout=60)
        except Exception:
            _log.debug("Failed to patch genai client for SSL bypass", exc_info=True)
