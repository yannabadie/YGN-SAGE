"""SSL bypass helper for corporate proxy environments.

Patches a ``google.genai.Client`` to disable SSL verification when the
environment indicates a corporate proxy with self-signed certificates.

Activation conditions (either triggers the patch):
- ``REQUESTS_CA_BUNDLE`` is set to ``""``
- ``SAGE_SSL_VERIFY`` is set to ``"0"``
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def patch_genai_ssl(client) -> None:
    """Patch a ``genai.Client`` to bypass SSL verification if needed.

    No-op when SSL bypass is not requested by the environment.
    """
    if os.environ.get("REQUESTS_CA_BUNDLE") == "" or os.environ.get("SAGE_SSL_VERIFY") == "0":
        try:
            import httpx
            client._api_client._httpx_client = httpx.Client(verify=False, timeout=60)
        except Exception:
            _log.debug("Failed to patch genai client for SSL bypass", exc_info=True)
