"""Dashboard security: auth + CORS + EventBus.clear()."""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    """Import the FastAPI app from ui/app.py."""
    import sys
    from pathlib import Path

    ui_dir = str(Path(__file__).resolve().parent.parent.parent / "ui")
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)

    # Re-import to pick up fresh state
    import importlib
    import app as app_mod
    importlib.reload(app_mod)
    return app_mod.app


@pytest.mark.asyncio
async def test_dashboard_requires_auth_when_token_set(app):
    """API routes must return 401 when SAGE_DASHBOARD_TOKEN is set but no bearer token provided."""
    with patch.dict(os.environ, {"SAGE_DASHBOARD_TOKEN": "secret-test-token"}):
        # Need to reload app module to pick up new env var
        import importlib
        import sys
        ui_dir = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "ui")
        if ui_dir not in sys.path:
            sys.path.insert(0, ui_dir)
        import app as app_mod
        importlib.reload(app_mod)

        transport = ASGITransport(app=app_mod.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/state")
            assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_dashboard_open_when_no_token(app):
    """API routes must be open when SAGE_DASHBOARD_TOKEN is not set (dev mode)."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SAGE_DASHBOARD_TOKEN", None)
        import importlib
        import sys
        ui_dir = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "ui")
        if ui_dir not in sys.path:
            sys.path.insert(0, ui_dir)
        import app as app_mod
        importlib.reload(app_mod)

        transport = ASGITransport(app=app_mod.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/state")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_cors_headers_present(app):
    """CORS middleware must be configured."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/api/state",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in resp.headers


def test_eventbus_clear():
    """EventBus must expose a public clear() method."""
    from sage.events.bus import EventBus
    from sage.agent_loop import AgentEvent
    import time

    bus = EventBus()
    bus.emit(AgentEvent(type="TEST", step=0, timestamp=time.time(), meta={}))
    assert len(bus.query(last_n=100)) == 1

    bus.clear()
    assert len(bus.query(last_n=100)) == 0
