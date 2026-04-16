"""Dashboard security: auth + CORS + EventBus.clear()."""
from __future__ import annotations

import os
import asyncio
import time
import pytest
from unittest.mock import patch

from httpx import AsyncClient, ASGITransport


def _reload_app_module():
    """Import ui/app.py and reload it to get fresh module state."""
    import sys
    from pathlib import Path

    ui_dir = str(Path(__file__).resolve().parent.parent.parent / "ui")
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)

    import importlib
    import app as app_mod
    importlib.reload(app_mod)
    return app_mod


@pytest.fixture
def app():
    """Import the FastAPI app from ui/app.py."""
    return _reload_app_module().app


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
async def test_reset_waits_for_inflight_run_before_clearing_state():
    """Reset must serialize behind run_lock so shared state is not cleared mid-run."""
    app_mod = _reload_app_module()
    app_mod._state.event_bus.emit(
        app_mod.AgentEvent(type="TEST", step=0, timestamp=time.time(), meta={"source": "pre-reset"})
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_lock():
        async with app_mod._state.run_lock:
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold_lock())
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    transport = ASGITransport(app=app_mod.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reset_task = asyncio.create_task(client.post("/api/reset"))
        await asyncio.sleep(0.05)
        assert not reset_task.done(), "reset should wait for the active run lock holder"

        release.set()
        resp = await asyncio.wait_for(reset_task, timeout=1.0)

    await asyncio.wait_for(holder, timeout=1.0)
    assert resp.status_code == 200
    assert app_mod._state.system is None
    assert len(app_mod._state.event_bus.query(last_n=10)) == 0


@pytest.mark.asyncio
async def test_reset_cancels_tracked_background_tasks():
    """Reset must cancel detached dashboard tasks such as benchmark runners."""
    app_mod = _reload_app_module()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def background():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = app_mod._track_background_task(asyncio.create_task(background()))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert task in app_mod._state._background_tasks

    transport = ASGITransport(app=app_mod.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/reset")

    assert resp.status_code == 200
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert task.done()
    assert task not in app_mod._state._background_tasks


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
