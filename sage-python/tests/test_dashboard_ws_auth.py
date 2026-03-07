"""Tests for dashboard WebSocket authentication -- audit response Task A2."""
import pytest


def test_dashboard_binds_localhost():
    """Dashboard should bind to 127.0.0.1 by default, not 0.0.0.0."""
    import pathlib
    app_py = pathlib.Path(__file__).parent.parent.parent / "ui" / "app.py"
    content = app_py.read_text()
    # The hardcoded default must be 127.0.0.1, not 0.0.0.0
    # 0.0.0.0 should only appear in the env var override context
    lines = content.split('\n')
    for line in lines:
        if 'uvicorn.run' in line and '0.0.0.0' in line:
            pytest.fail("uvicorn.run should not hardcode 0.0.0.0")


def test_websocket_has_auth_check():
    """WebSocket /ws endpoint should check DASHBOARD_TOKEN."""
    import pathlib
    app_py = pathlib.Path(__file__).parent.parent.parent / "ui" / "app.py"
    content = app_py.read_text()
    # Find the websocket_endpoint function and verify it checks token
    in_ws_func = False
    has_token_check = False
    for line in content.split('\n'):
        if 'async def websocket_endpoint' in line:
            in_ws_func = True
        elif in_ws_func and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            break  # left the function
        elif in_ws_func and ('DASHBOARD_TOKEN' in line or 'token' in line.lower()):
            has_token_check = True
    assert has_token_check, "WebSocket endpoint must check DASHBOARD_TOKEN"
