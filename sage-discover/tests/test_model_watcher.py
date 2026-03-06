"""Tests for the ModelWatcher module."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.model_watcher import ModelWatcher


# ---------------------------------------------------------------------------
# Helper: patch the lazy import of ModelRegistry inside check_new_models
# ---------------------------------------------------------------------------

def _patch_registry(**registry_kwargs):
    """Context manager that patches sage.providers.registry.ModelRegistry.

    Since ModelWatcher imports ModelRegistry inside the method body, we
    patch the *source* module rather than the consumer module.
    """
    mock_registry = MagicMock()
    mock_registry.refresh = AsyncMock()
    mock_registry._profiles = registry_kwargs.get("profiles", {})
    mock_registry.get = MagicMock(
        side_effect=lambda mid: mock_registry._profiles.get(mid)
    )

    return patch(
        "sage.providers.registry.ModelRegistry",
        return_value=mock_registry,
    ), mock_registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_new_models_returns_list():
    """check_new_models returns a list even when registry is empty."""
    watcher = ModelWatcher()
    patcher, _ = _patch_registry(profiles={})

    with patcher:
        results = await watcher.check_new_models()
        assert isinstance(results, list)
        assert len(results) == 0


@pytest.mark.asyncio
async def test_check_new_models_import_error():
    """When sage.providers is unavailable, returns empty list gracefully."""
    watcher = ModelWatcher()
    # Block the import so the except ImportError path fires
    with patch.dict(
        "sys.modules",
        {"sage.providers.registry": None, "sage.providers": None},
    ):
        results = await watcher.check_new_models()
        assert results == []


@pytest.mark.asyncio
async def test_check_new_models_detects_unprofiled():
    """Models available but without cost data are flagged as new."""
    watcher = ModelWatcher()

    # Create mock profiles: one profiled, one unprofiled
    profiled = MagicMock()
    profiled.id = "gemini-3.1-pro-preview"
    profiled.provider = "google"
    profiled.context_window = 1000000
    profiled.cost_input = 1.25
    profiled.cost_output = 10.0
    profiled.available = True

    unprofiled = MagicMock()
    unprofiled.id = "gemini-4.0-ultra"
    unprofiled.provider = "google"
    unprofiled.context_window = 2000000
    unprofiled.cost_input = 0.0
    unprofiled.cost_output = 0.0
    unprofiled.available = True

    patcher, _ = _patch_registry(
        profiles={
            "gemini-3.1-pro-preview": profiled,
            "gemini-4.0-ultra": unprofiled,
        }
    )

    with patcher:
        results = await watcher.check_new_models()
        assert len(results) == 1
        assert results[0]["id"] == "gemini-4.0-ultra"
        assert results[0]["provider"] == "google"
        assert results[0]["context_window"] == 2000000
        assert "NEW" in results[0]["status"]


@pytest.mark.asyncio
async def test_check_new_models_all_profiled():
    """When all available models have cost data, returns empty list."""
    watcher = ModelWatcher()

    profiled = MagicMock()
    profiled.id = "gemini-3.1-pro-preview"
    profiled.provider = "google"
    profiled.context_window = 1000000
    profiled.cost_input = 1.25
    profiled.cost_output = 10.0
    profiled.available = True

    patcher, _ = _patch_registry(
        profiles={"gemini-3.1-pro-preview": profiled}
    )

    with patcher:
        results = await watcher.check_new_models()
        assert len(results) == 0


@pytest.mark.asyncio
async def test_report_no_new_models():
    """Report indicates no action needed when all models are profiled."""
    watcher = ModelWatcher()
    with patch.object(
        watcher, "check_new_models", new_callable=AsyncMock, return_value=[]
    ):
        report = await watcher.report()
        assert "No action needed" in report


@pytest.mark.asyncio
async def test_report_with_new_models():
    """Report lists unprofiled models and action item."""
    watcher = ModelWatcher()
    mock_models = [
        {
            "id": "new-model-1",
            "provider": "google",
            "context_window": 1000000,
            "status": "NEW",
        },
        {
            "id": "new-model-2",
            "provider": "openai",
            "context_window": None,
            "status": "NEW",
        },
    ]
    with patch.object(
        watcher,
        "check_new_models",
        new_callable=AsyncMock,
        return_value=mock_models,
    ):
        report = await watcher.report()
        assert "2 New Unprofiled Models" in report
        assert "new-model-1" in report
        assert "new-model-2" in report
        assert "model_profiles.toml" in report


@pytest.mark.asyncio
async def test_report_context_window_formatting():
    """Report formats context windows with commas and handles None."""
    watcher = ModelWatcher()
    mock_models = [
        {
            "id": "model-with-ctx",
            "provider": "google",
            "context_window": 2000000,
            "status": "NEW",
        },
        {
            "id": "model-no-ctx",
            "provider": "openai",
            "context_window": None,
            "status": "NEW",
        },
    ]
    with patch.object(
        watcher,
        "check_new_models",
        new_callable=AsyncMock,
        return_value=mock_models,
    ):
        report = await watcher.report()
        assert "2,000,000" in report
        assert "ctx=?" in report
