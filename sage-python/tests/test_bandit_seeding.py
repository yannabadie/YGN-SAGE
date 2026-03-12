"""Test that bandit arms are seeded from registry, not hardcoded."""
import pytest
from unittest.mock import MagicMock


def test_bandit_seeded_from_registry():
    """Bandit should be seeded from ModelRegistry discovered models."""
    # The hardcoded bandit seeding list should not appear in boot.py's AgentSystem.run
    import inspect
    import sage.boot as boot_module
    source = inspect.getsource(boot_module.AgentSystem.run)

    # Check that the hardcoded 4-model bandit seeding list is gone.
    # The old code was:
    #   for model_id in ["gemini-2.5-flash", "gemini-3-flash-preview",
    #                    "gemini-3.1-pro-preview", "gemini-2.5-flash-lite"]:
    #       self.bandit.register_arm(model_id, template_type)
    assert '"gemini-3.1-pro-preview", "gemini-2.5-flash-lite"' not in source, (
        "Hardcoded Gemini model list should be replaced with registry-based seeding"
    )
    # Verify the registry-based seeding is in place
    assert "_rust_registry" in source, (
        "Registry-based seeding via _rust_registry should be present"
    )
    assert "list_ids()" in source, (
        "Rust ModelRegistry.list_ids() should be used for arm seeding"
    )
    assert "list_available()" in source, (
        "Python registry fallback via list_available() should be present"
    )
