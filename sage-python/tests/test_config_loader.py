import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
import os
from pathlib import Path


def test_load_models_from_toml(tmp_path):
    from sage.llm.config_loader import load_model_config
    toml_content = b'[tiers]\nfast = "gemini-test-fast"\ncodex = "gpt-test-codex"\n\n[defaults]\ntemperature = 0.5\n'
    config_file = tmp_path / "models.toml"
    config_file.write_bytes(toml_content)
    config = load_model_config(config_file)
    assert config["tiers"]["fast"] == "gemini-test-fast"
    assert config["tiers"]["codex"] == "gpt-test-codex"
    assert config["defaults"]["temperature"] == 0.5


def test_load_models_returns_empty_on_missing():
    from sage.llm.config_loader import load_model_config
    config = load_model_config(Path("/nonexistent/models.toml"))
    assert config == {}


def test_env_override():
    import os
    from sage.llm.config_loader import resolve_model_id
    os.environ["SAGE_MODEL_FAST"] = "override-model"
    try:
        result = resolve_model_id("fast", toml_tiers={"fast": "toml-model"})
        assert result == "override-model"
    finally:
        del os.environ["SAGE_MODEL_FAST"]


def test_toml_fallback():
    import os
    from sage.llm.config_loader import resolve_model_id
    os.environ.pop("SAGE_MODEL_FAST", None)
    result = resolve_model_id("fast", toml_tiers={"fast": "toml-model"})
    assert result == "toml-model"


def test_hardcoded_fallback():
    import os
    from sage.llm.config_loader import resolve_model_id
    os.environ.pop("SAGE_MODEL_FAST", None)
    result = resolve_model_id("fast", toml_tiers={}, hardcoded="default-model")
    assert result == "default-model"
