import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
import os


def test_router_loads_default_models():
    from sage.llm.router import ModelRouter
    assert "fast" in ModelRouter.MODELS
    assert "codex" in ModelRouter.MODELS


def test_router_get_config_returns_llmconfig():
    from sage.llm.router import ModelRouter
    config = ModelRouter.get_config("fast")
    assert config.model == ModelRouter.MODELS["fast"]
    assert config.provider == "google"


def test_router_env_override():
    from sage.llm.router import ModelRouter
    original = ModelRouter.MODELS.get("fast")
    os.environ["SAGE_MODEL_FAST"] = "test-override-model"
    try:
        ModelRouter._load_config()
        assert ModelRouter.MODELS["fast"] == "test-override-model"
        config = ModelRouter.get_config("fast")
        assert config.model == "test-override-model"
    finally:
        os.environ.pop("SAGE_MODEL_FAST", None)
        ModelRouter.MODELS["fast"] = original


def test_router_codex_provider():
    from sage.llm.router import ModelRouter
    config = ModelRouter.get_config("codex")
    assert config.provider == "codex"


def test_router_critical_is_reasoner():
    from sage.llm.router import ModelRouter
    config = ModelRouter.get_config("critical")
    assert config.model == ModelRouter.MODELS["reasoner"]
