import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.llm.base import LLMConfig, Message, Role, LLMResponse

def test_llm_config_has_json_schema():
    cfg = LLMConfig(provider="google", model="gemini-3-flash-preview")
    assert hasattr(cfg, 'json_schema')
    assert cfg.json_schema is None  # default

def test_llm_config_with_schema():
    cfg = LLMConfig(provider="google", model="test", json_schema={"type": "object"})
    assert cfg.json_schema == {"type": "object"}

def test_model_router_tiers():
    from sage.llm.router import ModelRouter
    fast = ModelRouter.get_config("fast")
    assert "flash" in fast.model.lower() or "lite" in fast.model.lower()

    mutator = ModelRouter.get_config("mutator")
    assert "flash" in mutator.model.lower()
    assert mutator.provider == "google"

    reasoner = ModelRouter.get_config("reasoner")
    assert "pro" in reasoner.model.lower()

    codex = ModelRouter.get_config("codex")
    assert codex.provider == "codex"

    budget = ModelRouter.get_config("budget")
    assert "lite" in budget.model.lower()

def test_model_router_with_schema():
    from sage.llm.router import ModelRouter
    cfg = ModelRouter.get_config("mutator", json_schema={"type": "object"})
    assert cfg.json_schema == {"type": "object"}

def test_model_router_fallback_tier():
    from sage.llm.router import ModelRouter
    fb = ModelRouter.get_config("fallback")
    assert "2.5" in fb.model or "flash" in fb.model.lower()
    assert fb.provider == "google"

def test_model_router_critical_maps_to_reasoner():
    from sage.llm.router import ModelRouter
    critical = ModelRouter.get_config("critical")
    reasoner = ModelRouter.get_config("reasoner")
    assert critical.model == reasoner.model

def test_model_router_codex_max():
    from sage.llm.router import ModelRouter
    cfg = ModelRouter.get_config("codex_max")
    assert cfg.provider == "codex"
    assert "5.2" in cfg.model
    assert cfg.extra.get("reasoning_effort") == "xhigh"

def test_pydantic_model_as_schema():
    from pydantic import BaseModel
    class MutationOutput(BaseModel):
        search: str
        replace: str
        description: str
        features: list[int]

    cfg = LLMConfig(provider="google", model="test", json_schema=MutationOutput)
    assert cfg.json_schema is MutationOutput
    assert hasattr(cfg.json_schema, 'model_json_schema')

def test_codex_schema_additional_properties():
    """Verify _ensure_additional_properties_false works."""
    from sage.llm.codex import _ensure_additional_properties_false
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    fixed = _ensure_additional_properties_false(schema)
    assert fixed["additionalProperties"] is False
