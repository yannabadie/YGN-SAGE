import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.llm.base import LLMConfig

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
    assert "flash" in fast.model.lower() or "lite" in fast.model.lower() or "nano" in fast.model.lower()

    mutator = ModelRouter.get_config("mutator")
    assert mutator.model  # mutator model is configured (may be gpt-5.4-mini or flash)

    reasoner = ModelRouter.get_config("reasoner")
    assert "pro" in reasoner.model.lower() or "reasoning" in reasoner.model.lower()

    codex = ModelRouter.get_config("codex")
    assert codex.provider == "openai"  # codex tier uses gpt-5.4 via OpenAI API

    budget = ModelRouter.get_config("budget")
    assert budget.model  # budget model is configured (may be deepseek-chat or lite)
    if budget.provider == "deepseek" and budget.model == "deepseek-v4-flash":
        assert budget.extra["thinking"] == "disabled"

def test_model_router_with_schema():
    from sage.llm.router import ModelRouter
    cfg = ModelRouter.get_config("mutator", json_schema={"type": "object"})
    assert cfg.json_schema == {"type": "object"}

def test_model_router_fallback_tier():
    from sage.llm.router import ModelRouter
    fb = ModelRouter.get_config("fallback")
    assert fb.model  # fallback model is configured (may be deepseek-chat, flash, etc.)
    if fb.provider == "deepseek" and fb.model == "deepseek-v4-flash":
        assert fb.extra["thinking"] == "disabled"

def test_model_router_critical_maps_to_reasoner():
    from sage.llm.router import ModelRouter
    critical = ModelRouter.get_config("critical")
    reasoner = ModelRouter.get_config("reasoner")
    assert critical.model == reasoner.model

def test_model_router_codex_max():
    from sage.llm.router import ModelRouter
    cfg = ModelRouter.get_config("codex_max")
    assert cfg.provider == "openai"  # codex_max tier uses gpt-5.4-pro via OpenAI API
    assert "pro" in cfg.model.lower() or "5." in cfg.model  # gpt-5.x-pro or similar
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


def test_codex_schema_additional_properties_nested():
    """Nested object schemas should also be patched recursively."""
    from sage.llm.codex import _ensure_additional_properties_false

    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "prefs": {
                        "type": "object",
                        "properties": {"theme": {"type": "string"}},
                    },
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            },
        },
    }

    fixed = _ensure_additional_properties_false(schema)
    assert fixed["additionalProperties"] is False
    assert fixed["properties"]["user"]["additionalProperties"] is False
    assert fixed["properties"]["user"]["properties"]["prefs"]["additionalProperties"] is False
    assert fixed["properties"]["items"]["items"]["additionalProperties"] is False


def test_codex_extract_text_from_jsonl_v1_and_v2():
    """Parser should read both legacy and evolved Codex JSONL message formats."""
    from sage.llm.codex import _extract_text_from_jsonl

    v1 = '\n'.join([
        '{"type":"turn.completed","usage":{}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"legacy text"}}',
    ])
    assert _extract_text_from_jsonl(v1) == "legacy text"

    v2 = '\n'.join([
        '{"type":"turn.completed","usage":{}}',
        '{"type":"item.completed","item":{"type":"agent_message","content":[{"type":"output_text","text":"new "},{"type":"output_text","text":"format"}]}}',
    ])
    assert _extract_text_from_jsonl(v2) == "new format"
