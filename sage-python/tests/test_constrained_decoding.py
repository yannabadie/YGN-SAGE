"""Tests for constrained decoding (JSON schema / structured output)."""
from __future__ import annotations

import asyncio
import sys
import pytest

from sage.llm.base import LLMConfig, LLMResponse, Message, Role


# ---------------------------------------------------------------------------
# Unit: LLMConfig carries json_schema
# ---------------------------------------------------------------------------

def test_llm_config_has_response_schema():
    """LLMConfig.json_schema stores an arbitrary JSON schema dict."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    config = LLMConfig(provider="google", model="gemini-2.5-flash", json_schema=schema)
    assert config.json_schema == schema
    assert "properties" in config.json_schema


def test_llm_config_json_schema_default_none():
    """json_schema is None by default — no forced structured output."""
    config = LLMConfig(provider="google", model="gemini-2.5-flash")
    assert config.json_schema is None


def test_llm_config_json_schema_pydantic_class():
    """json_schema accepts a Pydantic model class (duck-typing)."""
    class FakePydantic:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {"x": {"type": "integer"}}}

    config = LLMConfig(provider="google", model="gemini-2.5-flash", json_schema=FakePydantic)
    assert config.json_schema is FakePydantic  # stored as-is, expanded at generate() time


# ---------------------------------------------------------------------------
# Unit: GoogleProvider wires json_schema into GenerateContentConfig
# ---------------------------------------------------------------------------

def test_google_provider_respects_json_schema_in_source():
    """GoogleProvider.generate() sets response_mime_type and response_schema when json_schema is set.

    This test verifies the source code contains the expected wiring rather than
    executing the real API (which requires a valid key).
    """
    import inspect
    from sage.llm import google as google_module
    src = inspect.getsource(google_module.GoogleProvider.generate)
    assert "response_mime_type" in src, "generate() must set response_mime_type"
    assert "response_schema" in src, "generate() must set response_schema"
    assert "json_schema" in src, "generate() must read config.json_schema"
    assert "application/json" in src, "generate() must use application/json MIME type"


# ---------------------------------------------------------------------------
# Unit: OpenAICompatProvider wires json_schema into response_format
# ---------------------------------------------------------------------------

def test_openai_compat_respects_json_schema_in_source():
    """OpenAICompatProvider.generate() passes response_format when json_schema is set.

    Source-level check — avoids requiring an OpenAI API key in CI.
    """
    import inspect
    from sage.providers import openai_compat as oai_module
    src = inspect.getsource(oai_module.OpenAICompatProvider.generate)
    assert "response_format" in src, "generate() must set response_format"
    assert "json_schema" in src, "generate() must read config.json_schema"
    assert "json_schema" in src, "generate() must use json_schema response_format type"


def test_openai_compat_sets_response_format():
    """OpenAICompatProvider injects correct response_format dict when json_schema is present."""
    captured_params: dict = {}

    class FakeCompletions:
        @staticmethod
        async def create(**kwargs):
            captured_params.update(kwargs)

            class FakeMsg:
                content = '{"result": 42}'
                reasoning_content = None

            class FakeChoice:
                message = FakeMsg()

            class FakeResp:
                choices = [FakeChoice()]

            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kw):
            pass
        chat = FakeChat()

    # Save original module state
    orig_openai = sys.modules.get("openai")

    fake_openai = type(sys)("openai")
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    sys.modules["openai"] = fake_openai  # type: ignore[assignment]

    orig_compat = sys.modules.pop("sage.providers.openai_compat", None)

    try:
        from sage.providers.openai_compat import OpenAICompatProvider

        schema = {"type": "object", "properties": {"result": {"type": "integer"}}}
        config = LLMConfig(provider="openai", model="gpt-4", json_schema=schema)
        provider = OpenAICompatProvider(api_key="test-key", model_id="gpt-4")

        asyncio.run(provider.generate([], config=config))

        assert "response_format" in captured_params, (
            "response_format not set in OpenAI API call"
        )
        rf = captured_params["response_format"]
        assert rf["type"] == "json_schema", (
            f"Expected type='json_schema', got {rf['type']!r}"
        )
        assert rf["json_schema"]["schema"] == schema, (
            f"Schema mismatch: {rf['json_schema']['schema']!r} != {schema!r}"
        )
    finally:
        if orig_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = orig_openai
        sys.modules.pop("sage.providers.openai_compat", None)
        if orig_compat is not None:
            sys.modules["sage.providers.openai_compat"] = orig_compat


def test_openai_compat_no_response_format_without_schema():
    """OpenAICompatProvider must NOT inject response_format when json_schema is None."""
    captured_params: dict = {}

    class FakeCompletions:
        @staticmethod
        async def create(**kwargs):
            captured_params.update(kwargs)

            class FakeMsg:
                content = "hello"
                reasoning_content = None

            class FakeChoice:
                message = FakeMsg()

            class FakeResp:
                choices = [FakeChoice()]

            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kw):
            pass
        chat = FakeChat()

    orig_openai = sys.modules.get("openai")
    fake_openai = type(sys)("openai")
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    sys.modules["openai"] = fake_openai  # type: ignore[assignment]

    orig_compat = sys.modules.pop("sage.providers.openai_compat", None)

    try:
        from sage.providers.openai_compat import OpenAICompatProvider

        config = LLMConfig(provider="openai", model="gpt-4")  # no json_schema
        provider = OpenAICompatProvider(api_key="test-key", model_id="gpt-4")

        asyncio.run(provider.generate([], config=config))

        assert "response_format" not in captured_params, (
            "response_format must NOT be set when json_schema is None"
        )
    finally:
        if orig_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = orig_openai
        sys.modules.pop("sage.providers.openai_compat", None)
        if orig_compat is not None:
            sys.modules["sage.providers.openai_compat"] = orig_compat


# ---------------------------------------------------------------------------
# Unit: llm_caller synthesize_topology uses json_schema on both stages
# ---------------------------------------------------------------------------

def test_llm_caller_prompt_helpers():
    """build_role_prompt and build_structure_prompt produce expected JSON structure."""
    from sage.topology.llm_caller import build_role_prompt, build_structure_prompt

    role_p = build_role_prompt("design a classifier", max_agents=2)
    assert "roles" in role_p
    assert "2" in role_p  # max_agents reflected

    struct_p = build_structure_prompt(
        '{"roles": [{"name": "a", "model": "m", "system": 1, "capabilities": []}]}'
    )
    assert "adjacency" in struct_p
    assert "edge_types" in struct_p
    assert "template" in struct_p


def test_llm_caller_config_carries_schema():
    """synthesize_topology passes LLMConfig with json_schema on both Stage 1 and Stage 2 calls."""
    configs_used: list[LLMConfig] = []

    call_count = 0

    async def fake_generate(self_or_messages, messages=None, config=None, **kwargs):
        """Accept both bound (self, messages=...) and unbound (messages, ...) call styles."""
        nonlocal call_count
        if config is not None:
            configs_used.append(config)
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=(
                    '{"roles": [{"name": "planner", "model": "gemini-2.5-flash",'
                    ' "system": 2, "capabilities": ["plan"]}]}'
                ),
                model="gemini-2.5-flash",
            )
        return LLMResponse(
            content='{"adjacency": [[0]], "edge_types": [[""]], "template": "sequential"}',
            model="gemini-2.5-flash",
        )

    class FakeProvider:
        generate = fake_generate

    # Patch sage_core so parse_and_build_topology returns a stub
    orig_sage_core = sys.modules.get("sage_core")
    fake_sage_core = type(sys)("sage_core")

    class FakeGraph:
        def node_count(self): return 1
        def edge_count(self): return 0
        def add_node(self, n): pass
        def add_edge(self, i, j, e): pass

    fake_sage_core.TopologyGraph = lambda t: FakeGraph()
    fake_sage_core.TopologyNode = lambda *a: object()
    fake_sage_core.TopologyEdge = lambda *a: object()
    sys.modules["sage_core"] = fake_sage_core  # type: ignore[assignment]

    orig_caller = sys.modules.pop("sage.topology.llm_caller", None)

    try:
        from sage.topology.llm_caller import synthesize_topology

        asyncio.run(synthesize_topology(FakeProvider(), task="build a classifier"))

        assert len(configs_used) == 2, (
            f"Expected 2 LLM calls with config, got {len(configs_used)}"
        )
        stage1_config, stage2_config = configs_used

        # Stage 1 must carry a roles schema
        assert stage1_config.json_schema is not None, "Stage 1 config missing json_schema"
        s1_props = stage1_config.json_schema.get("properties", {})
        assert "roles" in s1_props, (
            f"Stage 1 schema must have 'roles' property, got: {s1_props}"
        )

        # Stage 2 must carry a structure schema
        assert stage2_config.json_schema is not None, "Stage 2 config missing json_schema"
        s2_props = stage2_config.json_schema.get("properties", {})
        assert "adjacency" in s2_props, (
            f"Stage 2 schema must have 'adjacency' property, got: {s2_props}"
        )
        assert "template" in s2_props, (
            f"Stage 2 schema must have 'template' property, got: {s2_props}"
        )
    finally:
        if orig_sage_core is None:
            sys.modules.pop("sage_core", None)
        else:
            sys.modules["sage_core"] = orig_sage_core
        sys.modules.pop("sage.topology.llm_caller", None)
        if orig_caller is not None:
            sys.modules["sage.topology.llm_caller"] = orig_caller


# ---------------------------------------------------------------------------
# Unit: planner.py is purely Python — no LLM call expected
# ---------------------------------------------------------------------------

def test_planner_no_llm_call():
    """TaskPlanner.plan_static() is pure Python — no LLM dependency."""
    from sage.contracts.planner import TaskPlanner

    steps = [
        {"id": "a", "description": "step a"},
        {"id": "b", "description": "step b", "depends_on": ["a"]},
    ]
    planner = TaskPlanner()
    result = planner.plan_static(steps)
    assert result.node_count == 2
    assert result.edge_count == 1
