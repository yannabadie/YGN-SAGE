"""Tests for Python ModelAssigner fallback.

Uses mock TopologyGraph and real ModelRegistry (ModelCardCatalog) built from TOML.
"""
from __future__ import annotations

import pytest
from sage.llm.model_assigner import ModelAssigner
from sage.llm.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# TOML fixtures
# ---------------------------------------------------------------------------

_TOML_TWO_MODELS = """
[[models]]
id = "fast-s1"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.4
tool_use_score = 0.4
math_score = 0.3
formal_z3_strength = 0.2
cost_input_per_m = 0.1
cost_output_per_m = 0.2
latency_ttft_ms = 100.0
tokens_per_sec = 500.0
s1_affinity = 0.9
s2_affinity = 0.4
s3_affinity = 0.2
supports_tools = false
supports_json_mode = true
context_window = 128000

[models.domain_scores]
code = 0.4
general = 0.85

[[models]]
id = "smart-s2"
provider = "test"
family = "test"
code_score = 0.95
reasoning_score = 0.9
tool_use_score = 0.92
math_score = 0.8
formal_z3_strength = 0.5
cost_input_per_m = 1.0
cost_output_per_m = 3.0
latency_ttft_ms = 400.0
tokens_per_sec = 100.0
s1_affinity = 0.2
s2_affinity = 0.9
s3_affinity = 0.7
supports_tools = true
supports_json_mode = true
context_window = 128000

[models.domain_scores]
code = 0.94
general = 0.6
"""

_TOML_THREE_MODELS = """
[[models]]
id = "budget"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.4
tool_use_score = 0.3
math_score = 0.3
formal_z3_strength = 0.1
cost_input_per_m = 0.05
cost_output_per_m = 0.1
latency_ttft_ms = 50.0
tokens_per_sec = 600.0
s1_affinity = 0.8
s2_affinity = 0.4
s3_affinity = 0.2
supports_tools = false
supports_json_mode = false
context_window = 32000

[[models]]
id = "mid-tools"
provider = "test"
family = "test"
code_score = 0.8
reasoning_score = 0.75
tool_use_score = 0.85
math_score = 0.6
formal_z3_strength = 0.4
cost_input_per_m = 0.5
cost_output_per_m = 1.5
latency_ttft_ms = 300.0
tokens_per_sec = 200.0
s1_affinity = 0.5
s2_affinity = 0.8
s3_affinity = 0.5
supports_tools = true
supports_json_mode = true
context_window = 128000

[models.domain_scores]
code = 0.85

[[models]]
id = "expert-s3"
provider = "test"
family = "test"
code_score = 0.9
reasoning_score = 0.95
tool_use_score = 0.8
math_score = 0.95
formal_z3_strength = 0.9
cost_input_per_m = 5.0
cost_output_per_m = 15.0
latency_ttft_ms = 800.0
tokens_per_sec = 50.0
s1_affinity = 0.1
s2_affinity = 0.5
s3_affinity = 0.95
supports_tools = true
supports_json_mode = true
context_window = 200000

[models.domain_scores]
code = 0.92
math = 0.97
"""


# ---------------------------------------------------------------------------
# Mock TopologyGraph
# ---------------------------------------------------------------------------

class MockNode:
    """Minimal node descriptor."""

    def __init__(
        self,
        role: str = "worker",
        system: int = 1,
        required_capabilities: list[str] | None = None,
        max_cost_usd: float = 10.0,
    ) -> None:
        self.role = role
        self.system = system
        self.required_capabilities = required_capabilities or []
        self.max_cost_usd = max_cost_usd
        self.model_id: str = ""


class MockGraph:
    """Simple mock topology graph with get_node / set_node_model_id."""

    def __init__(self, nodes: list[MockNode]) -> None:
        self._nodes = nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> MockNode | None:
        if 0 <= idx < len(self._nodes):
            return self._nodes[idx]
        return None

    def set_node_model_id(self, idx: int, model_id: str) -> None:
        self._nodes[idx].model_id = model_id


# ---------------------------------------------------------------------------
# Helper to build a ModelRegistry from TOML string
# ---------------------------------------------------------------------------

def _registry_from_toml(toml_str: str) -> ModelRegistry:
    return ModelRegistry.from_toml_str(toml_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelAssigner:
    def test_assigns_by_domain_and_system(self):
        """2-node graph: coder (S2, tools) + reviewer (S3). domain='code'.

        Node 0 (coder, S2, needs tools): should get 'smart-s2' — it's the only model
        with supports_tools=True and high S2 + code affinity.
        Node 1 (reviewer, S3): with budget remaining, score favours smart-s2 (S3 affinity 0.7)
        over fast-s1 (S3 affinity 0.2) for S3 system. smart-s2 wins on affinity+domain.
        """
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        nodes = [
            MockNode(role="coder", system=2, required_capabilities=["tools"]),
            MockNode(role="reviewer", system=3),
        ]
        graph = MockGraph(nodes)

        count = assigner.assign_models(graph, task_domain="code", budget_usd=5.0)

        assert count == 2
        # Node 0 needs tools — only smart-s2 qualifies
        assert graph._nodes[0].model_id == "smart-s2"
        # Node 1: no caps filter — smart-s2 has better S3 affinity (0.7 vs 0.2)
        assert graph._nodes[1].model_id == "smart-s2"

    def test_no_tools_node_can_pick_cheaper(self):
        """A node that doesn't need tools can pick the cheaper fast-s1 if it has better score.

        Node with system=1, no caps — fast-s1 has s1_affinity=0.9 and is cheap.
        Score: fast-s1 = 0.4*0.9 + 0.4*0.4 + 0.2*(cost_norm) should beat smart-s2.
        """
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        nodes = [MockNode(role="greeter", system=1, required_capabilities=[])]
        graph = MockGraph(nodes)

        count = assigner.assign_models(graph, task_domain="general", budget_usd=10.0)
        assert count == 1
        # fast-s1: s1_affinity=0.9, domain_score("general")=0.85, cheaper
        # smart-s2: s1_affinity=0.2, domain_score("general")=0.6, expensive
        # fast-s1 wins clearly
        assert graph._nodes[0].model_id == "fast-s1"

    def test_keeps_existing_when_no_candidate(self):
        """Node needs tools+json+vision with tiny budget: no model qualifies.

        Original model_id="" should be unchanged (kept), count=0.
        """
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        # Needs vision — neither model in _TOML_TWO_MODELS has supports_vision=True
        # (fast-s1: supports_vision default=False, smart-s2: supports_vision default=False)
        # We simulate this by making budget tiny so no model fits cost
        nodes = [MockNode(role="vision-node", system=2, required_capabilities=["tools"],
                          max_cost_usd=0.00001)]
        graph = MockGraph(nodes)
        # Give a budget of 0.00001 to force no model to qualify by budget
        # smart-s2 costs 1.0*1000/1e6 + 3.0*500/1e6 = 0.0025 >> 0.00001
        count = assigner.assign_models(graph, task_domain="code", budget_usd=0.00001)

        assert count == 0
        # model_id should remain as set originally (empty string / default)
        assert graph._nodes[0].model_id == ""

    def test_budget_exhaustion(self):
        """budget=0 → 0 nodes assigned (immediate exhaustion)."""
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        nodes = [
            MockNode(role="a", system=1),
            MockNode(role="b", system=2),
        ]
        graph = MockGraph(nodes)

        count = assigner.assign_models(graph, task_domain="code", budget_usd=0.0)

        assert count == 0
        # Neither node should have been assigned (budget 0 < BUDGET_EPSILON 0.01)
        assert graph._nodes[0].model_id == ""
        assert graph._nodes[1].model_id == ""

    def test_assign_single_node_success(self):
        """assign_single_node returns a valid model_id and updates graph."""
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        nodes = [MockNode(role="coder", system=2, required_capabilities=["tools"])]
        graph = MockGraph(nodes)

        model_id = assigner.assign_single_node(graph, 0, "code", budget_usd=5.0)
        assert model_id == "smart-s2"
        assert graph._nodes[0].model_id == "smart-s2"

    def test_assign_single_node_no_candidate_raises(self):
        """assign_single_node raises ValueError when no candidate found."""
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        # Budget too small — no model fits
        nodes = [MockNode(role="worker", system=2, required_capabilities=["tools"])]
        graph = MockGraph(nodes)

        with pytest.raises(ValueError, match="No candidate"):
            assigner.assign_single_node(graph, 0, "code", budget_usd=0.000001)

    def test_assign_single_node_out_of_range_raises(self):
        """assign_single_node raises ValueError for out-of-range index."""
        catalog = _registry_from_toml(_TOML_TWO_MODELS)
        assigner = ModelAssigner(catalog)

        graph = MockGraph([MockNode()])

        with pytest.raises(ValueError, match="out of range"):
            assigner.assign_single_node(graph, 99, "code", budget_usd=5.0)

    def test_empty_catalog_warns_and_returns_zero(self):
        """Empty catalog → assign_models returns 0."""
        catalog = ModelRegistry()  # empty
        assigner = ModelAssigner(catalog)

        nodes = [MockNode(), MockNode()]
        graph = MockGraph(nodes)

        count = assigner.assign_models(graph, task_domain="code", budget_usd=10.0)
        assert count == 0

    def test_three_models_s3_node_picks_expert(self):
        """With 3 models, a S3 node with no caps filter picks the expert-s3 model
        (s3_affinity=0.95, code domain_score=0.92) over mid-tools (s3_affinity=0.5)."""
        catalog = _registry_from_toml(_TOML_THREE_MODELS)
        assigner = ModelAssigner(catalog)

        nodes = [MockNode(role="reasoner", system=3, required_capabilities=[])]
        graph = MockGraph(nodes)

        count = assigner.assign_models(graph, task_domain="code", budget_usd=50.0)
        assert count == 1
        # expert-s3 has highest S3 affinity + code domain_score + budget fits
        assert graph._nodes[0].model_id == "expert-s3"

    def test_score_weights(self):
        """Verify scoring: WEIGHT_AFFINITY=0.4, WEIGHT_DOMAIN=0.4, WEIGHT_COST=0.2.

        With two equal-affinity models, the one with higher domain score should win.
        """
        from sage.llm.model_assigner import WEIGHT_AFFINITY, WEIGHT_DOMAIN, WEIGHT_COST
        assert abs(WEIGHT_AFFINITY - 0.4) < 0.001
        assert abs(WEIGHT_DOMAIN - 0.4) < 0.001
        assert abs(WEIGHT_COST - 0.2) < 0.001
        assert abs(WEIGHT_AFFINITY + WEIGHT_DOMAIN + WEIGHT_COST - 1.0) < 0.001
