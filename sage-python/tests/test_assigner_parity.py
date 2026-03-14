"""Parity test: Rust and Python ModelAssigner must produce identical assignments."""
from __future__ import annotations
import pytest

# Skip entire module if sage_core is not available
sage_core = pytest.importorskip("sage_core")


CARDS_TOML = '''
[[models]]
id = "cheap"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.5
tool_use_score = 0.5
math_score = 0.5
formal_z3_strength = 0.3
cost_input_per_m = 0.1
cost_output_per_m = 0.2
latency_ttft_ms = 100.0
tokens_per_sec = 200.0
s1_affinity = 0.9
s2_affinity = 0.3
s3_affinity = 0.1
recommended_topologies = ["sequential"]
supports_tools = false
supports_json_mode = false
supports_vision = false
context_window = 128000
[models.domain_scores]
code = 0.5
math = 0.4

[[models]]
id = "smart"
provider = "test"
family = "test"
code_score = 0.9
reasoning_score = 0.95
tool_use_score = 0.9
math_score = 0.9
formal_z3_strength = 0.8
cost_input_per_m = 5.0
cost_output_per_m = 15.0
latency_ttft_ms = 3000.0
tokens_per_sec = 50.0
s1_affinity = 0.1
s2_affinity = 0.9
s3_affinity = 0.95
recommended_topologies = ["avr"]
supports_tools = true
supports_json_mode = true
supports_vision = true
context_window = 1000000
[models.domain_scores]
code = 0.9
math = 0.95
'''


def _make_rust_pair():
    """Create Rust ModelRegistry + ModelAssigner."""
    from sage_core import ModelRegistry, ModelAssigner, TopologyGraph, TopologyNode, TopologyEdge
    registry = ModelRegistry.from_toml_str(CARDS_TOML)
    assigner = ModelAssigner(registry)
    return assigner, TopologyGraph, TopologyNode, TopologyEdge


def _make_python_pair():
    """Create Python ModelCardCatalog + ModelAssigner."""
    from sage.llm.model_registry import ModelCardCatalog
    from sage.llm.model_assigner import ModelAssigner
    catalog = ModelCardCatalog.from_toml_str(CARDS_TOML)
    assigner = ModelAssigner(catalog)
    return assigner, catalog


class MockNode:
    def __init__(self, role, model_id, system, required_capabilities, max_cost_usd=5.0):
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities
        self.max_cost_usd = max_cost_usd


class MockGraph:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def node_count(self):
        return len(self._nodes)

    def get_node(self, idx):
        return self._nodes[idx] if 0 <= idx < len(self._nodes) else None

    def set_node_model_id(self, idx, model_id):
        if 0 <= idx < len(self._nodes):
            self._nodes[idx].model_id = model_id
            return True
        return False


def test_parity_basic_assignment():
    """Rust and Python assign the same model_ids for a 2-node topology."""
    rust_assigner, TG, TN, TE = _make_rust_pair()

    # Rust graph
    rust_graph = TG("sequential")
    rust_graph.add_node(TN("coder", "", 2, ["tools"], 0, 5.0, 60.0))
    rust_graph.add_node(TN("reviewer", "", 3, [], 0, 5.0, 60.0))
    rust_graph.add_edge(0, 1, TE("control"))
    rust_n = rust_assigner.assign_models(rust_graph, "code", 10.0)
    rust_model_0 = rust_graph.get_node(0).model_id
    rust_model_1 = rust_graph.get_node(1).model_id

    # Python graph (mock)
    py_assigner, _ = _make_python_pair()
    py_graph = MockGraph([
        MockNode("coder", "", 2, ["tools"]),
        MockNode("reviewer", "", 3, []),
    ])
    py_n = py_assigner.assign_models(py_graph, "code", 10.0)
    py_model_0 = py_graph.get_node(0).model_id
    py_model_1 = py_graph.get_node(1).model_id

    assert rust_n == py_n, f"Rust assigned {rust_n}, Python assigned {py_n}"
    assert rust_model_0 == py_model_0, f"Node 0: Rust={rust_model_0}, Python={py_model_0}"
    assert rust_model_1 == py_model_1, f"Node 1: Rust={rust_model_1}, Python={py_model_1}"


def test_parity_tight_budget():
    """Both assigners handle budget smaller than cheapest model identically.

    cheap model costs ~0.0002 USD (0.1*1000 + 0.2*500) / 1e6.
    We pass 0.0001 total_budget → per_node_budget = 0.0001 < 0.0002 → no model fits.
    """
    rust_assigner, TG, TN, TE = _make_rust_pair()

    rust_graph = TG("sequential")
    rust_graph.add_node(TN("worker", "", 1, [], 0, 5.0, 60.0))
    # 0.0001 USD total budget < 0.0002 USD cost of cheapest model
    rust_n = rust_assigner.assign_models(rust_graph, "code", 0.0001)

    py_assigner, _ = _make_python_pair()
    py_graph = MockGraph([MockNode("worker", "", 1, [])])
    py_n = py_assigner.assign_models(py_graph, "code", 0.0001)

    assert rust_n == py_n == 0


def test_parity_missing_capabilities():
    """Both keep existing model_id when no candidate has required capabilities."""
    rust_assigner, TG, TN, TE = _make_rust_pair()

    rust_graph = TG("sequential")
    rust_graph.add_node(TN("special", "original", 2, ["tools", "json", "vision"], 0, 0.001, 60.0))
    rust_n = rust_assigner.assign_models(rust_graph, "code", 0.001)
    rust_model = rust_graph.get_node(0).model_id

    py_assigner, _ = _make_python_pair()
    py_graph = MockGraph([MockNode("special", "original", 2, ["tools", "json", "vision"], max_cost_usd=0.001)])
    py_n = py_assigner.assign_models(py_graph, "code", 0.001)
    py_model = py_graph.get_node(0).model_id

    assert rust_n == py_n == 0
    assert rust_model == py_model == "original"
