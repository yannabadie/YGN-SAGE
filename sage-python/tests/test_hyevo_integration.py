"""Tests for HyEvo integration: code nodes, cascaded eval, reflection, behavior descriptors.

Reference: HyEvo (arXiv 2603.19639) — Self-Evolving Hybrid Agentic Workflows.
"""
import pytest
from sage.verl.topology_schema import TopologySchema, TopologyNodeSchema, VALID_NODE_TYPES


# -- Code nodes --

YAML_HYBRID = (
    "difficulty: moderate\n"
    "reasoning: Hybrid LLM + code topology\n"
    "nodes:\n"
    "  - role: planner\n"
    "    node_type: llm\n"
    "    model_tier: reasoner\n"
    "    prompt: Plan the solution approach\n"
    "  - role: validator\n"
    "    node_type: code\n"
    "    code_spec: |\n"
    "      def validate(input_text):\n"
    "          import json\n"
    "          try:\n"
    "              data = json.loads(input_text)\n"
    "              return 'valid' if 'result' in data else 'invalid'\n"
    "          except:\n"
    "              return 'parse_error'\n"
    "    io_signature: 'str -> str'\n"
    "    deterministic: true\n"
    "  - role: synthesizer\n"
    "    node_type: llm\n"
    "    model_tier: fast\n"
    "    prompt: Produce final answer\n"
    "edges:\n"
    "  - from_idx: 0\n"
    "    to_idx: 1\n"
    "  - from_idx: 1\n"
    "    to_idx: 2\n"
)


class TestCodeNodes:
    def test_code_node_parsed(self):
        schema = TopologySchema.from_yaml(YAML_HYBRID)
        assert schema is not None
        assert len(schema.nodes) == 3
        assert schema.nodes[1].is_code_node
        assert schema.nodes[1].node_type == "code"
        assert "def validate" in schema.nodes[1].code_spec

    def test_llm_nodes_identified(self):
        schema = TopologySchema.from_yaml(YAML_HYBRID)
        assert schema.nodes[0].is_llm_node
        assert schema.nodes[2].is_llm_node

    def test_valid_node_types(self):
        assert "llm" in VALID_NODE_TYPES
        assert "code" in VALID_NODE_TYPES

    def test_code_node_validation_requires_code_spec(self):
        schema = TopologySchema(nodes=[
            TopologyNodeSchema(role="bad_code", node_type="code"),  # no code_spec
        ])
        errors = schema.validate()
        assert any("code_spec" in e for e in errors)

    def test_default_node_type_is_llm(self):
        yaml_no_type = "nodes:\n  - role: coder\n"
        schema = TopologySchema.from_yaml(yaml_no_type)
        assert schema.nodes[0].node_type == "llm"


class TestBehaviorDescriptors:
    def test_hybrid_ratios(self):
        schema = TopologySchema.from_yaml(YAML_HYBRID)
        assert schema.llm_ratio == pytest.approx(2/3)
        assert schema.code_ratio == pytest.approx(1/3)
        assert schema.has_code_nodes

    def test_all_llm_ratio(self):
        yaml_llm = "nodes:\n  - role: a\n  - role: b\n"
        schema = TopologySchema.from_yaml(yaml_llm)
        assert schema.llm_ratio == 1.0
        assert schema.code_ratio == 0.0

    def test_behavior_descriptor_tuple(self):
        schema = TopologySchema.from_yaml(YAML_HYBRID)
        bd = schema.behavior_descriptor()
        assert len(bd) == 4
        node_count, llm_r, code_r, prov_div = bd
        assert node_count == 3
        assert llm_r == pytest.approx(2/3)
        assert code_r == pytest.approx(1/3)


class TestCascadedEvaluation:
    def test_stage_1_valid_yaml(self):
        from sage.verl.cascaded_eval import stage_1_schema
        r = stage_1_schema(YAML_HYBRID)
        assert r.passed
        assert r.stage_name == "schema"

    def test_stage_1_invalid_yaml(self):
        from sage.verl.cascaded_eval import stage_1_schema
        r = stage_1_schema("{{not yaml")
        assert not r.passed
        assert r.stage_reached == 1

    def test_stage_2_safe_code(self):
        from sage.verl.cascaded_eval import stage_2_security
        r = stage_2_security(YAML_HYBRID)
        assert r.passed

    def test_stage_2_dangerous_code(self):
        from sage.verl.cascaded_eval import stage_2_security
        dangerous = (
            "nodes:\n"
            "  - role: evil\n"
            "    node_type: code\n"
            "    code_spec: 'os.system(\"rm -rf /\")'\n"
        )
        r = stage_2_security(dangerous)
        assert not r.passed
        assert any("os.system" in e for e in r.errors)

    def test_stage_3_smoke(self):
        from sage.verl.cascaded_eval import stage_3_smoke
        valid = "nodes:\n  - role: coder\n    model_tier: fast\nreasoning: test\n"
        r = stage_3_smoke(valid)
        assert r.passed
        assert r.score > 0

    def test_cascaded_full_pipeline(self):
        from sage.verl.cascaded_eval import cascaded_evaluate
        r = cascaded_evaluate(YAML_HYBRID, gamma=0.05)
        assert r.stage_reached >= 3
        assert r.score > 0

    def test_cascaded_stops_at_invalid_schema(self):
        from sage.verl.cascaded_eval import cascaded_evaluate
        r = cascaded_evaluate("not yaml at all")
        assert r.stage_reached == 1
        assert not r.passed


class TestReflection:
    def test_diagnose_empty_traces(self):
        from sage.verl.reflection import diagnose
        diag = diagnose(
            parent_yaml=YAML_HYBRID,
            parent_score=0.5,
            parent_traces=[],
        )
        assert diag.parent_score == 0.5
        assert isinstance(diag.recommendations, list)

    def test_diagnose_with_failures(self):
        from sage.verl.reflection import diagnose
        traces = [
            {"node_idx": 0, "status": "TIMEOUT", "cost": 0.1, "latency": 25000},
            {"node_idx": 1, "status": "OK", "cost": 0.0, "latency": 50},
        ]
        diag = diagnose(
            parent_yaml=YAML_HYBRID,
            parent_score=0.3,
            parent_traces=traces,
            budget=0.2,
        )
        assert "TIMEOUT" in str(diag.failure_types)
        assert diag.cost_ratio > 0.4
        assert any("Timeout" in r or "code node" in r.lower() for r in diag.recommendations)

    def test_format_reflection_prompt(self):
        from sage.verl.reflection import diagnose, format_reflection_prompt
        diag = diagnose(
            parent_yaml=YAML_HYBRID,
            parent_score=0.4,
            parent_traces=[{"node_idx": 0, "status": "ERROR", "cost": 0.0, "latency": 0}],
            top_score=0.9,
        )
        prompt = format_reflection_prompt(diag, YAML_HYBRID, "Write fibonacci")
        assert "fibonacci" in prompt.lower()
        assert "0.400" in prompt or "0.4" in prompt
        assert "0.9" in prompt

    def test_diagnose_recommends_code_nodes(self):
        """If top exemplar has code nodes but parent doesn't, recommend adding them."""
        from sage.verl.reflection import diagnose
        parent_llm_only = "nodes:\n  - role: a\n  - role: b\n  - role: c\n"
        diag = diagnose(
            parent_yaml=parent_llm_only,
            parent_score=0.4,
            parent_traces=[],
            top_yaml=YAML_HYBRID,
            top_score=0.8,
        )
        assert any("code node" in r.lower() for r in diag.recommendations)


class TestRewardHybridBonus:
    def test_hybrid_topology_scores_higher(self):
        from sage.verl.reward import _score_structure

        hybrid = YAML_HYBRID
        llm_only = (
            "difficulty: moderate\n"
            "reasoning: LLM only topology\n"
            "nodes:\n"
            "  - role: planner\n"
            "    model_tier: reasoner\n"
            "  - role: coder\n"
            "    model_tier: fast\n"
            "  - role: synthesizer\n"
            "    model_tier: fast\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "  - from_idx: 1\n"
            "    to_idx: 2\n"
        )

        score_hybrid = _score_structure(hybrid)
        score_llm = _score_structure(llm_only)
        assert score_hybrid > score_llm, (
            f"Hybrid ({score_hybrid}) should score higher than LLM-only ({score_llm})"
        )
