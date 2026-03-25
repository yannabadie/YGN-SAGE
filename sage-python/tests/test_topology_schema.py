"""Tests for shared TopologySchema contract.

Validates that the single source of truth for topology YAML format
is consistent and used by all consumers.
"""
import pytest
from sage.verl.topology_schema import (
    TopologySchema,
    TopologyNodeSchema,
    TopologyEdgeSchema,
    AdaptationSchema,
    VALID_MODEL_TIERS,
    VALID_FLOW_TYPES,
)


class TestTopologySchemaFromYAML:
    def test_valid_topology(self):
        yaml_text = (
            "difficulty: moderate\n"
            "reasoning: test\n"
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: reasoner\n"
            "    provider_hint: deepseek\n"
            "  - role: reviewer\n"
            "    model_tier: fast\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "    flow_type: message\n"
        )
        schema = TopologySchema.from_yaml(yaml_text)
        assert schema is not None
        assert len(schema.nodes) == 2
        assert len(schema.edges) == 1
        assert schema.nodes[0].model_tier == "reasoner"
        assert schema.nodes[0].provider_hint == "deepseek"
        assert schema.edges[0].from_idx == 0
        assert schema.difficulty == "moderate"
        assert schema.reasoning == "test"

    def test_invalid_yaml(self):
        assert TopologySchema.from_yaml("{{not yaml") is None

    def test_no_nodes_key(self):
        assert TopologySchema.from_yaml("reasoning: just text") is None

    def test_with_adaptation(self):
        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "adaptation:\n"
            "  checkpoints: [0]\n"
            "  max_upgrades: 1\n"
        )
        schema = TopologySchema.from_yaml(yaml_text)
        assert schema is not None
        assert schema.has_checkpoints
        assert schema.adaptation.checkpoints == [0]
        assert schema.adaptation.max_upgrades == 1


class TestSchemaValidation:
    def test_valid_schema(self):
        schema = TopologySchema(
            difficulty="moderate",
            nodes=[TopologyNodeSchema(role="coder", model_tier="reasoner")],
            edges=[],
        )
        errors = schema.validate()
        assert errors == []

    def test_no_nodes_fails(self):
        schema = TopologySchema(nodes=[])
        errors = schema.validate()
        assert any("No nodes" in e for e in errors)

    def test_invalid_tier(self):
        schema = TopologySchema(
            nodes=[TopologyNodeSchema(role="coder", model_tier="quantum_computer")],
        )
        errors = schema.validate()
        assert any("invalid model_tier" in e for e in errors)

    def test_invalid_edge_indices(self):
        schema = TopologySchema(
            nodes=[TopologyNodeSchema(role="coder")],
            edges=[TopologyEdgeSchema(from_idx=0, to_idx=5)],
        )
        errors = schema.validate()
        assert any("invalid indices" in e for e in errors)

    def test_checkpoint_out_of_range(self):
        schema = TopologySchema(
            nodes=[TopologyNodeSchema(role="coder")],
            adaptation=AdaptationSchema(checkpoints=[99]),
        )
        errors = schema.validate()
        assert any("out of range" in e for e in errors)


class TestSchemaProperties:
    def test_tier_ratio_all_valid(self):
        schema = TopologySchema(
            nodes=[
                TopologyNodeSchema(role="coder", model_tier="reasoner"),
                TopologyNodeSchema(role="reviewer", model_tier="fast"),
            ],
        )
        assert schema.tier_ratio == 1.0

    def test_tier_ratio_mixed(self):
        schema = TopologySchema(
            nodes=[
                TopologyNodeSchema(role="coder", model_tier="reasoner"),
                TopologyNodeSchema(role="reviewer", model_tier="garbage"),
            ],
        )
        assert schema.tier_ratio == 0.5

    def test_has_provider_hints(self):
        schema = TopologySchema(
            nodes=[
                TopologyNodeSchema(role="coder", provider_hint="deepseek"),
                TopologyNodeSchema(role="reviewer"),
            ],
        )
        assert schema.has_provider_hints is True

    def test_no_provider_hints(self):
        schema = TopologySchema(
            nodes=[TopologyNodeSchema(role="coder")],
        )
        assert schema.has_provider_hints is False


class TestSchemaUsedByReward:
    """Verify reward.py uses the shared schema for scoring."""

    def test_reward_respects_schema_tiers(self):
        from sage.verl.reward import _score_structure

        # Valid tiers → higher score
        valid = "nodes:\n  - role: coder\n    model_tier: reasoner\nreasoning: test\n"
        invalid = "nodes:\n  - role: coder\n    model_tier: skynet\nreasoning: test\n"
        assert _score_structure(valid) > _score_structure(invalid)

    def test_reward_respects_provider_hint(self):
        from sage.verl.reward import _score_structure

        with_hint = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: reasoner\n"
            "    provider_hint: deepseek\n"
            "reasoning: test\n"
        )
        without_hint = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: reasoner\n"
            "reasoning: test\n"
        )
        assert _score_structure(with_hint) > _score_structure(without_hint)


class TestSchemaContractConsistency:
    """Verify the schema constants are consistent across consumers."""

    def test_valid_tiers_match_reward(self):
        """VALID_MODEL_TIERS in schema must match reward.py usage."""
        from sage.verl.topology_schema import VALID_MODEL_TIERS as schema_tiers
        # reward.py imports from topology_schema, so they ARE the same object
        from sage.verl.reward import VALID_MODEL_TIERS as reward_tiers
        assert schema_tiers is reward_tiers
