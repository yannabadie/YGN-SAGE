"""Tests for veRL reward function (sage.verl.reward)."""
import pytest


class TestScoreFormat:
    def test_valid_yaml_with_nodes(self):
        from sage.verl.reward import _score_format
        assert _score_format("nodes:\n- role: coder\n  prompt: Write code\ndifficulty: simple") == 1.0

    def test_invalid_yaml(self):
        from sage.verl.reward import _score_format
        assert _score_format("not: [valid: yaml: {{") == -2.0

    def test_valid_yaml_no_nodes(self):
        from sage.verl.reward import _score_format
        assert _score_format("reasoning: just text") == -0.5

    def test_valid_yaml_not_dict(self):
        from sage.verl.reward import _score_format
        assert _score_format("- item1\n- item2") == -1.5

    def test_empty_nodes_list(self):
        from sage.verl.reward import _score_format
        assert _score_format("nodes: []") == -0.25


class TestScoreStructure:
    def test_complete_topology(self):
        from sage.verl.reward import _score_structure
        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "    prompt: Write code\n"
            "  - role: reviewer\n"
            "    prompt: Review code\n"
            "edges:\n"
            "  - from_idx: 0\n"
            "    to_idx: 1\n"
            "    flow_type: message\n"
            "reasoning: Need coder then reviewer\n"
            "difficulty: moderate\n"
        )
        score = _score_structure(yaml_text)
        assert score == pytest.approx(1.0)  # nodes(0.3) + edges(0.2) + roles(0.3) + reasoning(0.2)

    def test_minimal_one_node(self):
        from sage.verl.reward import _score_structure
        score = _score_structure("nodes:\n- role: coder")
        assert score == pytest.approx(0.6)  # nodes(0.3) + roles(0.3)

    def test_no_roles(self):
        from sage.verl.reward import _score_structure
        score = _score_structure("nodes:\n- name: foo")
        assert score == pytest.approx(0.3)  # nodes only

    def test_invalid_yaml(self):
        from sage.verl.reward import _score_structure
        assert _score_structure("{{broken") == 0.0

    def test_no_nodes_key(self):
        from sage.verl.reward import _score_structure
        assert _score_structure("edges: []") == 0.0


class TestScoreRustDensity:
    def test_valid_topology_returns_score(self):
        from sage.verl.reward import _score_rust_density
        score = _score_rust_density("nodes:\n- role: coder", {})
        # With sage_core: Rust-computed. Without: 0.5 fallback.
        assert 0.0 <= score <= 1.0

    def test_invalid_yaml_returns_zero(self):
        from sage.verl.reward import _score_rust_density
        assert _score_rust_density("{{broken", {}) == 0.0

    def test_no_nodes_returns_zero(self):
        from sage.verl.reward import _score_rust_density
        assert _score_rust_density("reasoning: text", {}) == 0.0


class TestComputeScore:
    def test_valid_topology(self):
        from sage.verl.reward import compute_score
        result = compute_score(
            data_source="sage_topology",
            solution_str="nodes:\n- role: coder\n  prompt: code\ndifficulty: simple",
            ground_truth="",
            extra_info={"task_id": "test/0", "difficulty": "simple"},
        )
        assert isinstance(result, float)
        # fmt_norm=1.0, struct=0.6, rust>=0.5 → combined >= 0.7
        assert 0.5 <= result <= 1.0

    def test_invalid_yaml_low_score(self):
        from sage.verl.reward import compute_score
        result = compute_score("sage_topology", "{{invalid", "", {})
        assert 0.0 <= result <= 0.1  # fmt=-2.0 → fmt_norm=0.0, rest=0.0

    def test_no_extra_info(self):
        from sage.verl.reward import compute_score
        result = compute_score("sage_topology", "nodes:\n- role: coder", "", None)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_returns_float_not_tensor(self):
        """veRL expects float, not torch.Tensor."""
        from sage.verl.reward import compute_score
        result = compute_score("sage_topology", "nodes:\n- role: coder", "", {})
        assert type(result) is float


class TestComputeScoreWithEdgeCredit:
    def test_passing_topology_gets_bonus(self):
        from sage.verl.reward import compute_score_with_edge_credit
        topos = [
            {"yaml": "nodes:\n- role: coder\n- role: reviewer\nedges:\n- from_idx: 0\n  to_idx: 1", "base_reward": 1.5},
            {"yaml": "nodes:\n- role: coder\nedges: []", "base_reward": 0.0},
        ]
        adjusted = compute_score_with_edge_credit(topos)
        assert len(adjusted) == 2
        assert adjusted[0] >= 1.5  # bonus from edge credit

    def test_single_topology_unchanged(self):
        from sage.verl.reward import compute_score_with_edge_credit
        topos = [{"yaml": "nodes:\n- role: coder", "base_reward": 0.8}]
        adjusted = compute_score_with_edge_credit(topos)
        assert len(adjusted) == 1
        assert adjusted[0] == pytest.approx(0.8, abs=0.01)
