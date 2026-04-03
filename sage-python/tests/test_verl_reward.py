"""Tests for veRL reward function (sage.verl.reward)."""
import pytest


class TestScoreFormat:
    def test_valid_yaml_with_nodes(self):
        from sage.verl.reward import _score_format
        # V8: plain YAML returns 0.5; 1.0 is reserved for <tool_call> format
        assert _score_format("nodes:\n- role: coder\n  prompt: Write code\ndifficulty: simple") == 0.5

    def test_invalid_yaml(self):
        from sage.verl.reward import _score_format
        assert _score_format("not: [valid: yaml: {{") == -2.0

    def test_valid_yaml_no_nodes(self):
        from sage.verl.reward import _score_format
        assert _score_format("reasoning: just text") == -0.5

    def test_valid_yaml_not_dict(self):
        from sage.verl.reward import _score_format
        # V8: YAML list has no topology markers, _extract_topology_data returns None,
        # _partial_credit returns -2.0 (no markers found)
        assert _score_format("- item1\n- item2") == -2.0

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
        # 1 node, no difficulty -> defaults to "moderate" (expected_min=2)
        # Base: nodes(0.3) + roles(0.3) = 0.6, then *0.5 trivial penalty = 0.3
        score = _score_structure("nodes:\n- role: coder")
        assert score == pytest.approx(0.3)

    def test_minimal_one_node_simple_no_penalty(self):
        from sage.verl.reward import _score_structure
        # 1 node with difficulty=simple -> expected_min=1, no penalty
        score = _score_structure("nodes:\n- role: coder\ndifficulty: simple")
        assert score == pytest.approx(0.6)  # nodes(0.3) + roles(0.3)

    def test_no_roles(self):
        from sage.verl.reward import _score_structure
        # 1 node, no difficulty -> defaults to "moderate" (expected_min=2)
        # Base: nodes(0.3), then *0.5 trivial penalty = 0.15
        score = _score_structure("nodes:\n- name: foo")
        assert score == pytest.approx(0.15)

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


class TestPartialCredit:
    """V5 reward shaping: truncated YAML gets partial credit instead of -2.0."""

    def test_pure_garbage_no_credit(self):
        from sage.verl.reward import _partial_credit
        assert _partial_credit("some random text about topology") == -2.0

    def test_empty_string_no_credit(self):
        from sage.verl.reward import _partial_credit
        assert _partial_credit("") == -2.0

    def test_only_nodes_key(self):
        from sage.verl.reward import _partial_credit
        # V8: nodes: marker gives +0.5 (was +1.0 in V5)
        assert _partial_credit("nodes:") == -1.5

    def test_nodes_plus_role_plus_list(self):
        from sage.verl.reward import _partial_credit
        score = _partial_credit("nodes:\n- role: coder\n  model_tier: bud")
        # V8: nodes(+0.5) + role(+0.2) + yaml_list(+0.1) = -2.0+0.8 = -1.2
        assert score == pytest.approx(-1.2)

    def test_all_markers_capped(self):
        from sage.verl.reward import _partial_credit
        score = _partial_credit("nodes:\n- role: coder\nreasoning: plan")
        # V8: nodes(+0.5) + role(+0.2) + yaml_list(+0.1) + reasoning(+0.1) = -2.0+0.9 = -1.1
        assert score == pytest.approx(-1.1)

    def test_never_exceeds_valid_yaml(self):
        """Partial credit must always be < -0.25 (valid-but-empty-nodes score)."""
        from sage.verl.reward import _partial_credit
        score = _partial_credit("nodes:\n- role: x\n- role: y\nreasoning: z\nedges:\n- from: 0")
        assert score <= -0.3

    def test_format_delegates_to_partial_credit(self):
        """_score_format uses _partial_credit for broken YAML with markers."""
        from sage.verl.reward import _score_format
        # Unbalanced flow mapping — definitely fails YAML parse
        score = _score_format("nodes:\n- {role: coder, prompt: 'trunca")
        assert -2.0 < score <= -0.3

    def test_format_still_minus_2_for_no_markers(self):
        from sage.verl.reward import _score_format
        assert _score_format("not: [valid: yaml: {{") == -2.0

    def test_code_fence_stripped_before_partial(self):
        from sage.verl.reward import _partial_credit
        score = _partial_credit("```yaml\nnodes:\n- role: coder")
        assert score > -2.0  # fence stripped, markers detected


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
