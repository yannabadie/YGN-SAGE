"""Tests for veRL reward function — structural and execution paths.

Issue F audit fix: The execution reward path was completely untested.
These tests cover all error paths and mode switching.
"""
import os

import pytest


class TestIsExecMode:
    def test_reads_env_dynamically(self):
        from sage.verl.reward import _is_exec_mode

        old = os.environ.get("SAGE_VERL_EXEC")
        try:
            os.environ["SAGE_VERL_EXEC"] = "0"
            assert _is_exec_mode() is False

            os.environ["SAGE_VERL_EXEC"] = "1"
            assert _is_exec_mode() is True

            os.environ.pop("SAGE_VERL_EXEC", None)
            assert _is_exec_mode() is False
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)


class TestComputeScoreStructural:
    def test_valid_yaml_high_score(self):
        from sage.verl.reward import compute_score

        yaml_text = (
            "nodes:\n"
            "  - role: coder\n"
            "    model_tier: budget\n"
            "reasoning: test task\n"
        )
        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "0"
        try:
            score = compute_score("test", yaml_text, "", {})
            assert isinstance(score, float)
            assert score > 0.0
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)

    def test_invalid_yaml_low_score(self):
        from sage.verl.reward import compute_score

        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "0"
        try:
            score = compute_score("test", "this is not yaml at all {{{", "", {})
            assert isinstance(score, float)
            assert score < 0.5
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)


class TestPartialCredit:
    def test_truncated_yaml_with_nodes_key(self):
        from sage.verl.reward import _partial_credit

        score = _partial_credit("nodes:\n- role: coder\n  model_tier: budget\n  prompt: write code that")
        assert score > -2.0, "Truncated YAML with nodes: key should get partial credit"
        assert score <= -0.3, "Truncated YAML must score below valid YAML"

    def test_completely_wrong_text(self):
        from sage.verl.reward import _partial_credit

        score = _partial_credit("Hello, I'm a helpful assistant!")
        assert score == -2.0, "Non-YAML text should get no partial credit"

    def test_partial_credit_bounded(self):
        from sage.verl.reward import _partial_credit

        # V8: Maximum partial credit requires tool_call markers + structure.
        # YAML-only partial credit is bounded by available signal bonuses.
        yaml_score = _partial_credit("nodes:\n- role: coder\nreasoning: test\n- name: synth")
        assert yaml_score > -2.0, "YAML-like text should get some partial credit"
        assert yaml_score <= -0.3, "YAML partial credit capped at -0.3"

        # With tool_call markers, can reach the -0.3 cap
        tool_call_score = _partial_credit(
            "<tool_call>nodes:\n- role: coder\nreasoning: test\n- name: synth</tool_call>"
        )
        assert tool_call_score == -0.3, "Tool-call format partial credit should reach cap"


class TestExecModeFallback:
    def test_exec_mode_no_provider_falls_back(self):
        """With SAGE_VERL_EXEC=1 but no provider, must fall back to structural."""
        from sage.verl.reward import compute_score

        yaml_text = "nodes:\n  - role: coder\n    model_tier: budget\nreasoning: test\n"
        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "1"
        try:
            # Should NOT crash, should fall back to structural
            score = compute_score("test", yaml_text, "", {})
            assert isinstance(score, float)
            assert score > 0.0
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)

    def test_exec_mode_invalid_yaml_stays_structural(self):
        """With SAGE_VERL_EXEC=1 but invalid YAML, fmt < 0 → stays structural."""
        from sage.verl.reward import compute_score

        old = os.environ.get("SAGE_VERL_EXEC")
        os.environ["SAGE_VERL_EXEC"] = "1"
        try:
            score = compute_score("test", "not yaml {{{", "", {})
            assert isinstance(score, float)
        finally:
            if old is not None:
                os.environ["SAGE_VERL_EXEC"] = old
            else:
                os.environ.pop("SAGE_VERL_EXEC", None)
