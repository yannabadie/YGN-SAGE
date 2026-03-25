"""Tests for Path 6 policy model loader — V1/V2 support.

Issue C audit fix: Path 6 now supports Nemotron-8B V2 (GRPO-merged)
alongside legacy Phi-4-mini V1 (SFT LoRA).
"""
import pytest

from sage.topology.llm_caller import (
    POLICY_V1,
    POLICY_V2,
    PolicyModelConfig,
    _format_prompt,
)


class TestPolicyModelConfig:
    def test_v2_config(self):
        assert POLICY_V2.repo == "yannabadie/sage-topology-policy-v2"
        assert POLICY_V2.base_model == "nvidia/Nemotron-Orchestrator-8B"
        assert POLICY_V2.chat_template == "qwen3"
        assert POLICY_V2.max_new_tokens == 512
        assert POLICY_V2.trust_remote_code is True

    def test_v1_config(self):
        assert POLICY_V1.repo == "yannabadie/sage-topology-policy"
        assert POLICY_V1.base_model == "microsoft/Phi-4-mini-instruct"
        assert POLICY_V1.chat_template == "phi4"
        assert POLICY_V1.max_new_tokens == 256
        assert POLICY_V1.trust_remote_code is False

    def test_v2_has_larger_context_than_v1(self):
        assert POLICY_V2.max_new_tokens > POLICY_V1.max_new_tokens


class TestFormatPrompt:
    def test_qwen3_template(self):
        prompt = _format_prompt("Write a fibonacci function", POLICY_V2)
        assert "<|im_start|>system" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "Write a fibonacci function" in prompt
        # Must NOT contain Phi-4 tokens
        assert "<|system|>" not in prompt
        assert "<|end|>" not in prompt

    def test_phi4_template(self):
        prompt = _format_prompt("Write a fibonacci function", POLICY_V1)
        assert "<|system|>" in prompt
        assert "<|end|>" in prompt
        assert "<|assistant|>" in prompt
        assert "Write a fibonacci function" in prompt
        # Must NOT contain Qwen3 tokens
        assert "<|im_start|>" not in prompt
        assert "<|im_end|>" not in prompt

    def test_truncation_at_2000_chars(self):
        long_task = "x" * 5000
        prompt = _format_prompt(long_task, POLICY_V2)
        # Task is truncated to 2000 chars
        assert "x" * 2000 in prompt
        assert "x" * 2001 not in prompt

    def test_system_prompt_contains_topology_instructions(self):
        prompt = _format_prompt("test", POLICY_V2)
        assert "topology" in prompt.lower()
        assert "nodes" in prompt.lower()
        assert "edges" in prompt.lower()


class TestPolicyV2Priority:
    def test_v2_tried_before_v1(self):
        """download_policy_model tries V2 (Nemotron) before V1 (Phi-4)."""
        # We can't test the actual download, but we verify the order
        # by inspecting the function source
        import inspect
        from sage.topology.llm_caller import download_policy_model
        source = inspect.getsource(download_policy_model)
        v2_pos = source.find("POLICY_V2")
        v1_pos = source.find("POLICY_V1")
        assert v2_pos < v1_pos, "V2 must be tried before V1 in download_policy_model"
