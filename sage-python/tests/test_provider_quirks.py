"""Tests for provider-specific quirk handling in OpenAICompatProvider."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sage.providers.openai_compat import OpenAICompatProvider
from sage.llm.base import Message, Role, LLMConfig


class TestInferProvider:
    def test_deepseek(self):
        assert OpenAICompatProvider._infer_provider("https://api.deepseek.com") == "deepseek"

    def test_xai(self):
        assert OpenAICompatProvider._infer_provider("https://api.x.ai/v1") == "xai"

    def test_minimax(self):
        assert OpenAICompatProvider._infer_provider("https://api.minimaxi.chat/v1") == "minimax"

    def test_kimi(self):
        assert OpenAICompatProvider._infer_provider("https://api.moonshot.ai/v1") == "kimi"

    def test_openai(self):
        assert OpenAICompatProvider._infer_provider("https://api.openai.com/v1") == "openai"

    def test_none_defaults_to_openai(self):
        assert OpenAICompatProvider._infer_provider(None) == "openai"

    def test_unknown_url(self):
        assert OpenAICompatProvider._infer_provider("https://unknown.api.com") == ""


class TestApplyQuirks:
    def test_deepseek_reasoner_strips_temperature(self):
        p = OpenAICompatProvider(api_key="k", provider_name="deepseek", model_id="deepseek-reasoner")
        params = p._apply_quirks({"model": "deepseek-reasoner", "temperature": 0.7, "max_tokens": 4096})
        assert "temperature" not in params

    def test_deepseek_chat_keeps_temperature(self):
        p = OpenAICompatProvider(api_key="k", provider_name="deepseek", model_id="deepseek-chat")
        params = p._apply_quirks({"model": "deepseek-chat", "temperature": 0.7, "max_tokens": 4096})
        assert params["temperature"] == 0.7

    def test_kimi_clamps_temperature(self):
        p = OpenAICompatProvider(api_key="k", provider_name="kimi", model_id="kimi-k2.5")
        params = p._apply_quirks({"model": "kimi-k2.5", "temperature": 1.5, "max_tokens": 4096})
        assert params["temperature"] <= 1.0

    def test_kimi_keeps_low_temperature(self):
        p = OpenAICompatProvider(api_key="k", provider_name="kimi", model_id="kimi-k2.5")
        params = p._apply_quirks({"model": "kimi-k2.5", "temperature": 0.3, "max_tokens": 4096})
        assert params["temperature"] == 0.3

    def test_unknown_no_quirks(self):
        p = OpenAICompatProvider(api_key="k", model_id="test")
        original = {"model": "test", "temperature": 0.7, "max_tokens": 4096}
        params = p._apply_quirks(original.copy())
        assert params == original


class TestReasoningExtraction:
    def test_extract_with_reasoning_content(self):
        p = OpenAICompatProvider(api_key="k", provider_name="deepseek")
        msg = MagicMock()
        msg.content = "Answer"
        msg.reasoning_content = "Step 1: think"
        reasoning, content = p._extract_reasoning(msg)
        assert reasoning == "Step 1: think"
        assert content == "Answer"

    def test_extract_without_reasoning_content(self):
        p = OpenAICompatProvider(api_key="k", provider_name="openai")
        msg = MagicMock()
        msg.content = "Answer"
        msg.reasoning_content = None
        reasoning, content = p._extract_reasoning(msg)
        assert reasoning == ""
        assert content == "Answer"

    def test_format_with_reasoning(self):
        p = OpenAICompatProvider(api_key="k")
        result = p._format_response("thinking...", "answer")
        assert result == "<think>thinking...</think>\nanswer"

    def test_format_without_reasoning(self):
        p = OpenAICompatProvider(api_key="k")
        result = p._format_response("", "answer")
        assert result == "answer"


class TestProviderNameParam:
    def test_explicit_provider_name(self):
        p = OpenAICompatProvider(api_key="k", provider_name="deepseek")
        assert p.provider_name == "deepseek"

    def test_inferred_from_url(self):
        p = OpenAICompatProvider(api_key="k", base_url="https://api.deepseek.com")
        assert p.provider_name == "deepseek"

    def test_explicit_overrides_inferred(self):
        p = OpenAICompatProvider(api_key="k", base_url="https://api.x.ai/v1", provider_name="custom")
        assert p.provider_name == "custom"
