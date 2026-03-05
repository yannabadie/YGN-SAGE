
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolDef
import asyncio
import time

class MockLLMProvider:
    """Mock LLM provider for benchmarking infrastructure overhead without network latency."""
    name = "mock"

    def __init__(self, latency: float = 0.5):
        self.latency = latency

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        file_search_store_names: list[str] | None = None,
    ) -> LLMResponse:
        # Simulate LLM reasoning time
        await asyncio.sleep(self.latency)
        return LLMResponse(
            content="This is a mock response for benchmarking.",
            tool_calls=[],
            model="mock-frontier"
        )


class MockProvider:
    """Mock LLM provider with configurable responses for testing."""
    name = "mock"

    def __init__(self, responses: list[str] | None = None, latency: float = 0.0):
        self._responses = list(responses) if responses else ["Mock response."]
        self._call_index = 0
        self.latency = latency

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        file_search_store_names: list[str] | None = None,
    ) -> LLMResponse:
        if self.latency > 0:
            await asyncio.sleep(self.latency)
        content = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return LLMResponse(
            content=content,
            tool_calls=[],
            model="mock-test",
        )
