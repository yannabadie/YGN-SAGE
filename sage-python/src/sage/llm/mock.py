
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
    ) -> LLMResponse:
        # Simulate LLM reasoning time
        await asyncio.sleep(self.latency)
        return LLMResponse(
            content="This is a mock response for benchmarking.",
            tool_calls=[],
            model="mock-frontier"
        )
