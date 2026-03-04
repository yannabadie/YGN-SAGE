
from typing import Literal
from sage.llm.base import LLMConfig

class ModelRouter:
    """SOTA 2026 Model Router for YGN-SAGE.
    
    Orchestrates between Flash-Lite (Speed/Cost) and Pro (Reasoning).
    """
    
    MODELS = {
        "fast": "gemini-3.1-flash-lite-preview",
        "reasoner": "gemini-3.1-pro-preview",
        "fallback": "gemini-2.0-flash"
    }

    @staticmethod
    def get_config(tier: Literal["fast", "reasoner", "critical"], temperature: float = 0.7) -> LLMConfig:
        if tier == "reasoner" or tier == "critical":
            return LLMConfig(
                provider="google",
                model=ModelRouter.MODELS["reasoner"],
                max_tokens=8192,
                temperature=temperature
            )
        else:
            return LLMConfig(
                provider="google",
                model=ModelRouter.MODELS["fast"],
                max_tokens=4096,
                temperature=temperature
            )
