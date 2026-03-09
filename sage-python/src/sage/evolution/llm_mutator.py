"""LLM-based code mutator using Gemini/Codex with structured JSON output."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel

from sage.llm.base import Message, Role
from sage.llm.router import ModelRouter

log = logging.getLogger(__name__)

MUTATION_SYSTEM_PROMPT = """You are an expert code evolution engine. Given source code and an objective,
generate SEARCH/REPLACE mutations that improve the code toward the objective.

Rules:
1. Each mutation must have an exact `search` string (verbatim from the code) and a `replace` string.
2. Mutations must be syntactically valid and maintain the function signature.
3. Provide `features` as a list of 2 integers (0-9) describing behavioral dimensions.
4. Provide brief `reasoning` explaining the improvement.
"""


class MutationItem(BaseModel):
    search: str
    replace: str
    description: str


class MutationResponse(BaseModel):
    mutations: list[MutationItem]
    features: list[int]
    reasoning: str


@dataclass
class MutationRequest:
    code: str
    objective: str
    context: str = ""


class LLMMutator:
    """Generates code mutations via LLM with structured JSON output."""

    def __init__(self, llm_tier: str = "mutator"):
        self.llm_tier = llm_tier

    def _build_mutation_prompt(self, code: str, objective: str, context: str) -> str:
        prompt = f"## Objective\n{objective}\n\n"
        if context:
            prompt += f"## SAMPO Directive\n{context}\n\n"
        prompt += f"## Source Code\n```\n{code}\n```\n\n"
        prompt += "Generate 1-3 mutations as SEARCH/REPLACE pairs. Respond in the required JSON format."
        return prompt

    async def mutate(self, request: MutationRequest) -> MutationResponse:
        """Generate mutations using LLM with structured output."""
        config = ModelRouter.get_config(
            self.llm_tier,
            temperature=0.8,
            json_schema=MutationResponse,
        )

        prompt = self._build_mutation_prompt(
            request.code, request.objective, request.context
        )

        messages = [
            Message(role=Role.SYSTEM, content=MUTATION_SYSTEM_PROMPT),
            Message(role=Role.USER, content=prompt),
        ]

        # Get provider based on tier
        if config.provider == "codex":
            from sage.llm.codex import CodexProvider
            provider = CodexProvider()
        else:
            from sage.llm.google import GoogleProvider
            provider = GoogleProvider()

        response = await provider.generate(messages, config=config)

        try:
            return MutationResponse.model_validate_json(response.content)
        except Exception as e:
            log.warning(f"Failed to parse mutation response: {e}")
            # Attempt lenient JSON extraction
            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return MutationResponse.model_validate_json(text[start:end])
            raise
