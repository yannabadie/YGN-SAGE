"""LLM-based code mutator using Gemini/Codex with structured JSON output.

Includes AdaptiveMutator (ShinkaEvolve-inspired, arXiv 2509.19349):
Thompson sampling over LLM tiers for mutation selection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
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
        self.evolution_memory = None  # CORAL: wire for skill injection

    def _build_mutation_prompt(self, code: str, objective: str, context: str) -> str:
        prompt = f"## Objective\n{objective}\n\n"
        if context:
            prompt += f"## SAMPO Directive\n{context}\n\n"
        prompt += f"## Source Code\n```\n{code}\n```\n\n"
        prompt += "Generate 1-3 mutations as SEARCH/REPLACE pairs. Respond in the required JSON format."
        return prompt

    async def _inject_skills(self, prompt: str, context: str) -> str:
        """CORAL: inject learned skills from EvolutionMemory into mutation prompt."""
        if not self.evolution_memory:
            return prompt
        try:
            import json as _json
            ctx = _json.loads(context) if isinstance(context, str) and context.startswith("{") else {}
            skills = await self.evolution_memory.query_skills(
                domain=ctx.get("domain", ""),
                parent_score=ctx.get("parent_score", 0.0),
                top_k=3,
            )
            if skills:
                prompt += "\n## Learned Patterns (from past evolution)\n"
                for s in skills:
                    prompt += f"- SAMPO action {s.sampo_action}: {s.pattern} "
                    prompt += f"(success {s.success_rate:.0%}, n={s.sample_count})\n"
        except Exception:
            pass  # Best-effort — don't break mutation
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
        prompt = await self._inject_skills(prompt, request.context)

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


class AdaptiveMutator:
    """Bandit-based LLM ensemble for mutations (ShinkaEvolve, arXiv 2509.19349).

    Maintains a Thompson sampling bandit over LLM tiers, selecting the tier
    most likely to produce improving mutations based on historical success rates.
    """

    def __init__(self, tiers: list[str] | None = None) -> None:
        self.tiers = tiers or ["budget", "fast", "mutator", "reasoner"]
        # Beta distribution priors: successes (alpha) and failures (beta)
        self._successes: dict[str, float] = {t: 1.0 for t in self.tiers}
        self._failures: dict[str, float] = {t: 1.0 for t in self.tiers}
        self._total_selections: dict[str, int] = {t: 0 for t in self.tiers}

    def select_tier(self) -> str:
        """Thompson sampling: sample from Beta(successes, failures) for each tier."""
        samples = {
            t: np.random.beta(self._successes[t], self._failures[t])
            for t in self.tiers
        }
        selected = max(samples, key=samples.get)
        self._total_selections[selected] += 1
        return selected

    def record(self, tier: str, improved: bool) -> None:
        """Update posterior for the selected tier based on mutation outcome."""
        if tier not in self._successes:
            self._successes[tier] = 1.0
            self._failures[tier] = 1.0
        if improved:
            self._successes[tier] += 1.0
        else:
            self._failures[tier] += 1.0
        # In-run observability: log every Thompson-sampling posterior update so
        # future evolution runs show per-tier arm statistics. Note:
        # AdaptiveMutator is NOT invoked on the pipeline runtime path today
        # (no call sites outside this module) -- the log is wired now so the
        # observability is ready when the offline evolution training path is
        # re-activated.
        a = self._successes[tier]
        b = self._failures[tier]
        log.info(
            "evolution.mutator.update tier=%s improved=%s alpha=%.1f beta=%.1f "
            "success_rate=%.3f",
            tier, "true" if improved else "false", a, b, a / (a + b),
        )

    def success_rate(self, tier: str) -> float:
        """Estimated success rate for a tier (mean of Beta posterior)."""
        a = self._successes.get(tier, 1.0)
        b = self._failures.get(tier, 1.0)
        return a / (a + b)

    def stats(self) -> dict[str, Any]:
        """Return bandit statistics for all tiers."""
        return {
            tier: {
                "successes": self._successes[tier],
                "failures": self._failures[tier],
                "selections": self._total_selections[tier],
                "success_rate": round(self.success_rate(tier), 3),
            }
            for tier in self.tiers
        }
