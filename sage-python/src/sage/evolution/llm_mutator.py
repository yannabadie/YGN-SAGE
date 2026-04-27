"""LLM-based code mutator using Gemini/Codex with structured JSON output.

Includes AdaptiveMutator (ShinkaEvolve-inspired, arXiv 2509.19349):
Thompson sampling over LLM tiers for mutation selection.
"""
from __future__ import annotations

import logging
import sqlite3
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
        provider: Any
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
        selected = max(samples, key=lambda t: samples[t])
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
        # (no call sites outside this module) — cgpro 2026-04-27 verdict:
        # keep + persist + wire into offline evolution path only, not the
        # runtime per-task agent loop. `save()` / `load()` and
        # `state_dict()` / `load_state_dict()` are wired below; the
        # offline-path call site remains a roadmap follow-up.
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

    def state_dict(self) -> dict[str, Any]:
        """Serialise Thompson posteriors + selection counts for persistence.

        cgpro 2026-04-27: AdaptiveMutator was the third instance of the
        bandit::restore_arm class — research-backed (ShinkaEvolve, arXiv
        2509.19349) but in-memory only. Persisting state lets the offline
        evolution path resume search from where it left off instead of
        cold-starting at uniform Beta(1,1) priors on every restart.
        """
        return {
            "tiers": list(self.tiers),
            "successes": dict(self._successes),
            "failures": dict(self._failures),
            "total_selections": dict(self._total_selections),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a ``state_dict()`` payload.

        Tiers absent from ``state`` keep their default Beta(1, 1) prior.
        Tiers present in ``state`` but not configured on this instance
        are added so callers don't lose history when the tier list is
        widened across versions.
        """
        loaded_tiers = state.get("tiers") or list(state.get("successes", {}).keys())
        for tier in loaded_tiers:
            if tier not in self.tiers:
                self.tiers = [*self.tiers, tier]
                self._successes.setdefault(tier, 1.0)
                self._failures.setdefault(tier, 1.0)
                self._total_selections.setdefault(tier, 0)
        for tier, value in (state.get("successes") or {}).items():
            self._successes[tier] = float(value)
        for tier, value in (state.get("failures") or {}).items():
            self._failures[tier] = float(value)
        for tier, value in (state.get("total_selections") or {}).items():
            self._total_selections[tier] = int(value)

    def save(self, db_path: str) -> None:
        """Persist Thompson posteriors + selection counts to SQLite.

        Schema: a single row per tier in ``adaptive_mutator_state``.
        Snapshot semantics: existing rows for tiers in this instance are
        replaced; tiers from earlier configurations remain untouched so
        widening the tier list later doesn't lose history.
        """
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS adaptive_mutator_state "
                "(tier TEXT PRIMARY KEY, successes REAL NOT NULL, "
                "failures REAL NOT NULL, total_selections INTEGER NOT NULL)"
            )
            rows = [
                (
                    tier,
                    self._successes[tier],
                    self._failures[tier],
                    self._total_selections[tier],
                )
                for tier in self.tiers
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO adaptive_mutator_state "
                "(tier, successes, failures, total_selections) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def load(self, db_path: str) -> None:
        """Restore from a SQLite file written by :meth:`save`.

        No-op if the file or table is missing (cold start). Existing
        in-memory state is overwritten only for tiers present in the DB.
        """
        import os

        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "adaptive_mutator_state" not in tables:
                return
            rows = list(
                conn.execute(
                    "SELECT tier, successes, failures, total_selections "
                    "FROM adaptive_mutator_state"
                )
            )
        finally:
            conn.close()

        if not rows:
            return
        state = {
            "tiers": [r[0] for r in rows],
            "successes": {r[0]: float(r[1]) for r in rows},
            "failures": {r[0]: float(r[2]) for r in rows},
            "total_selections": {r[0]: int(r[3]) for r in rows},
        }
        self.load_state_dict(state)
