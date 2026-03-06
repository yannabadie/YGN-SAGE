"""Dynamic agent factory -- create agents from LLM-generated blueprints."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sage.providers.registry import ModelProfile


@dataclass
class AgentBlueprint:
    name: str
    role: str = ""
    needs_code: bool = False
    needs_reasoning: bool = False
    needs_tools: bool = False
    tools: list[str] | None = None


class DynamicAgentFactory:
    """Creates ModelAgent instances from blueprints + model profiles."""

    def create(self, blueprint: AgentBlueprint, profile: ModelProfile) -> Any:
        from sage.orchestrator import ModelAgent
        prompt = self._build_prompt(blueprint)
        return ModelAgent(name=blueprint.name, model=profile, system_prompt=prompt)

    def _build_prompt(self, bp: AgentBlueprint) -> str:
        parts = [f"You are '{bp.name}'."]
        if bp.role:
            parts.append(f"Your role: {bp.role}")
        if bp.needs_code:
            parts.append("Focus on writing correct, efficient code.")
        if bp.needs_reasoning:
            parts.append("Use rigorous step-by-step reasoning. Show your work.")
        if bp.tools:
            parts.append(f"You have access to these tools: {', '.join(bp.tools)}")
        return " ".join(parts)

    def parse_blueprints(self, text: str) -> list[AgentBlueprint]:
        """Parse LLM decomposition output into blueprints."""
        blueprints = []
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if not line or len(line) < 10:
                continue
            needs_code = "[CODE]" in line.upper() or "code" in line.lower()
            needs_reasoning = "[REASON]" in line.upper() or any(
                w in line.lower() for w in ("prove", "verify", "reason", "analyze")
            )
            clean = re.sub(r"\[(CODE|REASON|GENERAL)\]", "", line, flags=re.IGNORECASE).strip()
            if clean:
                name = re.sub(r"[^a-zA-Z0-9_-]", "-", clean[:30]).strip("-").lower()
                blueprints.append(AgentBlueprint(
                    name=name or f"agent-{len(blueprints)}",
                    role=clean,
                    needs_code=needs_code,
                    needs_reasoning=needs_reasoning,
                ))
        return blueprints[:4]
