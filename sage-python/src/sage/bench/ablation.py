"""Ablation study framework for isolating pillar contributions.

Design from Gemini + Codex oracle consultation (March 10, 2026):
6 configurations to measure each pillar's delta:
1. full — all pillars enabled (reference)
2. baseline — bare LLM call (no framework)
3. no-memory — disable memory injection
4. no-avr — disable S2 Act-Verify-Refine loop
5. no-routing — force S2 for everything
6. no-guardrails — disable all guardrails
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class AblationConfig:
    """Which pillars to enable/disable."""
    memory: bool = True
    avr: bool = True
    routing: bool = True
    guardrails: bool = True
    label: str = "full"

    def apply(self, system: Any) -> None:
        """Apply this config to an AgentSystem by setting skip flags on the agent loop."""
        loop = system.agent_loop
        loop._skip_memory = not self.memory
        loop._skip_avr = not self.avr
        loop._skip_routing = not self.routing
        loop._skip_guardrails = not self.guardrails


ABLATION_CONFIGS = [
    AblationConfig(label="full"),
    AblationConfig(memory=False, avr=False, routing=False, guardrails=False, label="baseline"),
    AblationConfig(memory=False, label="no-memory"),
    AblationConfig(avr=False, label="no-avr"),
    AblationConfig(routing=False, label="no-routing"),
    AblationConfig(guardrails=False, label="no-guardrails"),
]
