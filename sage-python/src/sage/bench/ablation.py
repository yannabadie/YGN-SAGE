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
import math
import random
from dataclasses import dataclass
from itertools import combinations
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


def compute_ablation_stats(results: dict[str, list[int]]) -> dict:
    """Compute McNemar's test, Cohen's d, and bootstrap CI for ablation results.

    Uses only stdlib (math.erfc for chi2 p-value). No scipy/numpy dependency.

    Args:
        results: Mapping of config label -> list of binary pass/fail outcomes (1/0),
                 all lists must have the same length (paired tasks).

    Returns:
        Dict with "pairwise" key mapping "{a}_vs_{b}" to stats dicts containing:
        - mcnemar_p: p-value from McNemar's test with continuity correction
        - cohens_d: effect size (positive = a better than b)
        - bootstrap_ci_95: [lo, hi] 95% CI on mean difference (a - b)
        - discordant: {"b_wins": int, "c_wins": int} raw discordant cell counts
    """
    stats: dict[str, Any] = {"pairwise": {}}
    configs = list(results.keys())

    for a, b in combinations(configs, 2):
        ra, rb = results[a], results[b]
        n = len(ra)

        # McNemar's test (continuity correction, chi-squared df=1)
        # b_wins: a=1, b=0 (a correct, b wrong)
        # c_wins: a=0, b=1 (b correct, a wrong)
        b_wins = sum(1 for i in range(n) if ra[i] == 1 and rb[i] == 0)
        c_wins = sum(1 for i in range(n) if ra[i] == 0 and rb[i] == 1)
        if b_wins + c_wins == 0:
            p_value = 1.0
        else:
            chi2_stat = (abs(b_wins - c_wins) - 1) ** 2 / (b_wins + c_wins)
            # math.erfc(sqrt(x/2)) gives chi-squared survival function for df=1
            p_value = math.erfc(math.sqrt(chi2_stat / 2))

        # Cohen's d (pooled standard deviation, population variance)
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        var_a = sum((x - mean_a) ** 2 for x in ra) / n
        var_b = sum((x - mean_b) ** 2 for x in rb) / n
        pooled_std = math.sqrt((var_a + var_b) / 2)
        d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Bootstrap 95% CI on mean difference (10,000 resamples, seed=42)
        rng = random.Random(42)
        diffs = []
        for _ in range(10_000):
            idx = [rng.randint(0, n - 1) for _ in range(n)]
            sample_a = [ra[i] for i in idx]
            sample_b = [rb[i] for i in idx]
            diffs.append(sum(sample_a) / n - sum(sample_b) / n)
        diffs.sort()
        ci_lo = diffs[int(len(diffs) * 0.025)]
        ci_hi = diffs[int(len(diffs) * 0.975)]

        stats["pairwise"][f"{a}_vs_{b}"] = {
            "mcnemar_p": round(p_value, 4),
            "cohens_d": round(d, 4),
            "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "discordant": {"b_wins": b_wins, "c_wins": c_wins},
        }

    return stats
