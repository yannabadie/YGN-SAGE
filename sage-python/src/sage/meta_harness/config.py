"""HarnessConfig: parameterized harness search space.

Each field maps to a currently-hardcoded decision in the SAGE pipeline/runner.
Meta-Harness searches over HarnessConfig instances to find the combination
that maximizes benchmark scores.

The config is injected at runtime via monkey-patching (HarnessPatcher),
so the original source code is not modified — candidates are ephemeral overlays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ContextConfig:
    """How predecessor outputs are aggregated and presented to each node.

    Maps to: TopologyRunner._gather_predecessor_context()
              TopologyRunner._context_budget_per_predecessor()
              TopologyRunner._deduplicate_context()
    """

    # Format template for each predecessor's output
    # Available vars: {role}, {text}, {node_idx}, {model_id}
    predecessor_format: str = "[{role}]: {text}"

    # Separator between predecessor outputs
    predecessor_separator: str = "\n\n"

    # Injection template (the system message wrapping all predecessor context)
    # Available vars: {context}, {n_predecessors}, {task_preview}
    injection_template: str = "Context from previous agents:\n{context}"

    # Ratio of model context_window allocated to predecessor outputs (0.0-1.0)
    budget_ratio: float = 0.70

    # Minimum character budget per predecessor (floor)
    budget_floor_chars: int = 1000

    # Chars-per-token estimate for budget calculation
    chars_per_token: int = 4

    # Similarity gate: cosine threshold for deduplication
    # 0.0 = no dedup, 1.0 = exact only. Default 0.90
    similarity_threshold: float = 0.90

    # Dedup strategy: "semantic" (embeddings) or "jaccard" (word overlap)
    dedup_strategy: str = "semantic"

    # Context overflow compression strategy: "summarize", "truncate", "hierarchical"
    overflow_strategy: str = "summarize"


@dataclass
class PromptConfig:
    """System prompt engineering per node role.

    Maps to: TopologyRunner._execute_node() system prompt construction
    """

    # Default system prompt template when node has no custom prompt
    # Available vars: {role}, {capabilities}, {task_preview}, {n_predecessors}
    default_template: str = "You are acting as: {role}."

    # Capability suffix template (appended when node has capabilities)
    capability_template: str = " Your capabilities: {capabilities}."

    # Whether to inject task metadata into system prompt
    inject_task_metadata: bool = False

    # Task metadata template (when inject_task_metadata=True)
    task_metadata_template: str = ""

    # Per-role prompt overrides (role_name -> prompt_template)
    role_overrides: dict[str, str] = field(default_factory=dict)

    # Prefix/suffix wrappers applied to ALL system prompts
    global_prefix: str = ""
    global_suffix: str = ""


@dataclass
class ExecutionConfig:
    """Execution-level harness parameters.

    Maps to: CognitiveOrchestrationPipeline._stage_execute()
              TopologyRunner._execute_node() timeout/overflow
    """

    # FrugalGPT cascade: quality threshold below which retry with upgraded model
    quality_cascade_threshold: float = 0.30

    # Budget escalation factor on cascade retry
    cascade_budget_multiplier: float = 1.50

    # Per-node timeout (seconds)
    node_timeout_s: float = 60.0

    # Context overflow threshold (fraction of context_window)
    overflow_threshold: float = 0.85

    # Max debate rounds (multi-turn reset_node + open_gate)
    max_debate_rounds: int = 3

    # Whether to enable parallel execution of ready nodes
    enable_parallel: bool = True

    # Compression prompt for context overflow
    compression_prompt: str = (
        "Summarize concisely. Preserve all key facts, numbers, code, and conclusions."
    )


@dataclass
class TopologyConfig:
    """Topology selection overrides.

    Maps to: pipeline_stages.select_macro_topology()
    """

    # DAG feature thresholds for template selection
    omega_parallel_threshold: int = 3
    delta_deep_threshold: int = 4
    gamma_coupling_threshold: float = 0.5

    # Force specific template for certain domains (domain -> template_name)
    domain_template_overrides: dict[str, str] = field(default_factory=dict)

    # S1 skip threshold: system level at or below which topology is skipped
    s1_skip_threshold: int = 1


@dataclass
class HarnessConfig:
    """Complete harness configuration — the search space for Meta-Harness.

    Each field is a tunable knob that determines what context the LLM sees.
    The baseline config matches SAGE's current hardcoded behavior exactly.
    """

    id: str = "baseline"
    description: str = "SAGE default (hardcoded values)"
    parent_id: str = ""  # Which candidate this was derived from

    context: ContextConfig = field(default_factory=ContextConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str) -> HarnessConfig:
        data = json.loads(text)
        return cls(
            id=data.get("id", "unknown"),
            description=data.get("description", ""),
            parent_id=data.get("parent_id", ""),
            context=ContextConfig(**data.get("context", {})),
            prompts=PromptConfig(**data.get("prompts", {})),
            execution=ExecutionConfig(**data.get("execution", {})),
            topology=TopologyConfig(**data.get("topology", {})),
        )

    @classmethod
    def load(cls, path: Path) -> HarnessConfig:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def diff(self, other: HarnessConfig) -> dict[str, tuple[Any, Any]]:
        """Return fields that differ between self and other."""
        d1 = asdict(self)
        d2 = asdict(other)
        diffs: dict[str, tuple[Any, Any]] = {}

        def _compare(prefix: str, a: dict, b: dict) -> None:
            for key in set(a) | set(b):
                full_key = f"{prefix}.{key}" if prefix else key
                va, vb = a.get(key), b.get(key)
                if isinstance(va, dict) and isinstance(vb, dict):
                    _compare(full_key, va, vb)
                elif va != vb:
                    diffs[full_key] = (va, vb)

        _compare("", d1, d2)
        return diffs
