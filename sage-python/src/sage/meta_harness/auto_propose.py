"""Automated proposer: LLM-driven candidate generation.

While the primary Meta-Harness workflow uses Claude Code as a human-in-the-loop
proposer, this module enables fully automated search by using an LLM to
analyze traces and propose new HarnessConfig candidates.

Can run as a standalone loop or be called from Claude Code:
    python -m sage.meta_harness.auto_propose --iterations 10
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sage.meta_harness.config import HarnessConfig, ContextConfig, PromptConfig, ExecutionConfig
from sage.meta_harness.search_loop import MetaHarnessLoop, DEFAULT_WORKSPACE

log = logging.getLogger(__name__)

# Structured prompt for the proposer LLM
PROPOSER_SYSTEM = """You are an expert harness engineer optimizing a multi-agent LLM pipeline.
You will receive:
1. A leaderboard of prior candidates with scores
2. Execution traces from the best and worst candidates
3. The current best HarnessConfig (JSON)

Your task: propose a NEW HarnessConfig that improves the aggregate score.
Respond ONLY with valid JSON matching the HarnessConfig schema.

Key tuning dimensions:
- predecessor_format: how predecessor outputs are formatted ("[{role}]: {text}" is default)
- injection_template: how context is presented to each node
- budget_ratio: fraction of context window for predecessors (default 0.70)
- similarity_threshold: dedup gate threshold (default 0.90)
- default_template: system prompt template (default "You are acting as: {role}.")
- global_suffix: appended to all system prompts (default empty)
- quality_cascade_threshold: FrugalGPT retry trigger (default 0.30)
- overflow_strategy: "summarize" | "truncate" | "hierarchical"

Strategy guidelines:
- Change 1-2 parameters per proposal (isolate variables)
- If context seems truncated in traces, increase budget_ratio or try hierarchical overflow
- If predecessor outputs are confusing, try XML-tagged predecessor_format
- If nodes produce generic outputs, add a global_suffix with task-specific guidance
- If dedup removes too much, lower similarity_threshold
- If quality is borderline, adjust quality_cascade_threshold
"""


def _build_proposer_prompt(
    workspace: Path,
    loop: MetaHarnessLoop,
) -> str:
    """Build a diagnosis prompt from the filesystem state."""
    parts: list[str] = []

    # Leaderboard
    lb = loop._load_leaderboard()
    parts.append("## Leaderboard\n```json\n" + json.dumps(lb, indent=2) + "\n```\n")

    # Best config
    best = loop.best_config()
    if best:
        parts.append(
            "## Best Config (to derive from)\n```json\n"
            + best.to_json()
            + "\n```\n"
        )

    # Traces from best candidate
    if lb:
        best_id = lb[0]["candidate_id"]
        traces_path = (
            loop.candidates_dir / best_id / "traces.jsonl"
            if best_id != "baseline"
            else loop.baseline_dir / "traces.jsonl"
        )
        if traces_path.exists():
            trace_lines = traces_path.read_text(encoding="utf-8").strip().split("\n")
            # Include up to 15 trace lines
            traces_preview = "\n".join(trace_lines[:15])
            parts.append(
                f"## Traces from best candidate ({best_id})\n```jsonl\n{traces_preview}\n```\n"
            )

        # Also include worst candidate traces for contrast
        if len(lb) > 1:
            worst_id = lb[-1]["candidate_id"]
            worst_traces = loop.candidates_dir / worst_id / "traces.jsonl"
            if worst_traces.exists():
                wt_lines = worst_traces.read_text(encoding="utf-8").strip().split("\n")
                wt_preview = "\n".join(wt_lines[:10])
                parts.append(
                    f"## Traces from worst candidate ({worst_id})\n```jsonl\n{wt_preview}\n```\n"
                )

    # Failure summary
    failure_count: dict[str, int] = {}
    for candidate_dir in sorted(loop.candidates_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        scores_path = candidate_dir / "scores.json"
        if scores_path.exists():
            scores_data = json.loads(scores_path.read_text(encoding="utf-8"))
            per_task = scores_data.get("per_task", {})
            for task_id, score in per_task.items():
                if score < 0.5:
                    failure_count[task_id] = failure_count.get(task_id, 0) + 1

    if failure_count:
        persistent = {k: v for k, v in failure_count.items() if v > 1}
        if persistent:
            parts.append(
                "## Persistent failures (fail in 2+ candidates)\n```json\n"
                + json.dumps(persistent, indent=2)
                + "\n```\n"
            )

    return "\n".join(parts)


async def auto_propose(
    workspace: Path | None = None,
    model: str = "gemini-2.5-flash",
) -> HarnessConfig:
    """Use an LLM to propose a new HarnessConfig based on filesystem state.

    Returns the proposed config (also saved to candidates dir).
    """
    loop = MetaHarnessLoop(workspace=workspace or DEFAULT_WORKSPACE)

    # Build diagnosis context
    prompt = _build_proposer_prompt(loop.workspace, loop)

    next_id = loop.next_candidate_id()

    # Get best config as starting point
    best = loop.best_config() or HarnessConfig()
    parent_id = best.id

    # Call LLM for proposal
    from sage.llm.base import Message, Role

    messages = [
        Message(role=Role.SYSTEM, content=PROPOSER_SYSTEM),
        Message(
            role=Role.USER,
            content=(
                f"{prompt}\n\n"
                f"Propose candidate {next_id} (derived from {parent_id}). "
                f"Respond with ONLY the HarnessConfig JSON. "
                f'Set "id": "{next_id}", "parent_id": "{parent_id}", '
                f'and write a short "description" of your changes.'
            ),
        ),
    ]

    # Use available provider
    try:
        from sage.llm.google import GoogleProvider
        from sage.llm.base import LLMConfig

        provider = GoogleProvider()
        config = LLMConfig(model=model, temperature=0.7, max_tokens=4000)
        response = await provider.generate(messages=messages, config=config)
        response_text = response.content or ""
    except (ImportError, RuntimeError) as exc:
        log.error("Auto-propose LLM call failed: %s", exc)
        raise

    # Parse response
    try:
        # Extract JSON from response (handle markdown fences)
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        proposed = HarnessConfig.from_json(text)
        proposed.id = next_id
        proposed.parent_id = parent_id
    except (json.JSONDecodeError, KeyError) as exc:
        log.error("Failed to parse proposed config: %s\nRaw: %s", exc, response_text[:500])
        raise ValueError(f"LLM produced invalid config: {exc}") from exc

    # Save
    candidate_dir = loop.candidates_dir / next_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    proposed.save(candidate_dir / "config.json")
    (candidate_dir / "proposal.md").write_text(
        f"# Auto-proposed candidate {next_id}\n\n"
        f"Model: {model}\n"
        f"Parent: {parent_id}\n"
        f"Description: {proposed.description}\n\n"
        f"## LLM reasoning\n{response_text}\n",
        encoding="utf-8",
    )

    log.info("Auto-proposed candidate %s: %s", next_id, proposed.description)
    return proposed


async def auto_search(
    iterations: int = 10,
    bench_type: str = "masbench",
    axis: str = "depth",
    limit: int = 20,
    workspace: Path | None = None,
    model: str = "gemini-2.5-flash",
) -> None:
    """Run N iterations of automated Meta-Harness search.

    Each iteration: propose -> evaluate -> log -> repeat.
    """
    loop = MetaHarnessLoop(workspace=workspace or DEFAULT_WORKSPACE)

    # Ensure baseline exists
    if not loop.leaderboard_path.exists():
        loop.init_workspace()

    lb = loop._load_leaderboard()
    if not lb:
        print("Evaluating baseline first...")
        await loop.evaluate_baseline(bench_type=bench_type, axis=axis, limit=limit)

    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  Meta-Harness iteration {i+1}/{iterations}")
        print(f"{'='*60}")

        # Propose
        try:
            proposed = await auto_propose(workspace=loop.workspace, model=model)
        except (ValueError, RuntimeError) as exc:
            log.error("Proposal failed at iteration %d: %s", i + 1, exc)
            continue

        # Evaluate
        try:
            result = await loop.evaluate_candidate(
                proposed.id, bench_type=bench_type, axis=axis, limit=limit,
            )
            print(f"  -> Score: {result.aggregate_score:.3f} "
                  f"(pass: {result.aggregate_pass_rate*100:.1f}%)")
        except (RuntimeError, Exception) as exc:
            log.error("Evaluation failed at iteration %d: %s", i + 1, exc)
            continue

    print("\n" + loop.status())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto Meta-Harness search")
    parser.add_argument("--iterations", "-n", type=int, default=10)
    parser.add_argument("--bench", default="masbench")
    parser.add_argument("--axis", default="depth")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workspace", "-w", default=None)

    args = parser.parse_args()

    import asyncio
    asyncio.run(auto_search(
        iterations=args.iterations,
        bench_type=args.bench,
        axis=args.axis,
        limit=args.limit,
        model=args.model,
        workspace=Path(args.workspace).expanduser() if args.workspace else None,
    ))
