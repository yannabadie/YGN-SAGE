"""Generate topology SFT training data using GPT-5.4 structured output.

Uses GPT-5.4 with JSON schema enforcement to generate high-quality
topology YAML for diverse coding tasks. Much faster than execution-based
collection (seconds vs minutes per entry).

Usage:
    python scripts/generate_topology_sft.py --limit 500 --output data/topology_sft_gpt54.jsonl
    python scripts/generate_topology_sft.py --limit 5000 --model gpt-5.4 --workers 8
    python scripts/generate_topology_sft.py --list
"""
from __future__ import annotations

import os
import ssl

# Configure SSL with corporate CA bundle if available
_CA_BUNDLE = "C:/Code/certs/ca-bundle.pem"
if os.path.exists(_CA_BUNDLE):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA_BUNDLE)
    os.environ.setdefault("SSL_CERT_FILE", _CA_BUNDLE)
    os.environ.setdefault("CURL_CA_BUNDLE", _CA_BUNDLE)
    # Patch httpx to use the CA bundle (HuggingFace Hub uses httpx)
    import httpx as _hx
    _hx_orig = _hx.Client.__init__
    def _hx_patched(self, *a, **kw):
        kw.setdefault("verify", _CA_BUNDLE)
        _hx_orig(self, *a, **kw)
    _hx.Client.__init__ = _hx_patched

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("gen_sft")

TOPOLOGY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning about why this topology is appropriate for the task",
        },
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "Agent role (e.g., planner, coder, reviewer, debugger)"},
                    "prompt": {"type": "string", "description": "Detailed system prompt for this agent (2-5 sentences)"},
                    "model_tier": {"type": "string", "enum": ["fast", "reasoner", "budget"], "description": "Model capability tier"},
                },
                "required": ["role", "prompt", "model_tier"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 10,
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_idx": {"type": "integer", "description": "Source node index"},
                    "to_idx": {"type": "integer", "description": "Target node index"},
                    "flow_type": {"type": "string", "enum": ["control", "message", "state"]},
                },
                "required": ["from_idx", "to_idx", "flow_type"],
                "additionalProperties": False,
            },
        },
        "difficulty": {"type": "string", "enum": ["simple", "moderate", "complex"]},
    },
    "required": ["reasoning", "nodes", "edges", "difficulty"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an expert multi-agent topology designer for the YGN-SAGE framework.

Given a coding task, design an optimal agent topology as a directed acyclic graph (DAG).

Rules:
- Simple tasks (factual, single-function): 1-2 nodes. Don't over-engineer.
- Moderate tasks (multi-step, requires planning): 3-5 nodes with planner → coder → reviewer.
- Complex tasks (formal proofs, system design, multi-file): 5-10 nodes with specialized roles.
- Each node has a role, a detailed system prompt, and a model tier preference.
- Edges define information flow: control (execution order), message (data passing), state (shared context).
- Keep topologies sparse — fewer edges = lower token cost. Only add edges that are necessary.
- Use diverse roles: planner, coder, reviewer, debugger, tester, architect, researcher, synthesizer.
- Each prompt should be task-specific, not generic ("You review code" is bad, "You review Python code for off-by-one errors and edge cases in sorting algorithms" is good)."""


def _load_tasks(dataset: str, subset: str, limit: int | None):
    """Load task prompts from BigCodeBench, APPS, or GSM8K."""
    tasks = []
    if dataset == "gsm8k":
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            for i, row in enumerate(ds):
                if limit and i >= limit:
                    break
                tasks.append((f"GSM8K/{i}", row.get("question", "")))
        except Exception as exc:
            log.error("GSM8K load failed: %s", exc)
            sys.exit(1)
    elif dataset == "bigcodebench":
        try:
            from bigcodebench.data import get_bigcodebench
            problems = get_bigcodebench(subset=subset)
            for tid, task in list(problems.items())[:limit]:
                tasks.append((tid, task.get("instruct_prompt", "")))
        except ImportError:
            log.error("bigcodebench not installed")
            sys.exit(1)
    elif dataset == "apps":
        try:
            from datasets import load_dataset
            ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
            for i, row in enumerate(ds):
                if limit and i >= limit:
                    break
                tasks.append((f"APPS/{i}", row.get("question", "")))
        except ImportError:
            log.error("datasets not installed")
            sys.exit(1)
    return tasks


async def _generate_one(client, model: str, task_id: str, prompt: str) -> dict | None:
    """Generate one topology via GPT-5.4 structured output."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Design an optimal agent topology for this task:\n\n{prompt[:2000]}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "topology",
                    "strict": True,
                    "schema": TOPOLOGY_SCHEMA,
                },
            },
            temperature=0.7,  # Diversity in topologies
            max_completion_tokens=2000,
            reasoning_effort="high",  # Maximum quality for SFT training data
        )
        content = response.choices[0].message.content
        topology = json.loads(content)

        # Validate structure
        if not topology.get("nodes") or len(topology["nodes"]) == 0:
            return None

        # Convert to YAML-like format for SFT training
        import yaml
        topology_yaml = yaml.dump(topology, default_flow_style=False)

        return {
            "task_id": task_id,
            "prompt": prompt[:500],
            "topology": topology,
            "topology_yaml": topology_yaml,
            "node_count": len(topology["nodes"]),
            "edge_count": len(topology.get("edges", [])),
            "difficulty": topology.get("difficulty", "moderate"),
            "reasoning": topology.get("reasoning", ""),
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        log.warning("[%s] Generation failed: %s", task_id, str(exc)[:100])
        return None


async def _generate_batch(client, model: str, tasks: list, workers: int) -> list[dict]:
    """Generate topologies in parallel batches."""
    import concurrent.futures

    results = []
    semaphore = asyncio.Semaphore(workers)

    async def _bounded_generate(tid, prompt):
        async with semaphore:
            return await asyncio.to_thread(
                lambda: asyncio.run(_generate_one(client, model, tid, prompt))
            )

    # Simple sequential for now (OpenAI rate limits)
    for i, (tid, prompt) in enumerate(tasks):
        if not prompt:
            continue
        result = await asyncio.to_thread(
            lambda t=tid, p=prompt: _generate_one_sync(client, model, t, p)
        )
        if result:
            results.append(result)
        if (i + 1) % 10 == 0:
            log.info("Progress: %d/%d done, %d valid topologies", i + 1, len(tasks), len(results))

    return results


def _generate_one_sync(client, model, task_id, prompt):
    """Generate one topology using the Responses API (for GPT-5.x Pro models)."""
    try:
        # Use Responses API (v1/responses) — required for gpt-5.x-pro models
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=f"Design an optimal agent topology for this task:\n\n{prompt[:2000]}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "topology",
                    "strict": True,
                    "schema": TOPOLOGY_SCHEMA,
                },
            },
            reasoning={"effort": "high"},
        )
        content = response.output_text
        topology = json.loads(content)

        if not topology.get("nodes") or len(topology["nodes"]) == 0:
            return None

        import yaml
        topology_yaml = yaml.dump(topology, default_flow_style=False)

        return {
            "task_id": task_id,
            "prompt": prompt[:500],
            "topology": topology,
            "topology_yaml": topology_yaml,
            "node_count": len(topology["nodes"]),
            "edge_count": len(topology.get("edges", [])),
            "difficulty": topology.get("difficulty", "moderate"),
            "reasoning": topology.get("reasoning", ""),
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except RuntimeError as exc:
        if "closed" in str(exc):
            log.warning("[%s] Client closed, will retry with fresh client", task_id)
        else:
            log.warning("[%s] RuntimeError: %s", task_id, str(exc)[:100])
        return "RETRY"
    except Exception as exc:
        log.warning("[%s] Failed: %s", task_id, str(exc)[:100])
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate topology SFT data via GPT-5.4")
    parser.add_argument("--dataset", choices=["bigcodebench", "apps"], default="bigcodebench")
    parser.add_argument("--subset", choices=["full", "hard"], default="hard")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", type=str, default="gpt-5.4-pro")
    parser.add_argument("--output", type=str, default="data/topology_sft_gpt54.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.list:
        if output_path.exists():
            count = sum(1 for _ in open(output_path, encoding="utf-8"))
            log.info("SFT data: %d entries in %s", count, output_path)
        else:
            log.info("No SFT data at %s", output_path)
        return

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env
        try:
            from dotenv import load_dotenv
            for p in [Path.cwd(), Path.cwd().parent]:
                env_file = p / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    api_key = os.environ.get("OPENAI_API_KEY")
                    break
        except ImportError:
            pass

    if not api_key:
        log.error("OPENAI_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    import httpx
    from sage.llm._ssl import ssl_verify
    import openai
    client = openai.OpenAI(
        api_key=api_key,
        http_client=httpx.Client(verify=ssl_verify(), timeout=180),
    )
    log.info("OpenAI client ready (model=%s)", args.model)

    # Load tasks
    tasks = _load_tasks(args.dataset, args.subset, args.limit)
    log.info("Loaded %d tasks from %s/%s", len(tasks), args.dataset, args.subset)

    # Generate topologies
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for i, (tid, prompt) in enumerate(tasks):
            if not prompt:
                continue

            result = _generate_one_sync(client, args.model, tid, prompt)
            # Retry with fresh client if connection closed
            if result == "RETRY":
                client = openai.OpenAI(
                    api_key=api_key,
                    http_client=httpx.Client(verify=ssl_verify(), timeout=180),
                )
                log.info("Recreated OpenAI client, retrying %s", tid)
                result = _generate_one_sync(client, args.model, tid, prompt)
            if result and result != "RETRY":
                f.write(json.dumps(result, default=str) + "\n")
                f.flush()
                count += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60
                log.info(
                    "Progress: %d/%d done, %d valid (%.0f tasks/min, est. %.0fmin remaining)",
                    i + 1, len(tasks), count, rate,
                    (len(tasks) - i - 1) / max(rate, 0.1),
                )

    elapsed = time.time() - t0
    log.info(
        "Done: %d valid topologies from %d tasks in %.0fs (%.1f tasks/min), saved to %s",
        count, len(tasks), elapsed, len(tasks) / max(elapsed, 1) * 60, output_path,
    )


if __name__ == "__main__":
    main()
