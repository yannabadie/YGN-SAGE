"""MASBENCH Traced Benchmark — full topology introspection per task.

Captures for each SAGE task:
- Topology template, nodes, edges, source
- Model assignment per node
- Output per node (full text)
- Controller decisions per node
- Quality score per node
- Latency per node
"""
import asyncio
import json
import logging
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.environ.get(
    "MASBENCH_OUTPUT_DIR",
    os.path.join(REPO_ROOT, "docs", "benchmarks", "masbench-runs"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "masbench_traced.log")
JSON_FILE = os.path.join(OUTPUT_DIR, "masbench_traced_results.json")
TRACES_FILE = os.path.join(OUTPUT_DIR, "masbench_traces.jsonl")

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(name)-25s] %(levelname)-5s %(message)s"))

logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(fh)
logging.root.addHandler(ch)

for mod in [
    "sage.pipeline", "sage.pipeline_stages", "sage.topology.runner",
    "sage.topology_controller", "sage.strategy.knn_router",
    "sage.llm.provider_pool", "sage.bench.masbench", "sage.boot",
]:
    logging.getLogger(mod).setLevel(logging.DEBUG)

log = logging.getLogger("masbench_traced")


async def run_sage_traced(system, question: str, task_idx: int) -> dict:
    """Run one task through SAGE with full topology trace."""
    trace = {
        "task_idx": task_idx,
        "question": question[:200],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t0 = time.perf_counter()

    # Access the pipeline directly for introspection
    pipeline = system.pipeline
    if not pipeline:
        # Fallback: no pipeline
        result = await system.run(question)
        trace["execution_path"] = "legacy"
        trace["result"] = result[:500]
        trace["latency_s"] = round(time.perf_counter() - t0, 2)
        return trace

    # Run the pipeline stages manually for introspection
    from sage.pipeline import PipelineContext
    from sage.constants import DEFAULT_BUDGET_USD

    ctx = PipelineContext(task=question, budget=DEFAULT_BUDGET_USD)
    trace["budget"] = ctx.budget

    # Stage 0: CLASSIFY
    ctx = pipeline._stage_classify(ctx)
    trace["system"] = ctx.system
    trace["domain"] = ctx.domain

    # Stage 1: DECOMPOSE
    ctx = await pipeline._stage_decompose(ctx)
    if ctx.dag_features:
        trace["dag_features"] = {
            "omega": ctx.dag_features.omega,
            "delta": ctx.dag_features.delta,
            "gamma": round(ctx.dag_features.gamma, 3),
        }

    # Stage 2: SELECT TOPOLOGY
    ctx = pipeline._stage_select_topology(ctx)
    if ctx.topology:
        topo = ctx.topology
        trace["topology"] = {
            "template": getattr(topo, "template_type", "unknown"),
            "node_count": topo.node_count() if hasattr(topo, "node_count") else 0,
            "edge_count": topo.edge_count() if hasattr(topo, "edge_count") else 0,
            "nodes": [],
            "edges": [],
        }
        # Capture each node
        for i in range(topo.node_count()):
            node = topo.get_node(i)
            trace["topology"]["nodes"].append({
                "idx": i,
                "role": getattr(node, "role", "?"),
                "model_id": getattr(node, "model_id", ""),
                "system": getattr(node, "system", 0),
                "max_cost_usd": getattr(node, "max_cost_usd", 0),
                "prompt": getattr(node, "prompt", "")[:100],
            })
        # Capture edges
        try:
            edges = topo.get_edges()
            for from_idx, to_idx, flow in edges:
                trace["topology"]["edges"].append({
                    "from": from_idx, "to": to_idx, "type": flow,
                })
        except (AttributeError, RuntimeError):
            pass
    else:
        trace["topology"] = None

    # Stage 3: ASSIGN MODELS
    ctx = pipeline._stage_assign_models(ctx)
    trace["assignments"] = dict(ctx.assignments)
    # Update trace with assigned model_ids
    if ctx.topology and "nodes" in (trace.get("topology") or {}):
        for i in range(ctx.topology.node_count()):
            node = ctx.topology.get_node(i)
            if i < len(trace["topology"]["nodes"]):
                trace["topology"]["nodes"][i]["assigned_model"] = getattr(node, "model_id", "")

    # Stage 4: EXECUTE with traces
    try:
        from sage.topology.runner import TopologyRunner
        from sage_core import TopologyExecutor

        if ctx.topology and ctx.topology.node_count() > 1:
            executor = TopologyExecutor(ctx.topology)
            runner = TopologyRunner(
                graph=ctx.topology,
                executor=executor,
                llm_provider=pipeline.llm_provider,
                llm_config=pipeline.llm_config,
                provider_pool=pipeline.provider_pool,
                controller=pipeline.controller,
                axis_hint=ctx.axis_hint,
            )
            node_traces = await runner.run_traced(question)
            trace["node_traces"] = []
            for nt in node_traces:
                trace["node_traces"].append({
                    "node_idx": nt.get("node_idx"),
                    "role": nt.get("role"),
                    "output": nt.get("output", "")[:300],
                    "output_len": len(nt.get("output", "")),
                    "latency_s": round(nt.get("latency", 0), 2),
                    "model_id": nt.get("model_id", ""),
                })
            ctx.result = node_traces[-1]["output"] if node_traces else ""
        else:
            # Single node or no topology
            result = await system.run(question)
            ctx.result = result
            trace["node_traces"] = [{"role": "single_agent", "output": result[:300], "output_len": len(result)}]
    except (ImportError, RuntimeError) as exc:
        result = await system.run(question)
        ctx.result = result
        trace["node_traces"] = [{"role": "fallback", "output": result[:300], "error": str(exc)[:100]}]

    trace["result"] = ctx.result[:500]
    trace["latency_s"] = round(time.perf_counter() - t0, 2)

    return trace


async def main():
    log.info("=" * 70)
    log.info("MASBENCH TRACED BENCHMARK — %s", time.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 70)

    from sage.boot import boot_agent_system
    from sage.bench.masbench import _load_masbench, _parse_task, _check_answer

    axis = "breadth"  # Start with one axis for debugging
    limit = 20  # Smaller for detailed analysis
    tasks = _load_masbench(axis, limit=limit)

    system = boot_agent_system()
    log.info("System booted: pipeline=%s, budget=%.1f",
             system.pipeline is not None,
             system.pipeline.run.__defaults__[0] if system.pipeline else 0)

    # Verify budget
    from sage.constants import DEFAULT_BUDGET_USD
    log.info("DEFAULT_BUDGET_USD = %.1f", DEFAULT_BUDGET_USD)

    all_traces = []
    bare_passed = 0
    sage_passed = 0

    # Open JSONL for streaming writes
    traces_f = open(TRACES_FILE, "w", encoding="utf-8")

    for i, item in enumerate(tasks):
        question, ground_truth = _parse_task(item, axis=axis)

        # Bare
        from sage.providers.connector import get_available_providers
        from sage.providers.openai_compat import OpenAICompatProvider
        from sage.llm.base import LLMConfig, Message, Role

        available = get_available_providers()
        cfg = available[0]
        provider = OpenAICompatProvider(
            api_key=os.environ.get(cfg["api_key_env"], ""),
            base_url=cfg["base_url"],
            provider_name=cfg["provider"],
        )
        bare_config = LLMConfig(provider=cfg["provider"], model=cfg.get("default_model", "deepseek-chat"), max_tokens=2048)
        bare_resp = await provider.generate(
            messages=[Message(role=Role.USER, content=question)],
            config=bare_config,
        )
        bare_ok = _check_answer(bare_resp.content or "", ground_truth, axis=axis)
        if bare_ok:
            bare_passed += 1

        # SAGE traced
        trace = await run_sage_traced(system, question, i)
        sage_ok = _check_answer(trace.get("result", ""), ground_truth, axis=axis)
        if sage_ok:
            sage_passed += 1

        trace["ground_truth"] = ground_truth
        trace["bare_passed"] = bare_ok
        trace["sage_passed"] = sage_ok
        all_traces.append(trace)

        # Stream to JSONL
        traces_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        traces_f.flush()

        log.info(
            "[%d/%d] bare=%s sage=%s gt=%s topo=%s nodes=%d latency=%.1fs",
            i + 1, limit,
            bare_ok, sage_ok,
            ground_truth[:20],
            trace.get("topology", {}).get("template", "none") if trace.get("topology") else "none",
            len(trace.get("node_traces", [])),
            trace.get("latency_s", 0),
        )

    traces_f.close()

    # Summary
    log.info("\n" + "=" * 70)
    log.info("RESULTS — axis=%s, %d tasks", axis, limit)
    log.info("Bare:  %d/%d = %.1f%%", bare_passed, limit, bare_passed / limit * 100)
    log.info("SAGE:  %d/%d = %.1f%%", sage_passed, limit, sage_passed / limit * 100)
    log.info("Delta: %+.1fpp", (sage_passed - bare_passed) / limit * 100)
    log.info("Traces: %s", TRACES_FILE)

    # Save full results
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "axis": axis,
            "limit": limit,
            "bare_pct": round(bare_passed / limit * 100, 1),
            "sage_pct": round(sage_passed / limit * 100, 1),
            "traces": all_traces,
        }, f, indent=2, ensure_ascii=False)
    log.info("Full results: %s", JSON_FILE)


if __name__ == "__main__":
    asyncio.run(main())
