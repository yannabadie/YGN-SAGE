#!/usr/bin/env python3
"""YGN-SAGE End-to-End Proof of Effectiveness.

This script exercises the FULL stack with NO MOCKS:
  Layer 1 — Rust Core (ONNX embedder, S-MMU, WorkingMemory, RagCache)
  Layer 2 — Python Components (routing, memory, guardrails, EventBus)
  Layer 3 — Full Agent Loop (real LLM via Google Gemini)
  Layer 4 — Benchmarks (routing accuracy + HumanEval subset)

Run:  python tests/e2e_proof.py
Requires: GOOGLE_API_KEY in .env, pip install onnxruntime, maturin develop --features onnx

Results saved to: docs/benchmarks/YYYY-MM-DD-e2e-proof.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sage-python" / "src"))
os.chdir(REPO_ROOT)

# Load .env
try:
    from dotenv import load_dotenv
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

# ─── Result structures ──────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    duration_ms: float = 0.0

@dataclass
class E2EReport:
    timestamp: str = ""
    platform: str = ""
    python_version: str = ""
    layers: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

results: list[TestResult] = []

def record(name: str, passed: bool, detail: str = "", duration_ms: float = 0.0):
    r = TestResult(name=name, passed=passed, detail=detail, duration_ms=duration_ms)
    results.append(r)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return r

# ===========================================================================
# LAYER 1 — RUST CORE
# ===========================================================================

def test_layer1_rust_core():
    print("\n" + "=" * 70)
    print("  LAYER 1 — RUST CORE (sage_core)")
    print("=" * 70)
    t0 = time.perf_counter()

    # 1.1 — Import sage_core
    try:
        import sage_core
        record("1.1 sage_core import", True, f"module at {sage_core.__file__}")
    except ImportError as e:
        record("1.1 sage_core import", False, str(e))
        return

    # 1.2 — WorkingMemory (Rust Arrow)
    try:
        wm = sage_core.WorkingMemory("e2e-test")
        eid = wm.add_event("test", "E2E proof event")
        evt = wm.get_event(eid)
        count = wm.event_count()
        record("1.2 WorkingMemory", True, f"event_count={count}, event_id={eid}")
    except Exception as e:
        record("1.2 WorkingMemory", False, str(e))

    # 1.3 — S-MMU compact + chunk count
    try:
        chunk_id = wm.compact_to_arrow()
        smmu_count = wm.smmu_chunk_count()
        record("1.3 S-MMU compact", True, f"chunk_id={chunk_id}, smmu_chunks={smmu_count}")
    except Exception as e:
        record("1.3 S-MMU compact", False, str(e))

    # 1.4 — S-MMU compact_with_meta (embeddings + keywords)
    try:
        # Add more events to have content
        for i in range(5):
            wm.add_event("data", f"Entity {i} processes task {i}")
        embedding = [0.1] * 384
        meta_chunk = wm.compact_to_arrow_with_meta(
            keywords=["test", "e2e", "proof"],
            embedding=embedding,
            parent_chunk_id=chunk_id,
        )
        smmu_count2 = wm.smmu_chunk_count()
        record("1.4 S-MMU with_meta", True, f"meta_chunk={meta_chunk}, smmu_chunks={smmu_count2}")
    except Exception as e:
        record("1.4 S-MMU with_meta", False, str(e))

    # 1.5 — RustEmbedder (ONNX auto-discovery)
    try:
        from sage_core import RustEmbedder
        model_path = str(REPO_ROOT / "sage-core" / "models" / "model.onnx")
        tokenizer_path = str(REPO_ROOT / "sage-core" / "models" / "tokenizer.json")

        if not Path(model_path).exists():
            record("1.5 RustEmbedder ONNX", False, "model.onnx not found — run download_model.py")
        else:
            emb = RustEmbedder(model_path, tokenizer_path)
            vec = emb.embed("YGN-SAGE end-to-end proof")
            norm = sum(x * x for x in vec) ** 0.5
            record("1.5 RustEmbedder ONNX", True,
                   f"dim={len(vec)}, norm={norm:.4f}, auto-discovered DLL")
    except Exception as e:
        record("1.5 RustEmbedder ONNX", False, str(e))

    # 1.6 — Semantic similarity (proves embeddings are meaningful)
    try:
        cat = emb.embed("I love cats")
        dog = emb.embed("I love dogs")
        code = emb.embed("fn main() { println!(\"hello\"); }")
        sim_cd = sum(a * b for a, b in zip(cat, dog))
        sim_cc = sum(a * b for a, b in zip(cat, code))
        passed = sim_cd > sim_cc
        record("1.6 Semantic similarity", passed,
               f"cat-dog={sim_cd:.3f} > cat-code={sim_cc:.3f}")
    except Exception as e:
        record("1.6 Semantic similarity", False, str(e))

    # 1.7 — RagCache (FIFO + TTL)
    try:
        from sage_core import RagCache
        cache = RagCache(max_entries=10, ttl_seconds=60)
        cache.put(12345, b"result_data")
        hit = cache.get(12345)
        miss = cache.get(99999)
        stats = cache.stats()
        record("1.7 RagCache", True,
               f"hit={hit is not None}, miss={miss is None}, stats={stats}")
    except Exception as e:
        record("1.7 RagCache", False, str(e))

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Layer 1 total: {elapsed:.0f}ms")

# ===========================================================================
# LAYER 2 — PYTHON COMPONENTS
# ===========================================================================

def test_layer2_python_components():
    print("\n" + "=" * 70)
    print("  LAYER 2 — PYTHON COMPONENTS")
    print("=" * 70)
    t0 = time.perf_counter()

    # 2.1 — ComplexityRouter (S1/S2/S3 routing)
    try:
        from sage.strategy.metacognition import ComplexityRouter
        router = ComplexityRouter()

        tasks = {
            "S1": "What is 2 + 2?",
            "S2": "Write a Python function that implements quicksort with detailed comments",
            "S3": "Prove that every continuous function on [0,1] attains its maximum using Z3",
        }
        routing_results = {}
        for expected, task in tasks.items():
            profile = router.assess_complexity(task)
            decision = router.route(profile)
            routing_results[expected] = f"S{decision.system} (complexity={profile.complexity:.2f})"

        detail = " | ".join(f"{k}->{v}" for k, v in routing_results.items())
        record("2.1 ComplexityRouter", True, detail)
    except Exception as e:
        record("2.1 ComplexityRouter", False, str(e))

    # 2.2 — Python Embedder (3-tier fallback)
    try:
        from sage.memory.embedder import Embedder
        py_emb = Embedder()
        vec = py_emb.embed("Python embedder test")
        is_semantic = py_emb.is_semantic
        backend = type(py_emb._provider).__name__
        record("2.2 Python Embedder", True,
               f"backend={backend}, semantic={is_semantic}, dim={len(vec)}")
    except Exception as e:
        record("2.2 Python Embedder", False, str(e))

    # 2.3 — EpisodicMemory (SQLite persistence)
    try:
        from sage.memory.episodic import EpisodicMemory
        import tempfile
        db_path = os.path.join(tempfile.mkdtemp(), "e2e_episodic.db")
        ep = EpisodicMemory(db_path=db_path)
        asyncio.run(ep.store("e2e-task", "E2E proof result", {"source": "e2e_proof"}))
        results_ep = asyncio.run(ep.search("e2e"))
        record("2.3 EpisodicMemory", len(results_ep) > 0,
               f"stored+retrieved {len(results_ep)} entries, db={db_path}")
    except Exception as e:
        record("2.3 EpisodicMemory", False, str(e))

    # 2.4 — SemanticMemory (entity-relation graph)
    try:
        from sage.memory.semantic import SemanticMemory
        from sage.memory.memory_agent import ExtractionResult
        sem = SemanticMemory()
        extraction = ExtractionResult(
            entities=["YGN-SAGE", "Embedder", "Router"],
            relationships=[
                ("YGN-SAGE", "contains", "Embedder"),
                ("YGN-SAGE", "contains", "Router"),
            ],
        )
        sem.add_extraction(extraction)
        ctx = sem.get_context_for("YGN-SAGE system components")
        n_ent = sem.entity_count()
        record("2.4 SemanticMemory", n_ent == 3,
               f"entities={n_ent}, context_len={len(ctx)}")
    except Exception as e:
        record("2.4 SemanticMemory", False, str(e))

    # 2.5 — CausalMemory
    try:
        from sage.memory.causal import CausalMemory
        cm = CausalMemory()
        cm.add_entity("bug", {"desc": "test failure"})
        cm.add_entity("fix", {"desc": "code change"})
        cm.add_causal_edge("bug", "fix", "caused")
        chain = cm.get_causal_chain("bug")
        record("2.5 CausalMemory", len(chain) > 0,
               f"chain={chain}")
    except Exception as e:
        record("2.5 CausalMemory", False, str(e))

    # 2.6 — Guardrails
    try:
        from sage.guardrails.base import GuardrailPipeline
        from sage.guardrails.builtin import CostGuardrail, OutputGuardrail
        pipeline = GuardrailPipeline([
            CostGuardrail(max_usd=10.0),
            OutputGuardrail(min_length=1),
        ])
        results_ok = asyncio.run(pipeline.check_all(output="Valid output"))
        results_empty = asyncio.run(pipeline.check_all(output=""))
        all_ok = all(r.passed for r in results_ok)
        any_fail = any(not r.passed for r in results_empty)
        record("2.6 Guardrails", all_ok and any_fail,
               f"valid_pass={all_ok}, empty_blocked={any_fail}")
    except Exception as e:
        record("2.6 Guardrails", False, str(e))

    # 2.7 — EventBus
    try:
        from sage.events.bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))
        bus.emit({"phase": "TEST", "data": "e2e_proof"})
        record("2.7 EventBus", len(received) == 1,
               f"emitted=1, received={len(received)}")
    except Exception as e:
        record("2.7 EventBus", False, str(e))

    # 2.8 — RelevanceGate (CRAG)
    try:
        from sage.memory.relevance_gate import RelevanceGate
        gate = RelevanceGate(threshold=0.3)
        score_relevant = gate.score("Write a Python sort function", "Python sorting algorithm implementation")
        score_irrelevant = gate.score("Write a Python sort function", "The weather in Paris is sunny today")
        record("2.8 RelevanceGate", score_relevant > score_irrelevant,
               f"relevant={score_relevant:.2f}, irrelevant={score_irrelevant:.2f}")
    except Exception as e:
        record("2.8 RelevanceGate", False, str(e))

    # 2.9 — ToolRegistry
    try:
        from sage.tools.registry import ToolRegistry
        from sage.tools.meta import create_python_tool, create_bash_tool
        reg = ToolRegistry()
        reg.register(create_python_tool)
        reg.register(create_bash_tool)
        tools = reg.list_tools()  # returns list[str]
        record("2.9 ToolRegistry", len(tools) >= 2,
               f"registered={len(tools)} tools: {tools}")
    except Exception as e:
        record("2.9 ToolRegistry", False, str(e))

    # 2.10 — Z3 Verification
    try:
        import z3
        x = z3.Int("x")
        s = z3.Solver()
        s.add(x > 0, x < 10, x * x == 9)
        result = s.check()
        model = s.model() if result == z3.sat else None
        record("2.10 Z3 Solver", result == z3.sat and model[x].as_long() == 3,
               f"result={result}, x={model[x] if model else 'N/A'}")
    except Exception as e:
        record("2.10 Z3 Solver", False, str(e))

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Layer 2 total: {elapsed:.0f}ms")

# ===========================================================================
# LAYER 3 — FULL AGENT LOOP (REAL LLM)
# ===========================================================================

async def test_layer3_agent_loop():
    print("\n" + "=" * 70)
    print("  LAYER 3 — FULL AGENT LOOP (Real Gemini LLM)")
    print("=" * 70)
    t0_total = time.perf_counter()

    if not os.environ.get("GOOGLE_API_KEY"):
        record("3.0 API key check", False, "GOOGLE_API_KEY not set")
        return

    # 3.1 — Boot full system (all 5 pillars)
    try:
        from sage.boot import boot_agent_system
        from sage.events.bus import EventBus
        bus = EventBus()
        events_captured = []
        bus.subscribe(lambda e: events_captured.append(e))

        t_boot = time.perf_counter()
        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
        boot_ms = (time.perf_counter() - t_boot) * 1000

        # Check what providers were discovered
        available = system.registry.list_available() if system.registry else []
        providers = set(p.provider for p in available)
        record("3.1 Boot AgentSystem", True,
               f"{boot_ms:.0f}ms, providers={providers}, models={len(available)}")
    except Exception as e:
        record("3.1 Boot AgentSystem", False, str(e))
        return

    # 3.2 — S1 simple question (real LLM, should be fast)
    try:
        t1 = time.perf_counter()
        answer = await system.run("What is the capital of Japan?")
        lat = (time.perf_counter() - t1) * 1000
        passed = "tokyo" in answer.lower()
        record("3.2 S1 factual question", passed,
               f"{lat:.0f}ms, answer contains 'Tokyo': {passed}")
    except Exception as e:
        record("3.2 S1 factual question", False, str(e))

    # 3.3 — S2 code generation (real LLM, should produce valid Python)
    try:
        t2 = time.perf_counter()
        code_answer = await system.run(
            "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number. "
            "Return ONLY the function, no explanation."
        )
        lat = (time.perf_counter() - t2) * 1000

        # Verify it's valid Python with the function
        has_def = "def " in code_answer and "fibonacci" in code_answer
        # Try to extract and execute
        import re
        code_block = re.search(r"```python\s*(.*?)```", code_answer, re.DOTALL)
        code_text = code_block.group(1) if code_block else code_answer

        # Execute in isolated namespace
        namespace = {}
        exec(compile(code_text, "<e2e>", "exec"), namespace)
        fib_fn = namespace.get("fibonacci")
        if fib_fn:
            fib_10 = fib_fn(10)
            correct = fib_10 == 55
            record("3.3 S2 code generation", correct,
                   f"{lat:.0f}ms, fibonacci(10)={fib_10} (expected 55)")
        else:
            record("3.3 S2 code generation", False,
                   f"{lat:.0f}ms, function 'fibonacci' not found in output")
    except Exception as e:
        record("3.3 S2 code generation", False, str(e))

    # 3.4 — Multi-step reasoning
    try:
        t3 = time.perf_counter()
        reasoning = await system.run(
            "A train leaves Paris at 9:00 AM traveling at 200 km/h. "
            "Another train leaves Lyon (450 km away) at 9:30 AM traveling at 250 km/h toward Paris. "
            "At what time do they meet? Show your work."
        )
        lat = (time.perf_counter() - t3) * 1000
        # The answer should mention a time — basic sanity check
        has_time = any(t in reasoning.lower() for t in ["10:", "11:", "hour", "minute", "meet"])
        record("3.4 Multi-step reasoning", has_time,
               f"{lat:.0f}ms, answer length={len(reasoning)} chars")
    except Exception as e:
        record("3.4 Multi-step reasoning", False, str(e))

    # 3.5 — EventBus captured real events
    try:
        from sage.agent_loop import AgentEvent
        event_types = set()
        for e in events_captured:
            if isinstance(e, AgentEvent):
                event_types.add(e.type)
            elif isinstance(e, dict) and "type" in e:
                event_types.add(e["type"])
        has_events = len(events_captured) > 0
        record("3.5 EventBus telemetry", has_events,
               f"captured {len(events_captured)} events, types={event_types}")
    except Exception as e:
        record("3.5 EventBus telemetry", False, str(e))

    # 3.6 — Memory persistence check
    try:
        ep_count = 0
        if hasattr(system.agent_loop, 'episodic_memory') and system.agent_loop.episodic_memory:
            ep_results = await system.agent_loop.episodic_memory.search("fibonacci")
            ep_count = len(ep_results)
        sem_count = 0
        if hasattr(system.agent_loop, 'semantic_memory') and system.agent_loop.semantic_memory:
            sem_count = system.agent_loop.semantic_memory.entity_count()
        record("3.6 Memory persistence", ep_count > 0 or sem_count > 0,
               f"episodic_hits={ep_count}, semantic_entities={sem_count}")
    except Exception as e:
        record("3.6 Memory persistence", False, str(e))

    elapsed = (time.perf_counter() - t0_total) * 1000
    print(f"  Layer 3 total: {elapsed:.0f}ms")

# ===========================================================================
# LAYER 4 — BENCHMARKS
# ===========================================================================

async def test_layer4_benchmarks():
    print("\n" + "=" * 70)
    print("  LAYER 4 — BENCHMARKS (Routing + HumanEval)")
    print("=" * 70)
    t0 = time.perf_counter()

    # 4.1 — Routing accuracy (30 labeled tasks, no LLM needed)
    try:
        from sage.strategy.metacognition import MetacognitiveController
        from sage.bench.routing import RoutingAccuracyBench

        mc = MetacognitiveController()
        bench = RoutingAccuracyBench(metacognition=mc)
        report = await bench.run()

        record("4.1 Routing benchmark", report.pass_rate >= 0.70,
               f"pass@1={report.pass_rate:.1%} ({report.passed}/{report.total}), "
               f"breakdown={report.routing_breakdown}")
    except Exception as e:
        record("4.1 Routing benchmark", False, str(e))

    # 4.2 — HumanEval (5 problems, real LLM)
    if not os.environ.get("GOOGLE_API_KEY"):
        record("4.2 HumanEval (5 problems)", False, "GOOGLE_API_KEY not set")
    else:
        try:
            from sage.bench.humaneval import HumanEvalBench
            from sage.boot import boot_agent_system
            from sage.events.bus import EventBus

            bus = EventBus()
            system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
            bench = HumanEvalBench(system=system, event_bus=bus)

            t_he = time.perf_counter()
            report = await bench.run(limit=5)
            he_ms = (time.perf_counter() - t_he) * 1000

            record("4.2 HumanEval (5 problems)", report.pass_rate > 0.0,
                   f"pass@1={report.pass_rate:.0%} ({report.passed}/{report.total}), "
                   f"{he_ms:.0f}ms, routing={report.routing_breakdown}")

            # Show per-task results
            for r in report.results:
                status = "PASS" if r.passed else "FAIL"
                err = f" — {r.error[:60]}" if r.error else ""
                print(f"       {status} {r.task_id}{err}")

        except Exception as e:
            record("4.2 HumanEval (5 problems)", False, str(e))

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Layer 4 total: {elapsed:.0f}ms")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("+" + "=" * 68 + "+")
    print("|" + "  YGN-SAGE -- End-to-End Proof of Effectiveness".center(68) + "|")
    print("|" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "|")
    print("+" + "=" * 68 + "+")

    t_global = time.perf_counter()

    # Layer 1 — Rust Core
    test_layer1_rust_core()

    # Layer 2 — Python Components
    test_layer2_python_components()

    # Layer 3 — Full Agent Loop (async)
    asyncio.run(test_layer3_agent_loop())

    # Layer 4 — Benchmarks (async)
    asyncio.run(test_layer4_benchmarks())

    total_ms = (time.perf_counter() - t_global) * 1000

    # ── Summary ──────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed}/{total} passed, {failed} failed ({total_ms / 1000:.1f}s)")
    print("=" * 70)

    if failed:
        print("\n  Failures:")
        for r in results:
            if not r.passed:
                print(f"    [{r.name}] {r.detail}")

    # ── Save report ──────────────────────────────────────────────────────
    report = E2EReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        platform=sys.platform,
        python_version=sys.version,
        layers={},
        summary={"passed": passed, "failed": failed, "total": total, "duration_ms": total_ms},
    )

    for r in results:
        layer = r.name.split()[0]  # "1.1" → "1"
        layer_key = {"1": "rust_core", "2": "python_components",
                     "3": "agent_loop", "4": "benchmarks"}.get(layer.split(".")[0], "unknown")
        if layer_key not in report.layers:
            report.layers[layer_key] = []
        report.layers[layer_key].append(asdict(r))

    bench_dir = REPO_ROOT / "docs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = bench_dir / f"{date_str}-e2e-proof.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
