"""YGN-SAGE v2 Control Dashboard -- FastAPI backend with EventBus WebSocket push.

Replaces JSONL file polling with real-time EventBus streaming.
Backend boots AgentSystem lazily on first task submission.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path: ensure sage-python is importable
# ---------------------------------------------------------------------------
_sage_src = Path(__file__).resolve().parent.parent / "sage-python" / "src"
if str(_sage_src) not in sys.path:
    sys.path.insert(0, str(_sage_src))

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Mock sage_core (Rust extension) if not compiled -- pure-Python fallback
import types as _types
if "sage_core" not in sys.modules:
    _mock = _types.ModuleType("sage_core")

    class _MockEvent:
        def __init__(self, eid, etype, content):
            self.id = eid
            self.event_type = etype
            self.content = content
            self.timestamp_str = ""
            self.is_summary = False

    class _WM:
        def __init__(self, agent_id="", parent_id=None):
            self.agent_id = agent_id
            self.parent_id = parent_id
            self._events: list = []
            self._counter = 0

        def add_event(self, t, c):
            self._counter += 1
            eid = f"evt-{self._counter}"
            self._events.append(_MockEvent(eid, t, c))
            return eid

        def get_event(self, eid):
            return next((e for e in self._events if e.id == eid), None)

        def recent_events(self, n=10):
            return self._events[-n:]

        def event_count(self):
            return len(self._events)

        def add_child_agent(self, cid): pass
        def child_agents(self): return []
        def compress_old_events(self, k, s): self._events = self._events[-k:]
        def compact_to_arrow(self): return 0
        def compact_to_arrow_with_meta(self, kw, emb=None, parent=None): return 0
        def retrieve_relevant_chunks(self, cid, hops, w=None): return []
        def get_page_out_candidates(self, cid, hops, budget): return []
        def smmu_chunk_count(self): return 0
        def get_latest_arrow_chunk(self): return None

    _mock.WorkingMemory = _WM
    sys.modules["sage_core"] = _mock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from sage.events.bus import EventBus
from sage.agent_loop import AgentEvent

logger = logging.getLogger("ygn-sage.dashboard")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
event_bus = EventBus()
system = None  # lazily booted AgentSystem
_agent_task: asyncio.Task | None = None

app = FastAPI(title="YGN-SAGE v2 Control Dashboard")

# Mount static files (resolve path relative to this file, not cwd)
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_to_json(event: AgentEvent) -> str:
    """Serialize an AgentEvent to a JSON string."""
    d = asdict(event)
    return json.dumps(d, default=str)


def _boot_system() -> None:
    """Boot the AgentSystem (once), wiring the global event_bus."""
    global system
    if system is not None:
        return

    from sage.boot import boot_agent_system

    has_google = bool(os.environ.get("GOOGLE_API_KEY"))
    has_codex = shutil.which("codex") is not None

    if has_codex or has_google:
        system = boot_agent_system(
            use_mock_llm=False,
            llm_tier="auto",
            event_bus=event_bus,
        )
    else:
        logger.warning("No LLM provider available -- booting with mock LLM")
        system = boot_agent_system(
            use_mock_llm=True,
            event_bus=event_bus,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the single-file dashboard."""
    html_path = _static_dir / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/state")
async def get_state():
    """Return current dashboard stats derived from event_bus."""
    events = event_bus.query(last_n=5000)

    step_count = 0
    total_cost = 0.0
    last_system = 1
    last_model = None
    last_response = ""
    last_phase = None
    memory_events = 0
    llm_calls = 0
    z3_pass = 0
    z3_fail = 0
    last_latency = 0.0

    for evt in events:
        step_count = max(step_count, evt.step)
        if evt.cost_usd is not None:
            total_cost = max(total_cost, evt.cost_usd)
        if evt.system is not None:
            last_system = evt.system
        if evt.model is not None:
            last_model = evt.model
        if evt.latency_ms is not None:
            last_latency = evt.latency_ms

        etype = evt.type.upper()
        last_phase = etype.lower()

        if etype == "THINK" and evt.model:
            llm_calls += 1
        if etype == "THINK" and evt.meta.get("content"):
            last_response = evt.meta["content"]

        meta = evt.meta
        if "events" in meta:
            memory_events = meta["events"]
        if "r_path" in meta:
            if meta["r_path"] >= 0:
                z3_pass += 1
            else:
                z3_fail += 1
        if meta.get("response_text"):
            last_response = meta["response_text"]
        if meta.get("result") == "complete":
            last_phase = None

    return JSONResponse({
        "step_count": step_count,
        "total_cost_usd": round(total_cost, 4),
        "metacognitive_system": last_system,
        "model": last_model,
        "last_response": last_response[:2000] if last_response else "",
        "current_phase": last_phase,
        "memory_events": memory_events,
        "llm_calls": llm_calls,
        "z3_pass": z3_pass,
        "z3_fail": z3_fail,
        "latency_ms": round(last_latency, 1),
        "total_events": len(events),
        "agent_status": "running" if (_agent_task and not _agent_task.done()) else "idle",
    })


@app.post("/api/task")
async def submit_task(request: Request):
    """Accept a new task and run it through the AgentSystem in background."""
    global _agent_task

    body = await request.json()
    task = body.get("task", "")
    if not isinstance(task, str) or len(task) > 10_000:
        return JSONResponse(
            {"error": "task must be a string under 10,000 characters"},
            status_code=400,
        )

    if _agent_task and not _agent_task.done():
        return JSONResponse(
            {"error": "An agent is already running. Stop it first."},
            status_code=409,
        )

    async def _run_agent(task_text: str) -> None:
        try:
            _boot_system()
            result = await system.run(task_text)
            logger.info("Agent finished: %s", result[:200] if result else "(empty)")
        except asyncio.CancelledError:
            logger.info("Agent task cancelled")
        except Exception:
            logger.exception("Agent run failed")
            # Emit error event so the dashboard knows
            event_bus.emit(AgentEvent(
                type="ERROR",
                step=0,
                timestamp=time.time(),
                meta={"error": "Agent run failed"},
            ))

    _agent_task = asyncio.create_task(_run_agent(task))
    return JSONResponse({"status": "started"})


@app.post("/api/stop")
async def stop_agent():
    """Cancel the currently running agent task."""
    global _agent_task
    if _agent_task and not _agent_task.done():
        _agent_task.cancel()
    _agent_task = None
    return JSONResponse({"status": "stopped"})


@app.post("/api/reset")
async def reset_state():
    """Clear event bus buffer and reset system reference."""
    global system, _agent_task

    if _agent_task and not _agent_task.done():
        _agent_task.cancel()
    _agent_task = None

    # Clear the ring buffer (EventBus has no clear() -- replace it)
    event_bus._buffer.clear()
    event_bus._async_queues.clear()

    system = None
    return JSONResponse({"status": "reset"})


@app.get("/api/providers")
async def list_providers():
    """Return available LLM providers from ModelRegistry or fallback."""
    try:
        if system and hasattr(system, 'registry') and system.registry:
            available = system.registry.list_available()
            providers = [
                {"name": p.id, "provider": p.provider, "code": p.code_score,
                 "reasoning": p.reasoning_score, "cost": f"${p.cost_input:.3f}/${p.cost_output:.3f}",
                 "available": p.available}
                for p in available[:20]
            ]
            return JSONResponse(providers)
    except Exception:
        pass
    # Fallback to simple detection
    codex_ok = shutil.which("codex") is not None
    gemini_ok = bool(os.environ.get("GOOGLE_API_KEY"))
    return JSONResponse([
        {"name": "Gemini 3.1 Pro", "provider": "google", "available": gemini_ok},
        {"name": "GPT-5.3 Codex", "provider": "openai", "available": codex_ok},
    ])


@app.post("/api/benchmark")
async def run_benchmark(request: Request):
    """Launch a benchmark run in background."""
    body = await request.json()
    bench_type = body.get("type", "routing")
    limit = body.get("limit")

    async def _run_bench():
        try:
            if bench_type == "routing":
                from sage.strategy.metacognition import MetacognitiveController
                from sage.bench.routing import RoutingAccuracyBench
                mc = MetacognitiveController()
                bench = RoutingAccuracyBench(metacognition=mc)
                report = await bench.run()
            elif bench_type == "humaneval":
                from sage.bench.humaneval import HumanEvalBench
                _boot_system()
                bench = HumanEvalBench(system=system, event_bus=event_bus)
                report = await bench.run(limit=limit)
            else:
                return
            # Emit summary event
            event_bus.emit(AgentEvent(
                type="BENCH_SUMMARY",
                step=0,
                timestamp=time.time(),
                meta={
                    "benchmark": report.benchmark,
                    "pass_rate": report.pass_rate,
                    "total": report.total,
                    "passed": report.passed,
                    "avg_latency_ms": report.avg_latency_ms,
                    "routing_breakdown": report.routing_breakdown,
                },
            ))
        except Exception as e:
            logger.exception("Benchmark failed")
            event_bus.emit(AgentEvent(type="ERROR", step=0, timestamp=time.time(), meta={"error": str(e)}))

    asyncio.create_task(_run_bench())
    return JSONResponse({"status": "started", "type": bench_type})


@app.get("/api/memory/stats")
async def memory_stats():
    """Return 4-tier memory statistics."""
    stats = {"stm": 0, "episodic": 0, "semantic": 0, "exocortex": False}
    try:
        if system:
            loop = system.agent_loop
            stats["stm"] = loop.working_memory.event_count()
            if loop.episodic_memory:
                try:
                    stats["episodic"] = await loop.episodic_memory.count()
                except Exception:
                    pass
            if hasattr(loop, 'semantic_memory') and loop.semantic_memory:
                stats["semantic"] = loop.semantic_memory.entity_count()
            if loop.exocortex and loop.exocortex.is_available:
                stats["exocortex"] = True
    except Exception:
        pass
    return JSONResponse(stats)


@app.get("/api/topology")
async def get_topology():
    """Return active agent pool topology."""
    agents = []
    try:
        if system and hasattr(system.agent_loop, 'agent_pool'):
            pool = system.agent_loop.agent_pool
            if hasattr(pool, 'list_agents'):
                agents = pool.list_agents()
    except Exception:
        pass
    return JSONResponse({"agents": agents})


@app.get("/api/evolution")
async def get_evolution():
    """Return evolution engine state."""
    evo = {"generation": 0, "population_size": 0, "best_score": 0.0, "cells": []}
    try:
        if system and hasattr(system.agent_loop, 'topology_population'):
            pop = system.agent_loop.topology_population
            if pop and pop.size() > 0:
                evo["population_size"] = pop.size()
                try:
                    for (x, y), (genome, score) in pop._grid.items():
                        evo["cells"].append({"x": x, "y": y, "score": round(score, 2)})
                        evo["best_score"] = max(evo["best_score"], score)
                except Exception:
                    pass
    except Exception:
        pass
    return JSONResponse(evo)


# ---------------------------------------------------------------------------
# WebSocket -- push events from EventBus in real-time
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # 1. Send buffered events (initial state catch-up)
        buffered = event_bus.query(last_n=100)
        for evt in buffered:
            await websocket.send_text(_event_to_json(evt))

        # 2. Stream new events as they arrive
        async for evt in event_bus.stream():
            await websocket.send_text(_event_to_json(evt))

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception:
        logger.debug("WebSocket connection closed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting YGN-SAGE v2 Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
