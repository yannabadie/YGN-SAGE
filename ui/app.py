"""YGN-SAGE Control Dashboard -- FastAPI backend with WebSocket event streaming.

Wires the real AgentSystem (boot.py) so that POST /api/task runs the full
perceive->think->act->learn loop via Gemini (or Codex fallback).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure sage-python is importable when running from repo root
_sage_src = Path(__file__).resolve().parent.parent / "sage-python" / "src"
if str(_sage_src) not in sys.path:
    sys.path.insert(0, str(_sage_src))

# Mock sage_core (Rust extension) if not compiled — allows pure-Python operation
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

logger = logging.getLogger("ygn-sage.dashboard")

app = FastAPI(title="YGN-SAGE Control Dashboard")
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

LOG_FILE = Path("docs/plans/agent_stream.jsonl")

# Background agent task handle (so we can cancel on /api/stop)
_agent_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# In-memory state for dashboard polling.
# NOTE: Safe under single-worker uvicorn (asyncio is single-threaded).
# If running with multiple workers, guard with asyncio.Lock.
# ---------------------------------------------------------------------------
dashboard_state: dict = {
    "agent_status": "idle",        # idle | running | error | stopped
    "current_phase": None,         # perceive | think | act | learn
    "step_count": 0,
    "llm_calls": 0,
    "total_cost_usd": 0.0,
    "sub_agents": [],
    "evolution_stats": {
        "grid_size": 0,
        "best_fitness": 0.0,
        "generation": 0,
        "cells": [],               # list of {x, y, fitness} for heatmap
    },
    "memory_events": 0,
    "aio_ratio": 0.0,
    "metacognitive_system": 1,     # 1 or 3
    "z3_pass": 0,
    "z3_fail": 0,
    "inference_time_ms": 0.0,
    "wall_time_s": 0.0,
    "total_events": 0,
}


def _update_state_from_event(evt: dict) -> None:
    """Side-effect: mutate *dashboard_state* from a parsed JSONL event."""
    etype = evt.get("type", "").upper()
    meta = evt.get("meta", {})

    dashboard_state["total_events"] += 1

    # Phase tracking
    if etype in ("PERCEIVE", "THINK", "ACT", "LEARN"):
        dashboard_state["current_phase"] = etype.lower()
        dashboard_state["agent_status"] = "running"

    # Step count
    if "step" in evt:
        dashboard_state["step_count"] = max(
            dashboard_state["step_count"], evt["step"]
        )

    # AIO ratio
    if "aio_ratio" in meta:
        dashboard_state["aio_ratio"] = meta["aio_ratio"]

    # Memory events
    if "events" in meta:
        dashboard_state["memory_events"] = meta["events"]

    # LLM calls (each THINK event is an LLM call)
    if etype == "THINK" and "model" in meta:
        dashboard_state["llm_calls"] += 1

    # System routing
    if "system" in meta:
        dashboard_state["metacognitive_system"] = meta["system"]

    # Z3 verification
    if "r_path" in meta:
        if meta["r_path"] >= 0:
            dashboard_state["z3_pass"] += 1
        else:
            dashboard_state["z3_fail"] += 1

    # Completion
    if meta.get("result") == "complete":
        dashboard_state["agent_status"] = "idle"

    # Latency / cycle metrics (from HFT-style stream)
    if "latency_ms" in meta:
        dashboard_state["inference_time_ms"] = meta["latency_ms"]
    if "cycle" in meta:
        dashboard_state["step_count"] = max(
            dashboard_state["step_count"], meta["cycle"]
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the single-file dashboard."""
    with open("ui/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/state")
async def get_state():
    """Return the current dashboard state as JSON (polled by the frontend)."""
    return JSONResponse(dashboard_state)


@app.post("/api/task")
async def submit_task(request: Request):
    """Accept a new task and run it through the real AgentSystem."""
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

    dashboard_state["agent_status"] = "running"
    dashboard_state["current_phase"] = "perceive"

    async def _run_agent(task_text: str) -> None:
        try:
            from sage.boot import boot_agent_system
            system = boot_agent_system(use_mock_llm=False, llm_tier="auto")
            result = await system.run(task_text)
            dashboard_state["agent_status"] = "idle"
            dashboard_state["current_phase"] = None
            logger.info("Agent finished: %s", result[:200] if result else "(empty)")
        except Exception as exc:
            logger.exception("Agent run failed")
            dashboard_state["agent_status"] = "error"
            dashboard_state["current_phase"] = None

    _agent_task = asyncio.create_task(_run_agent(task))
    return JSONResponse({"status": "accepted", "task": task[:200]})


@app.post("/api/stop")
async def stop_agent():
    """Stop the currently running agent (cancel background task)."""
    global _agent_task
    if _agent_task and not _agent_task.done():
        _agent_task.cancel()
    _agent_task = None
    dashboard_state["agent_status"] = "stopped"
    dashboard_state["current_phase"] = None
    return JSONResponse({"status": "stopped"})


@app.post("/api/reset")
async def reset_state():
    """Reset all dashboard counters."""
    for key in (
        "step_count", "llm_calls", "memory_events",
        "z3_pass", "z3_fail", "total_events",
    ):
        dashboard_state[key] = 0
    dashboard_state["total_cost_usd"] = 0.0
    dashboard_state["aio_ratio"] = 0.0
    dashboard_state["inference_time_ms"] = 0.0
    dashboard_state["wall_time_s"] = 0.0
    dashboard_state["agent_status"] = "idle"
    dashboard_state["current_phase"] = None
    dashboard_state["evolution_stats"] = {
        "grid_size": 0, "best_fitness": 0.0,
        "generation": 0, "cells": [],
    }
    return JSONResponse({"status": "reset"})


@app.get("/api/providers")
async def list_providers():
    """Return available LLM providers and their status."""
    import shutil
    providers = []
    # Codex CLI
    codex_ok = shutil.which("codex") is not None
    providers.append({"name": "GPT-5.3 Codex", "tier": "codex", "available": codex_ok})
    providers.append({"name": "GPT-5.2", "tier": "codex_max", "available": codex_ok})
    # Google Gemini (needs API key)
    gemini_ok = bool(os.environ.get("GOOGLE_API_KEY"))
    providers.append({"name": "Gemini 3.1 Pro", "tier": "reasoner", "available": gemini_ok})
    providers.append({"name": "Gemini 3.1 Flash Lite", "tier": "fast", "available": gemini_ok})
    providers.append({"name": "Gemini 3 Flash", "tier": "mutator", "available": gemini_ok})
    providers.append({"name": "Gemini 2.5 Flash Lite", "tier": "budget", "available": gemini_ok})
    return JSONResponse(providers)


# ---------------------------------------------------------------------------
# WebSocket -- stream JSONL events in real-time
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1. Replay existing events (send last 200 lines to avoid overwhelming)
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Send only the tail for initial catch-up
        tail = lines[-200:] if len(lines) > 200 else lines
        for line in tail:
            stripped = line.strip()
            if stripped:
                await websocket.send_text(stripped)
                try:
                    _update_state_from_event(json.loads(stripped))
                except (json.JSONDecodeError, Exception):
                    pass

    # Use binary mode for tailing so byte offsets from stat() are consistent.
    last_pos = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0

    try:
        while True:
            if LOG_FILE.exists():
                current_size = LOG_FILE.stat().st_size
                # Handle file truncation / recreation
                if current_size < last_pos:
                    last_pos = 0
                if current_size > last_pos:
                    with open(LOG_FILE, "rb") as f:
                        f.seek(last_pos)
                        for raw_line in f:
                            stripped = raw_line.decode("utf-8", errors="replace").strip()
                            if stripped:
                                await websocket.send_text(stripped)
                                try:
                                    _update_state_from_event(json.loads(stripped))
                                except (json.JSONDecodeError, Exception):
                                    pass
                        last_pos = f.tell()
            await asyncio.sleep(0.1)
    except Exception:
        pass  # Client disconnected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting YGN-SAGE Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
