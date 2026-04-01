"""Learning, evolution, and consolidation helpers for AgentLoop.

Extracted from agent_loop.py _run_legacy() to reduce file size.
These are standalone functions that take the relevant objects as parameters
instead of relying on self.X attribute access.
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def compute_learn_meta(
    start_time: float,
    total_inference_time: float,
    total_cost_usd: float,
    working_memory: Any,
    agent_pool: Any,
    semantic_memory: Any,
    causal_memory: Any,
) -> dict[str, Any]:
    """Compute the in-loop LEARN metadata dict.

    Returns a dict with aio_ratio, events, wall_time_s, cost_usd,
    plus optional sub_agents, semantic_entities, causal_entities, causal_edges.
    """
    wall = time.perf_counter() - start_time
    aio = max(0.0, (wall - total_inference_time) / wall) if wall > 0 else 0.0

    learn_meta: dict[str, Any] = {
        "aio_ratio": aio,
        "events": working_memory.event_count(),
        "wall_time_s": round(wall, 1),
        "cost_usd": round(total_cost_usd, 4),
    }

    if agent_pool and hasattr(agent_pool, "list_agents"):
        learn_meta["sub_agents"] = agent_pool.list_agents()

    if semantic_memory:
        learn_meta["semantic_entities"] = semantic_memory.entity_count()

    if causal_memory:
        learn_meta["causal_entities"] = causal_memory.entity_count()
        learn_meta["causal_edges"] = len(causal_memory._causal_edges)

    return learn_meta


def collect_evolution_stats(
    learn_meta: dict[str, Any],
    auto_evolve: bool,
    topology_population: Any,
    cb_evo: Any,
) -> None:
    """Collect MAP-Elites evolution grid stats into learn_meta.

    Reads the topology_population grid and populates evo_cells,
    evo_best, and evo_grid_size in learn_meta.

    Mutates *learn_meta* in place.
    """
    if not auto_evolve or not topology_population:
        return
    if topology_population.size() <= 0 or cb_evo.should_skip():
        return
    try:
        cells = []
        best_fitness = 0.0
        for (x, y), (genome, score) in topology_population._grid.items():
            cells.append({"x": x, "y": y, "fitness": round(score, 2)})
            best_fitness = max(best_fitness, score)
        learn_meta["evo_cells"] = cells
        learn_meta["evo_best"] = round(best_fitness, 2)
        learn_meta["evo_grid_size"] = len(cells)
        cb_evo.record_success()
    except (RuntimeError, AttributeError) as e:
        cb_evo.record_failure(e)


def run_online_evolution(
    learn_meta: dict[str, Any],
    auto_evolve: bool,
    topology_engine: Any,
    cb_evo: Any,
) -> None:
    """SA-3: Online Evolution -- run evolve() when should_evolve() triggers.

    Checks topology_engine.should_evolve() and calls evolve() if triggered.
    Also collects archive cell count and coverage stats.

    Mutates *learn_meta* in place.
    """
    if not auto_evolve or not topology_engine or cb_evo.should_skip():
        return
    try:
        if hasattr(topology_engine, 'should_evolve') and topology_engine.should_evolve():
            from sage.constants import EVOLUTION_ONLINE_POP_SIZE, EVOLUTION_ONLINE_GENERATIONS
            topology_engine.evolve(
                pop_size=EVOLUTION_ONLINE_POP_SIZE,
                generations=EVOLUTION_ONLINE_GENERATIONS,
            )
            learn_meta["evo_online_run"] = True
        if hasattr(topology_engine, 'archive_cell_count'):
            learn_meta["evo_archive_cells"] = topology_engine.archive_cell_count()
            learn_meta["evo_archive_coverage"] = round(
                topology_engine.archive_coverage(), 3
            )
        cb_evo.record_success()
    except (ImportError, RuntimeError) as e:
        cb_evo.record_failure(e)


async def run_consolidation(
    learn_meta: dict[str, Any],
    consolidator: Any,
    step_count: int,
    consolidation_interval: int,
    skip_memory: bool,
) -> None:
    """Inter-tier consolidation: episodic -> semantic -> causal.

    Runs consolidator.consolidate() every consolidation_interval steps.
    Records processed count and entities_added in learn_meta.

    Mutates *learn_meta* in place.
    """
    if not consolidator or skip_memory:
        return
    if step_count % consolidation_interval != 0:
        return
    try:
        consolidation_result = await consolidator.consolidate()
        if consolidation_result.processed > 0:
            learn_meta["consolidation_processed"] = consolidation_result.processed
            learn_meta["consolidation_entities"] = consolidation_result.entities_added
    except (RuntimeError, AttributeError):
        pass  # Best-effort, never blocks the loop


async def record_outcome(
    emit_fn: Any,
    learn_meta: dict[str, Any],
    auto_evolve: bool,
    topology_population: Any,
    topology_engine: Any,
    consolidator: Any,
    step_count: int,
    consolidation_interval: int,
    skip_memory: bool,
    cb_evo: Any,
) -> None:
    """Record full learning outcome: evolution stats + online evolution + consolidation + emit.

    Convenience function that calls collect_evolution_stats, run_online_evolution,
    run_consolidation, and then emits the LEARN event.
    """
    from sage.agent_loop import LoopPhase

    collect_evolution_stats(learn_meta, auto_evolve, topology_population, cb_evo)
    run_online_evolution(learn_meta, auto_evolve, topology_engine, cb_evo)
    await run_consolidation(learn_meta, consolidator, step_count, consolidation_interval, skip_memory)
    emit_fn(LoopPhase.LEARN, **learn_meta)
