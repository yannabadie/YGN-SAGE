"""Full integration tests: Rust TopologyEngine + Python runtime.

Comprehensive tests for the Phase 6 wiring of TopologyEngine,
ContextualBandit, LLM synthesis, TopologyExecutor, and EventBus
integration. Tests work both with and without sage_core compiled.
"""
import pytest


@pytest.fixture
def system():
    """Boot a mock agent system."""
    from sage.boot import boot_agent_system
    return boot_agent_system(use_mock_llm=True)


def test_full_system_has_all_phase6_components(system):
    """Verify all Phase 6 components are wired."""
    assert hasattr(system, "topology_engine")
    assert hasattr(system, "bandit")
    assert hasattr(system.agent_loop, "topology_engine")
    # Phase 2 components should also be present
    assert hasattr(system, "template_store")
    assert hasattr(system, "verifier")


@pytest.mark.asyncio
async def test_full_cycle_generate_execute_record(system):
    """Full cycle: generate topology -> execute -> record outcome."""
    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    result = await system.run("Write a Python function that reverses a string")
    assert isinstance(result, str)

    # Verify topology was generated and cached
    assert system.topology_engine.topology_count() >= 1

    # Verify outcome was recorded (S-MMU should have chunks)
    assert system.topology_engine.smmu_chunk_count() >= 1


@pytest.mark.asyncio
async def test_multiple_runs_build_archive(system):
    """Multiple runs should populate the MAP-Elites archive."""
    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    tasks = [
        "What is 2+2?",
        "Write a sorting algorithm",
        "Explain quantum computing",
        "Debug this code: def f(): return None",
        "Prove that sqrt(2) is irrational",
    ]
    for task in tasks:
        await system.run(task)

    # Archive should have entries after 5 runs
    assert system.topology_engine.archive_cell_count() >= 1
    # S-MMU should have multiple chunks
    assert system.topology_engine.smmu_chunk_count() >= len(tasks)
    # Topologies should be cached (some tasks may share a template,
    # so count may be less than len(tasks), but at least 1 per system tier)
    assert system.topology_engine.topology_count() >= 1


@pytest.mark.asyncio
async def test_graceful_degradation_no_engine():
    """System works normally when topology_engine is None."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    system.topology_engine = None
    system.bandit = None

    result = await system.run("Hello")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_events_include_topology(system):
    """EventBus should have TOPOLOGY events after a run."""
    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    events = []
    system.event_bus.subscribe(lambda e: events.append(e))

    await system.run("Calculate 5 factorial")

    topo_events = [e for e in events if getattr(e, "type", "") == "TOPOLOGY"]
    assert len(topo_events) >= 1

    meta = topo_events[0].meta
    assert "topology_source" in meta
    assert "topology_confidence" in meta
    assert "topology_template" in meta
    assert "topology_id" in meta
    assert "topology_nodes" in meta


def test_topology_executor_from_python():
    """TopologyExecutor can be used from Python for node scheduling."""
    try:
        from sage_core import TopologyEngine, TopologyExecutor
    except ImportError:
        pytest.skip("sage_core not compiled")

    engine = TopologyEngine()
    result = engine.generate("test task", system=2)
    graph = result.topology

    executor = TopologyExecutor(graph)
    assert executor.mode() in ("static", "dynamic")
    assert not executor.is_done()

    # Execute all nodes
    while not executor.is_done():
        ready = executor.next_ready(graph)
        if not ready:
            break
        for idx in ready:
            executor.mark_running(idx)
            executor.mark_completed(idx)

    assert executor.is_done()


def test_bandit_learning_loop():
    """Bandit posteriors should shift after repeated observations."""
    try:
        from sage_core import ContextualBandit
    except ImportError:
        pytest.skip("sage_core not compiled")

    bandit = ContextualBandit(0.995, 0.1)
    bandit.register_arm("fast-model", "sequential")
    bandit.register_arm("slow-model", "avr")

    # Record high quality for fast-model
    for _ in range(20):
        decision = bandit.select(0.0)  # pure exploit
        if decision.model_id == "fast-model":
            bandit.record(decision.decision_id, 0.95, 0.001, 50.0)
        else:
            bandit.record(decision.decision_id, 0.3, 0.01, 500.0)

    # After training, fast-model should be preferred in exploit mode
    exploit_choices = [bandit.select(0.0).model_id for _ in range(10)]
    fast_count = exploit_choices.count("fast-model")
    assert fast_count >= 5, f"Expected fast-model preferred, got {fast_count}/10"


def test_llm_caller_prompt_generation():
    """LLM caller generates valid prompts."""
    from sage.topology.llm_caller import build_role_prompt, build_structure_prompt

    role_prompt = build_role_prompt(
        "Analyze sentiment in tweets",
        max_agents=3,
        available_models=["gemini-2.5-flash"],
    )
    assert "sentiment" in role_prompt.lower()
    assert "3" in role_prompt

    structure_prompt = build_structure_prompt('{"roles": [{"name": "a"}]}')
    assert "adjacency" in structure_prompt.lower()
    assert "template" in structure_prompt.lower()


def test_llm_caller_build_topology():
    """LLM caller builds valid TopologyGraph from JSON."""
    try:
        from sage_core import TopologyGraph  # noqa: F401
    except ImportError:
        pytest.skip("sage_core not compiled")

    import json
    from sage.topology.llm_caller import parse_and_build_topology

    roles = json.dumps({
        "roles": [
            {"name": "analyzer", "model": "gemini-2.5-flash", "system": 2, "capabilities": ["analysis"]},
            {"name": "synthesizer", "model": "gemini-2.5-flash", "system": 2, "capabilities": ["synthesis"]},
            {"name": "reviewer", "model": "gemini-3.1-pro-preview", "system": 2, "capabilities": ["review"]},
        ]
    })
    structure = json.dumps({
        "adjacency": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "edge_types": [["", "control", ""], ["", "", "control"], ["", "", ""]],
        "template": "sequential",
    })

    graph = parse_and_build_topology(roles, structure)
    assert graph is not None
    assert graph.node_count() == 3
    assert graph.edge_count() == 2
