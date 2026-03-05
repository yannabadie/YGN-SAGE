import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.agent_pool import AgentPool, SubAgentSpec

def test_create_sub_agent_spec():
    spec = SubAgentSpec(
        name="researcher", role="research",
        system_prompt="You research topics.", tools=["web_search"],
    )
    assert spec.name == "researcher"
    assert spec.role == "research"

def test_pool_register_and_list():
    pool = AgentPool()
    spec = SubAgentSpec(name="coder", role="code", system_prompt="Write code.")
    pool.register(spec)
    agents = pool.list_agents()
    assert len(agents) == 1
    assert agents[0]["name"] == "coder"

def test_pool_deregister():
    pool = AgentPool()
    spec = SubAgentSpec(name="temp", role="temp", system_prompt="Temporary.")
    pool.register(spec)
    pool.deregister("temp")
    assert len(pool.list_agents()) == 0

def test_pool_ensemble_empty():
    pool = AgentPool()
    results = pool.collect_results()
    assert results == {}

def test_pool_store_and_collect_results():
    pool = AgentPool()
    spec = SubAgentSpec(name="worker", role="work", system_prompt="Work.")
    pool.register(spec)
    pool.mark_running("worker")
    assert pool.list_agents()[0]["running"] is True
    pool.store_result("worker", "task completed")
    results = pool.collect_results()
    assert results["worker"] == "task completed"
    assert pool.list_agents()[0]["running"] is False
