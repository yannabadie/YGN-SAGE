import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

_mock_core = sys.modules["sage_core"]

if not hasattr(_mock_core, "WorkingMemory"):

    class _MockMemoryEvent:
        def __init__(self, id, event_type, content, timestamp_str, is_summary=False):
            self.id = id
            self.event_type = event_type
            self.content = content
            self.timestamp_str = timestamp_str
            self.is_summary = is_summary

    class _MockWorkingMemory:
        def __init__(self, agent_id, parent_id=None):
            self.agent_id = agent_id
            self.parent_id = parent_id
            self._events = []
            self._counter = 0
            self._children = []

        def add_event(self, event_type, content):
            self._counter += 1
            eid = f"evt-{self._counter}"
            import time
            self._events.append(_MockMemoryEvent(
                id=eid, event_type=event_type, content=content,
                timestamp_str=str(time.time()),
            ))
            return eid

        def get_event(self, event_id):
            for e in self._events:
                if e.id == event_id:
                    return e
            return None

        def recent_events(self, n):
            return self._events[-n:] if n > 0 else []

        def event_count(self):
            return len(self._events)

        def add_child_agent(self, child_id):
            self._children.append(child_id)

        def child_agents(self):
            return list(self._children)

        def compress_old_events(self, keep_recent, summary):
            kept = self._events[-keep_recent:] if keep_recent > 0 else []
            self._events = [_MockMemoryEvent(
                id="summary-0", event_type="summary", content=summary,
                timestamp_str="0", is_summary=True,
            )] + kept

        def compact_to_arrow(self):
            return 0

        def compact_to_arrow_with_meta(self, keywords, embedding, parent_chunk_id):
            return 0

        def retrieve_relevant_chunks(self, active_chunk_id, max_hops, weights):
            return []

        def get_page_out_candidates(self, active_chunk_id, max_hops, budget):
            return []

        def smmu_chunk_count(self):
            return 0

        def get_latest_arrow_chunk(self):
            return None

    _mock_core.WorkingMemory = _MockWorkingMemory

import pytest
from sage.boot import boot_agent_system, AgentSystem

def test_boot_creates_system():
    system = boot_agent_system(use_mock_llm=True)
    assert isinstance(system, AgentSystem)
    assert system.agent_loop is not None
    assert system.agent_pool is not None
    assert system.metacognition is not None
    assert system.topology_evolver is not None

@pytest.mark.asyncio
async def test_full_cycle_with_mock():
    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("What is 2+2?")
    assert result is not None
    assert system.agent_loop.step_count > 0


def test_boot_registers_meta_tools():
    """Boot sequence registers meta tools (sandboxed — SEC-01/02 fixed)."""
    system = boot_agent_system(use_mock_llm=True)
    tool_names = system.tool_registry.list_tools()
    # SEC-01/SEC-02 fixed: tools now execute in subprocess sandbox
    assert "create_python_tool" in tool_names
    assert "create_bash_tool" in tool_names
    # Memory tools still registered
    assert "search_memory" in tool_names
    assert "store_memory" in tool_names


def test_boot_registers_all_memory_tools():
    """Boot sequence registers all 7 AgeMem memory tools."""
    system = boot_agent_system(use_mock_llm=True)
    tool_names = system.tool_registry.list_tools()
    for expected in [
        "retrieve_context", "summarize_context", "filter_context",
        "search_memory", "store_memory", "update_memory", "delete_memory",
    ]:
        assert expected in tool_names, f"Missing tool: {expected}"


def test_boot_wires_compressor_and_sandbox():
    """Boot sequence wires MemoryCompressor, EpisodicMemory, and SandboxManager."""
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.memory_compressor is not None
    assert system.agent_loop.episodic_memory is not None
    assert system.agent_loop.sandbox_manager is not None
