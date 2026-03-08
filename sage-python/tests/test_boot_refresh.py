import sys, types

# Ensure sage_core mock exists with a WorkingMemory class.
# Other test files may have already inserted a bare ModuleType for sage_core,
# so we must always patch WorkingMemory onto whatever module is present.
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
from unittest.mock import patch, MagicMock


def test_boot_creates_registry_in_mock_mode():
    """Mock mode should still create a ModelRegistry instance."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)

    assert system.registry is not None
    # Verify it's actually a ModelRegistry
    from sage.providers.registry import ModelRegistry
    assert isinstance(system.registry, ModelRegistry)


def test_boot_refresh_called_in_real_mode(monkeypatch):
    """Real mode should call registry.refresh() at boot."""
    # Set GOOGLE_API_KEY so auto-detect picks Google provider
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-fake")

    refresh_called = []

    async def mock_refresh(self):
        refresh_called.append(True)

    # Patch ModelRegistry.refresh to track calls
    monkeypatch.setattr(
        "sage.providers.registry.ModelRegistry.refresh",
        mock_refresh,
    )

    # Patch GoogleProvider to avoid real API calls
    mock_provider = MagicMock()
    mock_provider.return_value = MagicMock()
    monkeypatch.setattr("sage.llm.google.GoogleProvider", mock_provider)

    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=False, llm_tier="fast")

    assert len(refresh_called) == 1, "registry.refresh() should be called once at boot"
    assert system.registry is not None
    from sage.providers.registry import ModelRegistry
    assert isinstance(system.registry, ModelRegistry)


def test_boot_refresh_failure_does_not_crash(monkeypatch):
    """If registry.refresh() fails, boot should continue with a warning."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-fake")

    async def mock_refresh_fail(self):
        raise ConnectionError("API unreachable")

    monkeypatch.setattr(
        "sage.providers.registry.ModelRegistry.refresh",
        mock_refresh_fail,
    )

    mock_provider = MagicMock()
    mock_provider.return_value = MagicMock()
    monkeypatch.setattr("sage.llm.google.GoogleProvider", mock_provider)

    from sage.boot import boot_agent_system

    # Should not raise — just warn and continue
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast")
    assert system.registry is not None
    assert system.orchestrator is not None
