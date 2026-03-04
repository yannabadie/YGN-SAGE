import time
import ulid
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PythonMemoryEvent:
    id: str
    event_type: str
    content: str
    timestamp: datetime
    is_summary: bool

class PythonWorkingMemory:
    """Pure Python implementation of working memory for benchmarking.
    
    Adheres to S-MMU guidelines:
    - Uses ULIDs to prevent UUID heap fragmentation.
    - Implements Cognitive Sync Pulses (Slicing) during compaction.
    """
    def __init__(self, agent_id: str, parent_id: str | None = None):
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.events = []
        self.children = []
        self.sync_pulses = 0

    def add_event(self, event_type: str, content: str) -> str:
        # SOTA Mandate: Use ULID instead of UUID string allocs
        event_id = str(ulid.new())
        event = PythonMemoryEvent(
            id=event_id,
            event_type=event_type,
            content=content,
            timestamp=datetime.now(),
            is_summary=False
        )
        self.events.append(event)
        
        # S-MMU Data Minimization: Prevent silent failures from context overflow
        # If memory exceeds 5000 events, force a semantic slice (mocked here by keeping recent)
        if len(self.events) > 5000:
            self.events = self.events[-1000:]
            
        return event_id

    def event_count(self) -> int:
        return len(self.events)

    def recent_events(self, n: int):
        return self.events[-n:]

    def compact_to_arrow(self) -> int:
        self.sync_pulses += 1
        return self.sync_pulses

    def get_latest_arrow_chunk(self):
        import pyarrow as pa
        if not self.events:
            return None
        # S-MMU Cognitive Sync Pulse
        data = {
            "agent_id": [self.agent_id for _ in self.events],
            "parent_id": [self.parent_id for _ in self.events],
            "id": [e.id for e in self.events],
            "event_type": [e.event_type for e in self.events],
            "content": [e.content for e in self.events],
            "timestamp": [e.timestamp.isoformat() for e in self.events],
            "is_summary": [e.is_summary for e in self.events],
        }
        batch = pa.RecordBatch.from_pydict(data)
        self.events.clear()
        return batch

    @property
    def _events(self):
        return [{"type": e.event_type, "content": e.content} for e in self.events]

    def add_child_agent(self, child_id: str) -> None:
        self.children.append(child_id)

    def child_agents(self) -> list[str]:
        return self.children
