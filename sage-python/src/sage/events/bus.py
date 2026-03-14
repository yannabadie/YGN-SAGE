"""EventBus — central in-process event dispatch system for YGN-SAGE.

Thread-safe event bus with:
- Synchronous subscriber callbacks
- Async stream() for WebSocket consumers
- Bounded ring buffer with configurable max size
- Phase-filtered queries
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from sage.agent_loop import AgentEvent

log = logging.getLogger(__name__)

DEFAULT_MAX_BUFFER = 5000


@dataclass(frozen=True)
class _AsyncConsumer:
    """Async stream consumer state bound to an owning event loop."""

    queue: asyncio.Queue[AgentEvent | None]
    loop: asyncio.AbstractEventLoop


class EventBus:
    """Central event bus for the YGN-SAGE agent framework.

    Args:
        max_buffer: Maximum number of events to retain in the ring buffer.
                    Oldest events are evicted when the limit is exceeded.
    """

    def __init__(self, max_buffer: int = DEFAULT_MAX_BUFFER) -> None:
        self.max_buffer = max_buffer
        self._buffer: deque[AgentEvent] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._subscribers: dict[str, Callable[[AgentEvent], Any]] = {}
        self._async_consumers: list[_AsyncConsumer] = []

    @staticmethod
    def _enqueue_async_event(q: asyncio.Queue[AgentEvent], event: AgentEvent) -> None:
        """Enqueue one event into an async consumer queue."""
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("EventBus: async queue full, dropping event")

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------
    def emit(self, event: AgentEvent) -> None:
        """Dispatch an event to all subscribers and append to buffer.

        Thread-safe. Subscriber exceptions are logged and swallowed.
        """
        with self._lock:
            self._buffer.append(event)
            subscribers = list(self._subscribers.values())
            consumers = list(self._async_consumers)

        # Dispatch outside the lock to avoid holding it during callbacks
        for cb in subscribers:
            try:
                cb(event)
            except Exception:
                log.exception("EventBus subscriber error")

        # Fan-out to async stream consumers
        stale_consumers: list[_AsyncConsumer] = []
        for consumer in consumers:
            try:
                # Queue/Future internals are loop-thread-affine.
                # Schedule queue.put_nowait() through the owning loop so
                # emit() remains safe when called from arbitrary producer threads.
                consumer.loop.call_soon_threadsafe(
                    self._enqueue_async_event,
                    consumer.queue,
                    event,
                )
            except RuntimeError:
                # Event loop may already be closed; prune stale consumer.
                stale_consumers.append(consumer)

        if stale_consumers:
            with self._lock:
                for consumer in stale_consumers:
                    try:
                        self._async_consumers.remove(consumer)
                    except ValueError:
                        pass

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------
    def subscribe(self, callback: Callable[[AgentEvent], Any]) -> str:
        """Register a synchronous callback. Returns a subscription ID."""
        sub_id = str(uuid.uuid4())
        with self._lock:
            self._subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription by ID. No-op if ID is unknown."""
        with self._lock:
            self._subscribers.pop(sub_id, None)

    # ------------------------------------------------------------------
    # Async Stream
    # ------------------------------------------------------------------
    async def stream(self) -> AsyncIterator[AgentEvent]:
        """Yield events as they arrive. For WebSocket consumers.

        Each call to stream() creates an independent consumer queue.
        The caller is responsible for breaking out of the loop when done.
        """
        q: asyncio.Queue[AgentEvent] = asyncio.Queue()
        consumer = _AsyncConsumer(q, asyncio.get_running_loop())
        with self._lock:
            self._async_consumers.append(consumer)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    if event is None:
                        break  # Sentinel from clear() — stop streaming
                    yield event
                except asyncio.TimeoutError:
                    continue  # No events for 30s — keep consumer alive
        finally:
            with self._lock:
                try:
                    self._async_consumers.remove(consumer)
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear all buffered events and async queues."""
        with self._lock:
            self._buffer.clear()
            # Signal consumers to stop before clearing
            for consumer in self._async_consumers:
                try:
                    consumer.queue.put_nowait(None)  # sentinel to stop consumer
                except Exception:
                    pass
            self._async_consumers.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        phase: str | None = None,
        last_n: int = 50,
    ) -> list[AgentEvent]:
        """Query buffered events.

        Args:
            phase: If set, filter to events where event.type == phase.
            last_n: Maximum number of events to return (most recent).

        Returns:
            List of matching events, oldest first.
        """
        with self._lock:
            events = list(self._buffer)

        if phase is not None:
            events = [e for e in events if e.type == phase]

        if last_n is not None and last_n < len(events):
            events = events[-last_n:]

        return events
