"""Memory context injection helpers for AgentLoop.

Extracted from agent_loop.py _run_legacy() to reduce file size.
These are standalone functions that take memory objects as parameters
instead of relying on self.X attribute access.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import Message, Role

log = logging.getLogger(__name__)


def inject_semantic_memory(
    messages: list[Message],
    task: str,
    semantic_memory: Any,
    relevance_gate: Any,
    cb_semantic: Any,
    skip_memory: bool,
) -> None:
    """Inject semantic memory context into messages (one-time, before loop).

    Inserts a SYSTEM message with relevant entity knowledge after the
    first system prompt message.  No-op if semantic_memory is None,
    circuit breaker is tripped, or skip_memory is set.

    Mutates *messages* in place.
    """
    if not semantic_memory or cb_semantic.should_skip() or skip_memory:
        return
    try:
        sem_context = semantic_memory.get_context_for(task)
        if sem_context and relevance_gate.is_relevant(task, sem_context):
            messages.insert(1, Message(
                role=Role.SYSTEM,
                content=f"Relevant knowledge from previous interactions:\n{sem_context}",
            ))
        cb_semantic.record_success()
    except (RuntimeError, AttributeError) as e:
        cb_semantic.record_failure(e)


def inject_causal_memory(
    messages: list[Message],
    task: str,
    causal_memory: Any,
    relevance_gate: Any,
    cb_causal: Any,
    skip_memory: bool,
) -> None:
    """Inject causal memory context (directed cause-effect chains).

    Inserts a SYSTEM message after semantic context (position 2 or
    end of messages if shorter).  No-op if causal_memory is None,
    circuit breaker is tripped, or skip_memory is set.

    Mutates *messages* in place.
    """
    if not causal_memory or cb_causal.should_skip() or skip_memory:
        return
    try:
        causal_context = causal_memory.get_context_for(task)
        if causal_context and relevance_gate.is_relevant(task, causal_context):
            messages.insert(
                min(2, len(messages)),
                Message(
                    role=Role.SYSTEM,
                    content=f"Causal relationships from previous interactions:\n{causal_context}",
                ),
            )
        cb_causal.record_success()
    except (RuntimeError, AttributeError) as e:
        cb_causal.record_failure(e)


def inject_smmu_context(
    messages: list[Message],
    task: str,
    working_memory: Any,
    relevance_gate: Any,
    cb_smmu: Any,
    skip_memory: bool,
) -> None:
    """Inject S-MMU context (graph-based retrieval from compacted chunks).

    Inserts a SYSTEM message at position min(2, len(messages)).
    No-op if circuit breaker is tripped or skip_memory is set.

    Mutates *messages* in place.
    """
    if cb_smmu.should_skip() or skip_memory:
        return
    try:
        from sage.memory.smmu_context import retrieve_smmu_context
        smmu_context = retrieve_smmu_context(working_memory)
        if smmu_context and relevance_gate.is_relevant(task, smmu_context):
            messages.insert(
                min(2, len(messages)),
                Message(role=Role.SYSTEM, content=smmu_context),
            )
        cb_smmu.record_success()
    except (ImportError, RuntimeError, AttributeError) as e:
        cb_smmu.record_failure(e)


def inject_memory_context(
    messages: list[Message],
    task: str,
    semantic_memory: Any,
    causal_memory: Any,
    working_memory: Any,
    relevance_gate: Any,
    cb_semantic: Any,
    cb_causal: Any,
    cb_smmu: Any,
    skip_memory: bool,
) -> None:
    """Inject all memory contexts into messages (semantic, causal, S-MMU).

    Convenience wrapper that calls all three injection functions in order.
    Mutates *messages* in place.
    """
    inject_semantic_memory(
        messages, task, semantic_memory, relevance_gate, cb_semantic, skip_memory,
    )
    inject_causal_memory(
        messages, task, causal_memory, relevance_gate, cb_causal, skip_memory,
    )
    inject_smmu_context(
        messages, task, working_memory, relevance_gate, cb_smmu, skip_memory,
    )
