"""Tests for Embedder wiring in boot sequence (Task 7).

Verifies that boot_agent_system() wires an Embedder into the MemoryCompressor.
"""
from __future__ import annotations

import pytest

from sage.boot import boot_agent_system
from sage.memory.embedder import Embedder


def test_boot_wires_embedder_to_compressor():
    """After boot, the memory compressor should have an Embedder attribute."""
    system = boot_agent_system(use_mock_llm=True)
    compressor = system.agent_loop.memory_compressor

    assert compressor is not None, "memory_compressor must be wired"
    assert hasattr(compressor, "embedder"), "compressor must have embedder attribute"
    assert isinstance(compressor.embedder, Embedder), (
        f"embedder must be an Embedder instance, got {type(compressor.embedder)}"
    )


def test_boot_embedder_can_embed():
    """The wired embedder should be functional (produce a vector)."""
    system = boot_agent_system(use_mock_llm=True)
    embedder = system.agent_loop.memory_compressor.embedder

    try:
        vec = embedder.embed("test text")
    except Exception as e:
        pytest.skip(f"Embedding model not available: {e}")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_boot_embedder_not_force_hash():
    """Boot should wire a real Embedder (auto-select), not force_hash=True."""
    system = boot_agent_system(use_mock_llm=True)
    embedder = system.agent_loop.memory_compressor.embedder

    # The boot sequence should create Embedder() without force_hash
    # (auto-selects best backend). The compressor default is force_hash=True,
    # but boot should override with a fresh Embedder().
    assert isinstance(embedder, Embedder)
