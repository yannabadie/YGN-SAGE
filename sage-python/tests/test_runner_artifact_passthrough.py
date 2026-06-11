"""F2 — TopologyRunner artifact pass-through (cgpro DESIGN_LOCKED
2026-06-11, conv cgpro_emission_fixes_design, sequence 3).

Side-effect rule under test: final-output override is allowed ONLY for
the verified patch profile (SAGE_TASK_ARTIFACT_PROFILE=unified_diff,
operator-set) PLUS a valid detected artifact. Detection itself is
universal and side-effect-free.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.topology.runner import TopologyRunner, _TopologyDoneEvent


class FakeNode:
    def __init__(self, role: str, model_id: str = "m", system: int = 1):
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities: list[str] = []


class FakeGraph:
    def __init__(self, nodes):
        self._nodes = nodes

    def node_count(self):
        return len(self._nodes)

    def get_node(self, idx):
        return self._nodes[idx]


class FakeExecutor:
    def __init__(self, order):
        self._batches = list(order)
        self._batch_idx = 0

    def next_ready(self, graph):
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx):
        pass

    def is_done(self):
        return self._batch_idx >= len(self._batches)


COMPLETE_DIFF = (
    "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n"
    "+b = 5\n c = 3\n"
)
TWO_HUNK_DIFF = (
    "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n"
    "+b = 5\n c = 3\n@@ -10,3 +10,3 @@\n x = 1\n-y = 2\n+y = 9\n z = 3\n"
)
DEGRADED_DIFF = (
    "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n"
    "+b = 5\n@@ -9,2 +9,2 @@"
)


def _runner(nodes=None) -> TopologyRunner:
    nodes = nodes or [FakeNode("coder"), FakeNode("synthesizer")]
    return TopologyRunner(
        FakeGraph(nodes), FakeExecutor([[i] for i in range(len(nodes))]),
        llm_provider=AsyncMock(),
    )


def test_patch_task_final_selects_prior_valid_diff_over_synthesizer_degraded_diff(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    r = _runner()
    r._capture_node_artifacts(0, "coder", "done\n```diff\n" + COMPLETE_DIFF + "```")
    r._capture_node_artifacts(
        1, "synthesizer",
        "Let me produce the patch based on the information I have\n"
        "```diff\n" + DEGRADED_DIFF + "\n```",
    )
    final, prov = r._select_final_output("synthesizer narration text")
    assert final.strip().endswith("c = 3")
    assert prov is not None and prov["node_idx"] == 0
    assert prov["parse_status"] == "complete"


def test_synthesizer_valid_diff_can_win_if_higher_score(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    r = _runner()
    r._capture_node_artifacts(0, "coder", "```diff\n" + COMPLETE_DIFF + "```")
    r._capture_node_artifacts(
        1, "synthesizer", "```diff\n" + TWO_HUNK_DIFF + "```"
    )
    final, prov = r._select_final_output("whatever")
    assert prov is not None and prov["node_idx"] == 1
    assert "y = 9" in final


def test_non_patch_task_does_not_override_final_with_illustrative_diff(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SAGE_TASK_ARTIFACT_PROFILE", raising=False)
    r = _runner()
    r._capture_node_artifacts(0, "coder", "```diff\n" + COMPLETE_DIFF + "```")
    final, prov = r._select_final_output("the real prose answer")
    assert final == "the real prose answer"
    assert prov is None


def test_artifact_payload_not_truncated(monkeypatch) -> None:
    """The prose channel truncates to the per-predecessor budget; the
    artifact channel carries the diff verbatim regardless."""
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    r = _runner()
    huge_prose = "x" * 200_000
    output = huge_prose + "\n```diff\n" + COMPLETE_DIFF + "```"
    r._node_outputs = {0: output}
    r._capture_node_artifacts(0, "coder", output)
    ctx = r._format_predecessor_context(1, [0])
    assert COMPLETE_DIFF in ctx  # verbatim, byte-for-byte
    assert "ARTIFACT" in ctx


def test_artifact_payload_not_similarity_deduped(monkeypatch) -> None:
    """Two predecessors with near-identical prose: dedup may drop prose,
    but BOTH artifacts survive in the artifact channel."""
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    nodes = [FakeNode("coder_a"), FakeNode("coder_b"), FakeNode("synth")]
    r = _runner(nodes)
    other_diff = COMPLETE_DIFF.replace("mod.py", "other.py")
    out_a = "identical analysis text\n```diff\n" + COMPLETE_DIFF + "```"
    out_b = "identical analysis text\n```diff\n" + other_diff + "```"
    r._node_outputs = {0: out_a, 1: out_b}
    r._capture_node_artifacts(0, "coder_a", out_a)
    r._capture_node_artifacts(1, "coder_b", out_b)
    ctx = r._format_predecessor_context(2, [0, 1])
    assert COMPLETE_DIFF in ctx
    assert other_diff in ctx


@pytest.mark.asyncio
async def test_topology_done_records_artifact_provenance(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    graph = FakeGraph([FakeNode("coder"), FakeNode("synthesizer")])
    executor = FakeExecutor([[0], [1]])
    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=[
        MagicMock(content="fix below\n```diff\n" + COMPLETE_DIFF + "```"),
        MagicMock(content="Great work everyone, summary only — no diff."),
    ])
    r = TopologyRunner(graph, executor, llm_provider=provider)
    done = None
    async for event in r._run_core("fix the bug"):
        if isinstance(event, _TopologyDoneEvent):
            done = event
    assert done is not None
    assert done.artifact_provenance is not None
    assert done.artifact_provenance["node_idx"] == 0
    assert done.final_output.strip().endswith("c = 3")
