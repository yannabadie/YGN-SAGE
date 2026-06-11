"""Element-web regression fixture — cgpro DESIGN integration acceptance
('Replay the element-web trace as a regression fixture before spending
money', conv cgpro_emission_fixes_design 2026-06-11).

Shape-faithful replay of the observed 4-node trace from
docs/benchmarks/2026-06-11-mini-2b-ab (payloads synthesized against a
local git fixture so apply-truth is real):

  node 0  -> '[sage: agent exited after 5 steps with no content]'
  node 1  -> analysis prose + raw git diff
  node 2  -> clean FENCED diff (the best artifact; applies)
  node 3  -> synthesizer narration ('the previous agent had the correct
             analysis... let me produce the patch') + REGENERATED diff
             that is structurally complete but does NOT apply (the
             count-mismatch class) — and this became final_result.

Two defense layers under test:
  L1 runner: artifact-aware final selection emits a complete artifact
     with provenance (the scorer cannot see count-correctness, so the
     regenerated diff may win here — that is WHY L2 exists);
  L2 bench:  when the final patch fails git apply --check, the rescue
     finds the node-2 patch that actually applies.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sage-python" / "scripts"))

import run_dryrun_arm_d as arm_d  # noqa: E402

from sage.topology.runner import TopologyRunner  # noqa: E402

SENTINEL = "[sage: agent exited after 5 steps with no content]"

GOOD_DIFF = (
    "--- a/src/views/Recording.tsx\n"
    "+++ b/src/views/Recording.tsx\n"
    "@@ -1,4 +1,7 @@\n"
    " const onGoLive = () => {\n"
    "+    if (state !== State.Idle) {\n"
    "+        return;\n"
    "+    }\n"
    "     store.start();\n"
    " };\n"
    " export default onGoLive;\n"
)
# Same change, REGENERATED with wrong hunk counts (claims 3 context
# lines that do not exist) — structurally complete, does not apply.
BAD_REGEN_DIFF = (
    "--- a/src/views/Recording.tsx\n"
    "+++ b/src/views/Recording.tsx\n"
    "@@ -1,9 +1,12 @@\n"
    " const onGoLive = () => {\n"
    "+    if (state !== State.Idle) {\n"
    "+        return;\n"
    "+    }\n"
    "     store.start();\n"
    " };\n"
    " export default onGoLive;\n"
    " // trailing context that is not in the file\n"
    " // second phantom line\n"
)

NODE_OUTPUTS = [
    SENTINEL,
    "Root cause analysis: the Go-live control lacks state validation.\n"
    "diff --git a/src/views/Recording.tsx b/src/views/Recording.tsx\n"
    + GOOD_DIFF,
    "```diff\n" + GOOD_DIFF + "```",
    "OK, I'm fully sandboxed. Let me produce the patch based on the "
    "information I have. The previous agent had the correct analysis and "
    "the source diff was complete.\n```diff\n" + BAD_REGEN_DIFF + "```",
]
ROLES = ["planner", "coder", "coder", "synthesizer"]


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "element-web"
    (repo / "src" / "views").mkdir(parents=True)
    (repo / "src" / "views" / "Recording.tsx").write_text(
        "const onGoLive = () => {\n    store.start();\n};\n"
        "export default onGoLive;\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(repo)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True,
                   capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c",
         "user.name=t", "commit", "-m", "init"],
        check=True, capture_output=True,
    )
    return repo


def test_l1_runner_selects_a_complete_artifact_with_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    from unittest.mock import AsyncMock

    class _G:
        def node_count(self):
            return 4

        def get_node(self, idx):
            class _N:
                role = ROLES[idx]
            return _N()

    class _E:
        def next_ready(self, g):
            return []

        def mark_completed(self, i):
            pass

        def is_done(self):
            return True

    r = TopologyRunner(_G(), _E(), llm_provider=AsyncMock())
    for idx, (role, out) in enumerate(zip(ROLES, NODE_OUTPUTS)):
        r._capture_node_artifacts(idx, role, out)
    final, prov = r._select_final_output(NODE_OUTPUTS[-1])
    assert prov is not None
    assert prov["parse_status"] == "complete"
    # The sentinel never wins; the selected payload is a real diff.
    assert "exited after" not in final
    assert "store.start();" in final


def test_l2_bench_rescue_recovers_the_applying_node_patch(tmp_path) -> None:
    repo = _fixture_repo(tmp_path)
    from sage.patch_artifacts import git_apply_check

    # Preconditions that make this fixture meaningful: the regenerated
    # final diff must NOT apply; the node-2 diff must apply.
    assert git_apply_check(BAD_REGEN_DIFF, str(repo))[0] is False
    assert git_apply_check(GOOD_DIFF, str(repo))[0] is True

    rescued, idx = arm_d._rescue_apply_failed_patch(
        BAD_REGEN_DIFF, NODE_OUTPUTS, str(repo)
    )
    assert rescued
    assert git_apply_check(rescued, str(repo))[0] is True
    assert idx == 2  # the clean fenced coder diff, scanned last-first


def test_l2_rescue_returns_empty_when_nothing_applies(tmp_path) -> None:
    repo = _fixture_repo(tmp_path)
    rescued, idx = arm_d._rescue_apply_failed_patch(
        BAD_REGEN_DIFF, [SENTINEL, "prose only"], str(repo)
    )
    assert rescued == "" and idx is None
