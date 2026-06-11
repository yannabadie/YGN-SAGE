"""Shared unified-diff artifact extractor — cgpro DESIGN_LOCKED test
contract (2026-06-11, conv cgpro_emission_fixes_design), sequence 1."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sage.patch_artifacts import (
    extract_unified_diff_artifacts,
    git_apply_check,
    select_best_artifact,
)

FENCED = """Here is my fix:

```diff
--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 5
 c = 3
```

Hope this helps."""

RAW_GIT = """Analysis done. Patch follows.
diff --git a/pkg/server.go b/pkg/server.go
--- a/pkg/server.go
+++ b/pkg/server.go
@@ -10,4 +10,5 @@ func main() {
 \tsrv := New()
+\tsrv.Validate()
 \tsrv.Run()
 }
"""

MARKDOWN_HR = """## Options

---

Use `--- option` to pass flags. Another rule:

---

Done."""

TRUNCATED = """```diff
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 5
@@ -10,2 +10,2 @@
```"""


def test_extracts_fenced_unified_diff() -> None:
    arts = extract_unified_diff_artifacts(FENCED)
    assert len(arts) == 1
    art = arts[0]
    assert art.parse_status == "complete"
    assert art.hunk_count == 1
    assert art.payload.startswith("--- a/mod.py")
    assert art.payload.endswith("c = 3\n")
    assert len(art.sha256) == 64
    assert art.source_output_len == len(FENCED)


def test_extracts_raw_git_diff() -> None:
    arts = extract_unified_diff_artifacts(RAW_GIT, node_idx=2, role="coder")
    assert len(arts) == 1
    art = arts[0]
    assert art.payload.startswith("diff --git")
    assert art.parse_status == "complete"
    assert art.node_idx == 2
    assert art.role == "coder"


def test_rejects_markdown_hr_false_positive() -> None:
    assert extract_unified_diff_artifacts(MARKDOWN_HR) == []
    assert extract_unified_diff_artifacts("just prose, no diff") == []
    assert extract_unified_diff_artifacts("") == []


def test_scores_complete_over_partial() -> None:
    complete = extract_unified_diff_artifacts(FENCED)[0]
    partial = extract_unified_diff_artifacts(TRUNCATED)[0]
    assert partial.parse_status == "partial"
    assert complete.score > partial.score
    best = select_best_artifact([partial, complete])
    assert best is complete


def test_tiebreak_last_valid_deterministic() -> None:
    a1 = extract_unified_diff_artifacts(FENCED, node_idx=1)[0]
    a2 = extract_unified_diff_artifacts(FENCED, node_idx=3)[0]
    assert a1.score == a2.score
    best = select_best_artifact([a1, a2])
    assert best is a2  # last valid wins on equal score
    assert select_best_artifact([]) is None


def test_git_apply_check_against_real_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True,
                   capture_output=True)
    (repo / "mod.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True,
                   capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c",
         "user.name=t", "commit", "-m", "init"],
        check=True, capture_output=True,
    )
    art = extract_unified_diff_artifacts(FENCED)[0]
    ok, detail = git_apply_check(art.payload, str(repo))
    assert ok is True and detail == "applies"
    ok2, _ = git_apply_check("", str(repo))
    assert ok2 is False
