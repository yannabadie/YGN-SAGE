"""sage.grounding — promoted arm-A provisioning helpers (GROUNDING block)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sage.grounding import files_block, parse_file_list, repo_file_tree


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    (repo / "README.md").write_text("# readme\n", encoding="utf-8")
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


def test_repo_file_tree_lists_and_caps(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    tree = repo_file_tree(str(repo))
    assert "src/mod.py" in tree and "README.md" in tree
    capped = repo_file_tree(str(repo), max_files=1)
    assert "more files)" in capped


def test_parse_file_list_validates_against_worktree(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    reply = (
        "- `src/mod.py`\n"
        "src/invented_by_the_model.py\n"   # path-grounding: rejected
        "1. README.md — the docs\n"
        "src/mod.py\n"                      # dedup
    )
    assert parse_file_list(reply, str(repo)) == ["src/mod.py", "README.md"]
    assert parse_file_list("", str(repo)) == []


def test_files_block_verbatim_and_capped(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    block = files_block(str(repo), ["src/mod.py", "missing.py", "README.md"])
    assert "### FILE: src/mod.py" in block
    assert "a = 1\nb = 2\n" in block          # verbatim bytes
    assert "missing.py" not in block           # unreadable skipped
    tiny = files_block(str(repo), ["src/mod.py"], max_chars_total=60)
    assert len(tiny) <= 80
