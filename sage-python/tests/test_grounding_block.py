"""G1 helper/gating contract — cgpro GROUNDING DESIGN_LOCKED (2026-06-11,
conv cgpro_emission_fixes_design, sequence 'G1 helper / gating')."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from sage.grounding import (
    build_grounding_block,
    prefilter_paths,
)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "billing").mkdir(parents=True)
    (repo / "src" / "ui").mkdir(parents=True)
    (repo / "src" / "billing" / "invoice.py").write_text(
        "def total(items):\n    return sum(items)\n", encoding="utf-8"
    )
    (repo / "src" / "ui" / "widget.py").write_text(
        "class Widget:\n    pass\n", encoding="utf-8"
    )
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


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(messages[-1].content)

        class _R:
            content = self.reply
            usage = {"input_tokens": 100, "output_tokens": 10}

        return _R()


def test_git_ls_files_prefilter_is_deterministic_and_capped() -> None:
    paths = [f"pkg/mod_{i:03d}.py" for i in range(50)] + [
        "src/billing/invoice.py",
        "src/billing/tax.py",
        "docs/billing.md",
    ]
    problem = "The billing invoice total is wrong when tax applies"
    out1 = prefilter_paths(paths, problem, cap=10)
    out2 = prefilter_paths(list(reversed(paths)), problem, cap=10)
    assert out1 == out2  # deterministic regardless of input order
    assert len(out1) == 10
    # problem-relevant paths rank first
    assert out1[0] == "src/billing/invoice.py"
    assert "src/billing/tax.py" in out1[:3]
    assert "docs/billing.md" in out1[:3]


def test_grounding_requires_verified_artifact_profile(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("SAGE_TASK_ARTIFACT_PROFILE", raising=False)
    repo = _make_repo(tmp_path)
    llm = _FakeLLM("src/billing/invoice.py")
    block, telemetry = asyncio.run(
        build_grounding_block(str(repo), "fix the invoice total", llm)
    )
    assert block == ""
    assert llm.calls == 0  # no localizer call without the profile
    assert telemetry["skipped_reason"] == "artifact_profile_inactive"


def test_localizer_called_once_max_six_paths_and_invalid_dropped(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    repo = _make_repo(tmp_path)
    llm = _FakeLLM(
        "src/billing/invoice.py\n"
        "src/invented_by_model.py\n"     # invalid -> dropped + telemetered
        "src/ui/widget.py\n"
    )
    block, telemetry = asyncio.run(
        build_grounding_block(str(repo), "fix the invoice total", llm)
    )
    assert llm.calls == 1
    assert telemetry["localizer_valid_paths"] == [
        "src/billing/invoice.py", "src/ui/widget.py"
    ]
    assert telemetry["localizer_dropped_paths"] == ["src/invented_by_model.py"]


def test_grounding_block_contains_verbatim_bytes_headers_and_instruction(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    repo = _make_repo(tmp_path)
    llm = _FakeLLM("src/billing/invoice.py")
    block, telemetry = asyncio.run(
        build_grounding_block(str(repo), "fix the invoice total", llm)
    )
    assert "### VERIFIED REPOSITORY CONTEXT" in block
    assert "### FILE: src/billing/invoice.py" in block
    assert "def total(items):\n    return sum(items)\n" in block  # verbatim
    assert "Patch only files shown below" in block
    assert telemetry["file_count"] == 1
    assert telemetry["total_bytes"] > 0
    assert telemetry["grounding_truncated_files"] == []


def test_grounding_block_total_cap_records_truncation_telemetry(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    repo = _make_repo(tmp_path)
    big = repo / "src" / "billing" / "big.py"
    big.write_text("x = 1\n" * 20000, encoding="utf-8")  # ~120k chars
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True,
                   capture_output=True)
    llm = _FakeLLM("src/billing/big.py")
    block, telemetry = asyncio.run(
        build_grounding_block(
            str(repo), "fix big", llm, max_chars_total=5000
        )
    )
    assert "TRUNCATED" in block
    assert telemetry["grounding_truncated_files"] == ["src/billing/big.py"]
    assert "use read_file" in block  # nudge to verify beyond the cut


def test_grounding_reads_bytes_from_base_checkout_not_prior_tool_memory(
    monkeypatch, tmp_path
) -> None:
    """prior_paths are HINTS for the localizer prompt only — the bytes
    always come from the on-disk checkout."""
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    repo = _make_repo(tmp_path)
    llm = _FakeLLM("src/billing/invoice.py")
    block, telemetry = asyncio.run(
        build_grounding_block(
            str(repo), "fix the invoice total", llm,
            prior_paths=["src/ui/widget.py", "not/a/file.py"],
        )
    )
    assert "src/ui/widget.py" in llm.prompts[0]      # hint visible
    assert "def total(items)" in block               # disk bytes win
    assert telemetry["localizer_valid_paths"] == ["src/billing/invoice.py"]
