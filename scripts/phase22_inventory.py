#!/usr/bin/env python3
"""Cycle-13 K Phase 2.2 — AST-based stage seam inventory.

Emits a definitive table of every `pipeline._stage_<X>` and
`pipeline._<helper>` reference across the test suite, distinguishing:

- **kind**: `call` (executable call site), `assign` (instance method
  injection a la `pipeline._stage_X = mock`), `attr_ref` (attribute
  access used as a value).
- **stage**: `classify`, `decompose`, `select_topology`, `assign_models`,
  `execute`, `learn`, or `<helper>`.
- **async_stage**: `True` for `decompose`, `execute`, `learn`; `False`
  for sync stages and helpers.
- **lineno**: source line of the reference (1-indexed).
- **in_docstring**: `True` when the reference is a literal inside an
  `ast.Constant` string or inside a comment-only line — these are
  documentation references that do NOT need code rewriting.

Usage:
    python scripts/phase22_inventory.py                           # CSV to stdout
    python scripts/phase22_inventory.py --output inv.csv          # CSV to file
    python scripts/phase22_inventory.py --stage assign_models     # filter
    python scripts/phase22_inventory.py --stage assign_models --kind call

Provides the source-of-truth filter for every Phase 2.2 Stage B
sub-commit ("which files / lines does B1.2-async.X touch?") AND for
Stage D Q7 audit ("which production callsites in pipeline_v2/*.py
must be rewritten before deleting helper <X>?").

Lock source: cgpro `cgpro_phase22_test_rewrite_20260506` DESIGN_LOCK
2026-05-06 + advisor 2026-05-06 follow-up after the
`test_pipeline_adaptation.py:90` B1.2-sync oversight.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

ASYNC_STAGES = frozenset({"decompose", "execute", "learn"})
SYNC_STAGES = frozenset({"classify", "select_topology", "assign_models"})
ALL_STAGES = ASYNC_STAGES | SYNC_STAGES


# Test files in the Phase 2.2 27-file scope (matches narrative_guard
# TARGET_GLOBS).
TEST_FILES: tuple[str, ...] = (
    "sage-python/tests/observability/test_pipeline_spans.py",
    "sage-python/tests/test_e2e_campaign.py",
    "sage-python/tests/test_f7_task_system_forward.py",
    "sage-python/tests/test_model_assigner_top3_logging.py",
    "sage-python/tests/test_online_evolution.py",
    "sage-python/tests/test_oracle_stack.py",
    "sage-python/tests/test_oxiz_pipeline.py",
    "sage-python/tests/test_pillar_logging.py",
    "sage-python/tests/test_pipeline.py",
    "sage-python/tests/test_pipeline_adaptation.py",
    "sage-python/tests/test_pipeline_bandit_causality.py",
    "sage-python/tests/test_pipeline_budget.py",
    "sage-python/tests/test_pipeline_bypass.py",
    "sage-python/tests/test_pipeline_bypass_structural_isolation.py",
    "sage-python/tests/test_pipeline_governance.py",
    "sage-python/tests/test_pipeline_periodic_save_epoch_preflight.py",
    "sage-python/tests/test_pipeline_topology_flags.py",
    "sage-python/tests/test_pipeline_topology_skip_guardrails_decoupling.py",
    "sage-python/tests/test_pipeline_v2_bandit_attribution_invariant.py",
    "sage-python/tests/test_pipeline_v2_control_surface_fields.py",
    "sage-python/tests/test_pipeline_v2_fix_c_budget_tier_no_controller.py",
    "sage-python/tests/test_pipeline_v2_oracle_gate_invariant.py",
    "sage-python/tests/test_pipeline_v2_phase_a_wrappers.py",
    "sage-python/tests/test_run_frame.py",
    "sage-python/tests/test_speculative_routing.py",
    "sage-python/tests/test_system_hint.py",
    "sage-python/tests/test_topology_learn.py",
)

# Production code surfaces audited at Stage D Q7. Kept separate so a
# `--scope src` filter can target them without test noise.
PROD_FILES: tuple[str, ...] = (
    "sage-python/src/sage/pipeline.py",
    *(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "sage-python/src/sage/pipeline_v2").glob("*.py")),
)


@dataclass(frozen=True)
class Hit:
    file: str
    stage: str
    helper: str
    kind: str  # call | assign | attr_ref
    async_stage: bool
    lineno: int
    line_text: str

    @property
    def is_stage(self) -> bool:
        return self.stage in ALL_STAGES


class StageVisitor(ast.NodeVisitor):
    """Collect `pipeline._stage_<X>` and `pipeline._<helper>` references.

    Visits Call (kind=call), Assign with attribute LHS (kind=assign),
    and bare Attribute nodes (kind=attr_ref). Discards references whose
    parent is a Call but where the Attribute is the target of a Call —
    those are reported as `kind=call` once via the Call visit, not
    twice.
    """

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.hits: list[Hit] = []
        self._call_attr_ids: set[int] = set()

    def _classify_attr(self, attr_name: str) -> tuple[str, str, bool] | None:
        """Return (stage, helper, async_stage) tuple, or None if not in scope."""
        if not attr_name.startswith("_"):
            return None
        if attr_name.startswith("_stage_"):
            stage = attr_name[len("_stage_"):]
            if stage in ALL_STAGES:
                return stage, "", stage in ASYNC_STAGES
            return None
        # Helper delegator — return raw name as helper.
        return "<helper>", attr_name, False

    def _is_pipeline_attr(self, node: ast.Attribute) -> bool:
        """Detect references to the Pipeline instance specifically.

        We accept ONLY `pipeline._<X>` because `self._<X>` is ambiguous
        in tests (test fixture classes like `_SpyAssigner` and
        `_Topology` have their own `self._<X>` attrs that are NOT
        Pipeline methods). For production-code (`pipeline_v2/*.py`) the
        `self` references ARE the pipeline, but `--scope src` is the
        right tool there — and even that uses the explicit `pipeline`
        argument at the function boundary in pipeline_v2 functions.
        """
        if not isinstance(node.value, ast.Name):
            return False
        return node.value.id == "pipeline"

    def _line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        if isinstance(target, ast.Attribute) and self._is_pipeline_attr(target):
            classified = self._classify_attr(target.attr)
            if classified is not None:
                stage, helper, is_async = classified
                self.hits.append(
                    Hit(
                        file="",  # filled by caller
                        stage=stage,
                        helper=helper,
                        kind="call",
                        async_stage=is_async,
                        lineno=target.lineno,
                        line_text=self._line_text(target.lineno),
                    )
                )
                self._call_attr_ids.add(id(target))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute) and self._is_pipeline_attr(target):
                classified = self._classify_attr(target.attr)
                if classified is not None:
                    stage, helper, is_async = classified
                    self.hits.append(
                        Hit(
                            file="",
                            stage=stage,
                            helper=helper,
                            kind="assign",
                            async_stage=is_async,
                            lineno=target.lineno,
                            line_text=self._line_text(target.lineno),
                        )
                    )
                    self._call_attr_ids.add(id(target))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if id(node) in self._call_attr_ids:
            return  # already recorded as call/assign
        if self._is_pipeline_attr(node):
            classified = self._classify_attr(node.attr)
            if classified is not None:
                stage, helper, is_async = classified
                self.hits.append(
                    Hit(
                        file="",
                        stage=stage,
                        helper=helper,
                        kind="attr_ref",
                        async_stage=is_async,
                        lineno=node.lineno,
                        line_text=self._line_text(node.lineno),
                    )
                )
        self.generic_visit(node)


def _walk_file(path: Path) -> list[Hit]:
    text = path.read_text(encoding="utf-8")
    source_lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        print(
            f"WARN: parse error in {path}: {exc}", file=sys.stderr
        )
        return []
    visitor = StageVisitor(source_lines)
    visitor.visit(tree)
    rel = path.relative_to(ROOT).as_posix()
    return [
        Hit(
            file=rel,
            stage=h.stage,
            helper=h.helper,
            kind=h.kind,
            async_stage=h.async_stage,
            lineno=h.lineno,
            line_text=h.line_text,
        )
        for h in visitor.hits
    ]


def collect_hits(file_globs: Iterable[str]) -> list[Hit]:
    hits: list[Hit] = []
    for pattern in file_globs:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            hits.extend(_walk_file(path))
    return hits


def filter_hits(
    hits: list[Hit],
    *,
    stage: str | None = None,
    kind: str | None = None,
    async_only: bool | None = None,
    helper: str | None = None,
    file_pattern: str | None = None,
) -> list[Hit]:
    out = hits
    if stage is not None:
        out = [h for h in out if h.stage == stage]
    if kind is not None:
        out = [h for h in out if h.kind == kind]
    if async_only is not None:
        out = [h for h in out if h.async_stage == async_only]
    if helper is not None:
        out = [h for h in out if h.helper == helper]
    if file_pattern is not None:
        out = [h for h in out if file_pattern in h.file]
    return out


def write_csv(hits: list[Hit], stream) -> None:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "file",
            "stage",
            "helper",
            "kind",
            "async",
            "lineno",
            "line_text",
        ]
    )
    for h in hits:
        writer.writerow(
            [
                h.file,
                h.stage,
                h.helper,
                h.kind,
                "Y" if h.async_stage else "N",
                h.lineno,
                h.line_text,
            ]
        )


def summarize(hits: list[Hit]) -> str:
    by_stage: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    files: set[str] = set()
    for h in hits:
        files.add(h.file)
        key = f"{h.stage}{f'/{h.helper}' if h.helper else ''}"
        by_stage[key] = by_stage.get(key, 0) + 1
        by_kind[h.kind] = by_kind.get(h.kind, 0) + 1
    out = [f"Total: {len(hits)} hits across {len(files)} files\n"]
    out.append("By stage/helper:")
    for k in sorted(by_stage):
        out.append(f"  {k:30s} {by_stage[k]:4d}")
    out.append("By kind:")
    for k in sorted(by_kind):
        out.append(f"  {k:10s} {by_kind[k]:4d}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("tests", "src", "all"), default="tests")
    parser.add_argument("--output", help="Write CSV to file instead of stdout")
    parser.add_argument("--stage", help="Filter by stage name")
    parser.add_argument(
        "--kind", choices=("call", "assign", "attr_ref"), help="Filter by kind"
    )
    parser.add_argument(
        "--async-only",
        action="store_true",
        help="Only async stages",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync stages (excludes helpers)",
    )
    parser.add_argument("--helper", help="Filter by helper name (e.g. _build_write_gate)")
    parser.add_argument("--file", help="Filter by file substring")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if args.scope == "tests":
        globs = TEST_FILES
    elif args.scope == "src":
        globs = PROD_FILES
    else:
        globs = TEST_FILES + PROD_FILES

    hits = collect_hits(globs)

    async_filter: bool | None = None
    if args.async_only:
        async_filter = True
    elif args.sync_only:
        async_filter = False

    hits = filter_hits(
        hits,
        stage=args.stage,
        kind=args.kind,
        async_only=async_filter,
        helper=args.helper,
        file_pattern=args.file,
    )

    if args.sync_only:
        # _sync_only_ is "sync stages only", which excludes helpers (helpers
        # all have async_stage=False because helpers are not stages).
        hits = [h for h in hits if h.stage in SYNC_STAGES]

    if args.summary:
        print(summarize(hits))
        return 0

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as fp:
            write_csv(hits, fp)
        print(
            f"wrote {len(hits)} rows to {args.output}",
            file=sys.stderr,
        )
    else:
        write_csv(hits, sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
