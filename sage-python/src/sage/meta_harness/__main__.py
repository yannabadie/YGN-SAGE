"""CLI entry point: ``python -m sage.meta_harness``."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def _load_env() -> None:
    """Load .env for API keys."""
    try:
        from dotenv import load_dotenv
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass


def cmd_init(args: argparse.Namespace) -> None:
    from sage.meta_harness.search_loop import MetaHarnessLoop

    workspace = Path(args.workspace).expanduser() if args.workspace else None
    loop = MetaHarnessLoop(workspace=workspace)
    loop.init_workspace()

    print(f"[OK] Workspace initialized at {loop.workspace}")
    print(f"  Baseline config: {loop.baseline_dir / 'config.json'}")
    print(f"  Proposer guide:  {loop.workspace / 'PROPOSER_INSTRUCTIONS.md'}")
    print()
    print("Next steps:")
    print("  1. python -m sage.meta_harness evaluate baseline  # Establish baseline")
    print("  2. python -m sage.meta_harness propose             # Create candidate template")
    print("  3. Edit candidates/<id>/config.json")
    print("  4. python -m sage.meta_harness evaluate <id>       # Score it")


def cmd_evaluate(args: argparse.Namespace) -> None:
    _load_env()
    from sage.meta_harness.search_loop import MetaHarnessLoop

    workspace = Path(args.workspace).expanduser() if args.workspace else None
    loop = MetaHarnessLoop(workspace=workspace)

    cid = args.candidate_id

    if cid == "baseline":
        result = asyncio.run(loop.evaluate_baseline(
            bench_type=args.bench, axis=args.axis, limit=args.limit,
        ))
    else:
        result = asyncio.run(loop.evaluate_candidate(
            candidate_id=cid, bench_type=args.bench,
            axis=args.axis, limit=args.limit,
        ))

    print(f"\n{'='*60}")
    print(f"  Candidate {cid}: {result.config.description}")
    print(f"{'='*60}")
    print(f"  Score:     {result.aggregate_score:.3f}")
    print(f"  Pass rate: {result.aggregate_pass_rate*100:.1f}%")
    print(f"  Latency:   {result.total_latency_ms:.0f}ms")
    print()
    print(loop.status())


def cmd_status(args: argparse.Namespace) -> None:
    from sage.meta_harness.search_loop import MetaHarnessLoop

    workspace = Path(args.workspace).expanduser() if args.workspace else None
    loop = MetaHarnessLoop(workspace=workspace)
    print(loop.status())


def cmd_propose(args: argparse.Namespace) -> None:
    from sage.meta_harness.search_loop import MetaHarnessLoop
    from sage.meta_harness.config import HarnessConfig

    workspace = Path(args.workspace).expanduser() if args.workspace else None
    loop = MetaHarnessLoop(workspace=workspace)

    next_id = loop.next_candidate_id()
    candidate_dir = loop.candidates_dir / next_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    best = loop.best_config()
    parent_id = best.id if best else "baseline"

    if best:
        template = HarnessConfig(
            id=next_id,
            description=f"Derived from {parent_id} -- [describe your changes]",
            parent_id=parent_id,
            context=best.context,
            prompts=best.prompts,
            execution=best.execution,
            topology=best.topology,
        )
    else:
        template = HarnessConfig(
            id=next_id,
            description="[describe your changes]",
            parent_id="baseline",
        )

    template.save(candidate_dir / "config.json")

    (candidate_dir / "proposal.md").write_text(
        f"""# Candidate {next_id} -- Proposal

## Parent: {parent_id}

## Hypothesis
[What failure pattern did you observe in parent traces?]

## Changes
[What did you change in config.json and why?]

## Expected Impact
[Which benchmark axes should improve? By how much?]
""",
        encoding="utf-8",
    )

    print(f"[OK] Template created at {candidate_dir}/")
    print(f"  config.json  (derived from {parent_id})")
    print(f"  proposal.md  (fill in your reasoning)")
    print()
    print(f"Edit config.json, then run:")
    print(f"  python -m sage.meta_harness evaluate {next_id}")


def cmd_apply(args: argparse.Namespace) -> None:
    from sage.meta_harness.search_loop import MetaHarnessLoop

    workspace = Path(args.workspace).expanduser() if args.workspace else None
    loop = MetaHarnessLoop(workspace=workspace)

    best = loop.best_config()
    if not best:
        print("No evaluated candidates. Run evaluations first.")
        sys.exit(1)

    output = Path(args.output) if args.output else Path("config/harness.json")
    best.save(output)

    print(f"[OK] Best config ({best.id}) saved to {output}")
    print(f"  Description: {best.description}")
    print()
    print("Integration in boot.py:")
    print("  from sage.meta_harness.config import HarnessConfig")
    print("  from sage.meta_harness.patcher import HarnessPatcher")
    print(f'  config = HarnessConfig.load(Path("{output}"))')
    print("  patcher = HarnessPatcher(config)")
    print("  patcher.patch_runner(runner)  # or use context manager")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.meta_harness",
        description="Meta-Harness: end-to-end optimization of SAGE's harness code",
    )
    parser.add_argument("--workspace", "-w", default=None)

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize workspace")
    sub.add_parser("propose", help="Create template for next candidate")

    eval_p = sub.add_parser("evaluate", help="Evaluate a candidate")
    eval_p.add_argument("candidate_id")
    eval_p.add_argument("--bench", default="masbench",
                        choices=["masbench", "bigcodebench"])
    eval_p.add_argument("--axis", default="depth")
    eval_p.add_argument("--limit", type=int, default=20)

    sub.add_parser("status", help="Show leaderboard")

    apply_p = sub.add_parser("apply", help="Apply best config to SAGE")
    apply_p.add_argument("--output", "-o", default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cmds = {
        "init": cmd_init,
        "propose": cmd_propose,
        "evaluate": cmd_evaluate,
        "status": cmd_status,
        "apply": cmd_apply,
    }

    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
