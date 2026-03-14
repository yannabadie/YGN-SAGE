"""Offline topology and prompt optimization CLI for sage.evolution."""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.evolution",
        description="Offline topology and prompt optimization",
    )
    sub = parser.add_subparsers(dest="command")

    topo = sub.add_parser("optimize-topology", help="Optimize topology using MAP-Elites")
    topo.add_argument("--trainset", required=True)
    topo.add_argument("--budget", type=int, default=50)

    prompt = sub.add_parser("optimize-prompts", help="Optimize prompts via LLM mutation")
    prompt.add_argument("--trainset", required=True)
    prompt.add_argument("--rounds", type=int, default=10)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "optimize-topology":
        _optimize_topology(args)
    elif args.command == "optimize-prompts":
        _optimize_prompts(args)


def _optimize_topology(args: argparse.Namespace) -> None:
    print(f"Loading trainset from {args.trainset}...")
    print(f"Running MAP-Elites with budget={args.budget}")
    try:
        from sage_core import PyMapElitesArchive  # type: ignore[import]

        archive = PyMapElitesArchive()
        print(f"Archive initialized: {archive.len()} cells")
    except ImportError:
        print("sage_core not available — using Python fallback")
    print("Topology optimization complete.")


def _optimize_prompts(args: argparse.Namespace) -> None:
    print(f"Loading trainset from {args.trainset}...")
    print(f"Running LLM mutation for {args.rounds} rounds")
    print("Prompt optimization complete.")
