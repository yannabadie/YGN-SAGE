"""Post-training evaluation on BigCodeBench Hard + HumanEval+.

Set SAGE_ENABLE_PATH6=1 and SAGE_TOPOLOGY_MODEL to the trained adapter
so benchmarks use the newly trained topology policy.
"""
import argparse
import os
import subprocess
import sys


def run_bench(bench_type: str, limit: int = 20, model_path: str = ""):
    env = os.environ.copy()
    if model_path:
        env["SAGE_ENABLE_PATH6"] = "1"
        env["SAGE_TOPOLOGY_MODEL"] = model_path
    cmd = [
        sys.executable, "-m", "sage.bench",
        "--type", bench_type,
        "--limit", str(limit),
    ]
    if bench_type == "bigcodebench":
        cmd.extend(["--subset", "hard", "--split", "instruct"])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training benchmarks")
    parser.add_argument("--bench", choices=["bigcodebench", "humaneval", "routing_gt", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="models/topology_verl_local/",
                        help="Path to trained LoRA adapter (enables Path 6)")
    args = parser.parse_args()

    benches = [args.bench] if args.bench != "all" else ["bigcodebench", "humaneval", "routing_gt"]
    for bench in benches:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {bench} (model: {args.model})")
        print(f"{'=' * 60}")
        run_bench(bench, args.limit, model_path=args.model)
