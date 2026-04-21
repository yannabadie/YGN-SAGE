"""N=50 SWE-bench Lite smoke — statistically meaningful pass-rate.

Run with:

    source .env  # load API keys
    python sage-python/scripts/swebench_n50_smoke.py

Or in background:

    source .env && python sage-python/scripts/swebench_n50_smoke.py \
      2>&1 | tee docs/benchmarks/$(date +%Y-%m-%d)-swebench-n50-full.log &

Expected wallclock: 2-4 h (50 tasks × ~150 s gen + ~30 s eval = ~2.5 h).
Expected API cost: $20-50 depending on minimax / gemini split.

Output:
  - docs/benchmarks/<YYYY-MM-DD>-swebench-n50-full.json (bench report)
  - docs/benchmarks/<YYYY-MM-DD>-swebench-n50-full.jsonl (truth pack)
  - docs/benchmarks/<YYYY-MM-DD>-swebench-n50-full-summary.json

Rationale: the 2026-04-21 v13 → v17 session showed N=10 is dominated
by per-task variance (~10pp per flip). N=50 reduces noise to ~3-5pp
standard error, enough to detect whether the fallback-fix +
repair-pipeline changes moved the real pass-rate.

Uses the same `python -m sage.bench --type swebench --dataset lite`
entry point that v13 and v17 used; just a larger `--limit`. The
`--offset` argument lets you avoid the first-10 tasks that v13/v17
already burned (those are well-characterized; extra offset samples
fresh variance).
"""
from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    today = datetime.now().strftime("%Y-%m-%d")
    bench_dir = repo_root / "docs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    out_json = bench_dir / f"{today}-swebench-n50-full.json"

    # Offset 0 includes the 10 v13/v17 used. If you want a fresh 50:
    #   OFFSET=10 python sage-python/scripts/swebench_n50_smoke.py
    offset = int(os.environ.get("OFFSET", "0"))

    cmd = [
        sys.executable, "-m", "sage.bench",
        "--type", "swebench",
        "--dataset", "lite",
        "--limit", "50",
        "--offset", str(offset),
        "--output", str(out_json),
    ]
    print(f"Launching: {' '.join(cmd)}")
    print(f"Output JSON: {out_json}")
    print(f"Offset: {offset}")
    print()

    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    sys.exit(main())
