"""Sweep datasets x memory systems."""

import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml


def load_config() -> dict:
    """Load config from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


# Load config
_CONFIG = load_config()
DATASETS = _CONFIG["datasets"]
MODELS = _CONFIG["models"]  # List of {model, api_base} dicts
BASELINE_NAMES = _CONFIG["memory_systems"]["baselines"]
PROPOSED_NAMES = _CONFIG["memory_systems"]["proposed"]
MEMORY_SYSTEMS = [(n, f"agents/{n}.py") for n in BASELINE_NAMES] + [
    (n, f"agents/{n}.py") for n in PROPOSED_NAMES
]
SEEDS = _CONFIG["benchmark"]["seeds"]
CONCURRENCY = _CONFIG["benchmark"]["concurrency"]
_DS_DEFAULTS = {k: _CONFIG["dataset"][k] for k in ("num_train", "num_val", "num_test")}
_DS_OVERRIDES = _CONFIG["dataset"].get("overrides", {})

DEFAULT_SEED = 42
_SKIP_MEMORY_FILES = {"__init__", "fewshot_memory"}


def discover_all_memory_systems() -> list[tuple[str, str]]:
    """Auto-discover all memory system .py files on disk."""
    base = Path(__file__).parent
    systems = []
    for f in sorted((base / "agents").glob("*.py")):
        name = f.stem
        if name in _SKIP_MEMORY_FILES:
            continue
        systems.append((name, f"agents/{name}.py"))
    return systems


def get_dataset_sizes(dataset: str) -> tuple[int, int, int]:
    """Return (num_train, num_val, num_test) for a dataset, applying overrides."""
    o = _DS_OVERRIDES.get(dataset, {})
    return (
        o.get("num_train", _DS_DEFAULTS["num_train"]),
        o.get("num_val", _DS_DEFAULTS["num_val"]),
        o.get("num_test", _DS_DEFAULTS["num_test"]),
    )


def _sanitize_filename(desc: str) -> str:
    return re.sub(r"[^\w\-.]", "_", desc)


def _print_failure(desc: str, log_path: Path) -> None:
    print(f"\nFAILED: {desc}")
    print(f"Log: {log_path}")
    try:
        lines = log_path.read_text().strip().split("\n")
    except OSError:
        return
    for line in lines[-8:]:
        print(f"  {line[:120]}")


async def _run_with_retries(
    cmd: list[str],
    log_path: Path,
    max_retries: int = 2,
    timeout: float = 7200,
) -> bool:
    cmd_str = " ".join(cmd)
    log_path.write_text(f"command: {cmd_str}\n\n")

    for attempt in range(max_retries + 1):
        if attempt > 0:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\nretry {attempt}\n{'=' * 60}\n")

        with log_path.open("a", encoding="utf-8") as f:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                code = await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                code = 124
            f.write(f"\nexit={code}\n")

        if code == 0:
            return True

    return False


async def run_all_jobs(
    runs: list[tuple[str, list[str]]],
    logs_dir: Path,
    concurrency: int,
    max_retries: int = 2,
) -> list[tuple[str, bool]]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run_one(idx: int, desc: str, cmd: list[str]) -> tuple[str, bool]:
        async with sem:
            log_path = logs_dir / f"{idx:02d}_{_sanitize_filename(desc)}.log"
            ok = await _run_with_retries(cmd, log_path, max_retries=max_retries)
            if not ok:
                _print_failure(desc, log_path)
            return desc, ok

    tasks = [
        asyncio.create_task(run_one(idx, desc, cmd))
        for idx, (desc, cmd) in enumerate(runs)
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# New hierarchical directory structure
# ---------------------------------------------------------------------------


def run_dir(
    base: Path, dataset: str, memory: str, model: str, seed: int = DEFAULT_SEED
) -> Path:
    """Construct hierarchical run directory path.

    logs/{dataset}/{memory}/{model}/          (default seed)
    logs/{dataset}/{memory}/{model}_seed{N}/  (non-default seed)
    """
    leaf = model if seed == DEFAULT_SEED else f"{model}_seed{seed}"
    return base / dataset / memory / leaf


def parse_run_path(base: Path, filepath: Path) -> dict | None:
    """Parse (dataset, memory, model, seed) from a result file under base."""
    try:
        rel = filepath.parent.relative_to(base)
        parts = rel.parts
        if len(parts) != 3:
            return None
        dataset, memory, model_leaf = parts
        m = re.match(r"^(.+)_seed(\d+)$", model_leaf)
        if m:
            model = m.group(1)
            seed = int(m.group(2))
        else:
            model = model_leaf
            seed = DEFAULT_SEED
        return {"dataset": dataset, "memory": memory, "model": model, "seed": seed}
    except (ValueError, IndexError):
        return None


def load_results(base_dir: Path, filename: str = "val.json") -> dict:
    """Load results from hierarchical dir structure.

    Globs base_dir/**/filename, parses path to extract (dataset, memory, model, seed).
    Returns dict: (model, dataset, memory) -> data dict (with 'accuracy' field).
    """
    results = {}
    for filepath in base_dir.rglob(filename):
        parsed = parse_run_path(base_dir, filepath)
        if not parsed:
            continue
        try:
            data = json.loads(filepath.read_text())
            key = (parsed["model"], parsed["dataset"], parsed["memory"])
            results[key] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return results


# Reference results from mce_reproduce (Qwen3.5-27B, 3 datasets)
# Context chars are avg across LawBench, Symptom2Disease, USPTO
MCE_REF_METHODS = [
    ("No Context", 0),
    ("Few-shot (N=4)", 1176),
    ("Few-shot (N=16)", 4113),
    ("Few-shot (N=64)", 17455),
    ("ACE (9)", 202968),
    ("MCE", 114028),
]
MCE_REFERENCE = {
    ("USPTO", "No Context"): 11.0,
    ("USPTO", "Few-shot (N=4)"): 13.0,
    ("USPTO", "Few-shot (N=16)"): 15.0,
    ("USPTO", "Few-shot (N=64)"): 12.0,
    ("USPTO", "ACE (9)"): 16.0,
    ("USPTO", "MCE"): 14.0,
    ("Symptom2Disease", "No Context"): 67.0,
    ("Symptom2Disease", "Few-shot (N=4)"): 67.5,
    ("Symptom2Disease", "Few-shot (N=16)"): 73.6,
    ("Symptom2Disease", "Few-shot (N=64)"): 79.7,
    ("Symptom2Disease", "ACE (9)"): 77.8,
    ("Symptom2Disease", "MCE"): 83.0,
    ("LawBench", "No Context"): 3.0,
    ("LawBench", "Few-shot (N=4)"): 5.0,
    ("LawBench", "Few-shot (N=16)"): 16.0,
    ("LawBench", "Few-shot (N=64)"): 17.0,
    ("LawBench", "ACE (9)"): 29.0,
    ("LawBench", "MCE"): 23.0,
}


def compute_pareto_frontier(
    points: list[tuple[str, float, int]],
) -> list[tuple[str, float, int]]:
    """Compute Pareto frontier for (name, accuracy, ctx_tokens).

    A point is Pareto-optimal if no other point has both
    higher accuracy AND lower ctx_tokens (maximize accuracy, minimize tokens).
    Returns points sorted by accuracy descending.
    """
    sorted_points = sorted(points, key=lambda x: (-x[1], x[2]))
    pareto = []
    min_tokens = float("inf")
    for name, acc, tok in sorted_points:
        if tok <= min_tokens:
            pareto.append((name, acc, tok))
            min_tokens = tok
    return pareto


def print_results(results: dict, metric_label: str = "val", pareto_only: bool = False):
    """Print one table per model: memory systems as rows, datasets as columns, sorted by avg."""
    if not results:
        print("No results found")
        return

    memory_names = sorted(set(mem for _, _, mem in results.keys()))

    ds_short = {
        "word_sorting": "word_sort",
        "MathEquationBalancer": "MathEqn",
        "bbh/causal_judgement": "bbh/caus",
        "Symptom2Disease": "Symptom",
    }

    models_in_results = sorted(set(m for m, _, _ in results.keys()))
    target_models = [get_model_short_name(m["model"]) for m in MODELS]
    models_to_show = [m for m in models_in_results if m in target_models]

    for model_name in models_to_show:
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}  [{metric_label}]")
        print("=" * 80)

        rows = []
        for mem in memory_names:
            accs = []
            ctx_tokens = []
            cells = []
            for ds in DATASETS:
                data = results.get((model_name, ds, mem))
                if data:
                    acc = data.get("accuracy")
                    ctx_tokens.append(data.get("memory_context_chars", 0))
                    if acc is not None:
                        cells.append(f"{acc * 100:.1f}")
                        accs.append(acc * 100)
                    else:
                        cells.append("-")
                else:
                    cells.append("-")
                    ctx_tokens.append(0)
            n_ds = len(DATASETS)
            avg_acc = sum(accs) / n_ds
            rows.append((avg_acc, mem.replace("_memory", ""), cells, ctx_tokens))

        rows.sort(key=lambda x: x[0])

        # Compute Pareto frontier
        pareto_points = []
        for avg_acc, mem, cells, ctx_tokens in rows:
            non_zero = [ct for ct in ctx_tokens if ct > 10]
            avg_ctx = int(sum(non_zero) / len(non_zero)) if non_zero else 0
            pareto_points.append((mem, avg_acc, avg_ctx))
        pareto_set = {name for name, _, _ in compute_pareto_frontier(pareto_points)}

        short_names = [ds_short.get(d, d[:8]) for d in DATASETS]
        col_w = 12
        header = (
            f"{'memory':<28}"
            + "".join(f"{d:>{col_w}}" for d in short_names)
            + f"{'avg':>7}{'ctx_len':>10}"
        )
        print(header)
        print("-" * len(header))

        row_by_mem = {}
        for avg_acc, mem, cells, ctx_tokens in rows:
            row_by_mem[mem] = (avg_acc, mem, cells, ctx_tokens)

        def print_row(avg_acc, mem, cells, ctx_tokens):
            non_zero = [ct for ct in ctx_tokens if ct > 10]
            avg_ctx = int(sum(non_zero) / len(non_zero)) if non_zero else 0
            ctx_str = f"{avg_ctx:,}" if avg_ctx > 0 else "-"
            marker = " *" if mem in pareto_set else ""
            display_mem = f"{mem}{marker}"
            print(
                f"{display_mem:<28}"
                + "".join(f"{c:>{col_w}}" for c in cells)
                + f"{avg_acc:>7.1f}{ctx_str:>10}"
            )

        def print_ref_row(method, ctx_chars):
            ref_cells = []
            ref_test = []
            for ds in DATASETS:
                acc = MCE_REFERENCE.get((ds, method))
                if acc is not None:
                    ref_test.append(acc)
                    ref_cells.append(f"{acc:.1f}")
                else:
                    ref_cells.append("-")
            avg_ref = sum(ref_test) / len(ref_test) if ref_test else 0
            len_str = f"{ctx_chars:,}" if ctx_chars > 0 else "-"
            print(
                f"{'[ref] ' + method:<28}"
                + "".join(f"{c:>{col_w}}" for c in ref_cells)
                + f"{avg_ref:>7.1f}{len_str:>10}"
            )

        # Group 1: No memory
        print_ref_row("No Context", 0)
        if "no" in row_by_mem:
            print_row(*row_by_mem["no"])
        print("." * len(header))

        # Group 2: Few-shot
        for method, ctx_chars in MCE_REF_METHODS:
            if "Few-shot" in method:
                print_ref_row(method, ctx_chars)
        fewshot_rows = sorted(
            [
                (t, m, c, ct)
                for m, (t, m, c, ct) in row_by_mem.items()
                if m.startswith("fewshot")
            ],
            key=lambda x: x[0],
        )
        for r in fewshot_rows:
            print_row(*r)
        print("." * len(header))

        # Group 3: ACE / MCE
        for method, ctx_chars in MCE_REF_METHODS:
            if method in ("ACE (9)", "MCE"):
                print_ref_row(method, ctx_chars)
        if "ace" in row_by_mem:
            print_row(*row_by_mem["ace"])
        print("." * len(header))

        # Group 4: Proposed (everything else)
        shown = {"no", "ace"} | {m for m in row_by_mem if m.startswith("fewshot")}
        proposed_rows = sorted(
            [
                (t, m, c, ct)
                for m, (t, m, c, ct) in row_by_mem.items()
                if m not in shown
            ],
            key=lambda x: x[0],
        )
        if pareto_only:
            proposed_rows = [r for r in proposed_rows if r[1] in pareto_set]
        for r in proposed_rows:
            print_row(*r)

        # Print Pareto frontier summary
        pareto_rows = compute_pareto_frontier(pareto_points)
        if len(pareto_rows) > 1:
            print("\n  Pareto frontier (* above):")
            for n, a, t in pareto_rows:
                print(
                    f"    {n} ({a:.1f}%, {t:,}ch)"
                    if t > 0
                    else f"    {n} ({a:.1f}%, 0ch)"
                )


def build_val_runs(
    logs_dir: Path,
    memory_systems: list[tuple[str, str]],
    datasets: list[str],
    models: list[dict],
    mode: str = "online",
    num_epochs: int = 1,
    temperature: float | None = None,
) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Build (description, command) pairs for val runs that need to run."""
    runs = []
    num_done = 0
    for model_cfg in models:
        model = model_cfg["model"]
        api_base = model_cfg.get("api_base")
        model_name = get_model_short_name(model)
        for dataset in datasets:
            n_train, n_val, n_test = get_dataset_sizes(dataset)
            for mem_name, mem_path in memory_systems:
                for seed in SEEDS:
                    rd = run_dir(logs_dir, dataset, mem_name, model_name, seed)
                    val_file = rd / "val.json"

                    if val_file.exists():
                        num_done += 1
                        continue

                    rd.mkdir(parents=True, exist_ok=True)
                    desc = f"val/{dataset}/{mem_name}/{model_name}"
                    cmd = [
                        "env",
                        "PYTHONPATH=..",
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "text_classification.inner_loop",
                        "--memory",
                        mem_path,
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--model",
                        model,
                        "--mode",
                        mode,
                        "--val-output",
                        str(val_file),
                        "--save-memory",
                        str(rd / "memory.json"),
                        "--log",
                        str(rd / "log.jsonl"),
                    ]
                    cmd.extend(
                        [
                            "--num-train",
                            str(n_train),
                            "--num-val",
                            str(n_val),
                            "--num-test",
                            str(n_test),
                        ]
                    )
                    if api_base:
                        cmd.extend(["--api-base", api_base])
                    if mode == "offline" and num_epochs > 1:
                        cmd.extend(["--num-epochs", str(num_epochs)])
                    if temperature is not None:
                        cmd.extend(["--temperature", str(temperature)])
                    runs.append((desc, cmd))
    random.shuffle(runs)
    return runs, len(runs), num_done


def build_test_runs(
    logs_dir: Path,
    results_dir: Path,
    memory_systems: list[tuple[str, str]],
    datasets: list[str],
    models: list[dict],
    mode: str = "online",
    num_epochs: int = 1,
    temperature: float | None = None,
) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Build (description, command) pairs for test runs that need to run."""
    runs = []
    num_done = 0
    for model_cfg in models:
        model = model_cfg["model"]
        api_base = model_cfg.get("api_base")
        model_name = get_model_short_name(model)
        for dataset in datasets:
            n_train, n_val, n_test = get_dataset_sizes(dataset)
            for mem_name, mem_path in memory_systems:
                for seed in SEEDS:
                    rd_results = run_dir(
                        results_dir, dataset, mem_name, model_name, seed
                    )
                    test_file = rd_results / "test.json"

                    if test_file.exists():
                        num_done += 1
                        continue

                    # Need saved memory from val run
                    rd_logs = run_dir(logs_dir, dataset, mem_name, model_name, seed)
                    memory_file = rd_logs / "memory.json"
                    if not memory_file.exists():
                        print(
                            f"  WARNING: no memory.json for {dataset}/{mem_name}/{model_name} (run val first)"
                        )
                        continue

                    rd_results.mkdir(parents=True, exist_ok=True)
                    desc = f"test/{dataset}/{mem_name}/{model_name}"
                    cmd = [
                        "env",
                        "PYTHONPATH=..",
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "text_classification.inner_loop",
                        "--memory",
                        mem_path,
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--model",
                        model,
                        "--mode",
                        mode,
                        "--load-memory",
                        str(memory_file),
                        "--test-output",
                        str(test_file),
                    ]
                    cmd.extend(
                        [
                            "--num-train",
                            str(n_train),
                            "--num-val",
                            str(n_val),
                            "--num-test",
                            str(n_test),
                        ]
                    )
                    if api_base:
                        cmd.extend(["--api-base", api_base])
                    if temperature is not None:
                        cmd.extend(["--temperature", str(temperature)])
                    runs.append((desc, cmd))
    random.shuffle(runs)
    return runs, len(runs), num_done


def print_frontier(
    logs_dir: Path,
    results_dir: Path,
    model_filter: str | None = None,
    metric: str = "val",
):
    """Print frontier (best system per dataset) and write frontier JSON."""
    if metric == "test":
        base_dir = results_dir
        filename = "test.json"
    else:
        base_dir = logs_dir
        filename = "val.json"

    results = load_results(base_dir, filename)
    if not results:
        print("No results found")
        return

    if model_filter:
        results = {k: v for k, v in results.items() if k[0] == model_filter}
        if not results:
            print(f"No results for model: {model_filter}")
            return

    # Compute best system per dataset
    by_dataset = defaultdict(list)
    for (model, dataset, memory), data in results.items():
        acc = (data.get("accuracy") or 0) * 100
        ctx_len = data.get("memory_context_chars", 0)
        by_dataset[dataset].append(
            {"memory": memory, "accuracy": acc, "ctx_len": ctx_len}
        )

    frontier = {}
    for dataset in DATASETS:
        if dataset in by_dataset:
            best = max(
                by_dataset[dataset], key=lambda x: (x["accuracy"], -x["ctx_len"])
            )
            frontier[dataset] = {
                "best_system": best["memory"],
                "accuracy": best["accuracy"],
                "ctx_len": best["ctx_len"],
            }

    # Print frontier
    print("\n" + "=" * 60)
    title = (
        f"FRONTIER [{metric}] (model: {model_filter})"
        if model_filter
        else f"FRONTIER [{metric}]"
    )
    print(title)
    print("=" * 60)
    for dataset in DATASETS:
        if dataset in frontier:
            info = frontier[dataset]
            acc = info["accuracy"]
            len_str = f", {info['ctx_len']:,} chars" if info["ctx_len"] > 0 else ""
            print(f"  {dataset}: {info['best_system']} ({acc:.1f}%{len_str})")
        else:
            print(f"  {dataset}: (no results)")

    # Aggregate Pareto frontier
    by_memory = defaultdict(lambda: {"accs": [], "ctx_lens": []})
    for (model, dataset, memory), data in results.items():
        acc = (data.get("accuracy") or 0) * 100
        ctx_len = data.get("memory_context_chars", 0)
        by_memory[memory]["accs"].append(acc)
        by_memory[memory]["ctx_lens"].append(ctx_len)

    points = []
    for mem, stats in by_memory.items():
        avg_acc = sum(stats["accs"]) / len(stats["accs"])
        non_zero = [t for t in stats["ctx_lens"] if t > 0]
        avg_len = int(sum(non_zero) / len(non_zero)) if non_zero else 0
        points.append((mem, avg_acc, avg_len))

    pareto = compute_pareto_frontier(points)
    print(f"\nPARETO FRONTIER [{metric}] (accuracy vs context length):")
    print(f"  {'system':<28} {'acc':>7} {'ctx_len':>10}")
    print(f"  {'-' * 47}")
    for name, acc, length in pareto:
        len_str = f"{length:,}" if length > 0 else "0"
        print(f"  {name:<28} {acc:>7.1f} {len_str:>10}")

    # Write frontier json
    frontier["_pareto"] = [
        {
            "system": name,
            "val_accuracy" if metric == "val" else "test_accuracy": round(acc, 1),
            "ctx_len": length,
        }
        for name, acc, length in pareto
    ]
    frontier_filename = "frontier_val.json" if metric == "val" else "frontier.json"
    frontier_path = logs_dir / frontier_filename
    frontier_path.write_text(json.dumps(frontier, indent=2))
    print(f"\nWrote {frontier_path}")


def update_summary(logs_dir: Path):
    """Auto-update logs/summary.json with all val results aggregated."""
    results = load_results(logs_dir, "val.json")
    if not results:
        return

    summary = {}
    for (model, dataset, memory), data in results.items():
        if dataset not in summary:
            summary[dataset] = {}
        summary[dataset][memory] = {
            "accuracy": data.get("accuracy"),
            "memory_context_chars": data.get("memory_context_chars", 0),
            "model": model,
        }

    summary_path = logs_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


def print_summary(logs_dir: Path, results_dir: Path):
    """Print total token usage from completed runs."""
    total_tokens = 0
    for base, fn in [(logs_dir, "val.json"), (results_dir, "test.json")]:
        for data in load_results(base, fn).values():
            total_tokens += data.get("llm_input_tokens", 0) + data.get(
                "llm_output_tokens", 0
            )
    if total_tokens > 0:
        print(f"\nTotal tokens: {total_tokens:,}")


def print_missing(
    logs_dir: Path, memory_systems: list, datasets: list, metric: str = "val"
):
    """Print missing results."""
    if metric == "test":
        results = load_results(logs_dir.parent / "results", "test.json")
    else:
        results = load_results(logs_dir, "val.json")
    all_memories = [n for n, _ in memory_systems]
    target_models = [get_model_short_name(m["model"]) for m in MODELS]

    missing = []
    for model in target_models:
        for ds in datasets:
            for mem in all_memories:
                if (model, ds, mem) not in results:
                    missing.append((model, ds, mem))

    if missing:
        print(f"\n{'=' * 60}")
        print(f"MISSING RESULTS ({len(missing)}) [{metric}]")
        print("=" * 60)
        for model, ds, mem in missing[:20]:
            print(f"  {model} / {ds} / {mem}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


async def main():
    parser = argparse.ArgumentParser(description="Sweep datasets x memory systems")
    parser.add_argument("--memory", type=str, help="Filter to one memory system")
    parser.add_argument("--dataset", type=str, help="Filter to one dataset")
    parser.add_argument("--model", type=str, help="Filter by model (for --frontier)")
    parser.add_argument(
        "--test", action="store_true", help="Run/show test mode (default: val)"
    )
    parser.add_argument(
        "--frontier", action="store_true", help="Print frontier + write analysis files"
    )
    parser.add_argument(
        "--results", action="store_true", help="Print results table only (no jobs)"
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Only show baselines + Pareto frontier systems",
    )
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default=_CONFIG["inner_loop"].get("mode", "online"),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=_CONFIG["inner_loop"].get("num_epochs", 1),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_CONFIG["inner_loop"].get("temperature"),
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Override logs directory (default: logs/)",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else base / "logs"
    results_dir = base / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    metric = "test" if args.test else "val"

    if args.frontier:
        print_frontier(logs_dir, results_dir, model_filter=args.model, metric=metric)
        if metric == "test":
            print_results(
                load_results(results_dir, "test.json"),
                metric_label="test",
                pareto_only=args.pareto,
            )
        else:
            print_results(
                load_results(logs_dir, "val.json"),
                metric_label="val",
                pareto_only=args.pareto,
            )
        return

    if args.results or args.pareto:
        if args.test:
            print_results(
                load_results(results_dir, "test.json"),
                metric_label="test",
                pareto_only=args.pareto,
            )
        else:
            results = load_results(logs_dir, "val.json")
            print_results(results, metric_label="val", pareto_only=args.pareto)
            if not args.pareto:
                update_summary(logs_dir)
        return

    # Auto-discover all memory systems on disk
    memory_systems = discover_all_memory_systems()

    if args.memory:
        name = Path(args.memory).stem
        memory_systems = [(n, p) for n, p in memory_systems if n == name]
        if not memory_systems:
            print(f"Error: '{args.memory}' not found on disk.")
            return

    datasets = DATASETS
    if args.dataset:
        datasets = [
            d for d in DATASETS if d == args.dataset or d.endswith(f"/{args.dataset}")
        ]
        if not datasets:
            print(f"Error: '{args.dataset}' not found. Available: {DATASETS}")
            return

    # Run from project root
    os.chdir(Path(__file__).parent.parent.parent)

    if args.test:
        runs, num_pending, num_done = build_test_runs(
            logs_dir,
            results_dir,
            memory_systems,
            datasets,
            MODELS,
            args.mode,
            args.num_epochs,
            args.temperature,
        )
    else:
        runs, num_pending, num_done = build_val_runs(
            logs_dir,
            memory_systems,
            datasets,
            MODELS,
            args.mode,
            args.num_epochs,
            args.temperature,
        )
    n_total = num_pending + num_done

    model_names = [get_model_short_name(m["model"]) for m in MODELS]
    mode_str = f"mode={args.mode}" + (
        f" epochs={args.num_epochs}" if args.mode == "offline" else ""
    )
    print(f"Status: {num_done}/{n_total} done, {num_pending} pending [{metric}]")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Datasets: {len(datasets)}, Memory: {len(memory_systems)}, Seeds: {SEEDS}")
    print(f"  {mode_str}")

    if args.test:
        print_results(load_results(results_dir, "test.json"), metric_label="test")
    else:
        results = load_results(logs_dir, "val.json")
        print_results(results, metric_label="val")
        update_summary(logs_dir)

    if num_pending == 0:
        print("\nAll done!")
        return

    print(f"\nLaunching {num_pending} jobs (concurrency={CONCURRENCY})...")

    launcher_logs = logs_dir / ".launcher"
    job_results = await run_all_jobs(
        runs=runs,
        logs_dir=launcher_logs,
        concurrency=CONCURRENCY,
        max_retries=2,
    )

    succeeded = sum(1 for _, ok in job_results if ok)
    print(f"\nCompleted: {succeeded}/{len(job_results)}")

    print_summary(logs_dir, results_dir)
    print_missing(logs_dir, memory_systems, datasets, metric=metric)

    if args.test:
        print_results(load_results(results_dir, "test.json"), metric_label="test")
    else:
        results = load_results(logs_dir, "val.json")
        print_results(results, metric_label="val")
        update_summary(logs_dir)


if __name__ == "__main__":
    asyncio.run(main())
