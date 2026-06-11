"""Network-reset-proof SWE-bench Pro grading via remote Modal orchestration.

Problem (2026-06-11): corporate middleboxes reset long-lived gRPC/h2
streams. The official grader (`swe_bench_pro_eval.py`) drives Modal
Sandboxes from the LOCAL machine over exactly such streams — two grading
attempts died with ``grpclib StreamTerminatedError`` while short unary
RPCs (``modal token info``, ``spawn``, ``FunctionCall.get``) pass fine.

Solution: run the UNMODIFIED official grader INSIDE a Modal Function —
every sandbox stream then lives datacenter-side. The local driver only
performs short, retryable unary calls:

    deploy (cached after first build) -> spawn() -> poll
    FunctionCall.get(timeout=5) every ~25s -> download one compact zip.

Usage (local driver):
    python sage-python/scripts/modal_remote_grader.py \
        --grader-repo external/SWE-bench_Pro-os \
        --raw-sample-csv <bundle>/grader_n5.csv \
        --predictions <bundle>/run/predictions.json \
        --output-dir <bundle>/grading \
        [--dockerhub-username jefzda] [--num-workers 3] [--prefix-arg]

The driver injects truststore (corporate MITM CA, Python 3.13 strict)
before any Modal call and retries transient stream resets around every
unary call. Spend: same Modal sandbox cost as local grading + one
2-cpu orchestrator function (~$0.01/10min).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import modal

log = logging.getLogger("sage.bench.modal_remote_grader")

_GRADER_REMOTE_PATH = "/grader"
_APP_NAME = "ygn-sage-remote-grader"

# Image: the grader's own requirements (pandas/tqdm/modal; docker SDK not
# needed in Modal mode) + the grader repo copied in. Pinned loosely — the
# grader repo's requirements.txt is the reference.
def _grader_ignore(path: Path) -> bool:
    """Skip everything the eval script does not need (cuts the upload from
    ~93MB to the eval essentials: eval script + helper_code + run_scripts
    + dockerfiles)."""
    parts = {p.lower() for p in path.parts}
    return bool(
        parts
        & {
            ".git",
            "__pycache__",
            "mini-swe-agent",
            "error_analysis",
            "traj",
            ".github",
        }
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas>=1.5.0", "tqdm>=4.64.0", "modal>=1.0")
    .add_local_dir(
        Path(__file__).resolve().parents[2] / "external" / "SWE-bench_Pro-os",
        remote_path=_GRADER_REMOTE_PATH,
        ignore=_grader_ignore,
    )
)

app = modal.App(_APP_NAME)


@app.function(image=image, timeout=3600, cpu=2.0, memory=2048)
def grade_remote(
    csv_text: str,
    predictions_text: str,
    dockerhub_username: str = "jefzda",
    num_workers: int = 3,
    redo: bool = True,
) -> dict[str, Any]:
    """Run the unmodified official grader datacenter-side.

    Returns ``{"eval_results": dict, "outputs_zip_b64": str}`` where the
    zip carries the full per-instance output tree (output.json, stdout/
    stderr logs, entryscripts, patch copies) for local archiving.
    """
    import os
    import runpy
    import tempfile

    workdir = tempfile.mkdtemp(prefix="remote_grade_")
    csv_path = os.path.join(workdir, "raw_sample.csv")
    pred_path = os.path.join(workdir, "predictions.json")
    out_dir = os.path.join(workdir, "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(csv_text)
    with open(pred_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(predictions_text)

    # The eval script resolves dockerfiles/ relative to cwd (2026-05-11
    # lesson) and imports helper_code as a sibling package.
    os.chdir(_GRADER_REMOTE_PATH)
    sys.path.insert(0, _GRADER_REMOTE_PATH)

    argv = [
        "swe_bench_pro_eval.py",
        "--raw_sample_path", csv_path,
        "--patch_path", pred_path,
        "--output_dir", out_dir,
        "--dockerhub_username", dockerhub_username,
        "--scripts_dir", os.path.join(_GRADER_REMOTE_PATH, "run_scripts"),
        "--num_workers", str(num_workers),
    ]
    if redo:
        argv.append("--redo")
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(
            os.path.join(_GRADER_REMOTE_PATH, "swe_bench_pro_eval.py"),
            run_name="__main__",
        )
    except SystemExit as exc:  # argparse/main exits are fine
        if exc.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv

    eval_results: dict[str, Any] = {}
    eval_path = os.path.join(out_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8", errors="replace") as fh:
            eval_results = json.load(fh)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, out_dir))
    return {
        "eval_results": eval_results,
        "outputs_zip_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


# ── Local driver (short unary RPCs only) ────────────────────────────────────


def _retry_unary(fn, *, attempts: int = 5, base_sleep: float = 3.0):
    """Retry a short Modal call across transient stream resets."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - includes grpclib resets
            last = exc
            log.warning("unary attempt %d/%d failed: %s", i + 1, attempts, exc)
            time.sleep(base_sleep * (i + 1))
    raise RuntimeError(f"unary call failed after {attempts} attempts: {last}")


def deploy_with_retries(attempts: int = 4) -> None:
    """Deploy the app, tolerating heartbeat deadline kills — the server-side
    image build caches progressively, so every retry resumes further."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            log.info("Deploy attempt %d/%d...", i + 1, attempts)
            app.deploy()
            log.info("Deploy OK.")
            return
        except Exception as exc:  # noqa: BLE001 - heartbeat/stream kills
            last = exc
            log.warning("deploy attempt %d failed: %s", i + 1, exc)
            time.sleep(5.0 * (i + 1))
    raise RuntimeError(f"deploy failed after {attempts} attempts: {last}")


def drive(
    *,
    raw_sample_csv: Path,
    predictions: Path,
    output_dir: Path,
    dockerhub_username: str,
    num_workers: int,
    poll_interval_s: float = 25.0,
    max_wait_s: float = 2700.0,
    skip_deploy: bool = False,
) -> int:
    csv_text = raw_sample_csv.read_text(encoding="utf-8")
    predictions_text = predictions.read_text(encoding="utf-8")

    if not skip_deploy:
        deploy_with_retries()

    fn = modal.Function.from_name(_APP_NAME, "grade_remote")
    call = _retry_unary(
        lambda: fn.spawn(
            csv_text,
            predictions_text,
            dockerhub_username=dockerhub_username,
            num_workers=num_workers,
        )
    )
    call_id = call.object_id
    log.info("Spawned remote grading: call_id=%s — polling every %.0fs",
             call_id, poll_interval_s)

    deadline = time.monotonic() + max_wait_s
    result: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            result = _retry_unary(
                lambda: modal.FunctionCall.from_id(call_id).get(timeout=5)
            )
            break
        except TimeoutError:
            log.info("still grading... (call %s)", call_id)
            time.sleep(poll_interval_s)
    if result is None:
        log.error("remote grading did not finish within %.0fs "
                  "(call %s still retrievable later via FunctionCall.from_id)",
                  max_wait_s, call_id)
        return 3

    output_dir.mkdir(parents=True, exist_ok=True)
    blob = base64.b64decode(result["outputs_zip_b64"])
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(output_dir)
    (output_dir / "eval_results.json").write_text(
        json.dumps(result["eval_results"], indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    resolved = sum(1 for v in result["eval_results"].values() if v)
    log.info("Remote grading complete: %d/%d resolved — outputs in %s "
             "(%d bytes zip)",
             resolved, len(result["eval_results"]), output_dir, len(blob))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sample-csv", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dockerhub-username", default="jefzda")
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--poll-interval-s", type=float, default=25.0)
    parser.add_argument("--max-wait-s", type=float, default=2700.0)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--deploy-only", action="store_true",
        help="Deploy the remote grader app and exit (retries built in).",
    )
    parser.add_argument(
        "--skip-deploy", action="store_true",
        help="Assume the app is already deployed (Function.from_name).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Corporate MITM CA + Python 3.13 strict validation (2026-06-10
    # diagnosis): trust the OS store before any Modal call. Local driver
    # only — the remote function runs on clean Modal infra.
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - clean networks need no injection
        pass
    if args.deploy_only:
        deploy_with_retries()
        return 0
    if not (args.raw_sample_csv and args.predictions and args.output_dir):
        parser.error(
            "--raw-sample-csv, --predictions and --output-dir are required "
            "unless --deploy-only"
        )
    return drive(
        raw_sample_csv=args.raw_sample_csv,
        predictions=args.predictions,
        output_dir=args.output_dir,
        dockerhub_username=args.dockerhub_username,
        num_workers=args.num_workers,
        poll_interval_s=args.poll_interval_s,
        max_wait_s=args.max_wait_s,
        skip_deploy=args.skip_deploy,
    )


if __name__ == "__main__":
    sys.exit(main())
