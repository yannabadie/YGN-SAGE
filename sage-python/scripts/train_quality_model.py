#!/usr/bin/env python3
"""Fine-tune DistilBERT on quality triples and export to ONNX for Rust inference.

Trains a learned QualityEstimator to replace the heuristic 5-signal scorer.
Takes JSONL training data produced by collect_quality_triples.py and outputs
an ONNX model compatible with the ort crate (load-dynamic) used by sage-core.

Input format (JSONL, one per line)::

    {"task_id": "HumanEval/0", "task": "...", "response": "...",
     "heuristic_score": 0.85, "ground_truth_score": 1.0,
     "base_passed": true, "plus_passed": true,
     "latency_ms": 15000, "system_used": 2}

Usage::

    # Default: train on data/quality_triples.jsonl, export to models/quality_estimator.onnx
    python scripts/train_quality_model.py

    # Custom data, more epochs
    python scripts/train_quality_model.py --data my_triples.jsonl --epochs 10

    # Train only, skip ONNX export
    python scripts/train_quality_model.py --no-export

Requires:
    pip install torch transformers

Optional (for ONNX verification):
    pip install onnxruntime
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check — fail fast with actionable message
# ---------------------------------------------------------------------------

_MISSING: list[str] = []

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    _MISSING.append("torch")

try:
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    _MISSING.append("transformers")

if _MISSING:
    print("=" * 60)
    print("  Missing required packages: " + ", ".join(_MISSING))
    print()
    print("  Install with:")
    print(f"    pip install {' '.join(_MISSING)}")
    print()
    print("  For GPU training (optional):")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("=" * 60)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("train_quality")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"
MIN_SAMPLES = 100
ONNX_OPSET = 14  # Compatible with ort crate (load-dynamic)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QualityDataset(Dataset):
    """PyTorch dataset for (task+response, ground_truth_score) pairs."""

    def __init__(
        self,
        entries: list[dict],
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 512,
    ) -> None:
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        task = entry.get("task", "")
        response = entry.get("response", "")

        # Concatenate task + response with [SEP] separator
        encoding = self.tokenizer(
            task,
            response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(entry["ground_truth_score"], dtype=torch.float32),
            "heuristic_score": torch.tensor(
                entry.get("heuristic_score", 0.0), dtype=torch.float32
            ),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file, skip malformed lines."""
    entries: list[dict] = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Validate required fields
                if "ground_truth_score" not in obj:
                    bad += 1
                    continue
                entries.append(obj)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    log.warning("Skipping malformed line %d", i)

    if bad > 0:
        log.warning("Skipped %d malformed lines out of %d total", bad, len(entries) + bad)

    return entries


def split_data(
    entries: list[dict], val_split: float, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(entries)))
    rng.shuffle(indices)

    n_val = max(1, int(len(entries) * val_split))
    val_idx = set(indices[:n_val])

    train = [entries[i] for i in range(len(entries)) if i not in val_idx]
    val = [entries[i] for i in range(len(entries)) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[float],
    targets: list[float],
) -> dict[str, float]:
    """Compute Pearson r, MAE, and R^2."""
    n = len(predictions)
    if n < 2:
        return {"pearson_r": 0.0, "mae": 0.0, "r2": 0.0}

    p_mean = sum(predictions) / n
    t_mean = sum(targets) / n

    # Pearson correlation
    cov = sum((p - p_mean) * (t - t_mean) for p, t in zip(predictions, targets)) / n
    p_std = (sum((p - p_mean) ** 2 for p in predictions) / n) ** 0.5
    t_std = (sum((t - t_mean) ** 2 for t in targets) / n) ** 0.5

    if p_std > 1e-8 and t_std > 1e-8:
        pearson_r = cov / (p_std * t_std)
    else:
        pearson_r = 0.0

    # MAE
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / n

    # R^2
    ss_res = sum((t - p) ** 2 for p, t in zip(predictions, targets))
    ss_tot = sum((t - t_mean) ** 2 for t in targets)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    return {"pearson_r": pearson_r, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # type: ignore[operator]
        optimizer.zero_grad()

        total_loss += loss.item()

        # Progress bar
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            print(
                f"  [{epoch + 1}/{total_epochs}] "
                f"batch {batch_idx + 1}/{n_batches} | "
                f"loss={loss.item():.4f} | "
                f"lr={lr_current:.2e}",
                flush=True,
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[float], list[float], float]:
    """Evaluate model, return (predictions, targets, heuristic_scores, avg_loss)."""
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    heuristic_scores: list[float] = []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        n_batches += 1

        # Clamp predictions to [0, 1]
        preds = outputs.logits.squeeze(-1).clamp(0.0, 1.0).cpu().tolist()
        if isinstance(preds, float):
            preds = [preds]
        predictions.extend(preds)
        targets.extend(batch["label"].tolist())
        heuristic_scores.extend(batch["heuristic_score"].tolist())

    avg_loss = total_loss / max(n_batches, 1)
    return predictions, targets, heuristic_scores, avg_loss


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    tokenizer: DistilBertTokenizerFast,
    output_path: str,
    max_length: int,
    device: torch.device,
) -> None:
    """Export model to ONNX with dynamic axes, save tokenizer alongside."""
    model.eval()
    out_path = Path(output_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dummy input for tracing
    dummy_text = "Write a function that adds two numbers."
    encoding = tokenizer(
        dummy_text,
        "def add(a, b): return a + b",
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    dummy_input_ids = encoding["input_ids"].to(device)
    dummy_attention_mask = encoding["attention_mask"].to(device)

    # Dynamic axes: both batch_size and sequence_length are variable
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "quality_score": {0: "batch_size"},
    }

    log.info("Exporting ONNX model to %s ...", out_path)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["quality_score"],
        dynamic_axes=dynamic_axes,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
    )

    onnx_size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("ONNX model saved: %s (%.1f MB)", out_path, onnx_size_mb)

    # Save tokenizer for Rust tokenizers crate
    tokenizer_path = out_dir / "tokenizer.json"
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(str(tokenizer_path))
        log.info("Tokenizer saved: %s", tokenizer_path)
    else:
        tokenizer.save_pretrained(str(out_dir))
        log.info("Tokenizer saved to directory: %s", out_dir)

    # Verify ONNX model loads and produces consistent outputs
    _verify_onnx(out_path, dummy_input_ids, dummy_attention_mask, model, device)


def _verify_onnx(
    onnx_path: Path,
    dummy_input_ids: torch.Tensor,
    dummy_attention_mask: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> None:
    """Verify ONNX model loads and produces outputs close to PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning(
            "onnxruntime not installed -- skipping ONNX verification. "
            "Install with: pip install onnxruntime"
        )
        return

    log.info("Verifying ONNX model with onnxruntime ...")
    session = ort.InferenceSession(str(onnx_path))

    # ONNX inference
    ort_inputs = {
        "input_ids": dummy_input_ids.cpu().numpy(),
        "attention_mask": dummy_attention_mask.cpu().numpy(),
    }
    ort_outputs = session.run(["quality_score"], ort_inputs)
    ort_score = ort_outputs[0].flatten()[0]

    # PyTorch inference
    model.eval()
    with torch.no_grad():
        pt_outputs = model(
            input_ids=dummy_input_ids.to(device),
            attention_mask=dummy_attention_mask.to(device),
        )
        pt_score = pt_outputs.logits.squeeze(-1).clamp(0.0, 1.0).cpu().item()

    diff = abs(float(ort_score) - float(pt_score))
    log.info(
        "ONNX verification: PyTorch=%.4f, ONNX=%.4f, diff=%.6f %s",
        pt_score,
        ort_score,
        diff,
        "(OK)" if diff < 0.01 else "(WARNING: large divergence)",
    )

    # Print ONNX model input/output info
    log.info("ONNX inputs:  %s", [(i.name, i.shape) for i in session.get_inputs()])
    log.info("ONNX outputs: %s", [(o.name, o.shape) for o in session.get_outputs()])


# ---------------------------------------------------------------------------
# Ship criterion
# ---------------------------------------------------------------------------

def evaluate_ship_criterion(
    learned_pearson: float,
    heuristic_pearson: float,
) -> None:
    """Report whether the learned model meets the +3-8% improvement ship criterion."""
    delta = learned_pearson - heuristic_pearson
    delta_pct = delta * 100

    print()
    print("=" * 60)
    print("  Ship Criterion: Pearson correlation improvement")
    print("=" * 60)
    print(f"  Heuristic correlation:  {heuristic_pearson:+.4f}")
    print(f"  Learned correlation:    {learned_pearson:+.4f}")
    print(f"  Delta:                  {delta_pct:+.1f}pp")
    print()

    if delta_pct >= 8.0:
        print("  SHIP: Strong improvement (>= +8pp)")
    elif delta_pct >= 3.0:
        print("  SHIP: Meets minimum criterion (>= +3pp)")
    elif delta_pct >= 0.0:
        print("  HOLD: Marginal improvement (< +3pp) -- collect more data or tune hyperparams")
    else:
        print("  NO-GO: Learned model is worse than heuristic -- investigate training data quality")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune DistilBERT on quality triples and export to ONNX. "
            "Produces a learned QualityEstimator for Rust inference via ort."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/quality_triples.jsonl",
        help="Path to JSONL training data (default: data/quality_triples.jsonl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/quality_estimator.onnx",
        help="Output ONNX model path (default: models/quality_estimator.onnx)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length (default: 512)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export, just train and evaluate",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────

    data_path = Path(args.data)
    if not data_path.exists():
        print("=" * 60)
        print(f"  Data file not found: {data_path.resolve()}")
        print()
        print("  Collect training data first:")
        print("    python scripts/collect_quality_triples.py")
        print()
        print("  Or specify a custom path:")
        print("    python scripts/train_quality_model.py --data /path/to/triples.jsonl")
        print("=" * 60)
        sys.exit(1)

    log.info("Loading data from %s ...", data_path)
    entries = load_jsonl(str(data_path))

    if len(entries) < MIN_SAMPLES:
        print("=" * 60)
        print(f"  Only {len(entries)} entries found (minimum: {MIN_SAMPLES})")
        print()
        print("  Collect more training data:")
        print("    python scripts/collect_quality_triples.py --dataset humaneval")
        print("    python scripts/collect_quality_triples.py --dataset mbpp")
        print()
        print("  Then concatenate:")
        print("    cat data/humaneval_triples.jsonl data/mbpp_triples.jsonl > data/quality_triples.jsonl")
        print("=" * 60)
        sys.exit(1)

    # Data statistics
    gt_scores = [e["ground_truth_score"] for e in entries]
    gt_mean = sum(gt_scores) / len(gt_scores)
    n_pass = sum(1 for s in gt_scores if s >= 1.0)
    n_partial = sum(1 for s in gt_scores if 0.0 < s < 1.0)
    n_fail = sum(1 for s in gt_scores if s <= 0.0)
    log.info(
        "Loaded %d entries: pass=%d, partial=%d, fail=%d (gt_mean=%.3f)",
        len(entries), n_pass, n_partial, n_fail, gt_mean,
    )

    # ── Split ──────────────────────────────────────────────────────────────

    train_entries, val_entries = split_data(entries, args.val_split)
    log.info("Split: %d train, %d val (%.0f%% val)", len(train_entries), len(val_entries), args.val_split * 100)

    # ── Tokenizer + model ──────────────────────────────────────────────────

    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %dM total, %dM trainable", param_count // 1_000_000, trainable_count // 1_000_000)

    # ── Datasets and dataloaders ───────────────────────────────────────────

    train_dataset = QualityDataset(train_entries, tokenizer, args.max_length)
    val_dataset = QualityDataset(val_entries, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows-safe
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # ── Optimizer + scheduler ──────────────────────────────────────────────

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    log.info(
        "Training: %d epochs, %d steps/epoch, %d total steps, %d warmup steps",
        args.epochs, len(train_loader), total_steps, warmup_steps,
    )

    # ── Checkpoint directory ───────────────────────────────────────────────

    out_path = Path(args.output)
    ckpt_dir = out_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────

    best_pearson = -1.0
    best_epoch = -1
    training_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        print(f"\n{'─' * 60}")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"{'─' * 60}")

        # Train
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.epochs,
        )

        # Evaluate
        predictions, targets, heuristic_scores, avg_val_loss = evaluate(
            model, val_loader, device,
        )

        # Metrics
        learned_metrics = compute_metrics(predictions, targets)
        heuristic_metrics = compute_metrics(heuristic_scores, targets)

        epoch_time = time.perf_counter() - epoch_start

        print(f"\n  Train loss:            {avg_train_loss:.4f}")
        print(f"  Val loss:              {avg_val_loss:.4f}")
        print(f"  Learned  Pearson r:    {learned_metrics['pearson_r']:.4f}")
        print(f"  Heuristic Pearson r:   {heuristic_metrics['pearson_r']:.4f}")
        print(f"  Learned  MAE:          {learned_metrics['mae']:.4f}")
        print(f"  Heuristic MAE:         {heuristic_metrics['mae']:.4f}")
        print(f"  Learned  R^2:          {learned_metrics['r2']:.4f}")
        print(f"  Heuristic R^2:         {heuristic_metrics['r2']:.4f}")
        print(f"  Epoch time:            {epoch_time:.1f}s")

        # Track best
        if learned_metrics["pearson_r"] > best_pearson:
            best_pearson = learned_metrics["pearson_r"]
            best_epoch = epoch + 1
            print(f"  ** New best Pearson r: {best_pearson:.4f} (epoch {best_epoch})")

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learned_metrics": learned_metrics,
                "heuristic_metrics": heuristic_metrics,
            },
            str(ckpt_path),
        )
        log.info("Checkpoint saved: %s", ckpt_path)

    total_time = time.perf_counter() - training_start

    # ── Final evaluation ───────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("  Training Complete")
    print(f"{'=' * 60}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Best Pearson r:  {best_pearson:.4f} (epoch {best_epoch})")
    print(f"  Device:          {device}")

    # Load best checkpoint if it's not the last epoch
    if best_epoch != args.epochs and best_epoch > 0:
        best_ckpt = ckpt_dir / f"epoch_{best_epoch}.pt"
        if best_ckpt.exists():
            log.info("Loading best checkpoint from epoch %d", best_epoch)
            state = torch.load(str(best_ckpt), map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])

    # Final val metrics with best model
    predictions, targets, heuristic_scores, _ = evaluate(model, val_loader, device)
    learned_final = compute_metrics(predictions, targets)
    heuristic_final = compute_metrics(heuristic_scores, targets)

    print(f"\n  Final validation metrics (best model, epoch {best_epoch}):")
    print(f"    Learned  Pearson r: {learned_final['pearson_r']:.4f}  MAE: {learned_final['mae']:.4f}  R^2: {learned_final['r2']:.4f}")
    print(f"    Heuristic Pearson r: {heuristic_final['pearson_r']:.4f}  MAE: {heuristic_final['mae']:.4f}  R^2: {heuristic_final['r2']:.4f}")

    # Ship criterion
    evaluate_ship_criterion(learned_final["pearson_r"], heuristic_final["pearson_r"])

    # ── ONNX export ────────────────────────────────────────────────────────

    if args.no_export:
        log.info("Skipping ONNX export (--no-export)")
    else:
        export_onnx(model, tokenizer, args.output, args.max_length, device)

    # ── Summary JSON ───────────────────────────────────────────────────────

    summary = {
        "model": MODEL_NAME,
        "data_path": str(data_path.resolve()),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "best_epoch": best_epoch,
        "device": str(device),
        "total_time_s": round(total_time, 1),
        "learned_pearson_r": round(learned_final["pearson_r"], 4),
        "learned_mae": round(learned_final["mae"], 4),
        "learned_r2": round(learned_final["r2"], 4),
        "heuristic_pearson_r": round(heuristic_final["pearson_r"], 4),
        "heuristic_mae": round(heuristic_final["mae"], 4),
        "heuristic_r2": round(heuristic_final["r2"], 4),
        "delta_pearson_pp": round(
            (learned_final["pearson_r"] - heuristic_final["pearson_r"]) * 100, 1
        ),
        "onnx_exported": not args.no_export,
        "onnx_path": str(out_path.resolve()) if not args.no_export else None,
        "onnx_opset": ONNX_OPSET,
    }

    summary_path = out_path.parent / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Training summary saved: %s", summary_path)

    print(f"\n  Artifacts:")
    if not args.no_export:
        print(f"    ONNX model:     {out_path.resolve()}")
        print(f"    Tokenizer:      {out_path.parent.resolve() / 'tokenizer.json'}")
    print(f"    Checkpoints:    {ckpt_dir.resolve()}/")
    print(f"    Summary:        {summary_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
