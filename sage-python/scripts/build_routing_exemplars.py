#!/usr/bin/env python3
"""Build kNN routing exemplar embeddings from ground truth JSON.

Usage:
    python scripts/build_routing_exemplars.py [--validate] [--output PATH]

Embeds all ground truth tasks using the best available embedder
(Rust ONNX > sentence-transformers) and saves to .npz for kNN routing.

With --validate, runs leave-one-out cross-validation to estimate accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _find_gt() -> Path:
    candidates = [
        Path.cwd() / "config" / "routing_ground_truth.json",
        Path.cwd() / "sage-python" / "config" / "routing_ground_truth.json",
        Path(__file__).parent.parent / "config" / "routing_ground_truth.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    print("ERROR: routing_ground_truth.json not found")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build kNN routing exemplars")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .npz path (default: config/routing_exemplars.npz)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run leave-one-out cross-validation",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="k for kNN validation (default: 5)",
    )
    args = parser.parse_args()

    # Load ground truth
    gt_path = _find_gt()
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    tasks = gt_data["tasks"]
    texts = [t["task"] for t in tasks]
    labels = np.array([t["expected_system"] for t in tasks], dtype=np.int32)
    print(f"Loaded {len(tasks)} ground truth tasks from {gt_path}")
    print(f"  S1: {(labels == 1).sum()}, S2: {(labels == 2).sum()}, S3: {(labels == 3).sum()}")

    # Initialize embedder
    from sage.memory.embedder import Embedder
    emb = Embedder()
    print(f"Embedder backend: {emb._backend}")

    if emb.is_hash_fallback:
        print("WARNING: Using hash embedder -- kNN routing will not be meaningful!")
        print("Install onnxruntime + download model, or install sentence-transformers.")

    # Embed all tasks
    print("Embedding tasks...")
    vectors = emb.embed_batch(texts)
    embeddings = np.array(vectors, dtype=np.float32)

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings /= norms
    print(f"Embeddings shape: {embeddings.shape}")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = gt_path.parent / "routing_exemplars.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, embeddings=embeddings, labels=labels)
    print(f"Saved exemplars to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Leave-one-out cross-validation
    if args.validate:
        print(f"\nRunning LOO-CV (k={args.k})...")
        correct = 0
        per_system = {1: {"total": 0, "correct": 0}, 2: {"total": 0, "correct": 0}, 3: {"total": 0, "correct": 0}}

        for i in range(len(embeddings)):
            query = embeddings[i]
            expected = labels[i]

            # Remove the held-out sample
            bank = np.delete(embeddings, i, axis=0)
            bank_labels = np.delete(labels, i, axis=0)

            # Cosine similarity (dot product on unit vectors)
            sims = bank @ query
            k = min(args.k, len(sims))
            top_k = np.argpartition(sims, -k)[-k:]

            # Weighted majority vote
            votes: dict[int, float] = {}
            for idx in top_k:
                label = int(bank_labels[idx])
                votes[label] = votes.get(label, 0.0) + float(sims[idx])

            predicted = max(votes, key=lambda s: votes[s])

            per_system[expected]["total"] += 1
            if predicted == expected:
                correct += 1
                per_system[expected]["correct"] += 1
            else:
                print(f"  MISS [{tasks[i]['id']:2d}] expected=S{expected} got=S{predicted}: {texts[i][:60]}")

        accuracy = correct / len(embeddings)
        print(f"\nLOO-CV Accuracy: {accuracy:.1%} ({correct}/{len(embeddings)})")
        for sys in [1, 2, 3]:
            s = per_system[sys]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            print(f"  S{sys}: {acc:.0%} ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
