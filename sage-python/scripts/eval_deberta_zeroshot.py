#!/usr/bin/env python3
"""Zero-shot evaluation of NVIDIA DeBERTa-v3-base prompt-task-and-complexity-classifier
on SAGE's 50 ground truth routing tasks.

Purpose: determine whether the pretrained classifier can serve as Stage 1 in
SAGE's AdaptiveRouter without fine-tuning.  If bootstrap 95% CI lower bound
>= 90%, wire as-is.  Otherwise, fine-tuning is required.

NVIDIA model: nvidia/prompt-task-and-complexity-classifier
  - 11-class task type head (Open QA, Closed QA, Code Generation, ...)
  - 6 complexity dimension heads (creativity, reasoning, contextual_knowledge,
    domain_knowledge, constraint_ct, number_of_few_shots)
  - Composite prompt_complexity_score (weighted ensemble, 0-1)

Mapping to S1/S2/S3 is configurable via --mapping-preset and individual
threshold flags.  Defaults were chosen to align with SAGE's Kahneman-style
cognitive systems:
  S1 = low complexity, factual/simple tasks
  S2 = medium complexity, code/multi-step/tools
  S3 = high complexity, formal proofs/verification

Usage:
    python scripts/eval_deberta_zeroshot.py
    python scripts/eval_deberta_zeroshot.py --offline          # mock classifier
    python scripts/eval_deberta_zeroshot.py --verbose
    python scripts/eval_deberta_zeroshot.py --s2-threshold 0.25 --s3-threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground truth search paths (same convention as knn_router.py)
# ---------------------------------------------------------------------------
_GT_SEARCH_PATHS = [
    Path.cwd() / "config" / "routing_ground_truth.json",
    Path.cwd() / "sage-python" / "config" / "routing_ground_truth.json",
    Path(__file__).parent.parent / "config" / "routing_ground_truth.json",
]

# ---------------------------------------------------------------------------
# Task-type groups for mapping to cognitive systems
# ---------------------------------------------------------------------------
# These can be overridden via --s3-task-types / --s1-task-types CLI flags.
DEFAULT_S1_TASK_TYPES = frozenset({
    "Open QA",
    "Closed QA",
    "Chatbot",
    "Extraction",
})

DEFAULT_S3_TASK_TYPES = frozenset({
    # None of the 11 NVIDIA categories map cleanly to "formal proof",
    # so S3 is driven primarily by complexity score + reasoning score.
})

# Thresholds for the composite complexity score (0-1 range).
DEFAULT_S2_THRESHOLD = 0.20   # complexity >= this AND not S3 => S2
DEFAULT_S3_THRESHOLD = 0.45   # complexity >= this => S3
DEFAULT_S3_REASONING = 0.40   # reasoning >= this can also trigger S3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ClassifierOutput:
    """Single-task output from DeBERTa classifier."""
    task_type_1: str
    task_type_2: str
    task_type_prob: float
    creativity: float
    reasoning: float
    contextual_knowledge: float
    domain_knowledge: float
    constraint_ct: float
    few_shots: float
    complexity_score: float


@dataclass
class MappingConfig:
    """Configurable mapping from DeBERTa outputs to S1/S2/S3."""
    s1_task_types: frozenset[str] = DEFAULT_S1_TASK_TYPES
    s3_task_types: frozenset[str] = DEFAULT_S3_TASK_TYPES
    s2_threshold: float = DEFAULT_S2_THRESHOLD
    s3_threshold: float = DEFAULT_S3_THRESHOLD
    s3_reasoning: float = DEFAULT_S3_REASONING

    def map_to_system(self, output: ClassifierOutput) -> int:
        """Map a ClassifierOutput to cognitive system 1, 2, or 3.

        Decision logic (evaluated in order):
        1. complexity >= s3_threshold  => S3
        2. reasoning >= s3_reasoning AND complexity >= s2_threshold => S3
        3. task_type in s3_task_types  => S3
        4. complexity >= s2_threshold  => S2
        5. task_type NOT in s1_task_types AND complexity >= s2_threshold * 0.8 => S2
        6. Otherwise => S1
        """
        # S3: high complexity
        if output.complexity_score >= self.s3_threshold:
            return 3

        # S3: high reasoning + non-trivial complexity
        if (output.reasoning >= self.s3_reasoning
                and output.complexity_score >= self.s2_threshold):
            return 3

        # S3: explicit task type match
        if output.task_type_1 in self.s3_task_types:
            return 3

        # S2: medium complexity
        if output.complexity_score >= self.s2_threshold:
            return 2

        # S2: non-simple task type with borderline complexity
        if (output.task_type_1 not in self.s1_task_types
                and output.complexity_score >= self.s2_threshold * 0.8):
            return 2

        # S1: everything else
        return 1


@dataclass
class EvalResult:
    """Aggregate evaluation result."""
    total: int = 0
    correct: int = 0
    per_system: dict[int, dict[str, int]] = field(default_factory=dict)
    confusion: dict[tuple[int, int], int] = field(default_factory=dict)
    misroutes: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


# ---------------------------------------------------------------------------
# DeBERTa classifier wrapper
# ---------------------------------------------------------------------------
class DeBERTaClassifier:
    """Wraps the NVIDIA DeBERTa prompt-task-and-complexity-classifier."""

    def __init__(self, model_name: str = "nvidia/prompt-task-and-complexity-classifier"):
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading model: {model_name}")
        t0 = time.perf_counter()

        # Load config.json directly -- AutoConfig fails because the NVIDIA
        # model has no standard model_type key.
        cfg_path = hf_hub_download(model_name, "config.json")
        with open(cfg_path) as f:
            _raw_cfg = json.load(f)

        class _CfgNamespace:
            """Thin wrapper so downstream code can do cfg.target_sizes etc."""
            def __init__(self, d: dict):
                self.__dict__.update(d)

        self._config = _CfgNamespace(_raw_cfg)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- Replicate the NVIDIA custom model architecture ----------------
        class MeanPooling(nn.Module):
            def forward(self, last_hidden_state, attention_mask):
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        class MulticlassHead(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x):
                return self.fc(x)

        class CustomModel(nn.Module, PyTorchModelHubMixin):
            def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
                super().__init__()
                self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
                self.target_sizes = target_sizes.values()
                self.task_type_map = task_type_map
                self.weights_map = weights_map
                self.divisor_map = divisor_map
                self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz)
                              for sz in self.target_sizes]
                for i, head in enumerate(self.heads):
                    self.add_module(f"head_{i}", head)
                self.pool = MeanPooling()

            def compute_results(self, preds, target, decimal=4):
                if target == "task_type":
                    top2_indices = torch.topk(preds, k=2, dim=1).indices
                    softmax_probs = torch.softmax(preds, dim=1)
                    top2_probs = softmax_probs.gather(1, top2_indices)
                    top2 = top2_indices.detach().cpu().tolist()
                    top2_prob = top2_probs.detach().cpu().tolist()
                    top2_strings = [
                        [self.task_type_map[str(idx)] for idx in sample]
                        for sample in top2
                    ]
                    top2_prob_rounded = [
                        [round(v, 3) for v in sublist] for sublist in top2_prob
                    ]
                    for i, sublist in enumerate(top2_prob_rounded):
                        if sublist[1] < 0.1:
                            top2_strings[i][1] = "NA"
                    return (
                        [s[0] for s in top2_strings],
                        [s[1] for s in top2_strings],
                        [s[0] for s in top2_prob_rounded],
                    )
                else:
                    preds = torch.softmax(preds, dim=1)
                    weights = np.array(self.weights_map[target])
                    weighted_sum = np.sum(
                        np.array(preds.detach().cpu()) * weights, axis=1
                    )
                    scores = weighted_sum / self.divisor_map[target]
                    scores = [round(float(v), decimal) for v in scores]
                    if target == "number_of_few_shots":
                        scores = [x if x >= 0.05 else 0 for x in scores]
                    return scores

            def process_logits(self, logits):
                result = {}
                tt = self.compute_results(logits[0], "task_type")
                result["task_type_1"] = tt[0]
                result["task_type_2"] = tt[1]
                result["task_type_prob"] = tt[2]
                for i, target in enumerate(
                    ["creativity_scope", "reasoning", "contextual_knowledge",
                     "number_of_few_shots", "domain_knowledge",
                     "no_label_reason", "constraint_ct"],
                    start=1,
                ):
                    result[target] = self.compute_results(logits[i], target)
                result["prompt_complexity_score"] = [
                    round(
                        0.35 * c + 0.25 * r + 0.15 * con + 0.15 * dk
                        + 0.05 * ck + 0.05 * fs, 5
                    )
                    for c, r, con, dk, ck, fs in zip(
                        result["creativity_scope"], result["reasoning"],
                        result["constraint_ct"], result["domain_knowledge"],
                        result["contextual_knowledge"],
                        result["number_of_few_shots"],
                    )
                ]
                return result

            def forward(self, batch):
                outputs = self.backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"])
                logits = [head(pooled) for head in self.heads]
                return self.process_logits(logits)

        # Load the pretrained weights
        self._model = CustomModel(
            target_sizes=self._config.target_sizes,
            task_type_map=self._config.task_type_map,
            weights_map=self._config.weights_map,
            divisor_map=self._config.divisor_map,
        ).from_pretrained(model_name)
        self._model.eval()
        self._torch = torch

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Model loaded in {elapsed:.0f}ms")

    def classify(self, text: str) -> ClassifierOutput:
        """Classify a single prompt and return structured output."""
        encoded = self._tokenizer(
            [f"Prompt: {text}"],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        with self._torch.no_grad():
            result = self._model(encoded)

        return ClassifierOutput(
            task_type_1=result["task_type_1"][0],
            task_type_2=result["task_type_2"][0],
            task_type_prob=result["task_type_prob"][0],
            creativity=result["creativity_scope"][0],
            reasoning=result["reasoning"][0],
            contextual_knowledge=result["contextual_knowledge"][0],
            domain_knowledge=result["domain_knowledge"][0],
            constraint_ct=result["constraint_ct"][0],
            few_shots=result["number_of_few_shots"][0],
            complexity_score=result["prompt_complexity_score"][0],
        )

    def classify_batch(self, texts: list[str], batch_size: int = 8) -> list[ClassifierOutput]:
        """Classify a batch of prompts (tokenizes in chunks)."""
        results: list[ClassifierOutput] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = self._tokenizer(
                [f"Prompt: {t}" for t in chunk],
                return_tensors="pt",
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            with self._torch.no_grad():
                raw = self._model(encoded)

            for i in range(len(chunk)):
                results.append(ClassifierOutput(
                    task_type_1=raw["task_type_1"][i],
                    task_type_2=raw["task_type_2"][i],
                    task_type_prob=raw["task_type_prob"][i],
                    creativity=raw["creativity_scope"][i],
                    reasoning=raw["reasoning"][i],
                    contextual_knowledge=raw["contextual_knowledge"][i],
                    domain_knowledge=raw["domain_knowledge"][i],
                    constraint_ct=raw["constraint_ct"][i],
                    few_shots=raw["number_of_few_shots"][i],
                    complexity_score=raw["prompt_complexity_score"][i],
                ))
        return results


# ---------------------------------------------------------------------------
# Mock classifier (--offline mode)
# ---------------------------------------------------------------------------
class MockClassifier:
    """Deterministic mock for testing the evaluation pipeline without HF model.

    Uses simple keyword heuristics to produce plausible-looking outputs.
    NOT meant to be accurate -- just exercises the full eval pipeline.
    """

    # Keywords that nudge toward code/reasoning/formal tasks
    _CODE_KW = {"python", "code", "implement", "function", "write", "debug",
                "refactor", "api", "sql", "test", "build", "parse", "script",
                "scraper", "decorator", "pipeline", "migration", "neural",
                "bloom", "concurrent", "asyncio"}
    _FORMAL_KW = {"prove", "proof", "verify", "induction", "invariant",
                  "z3", "smt", "ltl", "turing", "decidable", "riemann",
                  "cantor", "ramsey", "linearizability", "deadlock",
                  "lambda calculus", "type preservation", "completeness",
                  "correctness", "safety", "liveness", "sound"}

    def classify(self, text: str) -> ClassifierOutput:
        lower = text.lower()
        words = set(lower.split())

        formal_hits = sum(1 for kw in self._FORMAL_KW if kw in lower)
        code_hits = sum(1 for kw in self._CODE_KW if kw in words)

        if formal_hits >= 2:
            return ClassifierOutput(
                task_type_1="Text Generation", task_type_2="NA",
                task_type_prob=0.6, creativity=0.05, reasoning=0.85,
                contextual_knowledge=0.1, domain_knowledge=0.9,
                constraint_ct=0.7, few_shots=0, complexity_score=0.55,
            )
        elif formal_hits == 1:
            return ClassifierOutput(
                task_type_1="Text Generation", task_type_2="NA",
                task_type_prob=0.5, creativity=0.05, reasoning=0.65,
                contextual_knowledge=0.1, domain_knowledge=0.7,
                constraint_ct=0.5, few_shots=0, complexity_score=0.42,
            )
        elif code_hits >= 2:
            return ClassifierOutput(
                task_type_1="Code Generation", task_type_2="Text Generation",
                task_type_prob=0.75, creativity=0.1, reasoning=0.3,
                contextual_knowledge=0.1, domain_knowledge=0.8,
                constraint_ct=0.4, few_shots=0, complexity_score=0.30,
            )
        elif code_hits == 1:
            return ClassifierOutput(
                task_type_1="Code Generation", task_type_2="NA",
                task_type_prob=0.6, creativity=0.1, reasoning=0.15,
                contextual_knowledge=0.1, domain_knowledge=0.5,
                constraint_ct=0.3, few_shots=0, complexity_score=0.22,
            )
        else:
            return ClassifierOutput(
                task_type_1="Open QA", task_type_2="NA",
                task_type_prob=0.8, creativity=0.05, reasoning=0.05,
                contextual_knowledge=0.1, domain_knowledge=0.1,
                constraint_ct=0.1, few_shots=0, complexity_score=0.08,
            )

    def classify_batch(self, texts: list[str], batch_size: int = 8) -> list[ClassifierOutput]:
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------
def bootstrap_ci(
    correct: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for accuracy.

    Parameters
    ----------
    correct : np.ndarray of bool
        Per-task correctness indicators (True/False).
    n_resamples : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (mean, ci_low, ci_high) : tuple[float, float, float]
    """
    rng = np.random.RandomState(seed)
    n = len(correct)
    accuracies = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(correct, size=n, replace=True)
        accuracies[i] = sample.mean()

    alpha = 1.0 - ci
    lo = np.percentile(accuracies, 100 * alpha / 2)
    hi = np.percentile(accuracies, 100 * (1 - alpha / 2))
    return float(correct.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def load_ground_truth(gt_path: str | None) -> list[dict]:
    """Load ground truth tasks from JSON."""
    if gt_path:
        path = Path(gt_path)
    else:
        path = None
        for candidate in _GT_SEARCH_PATHS:
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        print(f"ERROR: Ground truth not found. Searched: {_GT_SEARCH_PATHS}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data["tasks"]
    print(f"Loaded {len(tasks)} ground truth tasks from {path}")
    return tasks


def run_evaluation(
    classifier: DeBERTaClassifier | MockClassifier,
    tasks: list[dict],
    mapping: MappingConfig,
    verbose: bool = False,
) -> EvalResult:
    """Run zero-shot evaluation."""
    result = EvalResult(total=len(tasks))
    for sys_id in [1, 2, 3]:
        result.per_system[sys_id] = {"total": 0, "correct": 0, "predicted": 0}

    texts = [t["task"] for t in tasks]

    print(f"\nClassifying {len(texts)} tasks...")
    t0 = time.perf_counter()

    # Batch classify (DeBERTa) or one-by-one (mock)
    if isinstance(classifier, DeBERTaClassifier):
        outputs = classifier.classify_batch(texts, batch_size=8)
    else:
        outputs = classifier.classify_batch(texts)

    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Classification done in {result.elapsed_ms:.0f}ms "
          f"({result.elapsed_ms / len(texts):.1f}ms/task)")

    # Evaluate each task
    for task_entry, output in zip(tasks, outputs):
        expected = task_entry["expected_system"]
        predicted = mapping.map_to_system(output)

        result.per_system[expected]["total"] += 1
        result.per_system[predicted]["predicted"] += 1

        cm_key = (expected, predicted)
        result.confusion[cm_key] = result.confusion.get(cm_key, 0) + 1

        pred_record = {
            "id": task_entry["id"],
            "task": task_entry["task"][:80],
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "task_type": output.task_type_1,
            "complexity": output.complexity_score,
            "reasoning": output.reasoning,
            "domain": task_entry.get("domain", ""),
        }
        result.predictions.append(pred_record)

        if predicted == expected:
            result.correct += 1
            result.per_system[expected]["correct"] += 1
        else:
            result.misroutes.append({
                **pred_record,
                "task_type_2": output.task_type_2,
                "task_type_prob": output.task_type_prob,
                "creativity": output.creativity,
                "domain_knowledge": output.domain_knowledge,
                "constraint_ct": output.constraint_ct,
            })

        if verbose:
            status = "OK" if predicted == expected else f"MISS (got S{predicted})"
            print(
                f"  [{task_entry['id']:2d}] S{expected} {status} | "
                f"type={output.task_type_1:<16s} "
                f"cmplx={output.complexity_score:.3f} "
                f"reason={output.reasoning:.3f} "
                f"| {task_entry['task'][:55]}"
            )

    return result


def print_results(result: EvalResult, mapping: MappingConfig) -> None:
    """Print formatted evaluation results."""
    width = 72
    print("\n" + "=" * width)
    print("  NVIDIA DeBERTa Zero-Shot Evaluation Results")
    print("=" * width)

    # Overall accuracy
    print(f"\n  Overall accuracy: {result.accuracy:.1%} ({result.correct}/{result.total})")
    print(f"  Inference time:   {result.elapsed_ms:.0f}ms total, "
          f"{result.elapsed_ms / result.total:.1f}ms/task")

    # Mapping config
    print(f"\n  Mapping config:")
    print(f"    S2 threshold (complexity): {mapping.s2_threshold:.2f}")
    print(f"    S3 threshold (complexity): {mapping.s3_threshold:.2f}")
    print(f"    S3 reasoning threshold:    {mapping.s3_reasoning:.2f}")
    print(f"    S1 task types: {sorted(mapping.s1_task_types)}")
    if mapping.s3_task_types:
        print(f"    S3 task types: {sorted(mapping.s3_task_types)}")

    # Per-system precision / recall / F1
    print(f"\n  {'System':<8s} {'Prec':>6s} {'Recall':>8s} {'F1':>6s}  "
          f"{'TP':>4s} {'FP':>4s} {'FN':>4s}  {'Support':>7s}")
    print("  " + "-" * 58)

    for sys_id in [1, 2, 3]:
        tp = result.per_system[sys_id]["correct"]
        support = result.per_system[sys_id]["total"]       # actual positives
        predicted = result.per_system[sys_id]["predicted"]  # predicted positives
        fn = support - tp
        fp = predicted - tp

        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        print(f"  S{sys_id:<7d} {precision:>5.1%} {recall:>7.1%} {f1:>5.1%}  "
              f"{tp:>4d} {fp:>4d} {fn:>4d}  {support:>7d}")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=expected, cols=predicted):")
    print(f"  {'':>12s} {'pred S1':>8s} {'pred S2':>8s} {'pred S3':>8s}")
    for exp in [1, 2, 3]:
        row = []
        for pred in [1, 2, 3]:
            row.append(result.confusion.get((exp, pred), 0))
        print(f"  {'expect S' + str(exp):>12s} {row[0]:>8d} {row[1]:>8d} {row[2]:>8d}")

    # Bootstrap CI
    correct_arr = np.array([p["correct"] for p in result.predictions], dtype=bool)
    mean_acc, ci_lo, ci_hi = bootstrap_ci(correct_arr, n_resamples=1000)
    print(f"\n  Bootstrap 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}] (mean={mean_acc:.1%})")
    if ci_lo >= 0.90:
        print("  >>> CI lower bound >= 90%: WIRE AS-IS (no fine-tuning needed)")
    elif ci_lo >= 0.80:
        print("  >>> CI lower bound >= 80%: PROMISING, but fine-tuning recommended")
    else:
        print("  >>> CI lower bound < 80%: FINE-TUNING REQUIRED")

    # Comparison with baselines
    print(f"\n  Baseline comparison:")
    print(f"    DeBERTa zero-shot:  {result.accuracy:.1%} ({result.correct}/{result.total})")
    print(f"    kNN (arctic-embed): 92.0% (46/50)  [current SAGE Stage 0.5]")
    print(f"    Keyword heuristic:  52.0% (26/50)  [original ComplexityRouter]")
    delta_knn = (result.accuracy - 0.92) * 100
    delta_heur = (result.accuracy - 0.52) * 100
    print(f"    Delta vs kNN:       {delta_knn:+.1f}pp")
    print(f"    Delta vs heuristic: {delta_heur:+.1f}pp")

    # Misroutes detail
    if result.misroutes:
        print(f"\n  Misrouted tasks ({len(result.misroutes)}):")
        print(f"  {'ID':>4s} {'Exp':>4s} {'Got':>4s} {'Type':<18s} "
              f"{'Cmplx':>6s} {'Reason':>6s} {'Task'}")
        print("  " + "-" * 68)
        for m in result.misroutes:
            print(f"  {m['id']:>4d} S{m['expected']}   S{m['predicted']}   "
                  f"{m['task_type']:<18s} {m['complexity']:>5.3f} "
                  f"{m['reasoning']:>5.3f}  {m['task'][:40]}")

    # Task type distribution
    print(f"\n  Task type distribution (DeBERTa predictions):")
    type_counts: dict[str, int] = {}
    for p in result.predictions:
        tt = p["task_type"]
        type_counts[tt] = type_counts.get(tt, 0) + 1
    for tt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {tt:<20s} {count:>3d} ({count / result.total:.0%})")

    # Complexity score distribution per expected system
    print(f"\n  Complexity score distribution by expected system:")
    for sys_id in [1, 2, 3]:
        scores = [
            p["complexity"] for p in result.predictions if p["expected"] == sys_id
        ]
        if scores:
            arr = np.array(scores)
            print(f"    S{sys_id}: mean={arr.mean():.3f}  "
                  f"std={arr.std():.3f}  "
                  f"min={arr.min():.3f}  max={arr.max():.3f}  "
                  f"n={len(arr)}")

    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of NVIDIA DeBERTa classifier on SAGE routing GT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_deberta_zeroshot.py
  python scripts/eval_deberta_zeroshot.py --offline
  python scripts/eval_deberta_zeroshot.py --verbose --s3-threshold 0.50
  python scripts/eval_deberta_zeroshot.py --model nvidia/prompt-task-and-complexity-classifier
""",
    )
    parser.add_argument(
        "--model", type=str,
        default="nvidia/prompt-task-and-complexity-classifier",
        help="HuggingFace model ID (default: nvidia/prompt-task-and-complexity-classifier)",
    )
    parser.add_argument(
        "--gt-path", type=str, default=None,
        help="Path to ground truth JSON (default: auto-detect config/routing_ground_truth.json)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use mock classifier (no model download). Tests the eval pipeline.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-task classification details.",
    )

    # Mapping thresholds
    thresh = parser.add_argument_group("mapping thresholds")
    thresh.add_argument(
        "--s2-threshold", type=float, default=DEFAULT_S2_THRESHOLD,
        help=f"Complexity score threshold for S2 (default: {DEFAULT_S2_THRESHOLD})",
    )
    thresh.add_argument(
        "--s3-threshold", type=float, default=DEFAULT_S3_THRESHOLD,
        help=f"Complexity score threshold for S3 (default: {DEFAULT_S3_THRESHOLD})",
    )
    thresh.add_argument(
        "--s3-reasoning", type=float, default=DEFAULT_S3_REASONING,
        help=f"Reasoning score threshold for S3 override (default: {DEFAULT_S3_REASONING})",
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save detailed results to JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 72)
    print("  NVIDIA DeBERTa Zero-Shot Routing Evaluation")
    print("  Model: " + args.model)
    print("=" * 72)

    # Load ground truth
    tasks = load_ground_truth(args.gt_path)
    sys_counts = {1: 0, 2: 0, 3: 0}
    for t in tasks:
        sys_counts[t["expected_system"]] += 1
    print(f"  S1: {sys_counts[1]}, S2: {sys_counts[2]}, S3: {sys_counts[3]}")

    # Initialize classifier
    use_offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"

    if use_offline:
        print("\n[OFFLINE MODE] Using mock classifier (keyword heuristic)")
        print("  Results are NOT representative of actual DeBERTa performance.")
        classifier: DeBERTaClassifier | MockClassifier = MockClassifier()
    else:
        try:
            classifier = DeBERTaClassifier(args.model)
        except Exception as exc:
            print(f"\nERROR: Failed to load model: {exc}")
            print("Possible causes:")
            print("  - No internet / corporate proxy blocking HuggingFace")
            print("  - Missing packages: pip install transformers torch huggingface_hub")
            print("  - HF_HUB_OFFLINE=1 set in environment")
            print("\nRe-run with --offline to test the pipeline with a mock classifier.")
            sys.exit(1)

    # Configure mapping
    mapping = MappingConfig(
        s2_threshold=args.s2_threshold,
        s3_threshold=args.s3_threshold,
        s3_reasoning=args.s3_reasoning,
    )

    # Run evaluation
    result = run_evaluation(classifier, tasks, mapping, verbose=args.verbose)

    # Print results
    print_results(result, mapping)

    # Save to JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": args.model,
            "offline": use_offline,
            "mapping": {
                "s2_threshold": mapping.s2_threshold,
                "s3_threshold": mapping.s3_threshold,
                "s3_reasoning": mapping.s3_reasoning,
                "s1_task_types": sorted(mapping.s1_task_types),
            },
            "accuracy": result.accuracy,
            "correct": result.correct,
            "total": result.total,
            "elapsed_ms": result.elapsed_ms,
            "per_system": result.per_system,
            "confusion_matrix": {
                f"S{e}_to_S{p}": c for (e, p), c in result.confusion.items()
            },
            "predictions": result.predictions,
            "misroutes": result.misroutes,
            "bootstrap_95ci": {
                "mean": float(np.mean([p["correct"] for p in result.predictions])),
                "ci_low": bootstrap_ci(
                    np.array([p["correct"] for p in result.predictions], dtype=bool)
                )[1],
                "ci_high": bootstrap_ci(
                    np.array([p["correct"] for p in result.predictions], dtype=bool)
                )[2],
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed results saved to {out_path}")

    # Exit code: 0 if accuracy >= 80%, 1 otherwise (for CI gating)
    sys.exit(0 if result.accuracy >= 0.80 else 1)


if __name__ == "__main__":
    main()
