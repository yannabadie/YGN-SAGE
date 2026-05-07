"""Routing GT invariants — pins routing.knn_92pct + routing.system_router_88pct.

This file pins the headline routing-accuracy claims to strict-equal
integer-count CI floors on the 60-task human-labeled ground truth at
``sage-python/config/routing_ground_truth.json``.

Two independent tests:

* ``test_knn_routing_exemplars_loo_cv_floor`` — pure-NumPy LOO-CV over the
  committed embedding artifact ``sage-python/config/routing_exemplars.npz``,
  byte-equivalent to ``scripts/build_routing_exemplars.py --validate``.
* ``test_system_router_gt_floor`` — Rust legacy ``SystemRouter.route(task, 1.0)``
  over the same 60-task GT, comparing the returned cognitive system against
  the human label. Budget value is part of the invariant.

Floor philosophy (cgpro DESIGN_LOCK + amendment 2026-05-07,
``cgpro_routing_claims_evidence_20260507``):

This test pins the committed routing baseline. If
``routing_ground_truth.json``, ``routing_exemplars.npz``, ``cards.toml``, or
the scoring/router logic changes intentionally, update the artifact and
these floor constants in the same commit, with the new measured baseline
documented in ``docs/claims/routing.yaml``. Strict-equal floors are
chosen — not loose-margin floors — because the dataset and scoring rule
are frozen, hence the only sources of variance are intentional artifact
regeneration or unintended router/scoring drift. Either should be visible
in CI immediately, not absorbed by a margin.

The historical ~92% / ~88% figures were measured on an earlier 50-task GT
subset and are *provenance only*, not recertified by these floors. See
``docs/claims/routing.yaml``.

Out of scope (locked):

* No regeneration of ``routing_exemplars.npz``.
* No refactor of ``sage.strategy.knn_router`` or
  ``sage-core/src/routing/system_router.rs``.
* No change to ``route()`` / ``route_integrated()`` / ``route_constrained()`` signatures.
* No live embedder load.
* No benchmark JSON written by pytest (would dirty the working tree).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GT_PATH = _REPO_ROOT / "sage-python" / "config" / "routing_ground_truth.json"
_NPZ_PATH = _REPO_ROOT / "sage-python" / "config" / "routing_exemplars.npz"
_CARDS_PATH = _REPO_ROOT / "sage-core" / "config" / "cards.toml"

GT_TOTAL = 60
GT_PER_SYSTEM = {1: 20, 2: 20, 3: 20}

# Strict-equal empirical floors locked by cgpro
# (cgpro_routing_claims_evidence_20260507 amendment, 2026-05-07).
#
# Empirical baseline measured on HEAD 373ca63a:
#   kNN k=5 LOO    : 50/60 overall, S1=16/20, S2=15/20, S3=19/20
#   SystemRouter   : 52/60 overall (budget=1.0 and budget=1_000_000.0 give
#                    identical results on the current cards.toml; cgpro
#                    nonetheless mandates explicit budget=1.0 because budget
#                    is part of the legacy route() invariant).
KNN_K = 5
KNN_MIN_OVERALL_CORRECT = 50
KNN_MIN_PER_SYSTEM_CORRECT = {1: 16, 2: 15, 3: 19}

SYSTEM_ROUTER_BUDGET = 1.0
SYSTEM_ROUTER_MIN_OVERALL_CORRECT = 52


# ── Shared helpers ───────────────────────────────────────────────────────────


def _load_gt() -> tuple[list[dict], np.ndarray]:
    """Load the GT JSON. Asserts shape + label distribution + id sequence."""
    with open(_GT_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    tasks = data["tasks"]
    assert len(tasks) == GT_TOTAL, f"GT length drift: {len(tasks)} != {GT_TOTAL}"
    ids = [t["id"] for t in tasks]
    assert ids == list(range(1, GT_TOTAL + 1)), "GT ids no longer == [1..60]"
    labels = np.array([t["expected_system"] for t in tasks], dtype=np.int32)
    counts = {s: int((labels == s).sum()) for s in (1, 2, 3)}
    assert counts == GT_PER_SYSTEM, f"GT label counts drifted: {counts} != {GT_PER_SYSTEM}"
    return tasks, labels


def _load_npz_aligned(gt_labels: np.ndarray) -> np.ndarray:
    """Load the exemplar .npz, validate shape/dtype/L2-norm/label alignment.

    Returns the embedding matrix. The labels in the .npz must match the GT
    label sequence row-by-row; otherwise the LOO-CV would silently route
    against the wrong labels.
    """
    if not _NPZ_PATH.exists():
        pytest.skip(f"routing_exemplars.npz not found at {_NPZ_PATH}")
    blob = np.load(_NPZ_PATH)
    assert "embeddings" in blob.files and "labels" in blob.files, (
        f"routing_exemplars.npz missing required keys (got {blob.files!r})"
    )
    emb = blob["embeddings"]
    npz_labels = blob["labels"]
    assert emb.shape == (GT_TOTAL, 768), (
        f"embedding shape drift: {emb.shape} != ({GT_TOTAL}, 768)"
    )
    assert emb.dtype == np.float32, f"embedding dtype drift: {emb.dtype} != float32"
    assert npz_labels.shape == (GT_TOTAL,), (
        f"npz labels shape drift: {npz_labels.shape} != ({GT_TOTAL},)"
    )
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"npz embeddings not L2-normalized: norms range "
        f"[{float(norms.min()):.4f}, {float(norms.max()):.4f}]"
    )
    assert np.array_equal(npz_labels.astype(np.int32), gt_labels), (
        "npz labels do not match GT expected_system sequence — row-order drift"
    )
    return emb


def _knn_loo_predictions(emb: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Byte-equivalent LOO-CV mirroring scripts/build_routing_exemplars.py.

    For each row i: held-out row removed; cosine similarity over remaining
    rows (dot product on unit vectors); top-k via np.argpartition; weighted
    majority vote with raw similarity as weight; argmax over vote dict.
    """
    preds = np.empty(len(emb), dtype=np.int32)
    for i in range(len(emb)):
        bank = np.delete(emb, i, axis=0)
        bank_labels = np.delete(labels, i, axis=0)
        sims = bank @ emb[i]
        top_k = np.argpartition(sims, -k)[-k:]
        votes: dict[int, float] = {}
        for idx in top_k:
            label = int(bank_labels[idx])
            votes[label] = votes.get(label, 0.0) + float(sims[idx])
        preds[i] = max(votes, key=lambda s: votes[s])
    return preds


def _miss_lines(
    tasks: list[dict],
    expected: Iterable[int],
    predicted: Iterable[int],
) -> list[str]:
    """Render miss lines for failure messages."""
    out: list[str] = []
    for t, exp, pred in zip(tasks, expected, predicted, strict=True):
        if int(exp) != int(pred):
            text = t["task"][:60]
            out.append(f"  [{t['id']:2d}] expected=S{int(exp)} got=S{int(pred)}: {text}")
    return out


def _coerce_system(value: object) -> int:
    """Coerce CognitiveSystem (PyO3 enum) / int / str into an int 1, 2, or 3.

    The PyO3 binding for ``CognitiveSystem`` is declared with ``eq_int`` so
    ``value == 1`` works directly, but the underlying object is not int. Try
    integer comparison first, then ``str(value)``, then ``repr(value)``.
    """
    for s in (1, 2, 3):
        try:
            if value == s:
                return s
        except Exception:  # noqa: BLE001 — defensive: PyO3 may raise on cmp
            pass
    text = str(value)
    if text in {"S1", "1", "CognitiveSystem.S1"}:
        return 1
    if text in {"S2", "2", "CognitiveSystem.S2"}:
        return 2
    if text in {"S3", "3", "CognitiveSystem.S3"}:
        return 3
    raise AssertionError(f"unrecognised CognitiveSystem repr: {value!r}")


# ── Test 1 — kNN LOO-CV floor ────────────────────────────────────────────────


def test_knn_routing_exemplars_loo_cv_floor() -> None:
    """Pin routing.knn_92pct as a CI-defended floor on the frozen 60-task GT.

    Pure-NumPy, no embedder load, no I/O beyond reading the committed artifact.
    """
    tasks, labels = _load_gt()
    emb = _load_npz_aligned(labels)
    preds = _knn_loo_predictions(emb, labels, k=KNN_K)

    overall = int((preds == labels).sum())
    misses = _miss_lines(tasks, labels.tolist(), preds.tolist())
    assert overall >= KNN_MIN_OVERALL_CORRECT, (
        f"kNN k={KNN_K} LOO-CV: {overall}/{GT_TOTAL} correct, "
        f"floor is {KNN_MIN_OVERALL_CORRECT}/{GT_TOTAL}\n"
        f"misses ({len(misses)}):\n" + "\n".join(misses)
    )

    per_sys: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for exp, pred in zip(labels.tolist(), preds.tolist(), strict=True):
        if int(exp) == int(pred):
            per_sys[int(exp)] += 1

    for s in (1, 2, 3):
        floor = KNN_MIN_PER_SYSTEM_CORRECT[s]
        got = per_sys[s]
        total = GT_PER_SYSTEM[s]
        assert got >= floor, (
            f"kNN k={KNN_K} LOO-CV S{s}: {got}/{total} correct, "
            f"floor is {floor}/{total}\n"
            f"misses ({len(misses)}):\n" + "\n".join(misses)
        )


# ── Test 2 — Rust SystemRouter floor ────────────────────────────────────────


def test_system_router_gt_floor() -> None:
    """Pin routing.system_router_88pct as a CI-defended floor.

    Builds the live ``sage_core.SystemRouter(registry)`` from the canonical
    ``sage-core/config/cards.toml``, runs ``router.route(task, budget)`` over
    the 60-task GT, and asserts the floor on cognitive-system match counts.
    """
    sage_core = pytest.importorskip(
        "sage_core",
        reason="sage_core PyO3 extension not available (build with maturin develop)",
    )

    if not hasattr(sage_core, "ModelRegistry"):
        pytest.skip("sage_core.ModelRegistry not exposed in this build")
    if not hasattr(sage_core, "SystemRouter"):
        pytest.skip("sage_core.SystemRouter not exposed in this build")
    if not _CARDS_PATH.exists():
        pytest.skip(f"cards.toml not found at {_CARDS_PATH}")

    tasks, labels = _load_gt()

    registry = sage_core.ModelRegistry.from_toml_file(str(_CARDS_PATH))
    router = sage_core.SystemRouter(registry)

    preds: list[int] = []
    for task_obj in tasks:
        decision = router.route(task_obj["task"], SYSTEM_ROUTER_BUDGET)
        preds.append(_coerce_system(decision.system))

    overall = sum(1 for exp, pred in zip(labels.tolist(), preds, strict=True)
                  if int(exp) == int(pred))
    misses = _miss_lines(tasks, labels.tolist(), preds)
    assert overall >= SYSTEM_ROUTER_MIN_OVERALL_CORRECT, (
        f"SystemRouter.route(task, budget={SYSTEM_ROUTER_BUDGET:.0f}): "
        f"{overall}/{GT_TOTAL} correct, "
        f"floor is {SYSTEM_ROUTER_MIN_OVERALL_CORRECT}/{GT_TOTAL}\n"
        f"misses ({len(misses)}):\n" + "\n".join(misses)
    )
