"""kNN-based S1/S2/S3 routing using pre-computed exemplar embeddings.

Research backing: arXiv 2505.12601 shows simple kNN on embeddings
outperforms MLP, GNN, and attention-based routers for LLM routing.

At routing time, embeds the input task, finds k nearest neighbors
from the exemplar bank, and uses distance-weighted majority vote.

Falls back to None when:
- Exemplar file not found or corrupt
- Embedder produces non-semantic (hash) embeddings
- All neighbors below confidence threshold
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.memory.embedder import Embedder

import numpy as np

_log = logging.getLogger(__name__)

# Rust kNN hot-path acceleration
try:
    from sage_core import RustKnnRouter as _RustKnn
    _HAS_RUST_KNN = True
except ImportError:
    _HAS_RUST_KNN = False

# Search paths for pre-computed exemplar embeddings
_EXEMPLAR_SEARCH_PATHS = [
    Path.cwd() / "config" / "routing_exemplars.npz",
    Path.cwd() / "sage-python" / "config" / "routing_exemplars.npz",
    Path(__file__).parent.parent.parent.parent / "config" / "routing_exemplars.npz",
    Path.home() / ".sage" / "routing_exemplars.npz",
]

# Search paths for ground truth JSON (used for in-memory build)
_GT_SEARCH_PATHS = [
    Path.cwd() / "config" / "routing_ground_truth.json",
    Path.cwd() / "sage-python" / "config" / "routing_ground_truth.json",
    Path(__file__).parent.parent.parent.parent / "config" / "routing_ground_truth.json",
]


@dataclass
class KnnRoutingResult:
    """Result from kNN routing."""
    system: int           # 1, 2, or 3
    confidence: float     # 0.0-1.0 (similarity-weighted)
    nearest_distance: float  # cosine similarity of nearest neighbor
    k_distances: list[float] = field(default_factory=list)
    k_labels: list[int] = field(default_factory=list)
    method: str = "knn"


class KnnRouter:
    """kNN-based cognitive system router.

    Parameters
    ----------
    exemplar_path : Path, optional
        Path to pre-computed exemplar .npz file.
    embedder : Embedder, optional
        Shared embedder instance. Created lazily if not provided.
    k : int
        Number of nearest neighbors for majority vote.
    distance_threshold : float
        Minimum cosine similarity to accept routing (OOD rejection).
    """

    def __init__(
        self,
        exemplar_path: Path | None = None,
        embedder: Embedder | None = None,
        k: int = 5,
        distance_threshold: float = 0.3,
    ):
        self._k = k
        self._distance_threshold = distance_threshold
        self._embedder = embedder

        # Exemplar data (N x 768 float32, N int labels)
        self._exemplar_embeddings: np.ndarray | None = None
        self._exemplar_labels: np.ndarray | None = None
        self._exemplar_count = 0

        # Try loading pre-computed exemplars
        if exemplar_path is not None:
            self._load_exemplars(exemplar_path)
        else:
            self._try_auto_load()

        # Rust kNN hot-path: load exemplars into RustKnnRouter if available
        self._rust_knn = None
        if _HAS_RUST_KNN and self._exemplar_embeddings is not None:
            try:
                self._rust_knn = _RustKnn(k=self._k, distance_threshold=self._distance_threshold)
                self._rust_knn.load_exemplars(
                    self._exemplar_embeddings.flatten().tolist(),
                    self._exemplar_labels.tolist(),
                    self._exemplar_embeddings.shape[1],
                )
                _log.info("kNN: Rust acceleration active (%d exemplars)", self._rust_knn.exemplar_count())
            except Exception as e:
                _log.warning("kNN: Rust acceleration unavailable: %s", e)
                self._rust_knn = None

    @property
    def is_ready(self) -> bool:
        """True if exemplars are loaded and embedder is semantic."""
        emb = self._get_embedder()
        return (
            self._exemplar_embeddings is not None
            and self._exemplar_count > 0
            and emb is not None
            and not getattr(emb, "is_hash_fallback", True)
        )

    @property
    def exemplar_count(self) -> int:
        return self._exemplar_count

    @property
    def embedder_backend(self) -> str:
        emb = self._get_embedder()
        if emb is None:
            return "none"
        return getattr(emb, '_backend', 'unknown')

    def route(self, task: str) -> KnnRoutingResult | None:
        """Route a task using kNN on embeddings.

        Returns None if routing is not possible or confidence is too low.
        """
        if self._exemplar_embeddings is None:
            return None

        emb = self._get_embedder()
        if emb is None:
            return None

        # Refuse to route on hash embeddings (per CLAUDE.md policy)
        if emb.is_hash_fallback:
            return None

        # Rust hot-path: try RustKnnRouter first (avoids Python numpy overhead)
        if self._rust_knn is not None and self._rust_knn.has_exemplars():
            try:
                query_vec = np.array(emb.embed(task), dtype=np.float32)
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec /= norm
                result = self._rust_knn.route(query_vec.tolist())
                if result is not None:
                    system, confidence, nearest_dist = result
                    return KnnRoutingResult(
                        system=int(system),
                        confidence=confidence,
                        nearest_distance=nearest_dist,
                        method="rust_knn",
                    )
            except Exception as e:
                _log.warning("kNN: Rust route failed, falling back to Python: %s", e)

        # Embed the query task (Python fallback path)
        try:
            query_vec = np.array(emb.embed(task), dtype=np.float32)
        except Exception as exc:
            _log.warning("kNN embed failed: %s", exc)
            return None

        # L2-normalize query (arctic-embed-m outputs are already normalized,
        # but belt-and-suspenders for other backends)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Cosine similarity = dot product (both vectors are unit-normalized)
        similarities = self._exemplar_embeddings @ query_vec  # shape: (N,)

        # Find top-k indices
        k = min(self._k, len(similarities))
        top_k_idx = np.argpartition(similarities, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]

        k_distances = similarities[top_k_idx].tolist()
        k_labels = self._exemplar_labels[top_k_idx].tolist()

        # OOD rejection: if nearest neighbor is below threshold
        if k_distances[0] < self._distance_threshold:
            _log.debug(
                "kNN: OOD rejection (max_sim=%.3f < threshold=%.3f)",
                k_distances[0], self._distance_threshold,
            )
            return None

        # Distance-weighted majority vote
        votes: dict[int, float] = {}
        for dist, label in zip(k_distances, k_labels):
            weight = max(dist, 0.0)  # only positive similarities count
            votes[label] = votes.get(label, 0.0) + weight

        if not votes:
            return None

        winner = max(votes, key=lambda s: votes[s])
        total_weight = sum(votes.values())
        confidence = votes[winner] / total_weight if total_weight > 0 else 0.0

        return KnnRoutingResult(
            system=int(winner),
            confidence=confidence,
            nearest_distance=k_distances[0],
            k_distances=k_distances,
            k_labels=k_labels,
        )

    def build_from_ground_truth(
        self, gt_path: Path | None = None, save_path: Path | None = None,
    ) -> bool:
        """Build exemplar embeddings from ground truth JSON.

        Can be called at runtime (e.g., during boot) if .npz is missing.
        Returns True on success.
        """
        emb = self._get_embedder()
        if emb is None or emb.is_hash_fallback:
            _log.warning("kNN: cannot build exemplars without semantic embedder")
            return False

        # Find ground truth file
        path = gt_path
        if path is None:
            for candidate in _GT_SEARCH_PATHS:
                if candidate.exists():
                    path = candidate
                    break
        if path is None or not path.exists():
            _log.warning("kNN: ground truth file not found")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            tasks = gt_data["tasks"]
            texts = [t["task"] for t in tasks]
            labels = [t["expected_system"] for t in tasks]

            _log.info("kNN: embedding %d ground truth tasks...", len(texts))
            vectors = emb.embed_batch(texts)

            self._exemplar_embeddings = np.array(vectors, dtype=np.float32)
            # L2-normalize rows
            norms = np.linalg.norm(self._exemplar_embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            self._exemplar_embeddings /= norms

            self._exemplar_labels = np.array(labels, dtype=np.int32)
            self._exemplar_count = len(labels)

            _log.info(
                "kNN: built %d exemplars (embedder=%s)",
                self._exemplar_count, emb._backend,
            )

            # Optionally save to .npz
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    save_path,
                    embeddings=self._exemplar_embeddings,
                    labels=self._exemplar_labels,
                )
                _log.info("kNN: saved exemplars to %s", save_path)

            return True
        except Exception as exc:
            _log.warning("kNN: build_from_ground_truth failed: %s", exc)
            return False

    # -- Private ---------------------------------------------------------------

    def _get_embedder(self) -> Embedder | None:
        """Lazy-load embedder if not provided."""
        if self._embedder is None:
            try:
                from sage.memory.embedder import Embedder
                self._embedder = Embedder()
            except Exception:
                return None
        return self._embedder

    def _load_exemplars(self, path: Path) -> bool:
        """Load pre-computed exemplars from .npz file."""
        try:
            data = np.load(path)
            embeddings = data["embeddings"]
            labels = data["labels"]

            if embeddings.ndim != 2 or embeddings.shape[1] != 768:
                _log.warning(
                    "kNN: invalid exemplar shape %s (expected Nx768)",
                    embeddings.shape,
                )
                return False

            if len(embeddings) != len(labels):
                _log.warning("kNN: embedding/label count mismatch")
                return False

            # Ensure L2-normalized
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            self._exemplar_embeddings = (embeddings / norms).astype(np.float32)
            self._exemplar_labels = labels.astype(np.int32)
            self._exemplar_count = len(labels)

            _log.info("kNN: loaded %d exemplars from %s", self._exemplar_count, path)
            return True
        except Exception as exc:
            _log.warning("kNN: failed to load exemplars from %s: %s", path, exc)
            return False

    def _try_auto_load(self) -> None:
        """Try to find and load exemplars from standard paths."""
        for path in _EXEMPLAR_SEARCH_PATHS:
            if path.exists():
                if self._load_exemplars(path):
                    return
