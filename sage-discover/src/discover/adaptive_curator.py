"""src/discover/adaptive_curator.py — kNN + LinUCB bandit + self-feedback curation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from discover.curator import CuratedPaper, heuristic_filter, llm_score
from discover.discovery import PaperCandidate

logger = logging.getLogger(__name__)


@dataclass
class CurationSignals:
    knn_score: float
    llm_score: float
    heuristic_score: float


class KnnCurator:
    """kNN-based paper relevance scorer."""

    def __init__(self, exemplar_embeddings: np.ndarray, exemplar_labels: np.ndarray):
        self._embeddings = exemplar_embeddings
        self._labels = exemplar_labels

    def score(self, query_embedding: np.ndarray, k: int = 7) -> float:
        if len(self._embeddings) == 0:
            return 0.5
        k = min(k, len(self._embeddings))
        norms_e = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms_q = np.linalg.norm(query_embedding)
        if norms_q == 0:
            return 0.5
        similarities = (self._embeddings @ query_embedding) / (norms_e.squeeze() * norms_q + 1e-8)
        top_k_idx = np.argsort(-similarities)[:k]
        distances = 1.0 - similarities[top_k_idx]
        weights = 1.0 / (distances + 1e-6)
        labels = self._labels[top_k_idx]
        return float(np.average(labels, weights=weights))

    def add_exemplar(self, embedding: np.ndarray, label: int) -> None:
        self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)]) if len(self._embeddings) > 0 else embedding.reshape(1, -1)
        self._labels = np.append(self._labels, label)


class CurationBandit:
    """LinUCB bandit for fusing 3 curation signals."""

    def __init__(self, n_features: int = 3, alpha: float = 0.25):
        self.alpha = alpha
        self.n = n_features
        self.A = np.eye(n_features)
        self.b = np.zeros(n_features)
        self.A_inv = np.eye(n_features)
        self.theta = np.ones(n_features) / n_features

    def decide(self, signals: CurationSignals) -> tuple[bool, float]:
        x = np.array([signals.knn_score, signals.llm_score, signals.heuristic_score])
        score = float(self.theta @ x + self.alpha * np.sqrt(x @ self.A_inv @ x))
        return score > 0.5, score

    def update(self, signals: CurationSignals, reward: float) -> None:
        x = np.array([signals.knn_score, signals.llm_score, signals.heuristic_score])
        self.A += np.outer(x, x)
        self.b += reward * x
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b


_bandit = CurationBandit()


async def adaptive_curate(
    candidates: list[PaperCandidate],
    llm: Any,
    embedder: Any | None = None,
    knn_curator: KnnCurator | None = None,
    bandit: CurationBandit | None = None,
) -> list[CuratedPaper]:
    if bandit is None:
        bandit = _bandit

    filtered = heuristic_filter(candidates)
    passed_ids = {c.paper_id for c in filtered}

    llm_curated = await llm_score(filtered, llm) if llm else []
    llm_map = {cp.candidate.paper_id: cp for cp in llm_curated}

    results = []
    for candidate in filtered:
        h_score = 1.0 if candidate.paper_id in passed_ids else 0.0

        l_score = 0.5
        cp = llm_map.get(candidate.paper_id)
        if cp:
            l_score = cp.relevance_score / 10.0

        k_score = 0.5
        if embedder and knn_curator:
            dense, _ = embedder.embed_paper(candidate.title, candidate.abstract)
            k_score = knn_curator.score(dense)

        signals = CurationSignals(knn_score=k_score, llm_score=l_score, heuristic_score=h_score)
        accept, confidence = bandit.decide(signals)

        if accept:
            curated = cp if cp else CuratedPaper(
                candidate=candidate,
                relevance_score=int(confidence * 10),
                reason="bandit-accepted",
            )
            results.append(curated)

    return results
