"""Adaptive Router -- multi-stage learned routing pipeline.

Stage 0: Structural features (keyword complexity/uncertainty) -- Python.
Stage 0.5: kNN on pre-computed embeddings (arXiv 2505.12601) -- Python/numpy.
Stage 2: Entropy probe (logprobs or token diversity) -- async, optional.

Duck-type compatible with ComplexityRouter for seamless drop-in integration.

Notes
-----
Rust acceleration (Stage 1 BERT ONNX, Rust kNN hot-path) is available in the
full YGN-SAGE installation. In this standalone package only the Python paths
are active.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from sage_router.metacognition import CognitiveProfile, RoutingDecision
from sage_router.structural_features import StructuralFeatures  # noqa: F401

log = logging.getLogger(__name__)


@dataclass
class AdaptiveRoutingResult:
    """Extended routing result with stage info."""

    decision: RoutingDecision
    profile: CognitiveProfile
    stage: int  # 0=structural, 1=BERT, 2=entropy, 3=reserved
    confidence: float  # 0.0-1.0
    method: str  # "knn", "structural", "entropy_s2", "heuristic"


class AdaptiveRouter:
    """Multi-stage adaptive router, duck-type compatible with ComplexityRouter.

    Stages: structural -> kNN embeddings -> entropy probe -> cascade.

    Implements: route(), assess_complexity(), assess_complexity_async(),
    record_output_entropy(), should_brake()
    """

    def __init__(
        self,
        s1_complexity_ceil: float = 0.50,
        s1_uncertainty_ceil: float = 0.3,
        s3_complexity_floor: float = 0.65,
        s3_uncertainty_floor: float = 0.6,
        brake_window: int = 3,
        brake_entropy_threshold: float = 0.15,
        llm_provider: Any = None,
        c0_threshold: float = 0.85,
        c1_threshold: float = 0.70,
        classifier_path: str | None = None,
        tokenizer_path: str | None = None,
        enable_entropy_probe: bool = False,
        knn_router: Any = None,
    ):
        self._llm_provider = llm_provider
        self._enable_entropy = enable_entropy_probe

        # CGRS self-braking (same as ComplexityRouter)
        self.brake_window = brake_window
        self.brake_entropy_threshold = brake_entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=10)

        # ComplexityRouter thresholds (used by heuristic fallback)
        self.s1_complexity_ceil = s1_complexity_ceil
        self.s1_uncertainty_ceil = s1_uncertainty_ceil
        self.s3_complexity_floor = s3_complexity_floor
        self.s3_uncertainty_floor = s3_uncertainty_floor

        # Stage 0.5: kNN router (arXiv 2505.12601)
        self._knn = knn_router

        log.info("AdaptiveRouter: Python-only path (knn=%s)", knn_router is not None)

    # -- ComplexityRouter-compatible interface --------------------------------

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Route based on a pre-assessed profile (ComplexityRouter compat)."""
        return self._route_from_profile(profile)

    def assess_complexity(self, task: str) -> CognitiveProfile:
        """Synchronous assessment: Python kNN > structural features > heuristic."""
        # Stage 0.5: Python kNN
        knn_profile = self._try_knn_profile(task)
        if knn_profile is not None:
            return knn_profile

        # Stage 0: structural features
        return self._assess_structural(task)

    async def assess_complexity_async(self, task: str) -> CognitiveProfile:
        """Async assessment -- same as sync (entropy probe optional, requires LLM)."""
        return self.assess_complexity(task)

    def record_output_entropy(self, entropy: float) -> None:
        """Record entropy for CGRS self-braking."""
        self._entropy_history.append(entropy)

    def should_brake(self) -> bool:
        """CGRS: stop if recent outputs all have low entropy."""
        if len(self._entropy_history) < self.brake_window:
            return False
        recent = list(self._entropy_history)[-self.brake_window:]
        return all(e < self.brake_entropy_threshold for e in recent)

    # -- Extended API --------------------------------------------------------

    def route_adaptive(self, task: str) -> AdaptiveRoutingResult:
        """Full adaptive routing with stage info."""
        # Stage 0.5: Python kNN
        knn_result = self._try_knn(task)
        if knn_result is not None:
            profile = self._knn_to_profile(knn_result)
            decision = self._route_from_profile(profile)
            return AdaptiveRoutingResult(
                decision=decision,
                profile=profile,
                stage=0,
                confidence=knn_result.confidence,
                method="knn",
            )

        # Stage 0: structural features
        features = StructuralFeatures.extract(task)
        profile = CognitiveProfile(
            complexity=features.keyword_complexity,
            uncertainty=features.keyword_uncertainty,
            tool_required=features.tool_required,
            reasoning="structural_s0",
        )
        decision = self._route_from_profile(profile)
        return AdaptiveRoutingResult(
            decision=decision,
            profile=profile,
            stage=0,
            confidence=0.6,
            method="structural",
        )

    async def route_adaptive_async(self, task: str) -> AdaptiveRoutingResult:
        """Async adaptive routing (Stages 0-2)."""
        result = self.route_adaptive(task)

        # Stage 2: entropy probe if enabled and confidence low
        if (
            self._enable_entropy
            and self._llm_provider is not None
            and result.confidence < 0.85
        ):
            try:
                entropy_result = await self._entropy_probe(task, result)
                if entropy_result is not None:
                    return entropy_result
            except Exception as e:
                log.warning("Entropy probe failed: %s", e)

        return result

    def record_feedback(
        self,
        task: str,
        routed_tier: int,
        actual_quality: float,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record routing feedback (no-op in standalone; used for API compat)."""
        pass

    @property
    def has_rust(self) -> bool:
        """Whether the Rust backend is available (always False in standalone)."""
        return False

    @property
    def has_classifier(self) -> bool:
        """Whether an ONNX classifier is loaded (always False in standalone)."""
        return False

    @property
    def has_knn(self) -> bool:
        """Whether kNN routing is available."""
        return self._knn is not None and getattr(self._knn, "is_ready", False)

    # -- Private -------------------------------------------------------------

    def _route_from_profile(self, profile: CognitiveProfile) -> RoutingDecision:
        """Convert profile to routing decision (same logic as ComplexityRouter.route)."""
        c, u = profile.complexity, profile.uncertainty

        if c > self.s3_complexity_floor or u > self.s3_uncertainty_floor:
            tier = "codex" if c > 0.8 else "reasoner"
            return RoutingDecision(
                system=3,
                llm_tier=tier,
                max_tokens=8192,
                use_z3=True,
                validation_level=3,
            )

        if (
            c <= self.s1_complexity_ceil
            and u <= self.s1_uncertainty_ceil
            and not profile.tool_required
        ):
            return RoutingDecision(
                system=1,
                llm_tier="fast",
                max_tokens=2048,
                use_z3=False,
                validation_level=1,
            )

        tier = "reasoner" if c > 0.55 else "mutator"
        return RoutingDecision(
            system=2,
            llm_tier=tier,
            max_tokens=4096,
            use_z3=False,
            validation_level=2,
        )

    def _assess_structural(self, task: str) -> CognitiveProfile:
        """Stage 0: structural feature extraction."""
        features = StructuralFeatures.extract(task)
        return CognitiveProfile(
            complexity=features.keyword_complexity,
            uncertainty=features.keyword_uncertainty,
            tool_required=features.tool_required,
            reasoning="structural_s0",
        )

    def _try_knn(self, task: str):
        """Try Python kNN routing, return KnnRoutingResult or None."""
        if self._knn is None or not getattr(self._knn, "is_ready", False):
            return None
        try:
            return self._knn.route(task)
        except Exception as e:
            log.warning("kNN route failed: %s", e)
            return None

    def _try_knn_profile(self, task: str) -> CognitiveProfile | None:
        """Try Python kNN routing and convert to CognitiveProfile."""
        result = self._try_knn(task)
        if result is None:
            return None
        return self._knn_to_profile(result)

    @staticmethod
    def _knn_to_profile(knn_result) -> CognitiveProfile:
        """Convert KnnRoutingResult to CognitiveProfile."""
        return AdaptiveRouter._knn_to_profile_from_tier(knn_result.system, knn_result.confidence)

    @staticmethod
    def _knn_to_profile_from_tier(tier: int, confidence: float = 1.0) -> CognitiveProfile:
        """Convert kNN tier to CognitiveProfile for _route_from_profile."""
        if tier == 1:
            return CognitiveProfile(
                complexity=0.2, uncertainty=0.1,
                tool_required=False, reasoning="knn_s1",
            )
        elif tier == 3:
            return CognitiveProfile(
                complexity=0.8, uncertainty=0.7,
                tool_required=False, reasoning="knn_s3",
            )
        else:  # S2
            return CognitiveProfile(
                complexity=0.5, uncertainty=0.4,
                tool_required=False, reasoning="knn_s2",
            )

    async def _entropy_probe(
        self, task: str, current: AdaptiveRoutingResult
    ) -> AdaptiveRoutingResult | None:
        """Stage 2: entropy probe using logprobs if available.

        Prefers Shannon entropy from actual token log-probabilities (OpenAI,
        Groq, xAI expose these).  Falls back to a perplexity-like heuristic
        based on unique-token ratio over the first 5-10 tokens.

        Requires a provider injected via llm_provider that exposes:
          async generate_raw(prompt, max_tokens, temperature) -> response
        where response has .content and optionally .logprobs attributes.
        """
        if self._llm_provider is None:
            return None

        try:
            response = await self._llm_provider.generate_raw(
                task[:500], max_tokens=5, temperature=1.0
            )

            # Try logprobs first (OpenAI, Groq, xAI support this)
            entropy = None
            if hasattr(response, "logprobs") and response.logprobs:
                probs = [
                    math.exp(lp)
                    for lp in response.logprobs
                    if lp is not None
                ]
                if probs:
                    entropy = -sum(
                        p * math.log(p + 1e-10) for p in probs
                    ) / max(len(probs), 1)

            if entropy is None:
                # Fallback: estimate from content diversity (5-10 tokens)
                content = getattr(response, "content", str(response)).strip()
                if not content:
                    return None
                tokens = content.split()
                if len(tokens) < 2:
                    return None
                unique_ratio = len(set(tokens)) / len(tokens)
                entropy = unique_ratio  # rough proxy

            # Map entropy to routing adjustment
            if entropy < 0.3:
                conf = 0.75
            elif entropy > 0.7:
                conf = 0.65
            else:
                conf = 0.60

            profile = current.profile
            decision = self._route_from_profile(
                CognitiveProfile(
                    complexity=profile.complexity,
                    uncertainty=entropy,
                    tool_required=profile.tool_required,
                    reasoning="entropy_probe",
                )
            )
            return AdaptiveRoutingResult(
                decision=decision,
                profile=profile,
                stage=2,
                confidence=conf,
                method="entropy_s2",
            )
        except Exception:
            return None
