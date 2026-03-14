"""Adaptive Router -- 4-stage learned routing pipeline.

Stage 0: Structural features (keyword complexity/uncertainty) -- Rust or Python.
Stage 0.5: kNN on pre-computed embeddings (arXiv 2505.12601) -- Python/numpy.
Stage 1: ONNX BERT classifier (routellm/bert) -- Rust only.
Stage 2: Entropy probe (logprobs or token diversity) -- Python async.
Stage 3: Reserved for future online learning (not yet implemented).

Duck-type compatible with ComplexityRouter for seamless drop-in integration.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from sage.strategy.metacognition import CognitiveProfile, RoutingDecision
from sage.constants import (
    S1_COMPLEXITY_CEIL,
    S1_UNCERTAINTY_CEIL,
    S3_COMPLEXITY_FLOOR,
    S3_UNCERTAINTY_FLOOR,
    S2_REASONER_COMPLEXITY_FLOOR,
    S3_CODEX_COMPLEXITY_FLOOR,
    BRAKE_WINDOW,
    BRAKE_ENTROPY_THRESHOLD,
    BRAKE_HISTORY_MAXLEN,
    ADAPTIVE_C0_THRESHOLD,
    ADAPTIVE_C1_THRESHOLD,
    HEURISTIC_COMPLEXITY_DENOM,
    HEURISTIC_QUESTION_UNCERTAINTY,
    HEURISTIC_DEFAULT_UNCERTAINTY,
    HEURISTIC_FALLBACK_CONFIDENCE,
    KNN_S1_COMPLEXITY,
    KNN_S1_UNCERTAINTY,
    KNN_S2_COMPLEXITY,
    KNN_S2_UNCERTAINTY,
    KNN_S3_COMPLEXITY,
    KNN_S3_UNCERTAINTY,
    KNN_DISTANCE_THRESHOLD,
    MAX_TOKENS_S1,
    MAX_TOKENS_S2,
    MAX_TOKENS_S3,
    ENTROPY_LOW_THRESHOLD,
    ENTROPY_HIGH_THRESHOLD,
    ENTROPY_LOW_CONFIDENCE,
    ENTROPY_HIGH_CONFIDENCE,
    ENTROPY_MID_CONFIDENCE,
)

log = logging.getLogger(__name__)

# ── StructuralFeatures: Python implementation (Phase 1 rationalization) ───────
# Rust features.rs still compiled as internal dep of router.rs, but Python callers
# use the Python implementation directly.
from sage.strategy.structural_features import StructuralFeatures  # noqa: F401

# ── Rust AdaptiveRouter backend ───────────────────────────────────────────────
# Try Rust backend
_rust_available = False
try:
    from sage_core import AdaptiveRouter as _RustRouter

    _rust_available = True
except ImportError:
    pass


@dataclass
class AdaptiveRoutingResult:
    """Extended routing result with stage info."""

    decision: RoutingDecision
    profile: CognitiveProfile
    stage: int  # 0=structural, 1=BERT, 2=entropy, 3=reserved
    confidence: float  # 0.0-1.0
    method: str  # "rust_s0", "rust_s1", "entropy_s2", "heuristic"


class AdaptiveRouter:
    """4-stage adaptive router, duck-type compatible with ComplexityRouter.

    Stages: structural -> kNN embeddings -> BERT ONNX -> entropy probe.
    Stage 3 (online learning) reserved for future work. Falls back to cascade.

    Implements: route(), assess_complexity(), assess_complexity_async(),
    record_output_entropy(), should_brake()
    """

    def __init__(
        self,
        s1_complexity_ceil: float = S1_COMPLEXITY_CEIL,
        s1_uncertainty_ceil: float = S1_UNCERTAINTY_CEIL,
        s3_complexity_floor: float = S3_COMPLEXITY_FLOOR,
        s3_uncertainty_floor: float = S3_UNCERTAINTY_FLOOR,
        brake_window: int = BRAKE_WINDOW,
        brake_entropy_threshold: float = BRAKE_ENTROPY_THRESHOLD,
        llm_provider: Any = None,
        c0_threshold: float = ADAPTIVE_C0_THRESHOLD,
        c1_threshold: float = ADAPTIVE_C1_THRESHOLD,
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
        self._entropy_history: deque[float] = deque(maxlen=BRAKE_HISTORY_MAXLEN)

        # ComplexityRouter thresholds (used by heuristic fallback)
        self.s1_complexity_ceil = s1_complexity_ceil
        self.s1_uncertainty_ceil = s1_uncertainty_ceil
        self.s3_complexity_floor = s3_complexity_floor
        self.s3_uncertainty_floor = s3_uncertainty_floor

        # Stage 0.5: kNN router (arXiv 2505.12601)
        self._knn = knn_router

        # Rust backend
        self._rust = None
        if _rust_available:
            try:
                self._rust = _RustRouter(
                    c0_threshold=c0_threshold,
                    c1_threshold=c1_threshold,
                    classifier_path=classifier_path,
                    tokenizer_path=tokenizer_path,
                )
                log.info(
                    "AdaptiveRouter: Rust backend (classifier=%s)",
                    self._rust.has_classifier(),
                )
            except Exception as e:
                log.warning("Rust AdaptiveRouter init failed: %s", e)

        if self._rust is None:
            log.info("AdaptiveRouter: Python heuristic fallback")

    # -- ComplexityRouter-compatible interface --------------------------------

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Route based on a pre-assessed profile (ComplexityRouter compat)."""
        return self._route_from_profile(profile)

    def assess_complexity(self, task: str) -> CognitiveProfile:
        """Synchronous assessment: kNN (primary, 92% acc) > Rust structural > heuristic (52% acc, degraded).

        kNN is the real primary router. Heuristic only activates when both kNN paths
        are unavailable (no embedder, no exemplars). Per user mandate: no arbitrary
        heuristics when a learned model is available.
        """
        # PRIMARY PATH: Rust kNN (embed in Python, kNN search in Rust SIMD)
        if self._rust is not None and self._rust.has_knn():
            knn_profile = self._try_rust_knn_profile(task)
            if knn_profile is not None:
                log.debug("assess_complexity: Rust kNN → %s", knn_profile.reasoning)
                return knn_profile

        # PRIMARY PATH (Python fallback): Python kNN on pre-computed exemplars
        knn_profile = self._try_knn_profile(task)
        if knn_profile is not None:
            log.debug("assess_complexity: Python kNN → %s", knn_profile.reasoning)
            return knn_profile

        # DEGRADED PATH: Rust structural features (no embeddings needed)
        if self._rust is not None:
            result = self._rust.route(task)
            log.debug("assess_complexity: Rust structural (stage %d) → complexity=%.2f",
                      result.stage, result.features.keyword_complexity)
            return CognitiveProfile(
                complexity=result.features.keyword_complexity,
                uncertainty=result.features.keyword_uncertainty,
                tool_required=result.features.tool_required,
                reasoning=f"adaptive_stage{result.stage}",
            )

        # LAST RESORT: keyword-count heuristic (52% accuracy — issues a UserWarning)
        log.warning("assess_complexity: falling back to keyword heuristic (kNN unavailable)")
        return self._assess_heuristic(task)

    async def assess_complexity_async(self, task: str) -> CognitiveProfile:
        """Async assessment -- Rust fast path, then optional entropy probe."""
        return self.assess_complexity(task)

    def record_output_entropy(self, entropy: float) -> None:
        """Record entropy for CGRS self-braking."""
        self._entropy_history.append(entropy)

    def should_brake(self) -> bool:
        """CGRS: stop if recent outputs all have low entropy."""
        if len(self._entropy_history) < self.brake_window:
            return False
        recent = list(self._entropy_history)[-self.brake_window :]
        return all(e < self.brake_entropy_threshold for e in recent)

    # -- Extended API --------------------------------------------------------

    def route_adaptive(self, task: str) -> AdaptiveRoutingResult:
        """Full adaptive routing with stage info."""
        # Best path: Rust kNN (embed in Python, kNN in Rust SIMD)
        if self._rust is not None and self._rust.has_knn():
            rust_knn = self._try_rust_knn(task)
            if rust_knn is not None:
                profile = self._knn_to_profile_from_tier(rust_knn.tier, rust_knn.confidence)
                decision = self._route_from_profile(profile)
                return AdaptiveRoutingResult(
                    decision=decision,
                    profile=profile,
                    stage=0,
                    confidence=rust_knn.confidence,
                    method="rust_knn",
                )

        # Fallback: Python kNN
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

        if self._rust is not None:
            result = self._rust.route(task)
            profile = CognitiveProfile(
                complexity=result.features.keyword_complexity,
                uncertainty=result.features.keyword_uncertainty,
                tool_required=result.features.tool_required,
                reasoning=f"adaptive_stage{result.stage}",
            )
            decision = self._route_from_profile(profile)
            return AdaptiveRoutingResult(
                decision=decision,
                profile=profile,
                stage=result.stage,
                confidence=result.confidence,
                method=f"rust_s{result.stage}",
            )

        profile = self._assess_heuristic(task)
        decision = self._route_from_profile(profile)
        return AdaptiveRoutingResult(
            decision=decision,
            profile=profile,
            stage=0,
            confidence=HEURISTIC_FALLBACK_CONFIDENCE,
            method="heuristic",
        )

    async def route_adaptive_async(self, task: str) -> AdaptiveRoutingResult:
        """Async adaptive routing (Stages 0-2)."""
        result = self.route_adaptive(task)

        # Stage 2: entropy probe if enabled and confidence low
        if (
            self._enable_entropy
            and self._llm_provider is not None
            and result.confidence < ADAPTIVE_C0_THRESHOLD
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
        """Record routing feedback for online learning."""
        if self._rust is not None:
            self._rust.record_feedback(
                task, routed_tier, actual_quality, latency_ms, cost_usd
            )

    @property
    def has_rust(self) -> bool:
        """Whether the Rust backend is available."""
        return self._rust is not None

    @property
    def has_classifier(self) -> bool:
        """Whether an ONNX classifier is loaded (Stage 1 available)."""
        return self._rust is not None and self._rust.has_classifier()

    @property
    def has_knn(self) -> bool:
        """Whether kNN routing is available."""
        return self._knn is not None and self._knn.is_ready

    # -- Private -------------------------------------------------------------

    def _route_from_profile(self, profile: CognitiveProfile) -> RoutingDecision:
        """Convert profile to routing decision (same logic as ComplexityRouter.route)."""
        c, u = profile.complexity, profile.uncertainty

        if c > self.s3_complexity_floor or u > self.s3_uncertainty_floor:
            tier = "codex" if c > S3_CODEX_COMPLEXITY_FLOOR else "reasoner"
            return RoutingDecision(
                system=3,
                llm_tier=tier,
                max_tokens=MAX_TOKENS_S3,
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
                max_tokens=MAX_TOKENS_S1,
                use_z3=False,
                validation_level=1,
            )

        tier = "reasoner" if c > S2_REASONER_COMPLEXITY_FLOOR else "mutator"
        return RoutingDecision(
            system=2,
            llm_tier=tier,
            max_tokens=MAX_TOKENS_S2,
            use_z3=False,
            validation_level=2,
        )

    def _try_knn(self, task: str):
        """Try Python kNN routing, return KnnRoutingResult or None."""
        if self._knn is None or not self._knn.is_ready:
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

    def _try_rust_knn(self, task: str):
        """Try Rust kNN: embed in Python, search in Rust. Returns RoutingResult or None."""
        if self._knn is None or self._rust is None:
            return None
        try:
            emb = self._knn._get_embedder()
            if emb is None or getattr(emb, "is_hash_fallback", True):
                return None
            query_vec = emb.embed(task)
            result = self._rust.route_with_embedding(task, query_vec)
            # Accept if confidence exceeds OOD threshold (proxy for kNN contributing)
            if result is not None and result.confidence > KNN_DISTANCE_THRESHOLD:
                return result
            return None
        except Exception as e:
            log.warning("Rust kNN route failed: %s", e)
            return None

    def _try_rust_knn_profile(self, task: str) -> CognitiveProfile | None:
        """Try Rust kNN routing and convert to CognitiveProfile."""
        result = self._try_rust_knn(task)
        if result is None:
            return None
        return self._knn_to_profile_from_tier(result.tier, result.confidence)

    @staticmethod
    def _knn_to_profile(knn_result) -> CognitiveProfile:
        """Convert Python KnnRoutingResult to CognitiveProfile."""
        return AdaptiveRouter._knn_to_profile_from_tier(knn_result.system, knn_result.confidence)

    @staticmethod
    def _knn_to_profile_from_tier(tier: int, confidence: float = 1.0) -> CognitiveProfile:
        """Convert kNN tier to CognitiveProfile for _route_from_profile."""
        if tier == 1:
            return CognitiveProfile(
                complexity=KNN_S1_COMPLEXITY, uncertainty=KNN_S1_UNCERTAINTY,
                tool_required=False, reasoning="knn_s1",
            )
        elif tier == 3:
            return CognitiveProfile(
                complexity=KNN_S3_COMPLEXITY, uncertainty=KNN_S3_UNCERTAINTY,
                tool_required=False, reasoning="knn_s3",
            )
        else:  # S2
            return CognitiveProfile(
                complexity=KNN_S2_COMPLEXITY, uncertainty=KNN_S2_UNCERTAINTY,
                tool_required=False, reasoning="knn_s2",
            )

    def _assess_heuristic(self, task: str) -> CognitiveProfile:
        """Degraded keyword-count fallback (52% accuracy, no regex).

        LAST RESORT only — runs when kNN (primary, 92% accuracy) is unavailable.
        Issues a UserWarning to surface the degradation to callers and logs.
        """
        import warnings
        warnings.warn(
            "Using degraded keyword-count heuristic (52% accuracy). "
            "Install sage_core[onnx] or build kNN exemplars for accurate routing.",
            UserWarning,
            stacklevel=2,
        )
        words = task.lower().split()
        complex_kw = {"implement", "feature", "module", "component", "service",
                      "algorithm", "optimize", "distributed", "concurrent",
                      "debug", "fix", "race", "deadlock", "proof", "verify", "formal"}
        hits = sum(1 for w in words if w in complex_kw)
        return CognitiveProfile(
            complexity=min(hits / HEURISTIC_COMPLEXITY_DENOM, 1.0),
            uncertainty=HEURISTIC_QUESTION_UNCERTAINTY if "?" in task else HEURISTIC_DEFAULT_UNCERTAINTY,
            tool_required=False,
            reasoning="degraded_heuristic",
        )

    async def _entropy_probe(
        self, task: str, current: AdaptiveRoutingResult
    ) -> AdaptiveRoutingResult | None:
        """Stage 2: entropy probe using logprobs if available.

        Prefers Shannon entropy from actual token log-probabilities (OpenAI,
        Groq, xAI expose these).  Falls back to a perplexity-like heuristic
        based on unique-token ratio over the first 5-10 tokens.
        """
        if self._llm_provider is None:
            return None

        from sage.llm.base import LLMConfig, Message, Role

        try:
            response = await self._llm_provider.generate(
                messages=[Message(role=Role.USER, content=task[:500])],
                config=LLMConfig(
                    provider="auto", model="auto", temperature=1.0, max_tokens=5
                ),
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
                content = response.content.strip()
                if not content:
                    return None
                tokens = content.split()
                if len(tokens) < 2:
                    return None
                unique_ratio = len(set(tokens)) / len(tokens)
                entropy = unique_ratio  # rough proxy

            # Map entropy to routing adjustment
            if entropy < ENTROPY_LOW_THRESHOLD:
                conf = ENTROPY_LOW_CONFIDENCE
            elif entropy > ENTROPY_HIGH_THRESHOLD:
                conf = ENTROPY_HIGH_CONFIDENCE
            else:
                conf = ENTROPY_MID_CONFIDENCE

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
