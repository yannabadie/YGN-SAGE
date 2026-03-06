"""Model registry: auto-discovery + TOML knowledge base merge.

Provides :class:`ModelRegistry` that combines live provider discovery with
curated benchmark data from ``config/model_profiles.toml``.
"""
from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sage.providers.connector import ProviderConnector, DiscoveredModel, PROVIDER_CONFIGS

logger = logging.getLogger(__name__)

# ── TOML search path (mirrors config_loader.py) ──────────────────────────────
_TOML_FILENAME = "model_profiles.toml"


def _toml_search_paths() -> list[Path]:
    """Return ordered search paths for the model profiles TOML file."""
    return [
        Path.cwd() / "config" / _TOML_FILENAME,
        Path(__file__).parent.parent.parent.parent / "config" / _TOML_FILENAME,
        Path.home() / ".sage" / _TOML_FILENAME,
    ]


@dataclass
class ModelProfile:
    """Full profile for a single LLM model.

    Fields without TOML data default to 0.5 (scores) or 0.0 (economics).
    """

    # Identity
    id: str
    provider: str
    family: str = ""
    available: bool = False

    # Capability scores (0.0 - 1.0)
    code_score: float = 0.5
    reasoning_score: float = 0.5
    tool_use_score: float = 0.5

    # Economics ($/1M tokens)
    cost_input: float = 0.0
    cost_output: float = 0.0

    # Performance
    latency_ttft_ms: int = 0
    tokens_per_second: int = 0

    # Context
    context_window: int | None = None
    max_output_tokens: int | None = None

    # Compatibility flags
    supports_structured_output: bool = False
    supports_tools: bool = False
    structured_output_tools_compatible: bool = False
    supports_file_search: bool = False
    supports_thinking: bool = False

    # Raw metadata from API discovery
    raw_meta: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Central registry of known and available LLM models.

    Merges live API discovery (which models are reachable right now) with
    curated TOML knowledge base (benchmark scores, pricing, compatibility).

    Usage::

        registry = ModelRegistry()
        await registry.refresh()

        # Score-based selection
        best = registry.select({"code": 0.8, "reasoning": 0.3, "max_cost_per_1m": 5.0})

        # List everything available
        for m in registry.list_available():
            print(f"{m.id}: code={m.code_score}, ${m.cost_input}/{m.cost_output}")
    """

    def __init__(self):
        self._profiles: dict[str, ModelProfile] = {}
        self._connector = ProviderConnector()

    # ── Public API ────────────────────────────────────────────────────────

    async def refresh(self) -> None:
        """Discover available models and merge with TOML knowledge base.

        Called at boot. Safe to call multiple times (idempotent).
        """
        knowledge = self._load_toml()
        discovered = await self._connector.discover_all()

        # Track which TOML entries were seen via discovery
        seen_ids: set[str] = set()

        for dm in discovered:
            seen_ids.add(dm.id)
            profile = self._merge(dm, knowledge)
            self._profiles[dm.id] = profile

        # Add TOML-only models as unavailable (not discovered but known)
        for model_id, toml_data in knowledge.items():
            if model_id not in seen_ids:
                profile = self._profile_from_toml(model_id, toml_data)
                profile.available = False
                self._profiles[model_id] = profile
                logger.debug("TOML model %s not discovered (marked unavailable)", model_id)

        # Warn about discovered models missing from TOML
        for dm in discovered:
            if dm.id not in knowledge:
                logger.info(
                    "New model discovered: %s (%s) -- no TOML profile, using defaults",
                    dm.id, dm.provider,
                )

        logger.info(
            "ModelRegistry: %d total profiles, %d available",
            len(self._profiles),
            sum(1 for p in self._profiles.values() if p.available),
        )

    def select(self, needs: dict[str, float]) -> ModelProfile | None:
        """Select the best available model for given task needs.

        Args:
            needs: Dict with optional keys:
                - ``code``: weight for code_score (0.0-1.0)
                - ``reasoning``: weight for reasoning_score (0.0-1.0)
                - ``tool_use``: weight for tool_use_score (0.0-1.0)
                - ``max_cost_per_1m``: maximum acceptable input cost $/1M tokens
                - ``min_context``: minimum context window size
                - ``require_tools``: if True, filter to models with tool support
                - ``require_structured_output``: if True, filter accordingly

        Returns:
            Best matching ModelProfile, or None if no models match.

        The scoring formula is::

            quality = sum(need_weight * model_score for each dimension)
            cost_factor = max(cost_input, 0.01)  # avoid div-by-zero
            score = quality / cost_factor
        """
        candidates: list[ModelProfile] = []

        max_cost = needs.get("max_cost_per_1m")
        min_context = needs.get("min_context")
        require_tools = needs.get("require_tools", False)
        require_structured = needs.get("require_structured_output", False)

        for profile in self._profiles.values():
            if not profile.available:
                continue
            if max_cost is not None and profile.cost_input > max_cost:
                continue
            if min_context is not None and (
                profile.context_window is None or profile.context_window < min_context
            ):
                continue
            if require_tools and not profile.supports_tools:
                continue
            if require_structured and not profile.supports_structured_output:
                continue
            candidates.append(profile)

        if not candidates:
            return None

        def _score(p: ModelProfile) -> float:
            quality = 0.0
            quality += needs.get("code", 0.0) * p.code_score
            quality += needs.get("reasoning", 0.0) * p.reasoning_score
            quality += needs.get("tool_use", 0.0) * p.tool_use_score
            cost_factor = max(p.cost_input, 0.01)
            return quality / cost_factor

        return max(candidates, key=_score)

    def select_for_tier(self, tier: str) -> ModelProfile | None:
        """Legacy: select by tier name (fast, reasoner, codex, etc.).

        Maps tier names to need profiles and delegates to :meth:`select`.
        """
        tier_needs: dict[str, dict[str, float]] = {
            "fast": {"code": 0.3, "reasoning": 0.3, "max_cost_per_1m": 1.0},
            "budget": {"code": 0.2, "reasoning": 0.2, "max_cost_per_1m": 0.1},
            "mutator": {"code": 0.8, "reasoning": 0.5, "max_cost_per_1m": 5.0},
            "reasoner": {"code": 0.5, "reasoning": 1.0, "max_cost_per_1m": 15.0},
            "codex": {"code": 1.0, "reasoning": 0.8, "max_cost_per_1m": 20.0},
            "codex_max": {"code": 1.0, "reasoning": 1.0},
            "critical": {"code": 0.5, "reasoning": 1.0, "max_cost_per_1m": 15.0},
            "fallback": {"code": 0.3, "reasoning": 0.3, "max_cost_per_1m": 1.0},
        }
        return self.select(tier_needs.get(tier, {"code": 0.5, "reasoning": 0.5}))

    def list_available(self) -> list[ModelProfile]:
        """Return all available models sorted by input cost (ascending)."""
        return sorted(
            (p for p in self._profiles.values() if p.available),
            key=lambda p: p.cost_input,
        )

    def get(self, model_id: str) -> ModelProfile | None:
        """Get a specific model profile by ID."""
        return self._profiles.get(model_id)

    @property
    def profiles(self) -> dict[str, ModelProfile]:
        """Direct access to the internal profiles dict (read-only intent)."""
        return self._profiles

    # ── TOML loading ──────────────────────────────────────────────────────

    def _load_toml(self) -> dict[str, dict[str, Any]]:
        """Load model_profiles.toml, searching the standard paths.

        Returns:
            Dict mapping model_id -> profile fields from TOML.
        """
        for path in _toml_search_paths():
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                models_section = data.get("models", {})
                if models_section:
                    logger.info("Loaded model profiles from %s (%d entries)", path, len(models_section))
                    return models_section
            except (FileNotFoundError, OSError):
                continue
            except tomllib.TOMLDecodeError as exc:
                logger.warning("Invalid TOML in %s: %s", path, exc)
                continue

        logger.warning("No model_profiles.toml found in search paths")
        return {}

    # ── Merge helpers ─────────────────────────────────────────────────────

    def _merge(
        self, dm: DiscoveredModel, knowledge: dict[str, dict[str, Any]]
    ) -> ModelProfile:
        """Merge a discovered model with TOML knowledge base entry."""
        toml_data = knowledge.get(dm.id, {})

        profile = ModelProfile(
            id=dm.id,
            provider=toml_data.get("provider", dm.provider),
            family=toml_data.get("family", ""),
            available=True,
            # Scores
            code_score=toml_data.get("code_score", 0.5),
            reasoning_score=toml_data.get("reasoning_score", 0.5),
            tool_use_score=toml_data.get("tool_use_score", 0.5),
            # Economics
            cost_input=toml_data.get("cost_input", 0.0),
            cost_output=toml_data.get("cost_output", 0.0),
            # Performance
            latency_ttft_ms=toml_data.get("latency_ttft_ms", 0),
            tokens_per_second=toml_data.get("tokens_per_second", 0),
            # Context (prefer discovery data, fallback to TOML)
            context_window=dm.context_window,
            max_output_tokens=dm.max_output_tokens,
            # Compatibility
            supports_structured_output=toml_data.get("supports_structured_output", False),
            supports_tools=toml_data.get("supports_tools", False),
            structured_output_tools_compatible=toml_data.get("structured_output_tools_compatible", False),
            supports_file_search=toml_data.get("supports_file_search", False),
            supports_thinking=dm.supports_thinking,
            # Raw
            raw_meta=dm.raw_meta,
        )
        return profile

    def _profile_from_toml(
        self, model_id: str, toml_data: dict[str, Any]
    ) -> ModelProfile:
        """Create a ModelProfile purely from TOML data (no discovery)."""
        return ModelProfile(
            id=model_id,
            provider=toml_data.get("provider", "unknown"),
            family=toml_data.get("family", ""),
            available=False,
            code_score=toml_data.get("code_score", 0.5),
            reasoning_score=toml_data.get("reasoning_score", 0.5),
            tool_use_score=toml_data.get("tool_use_score", 0.5),
            cost_input=toml_data.get("cost_input", 0.0),
            cost_output=toml_data.get("cost_output", 0.0),
            latency_ttft_ms=toml_data.get("latency_ttft_ms", 0),
            tokens_per_second=toml_data.get("tokens_per_second", 0),
            supports_structured_output=toml_data.get("supports_structured_output", False),
            supports_tools=toml_data.get("supports_tools", False),
            structured_output_tools_compatible=toml_data.get("structured_output_tools_compatible", False),
            supports_file_search=toml_data.get("supports_file_search", False),
        )
