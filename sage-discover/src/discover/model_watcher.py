"""Watch for new model releases across all configured providers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Ensure sage-python is importable
_sage_src = Path(__file__).resolve().parent.parent.parent.parent / "sage-python" / "src"
if str(_sage_src) not in sys.path:
    sys.path.insert(0, str(_sage_src))


class ModelWatcher:
    """Detect new models not yet in TOML profiles."""

    async def check_new_models(self) -> list[dict]:
        """Compare discovered models vs TOML. Return unprofiled models."""
        try:
            from sage.providers.registry import ModelRegistry
        except ImportError:
            log.warning("sage.providers not available")
            return []

        registry = ModelRegistry()
        await registry.refresh()

        # Models with cost > 0 have TOML profiles
        profiled_ids = {
            p.id
            for p in registry._profiles.values()
            if p.cost_input > 0 or p.cost_output > 0
        }
        available_ids = {
            p.id for p in registry._profiles.values() if p.available
        }
        new_models = available_ids - profiled_ids

        results = []
        for model_id in sorted(new_models):
            profile = registry.get(model_id)
            if profile:
                results.append(
                    {
                        "id": model_id,
                        "provider": profile.provider,
                        "context_window": profile.context_window,
                        "status": "NEW - needs TOML profile",
                    }
                )

        if results:
            log.info(
                "ModelWatcher: %d new unprofiled models detected", len(results)
            )
        else:
            log.info("ModelWatcher: all discovered models are profiled")

        return results

    async def report(self) -> str:
        """Generate a human-readable report of new models."""
        models = await self.check_new_models()
        if not models:
            return "All discovered models have TOML profiles. No action needed."

        lines = [f"=== {len(models)} New Unprofiled Models ===", ""]
        for m in models:
            ctx = f"{m['context_window']:,}" if m.get("context_window") else "?"
            lines.append(
                f"  {m['id']:50s} provider={m['provider']:10s} ctx={ctx}"
            )
        lines.append("")
        lines.append(
            "Action: Add profiles to sage-python/config/model_profiles.toml"
        )
        return "\n".join(lines)
