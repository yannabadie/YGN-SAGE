"""Evolution engine: AlphaEvolve-inspired LLM-driven code evolution."""
from sage.evolution.population import Population, Individual
from sage.evolution.mutator import Mutator, Mutation
from sage.evolution.evaluator import Evaluator, EvalResult, validate_evolution
from sage.evolution.engine import EvolutionEngine
from sage.evolution.llm_mutator import LLMMutator, AdaptiveMutator

__all__ = [
    "Population",
    "Individual",
    "Mutator",
    "Mutation",
    "Evaluator",
    "EvalResult",
    "EvolutionEngine",
    "LLMMutator",
    "AdaptiveMutator",
    "validate_evolution",
]
