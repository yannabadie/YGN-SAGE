"""LLM-based Mutator for MAP-Elites evolution.

Connects the BaseLLM to the Mutator pipeline, generating new code variants
and their behavioral feature descriptors for the MAP-Elites population grid.
"""
from __future__ import annotations

import logging
from typing import Tuple

from sage.llm.base import LLMProvider, Message, Role
from sage.evolution.mutator import Mutator, Mutation


class LLMMutator:
    """Generates code mutations using an LLM."""

    def __init__(self, llm: LLMProvider, mutator: Mutator | None = None):
        self.llm = llm
        self.mutator = mutator or Mutator()
        self.logger = logging.getLogger(__name__)

    async def mutate(self, code: str, objective: str) -> Tuple[str, Tuple[int, int]]:
        """Mutates code using LLM and returns (new_code, features).
        
        Features are a 2D tuple representing:
        (complexity_score, creativity_score) mapped to 0-9 bins.
        """
        prompt = f"""You are an expert AI code optimizer.
Your objective is: {objective}

Given the following code, propose a mutation to improve it towards the objective.
You MUST output your mutation in SEARCH/REPLACE format.
You MUST also provide two behavioral scores from 0 to 9 for the resulting code:
1. Complexity: 0 (very simple) to 9 (highly complex).
2. Creativity: 0 (standard approach) to 9 (highly novel/creative approach).

Format:
<<<SEARCH
[exact original code lines]
===
[new code lines]
>>>REPLACE: [brief description]

FEATURES: Complexity=<0-9>, Creativity=<0-9>

CODE:
```python
{code}
```"""

        messages = [Message(role=Role.USER, content=prompt)]
        response = await self.llm.generate(messages)
        content = response.content or ""

        # Parse Diff
        mutations = self.mutator.parse_diff(content)
        new_code = self.mutator.apply_mutations(code, mutations)

        # Parse Features
        complexity = 5
        creativity = 5
        try:
            for line in content.split("\n"):
                if line.startswith("FEATURES:"):
                    parts = line.replace("FEATURES:", "").split(",")
                    for part in parts:
                        key, val = part.split("=")
                        key = key.strip().lower()
                        if key == "complexity":
                            complexity = int(val.strip())
                        elif key == "creativity":
                            creativity = int(val.strip())
        except Exception as e:
            self.logger.warning(f"Failed to parse features, using default 5,5: {e}")

        # Clamp features to 0-9 (assuming 10 bins)
        complexity = max(0, min(9, complexity))
        creativity = max(0, min(9, creativity))

        return new_code, (complexity, creativity)
