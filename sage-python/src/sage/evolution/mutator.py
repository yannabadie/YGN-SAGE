"""LLM-driven mutation pipeline using SEARCH/REPLACE diff format.

Inspired by AlphaEvolve's mutation strategy: use LLMs to propose
targeted code modifications rather than random mutations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Mutation:
    """A proposed code mutation."""
    search: str  # Original code to find
    replace: str  # Replacement code
    description: str = ""


class Mutator:
    """Applies LLM-proposed mutations to code using SEARCH/REPLACE diffs."""

    def apply_mutation(self, code: str, mutation: Mutation) -> str:
        """Apply a single mutation to code. Returns modified code."""
        if mutation.search not in code:
            raise ValueError(f"Search pattern not found in code: {mutation.search!r}")
        return code.replace(mutation.search, mutation.replace, 1)

    def apply_mutations(self, code: str, mutations: list[Mutation]) -> str:
        """Apply a list of mutations sequentially."""
        result = code
        for mutation in mutations:
            result = self.apply_mutation(result, mutation)
        return result

    def parse_diff(self, diff_text: str) -> list[Mutation]:
        """Parse a SEARCH/REPLACE diff block into mutations.

        Expected format:
        <<<SEARCH
        original code here
        ===
        replacement code here
        >>>REPLACE

        Optionally with a description after >>>REPLACE:
        >>>REPLACE: Improved performance by using binary search
        """
        mutations = []
        pattern = r"<<<SEARCH\n(.*?)\n===\n(.*?)\n>>>REPLACE(?::?\s*(.*?))?\s*$"
        for match in re.finditer(pattern, diff_text, re.DOTALL | re.MULTILINE):
            search = match.group(1)
            replace = match.group(2)
            description = match.group(3) or ""
            mutations.append(Mutation(
                search=search.strip(),
                replace=replace.strip(),
                description=description.strip(),
            ))
        return mutations

    def generate_diff(self, original: str, modified: str, description: str = "") -> str:
        """Generate a SEARCH/REPLACE diff string from original and modified code."""
        desc_suffix = f": {description}" if description else ""
        return f"<<<SEARCH\n{original}\n===\n{modified}\n>>>REPLACE{desc_suffix}"
