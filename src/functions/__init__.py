"""Functions (single-pass nodes) for the health claim verification pipeline.

Functions execute a fixed sequence of steps with at most one LLM call.
They do NOT have a reasoning loop or tool-use capability.

Modules:
    decomposer: Claim → PICO + sub-claims (NER → PICO → LLM decompose)
    safety_checker: Claim + verdict → safety flags (single LLM call)
"""

from src.functions import decomposer, safety_checker

__all__ = ["decomposer", "safety_checker"]
