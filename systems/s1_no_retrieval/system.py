"""S1: LLM-only baseline — no retrieval.

Uses the LLM's parametric knowledge to produce a verdict.
No evidence retrieval, no sub-claim decomposition.
"""

from src.models import FactCheckState


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim using LLM knowledge only (no retrieval).

    Args:
        claim: The health claim to verify.

    Returns:
        Final state with verdict but no retrieved evidence.
    """
    raise NotImplementedError("S1 (no retrieval) not yet implemented")
