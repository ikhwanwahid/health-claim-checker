"""S2: Simple RAG — static corpus with vector search.

Embeds a pre-downloaded corpus of vaccine articles, retrieves via
vector similarity, and passes top-k chunks to the LLM for a verdict.
"""

from src.models import FactCheckState


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim using simple vector-search RAG.

    Args:
        claim: The health claim to verify.

    Returns:
        Final state with verdict and vector-search evidence.
    """
    raise NotImplementedError("S2 (simple RAG) not yet implemented")
