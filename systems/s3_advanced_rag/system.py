"""S3: Advanced RAG — multi-source retrieval with function pipeline.

PICO query reformulation, multi-source API search (PubMed, Semantic
Scholar, Cochrane), cross-encoder re-ranking, LLM verdict. No agent
reasoning loop — pure function pipeline.
"""

from src.models import FactCheckState


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim using advanced RAG (function pipeline).

    Args:
        claim: The health claim to verify.

    Returns:
        Final state with verdict and multi-source evidence.
    """
    raise NotImplementedError("S3 (advanced RAG) not yet implemented")
