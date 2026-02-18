"""S5: Alternative agent platform implementation.

Same pipeline as S4 but built on a different agent framework
(CrewAI, AutoGen, smolagents, or PydanticAI — team picks 1-2).
"""

from src.models import FactCheckState


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim using an alternative agent platform.

    Args:
        claim: The health claim to verify.

    Returns:
        Final state with verdict and evidence.
    """
    raise NotImplementedError("S5 (alt platform) not yet implemented")
