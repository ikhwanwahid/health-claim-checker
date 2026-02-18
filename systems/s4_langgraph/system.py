"""S4 / S6: LangGraph multi-agent system entry point.

S4 = advanced retrieval with LangGraph agents (VLM disabled).
S6 = full agentic RAG with deep search + VLM (VLM enabled).

The workflow is the same; a config flag controls whether VLM and
deep search are active.
"""

from src.models import FactCheckState
from systems.s4_langgraph.workflow import verify_claim as _verify_claim


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim using the LangGraph multi-agent pipeline.

    Args:
        claim: The health claim to verify.

    Returns:
        Final pipeline state with verdict, evidence, and traces.
    """
    return await _verify_claim(claim)
