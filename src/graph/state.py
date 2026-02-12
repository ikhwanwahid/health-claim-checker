"""LangGraph state definitions for the fact-checking pipeline."""

from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel


class PICO(BaseModel):
    """PICO framework extraction."""
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparison: Optional[str] = None
    outcome: Optional[str] = None


class SubClaim(BaseModel):
    """A decomposed sub-claim."""
    id: str
    text: str
    pico: Optional[PICO] = None
    verdict: Optional[str] = None
    evidence: List[str] = []
    confidence: float = 0.0


class Evidence(BaseModel):
    """A piece of retrieved evidence."""
    id: str
    source: str  # pubmed, cochrane, guideline, etc.
    retrieval_method: str  # api, cross_encoder, deep_search, vlm
    title: str
    content: str
    url: Optional[str] = None
    study_type: Optional[str] = None
    quality_score: float = 0.0
    pmid: Optional[str] = None


class ToolCall(BaseModel):
    """Record of a single tool call made by an agent."""
    tool: str
    input_summary: str
    output_summary: str
    duration_seconds: float
    success: bool


class AgentTrace(BaseModel):
    """Trace of a single agent's execution."""
    agent: str
    node_type: str = "function"  # "function" or "agent"
    duration_seconds: float
    cost_usd: float
    input_summary: str
    output_summary: str
    success: bool
    tools_called: List[str] = []  # e.g., ["pubmed_api", "cross_encoder"]
    tool_calls: List[ToolCall] = []  # detailed per-call trace
    reasoning_steps: int = 0  # number of LLM reasoning steps (for agents)


class FactCheckState(TypedDict):
    """Main state passed through the LangGraph workflow."""
    # Input
    claim: str
    
    # Decomposition
    pico: Optional[PICO]
    sub_claims: List[SubClaim]
    entities: dict
    
    # Retrieval planning
    retrieval_plan: dict
    
    # Evidence
    evidence: List[Evidence]
    
    # VLM
    extracted_figures: List[dict]
    
    # Grading
    evidence_quality: dict
    
    # Verdict
    verdict: str
    confidence: float
    explanation: str
    
    # Safety
    safety_flags: List[str]
    is_dangerous: bool
    
    # Tracing
    agent_trace: List[AgentTrace]
    total_cost_usd: float
    total_duration_seconds: float
