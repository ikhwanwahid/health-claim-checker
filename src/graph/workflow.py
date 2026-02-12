"""Main LangGraph workflow for health claim verification."""

from langgraph.graph import StateGraph, END
from src.graph.state import FactCheckState
from src.functions import decomposer, safety_checker
from src.agents import (
    retrieval_planner,
    evidence_retriever,
    vlm_extractor,
    evidence_grader,
    verdict_agent,
)


def create_workflow() -> StateGraph:
    """Create the fact-checking workflow graph."""
    
    workflow = StateGraph(FactCheckState)
    
    # Add nodes
    workflow.add_node("decomposer", decomposer.run_decomposer)
    workflow.add_node("retrieval_planner", retrieval_planner.run_retrieval_planner)
    workflow.add_node("evidence_retriever", evidence_retriever.run_evidence_retriever)
    workflow.add_node("vlm_extractor", vlm_extractor.run_vlm_extractor)
    workflow.add_node("evidence_grader", evidence_grader.run_evidence_grader)
    workflow.add_node("verdict_agent", verdict_agent.run_verdict_agent)
    workflow.add_node("safety_checker", safety_checker.run_safety_checker)
    
    # Define edges
    workflow.set_entry_point("decomposer")
    workflow.add_edge("decomposer", "retrieval_planner")
    workflow.add_edge("retrieval_planner", "evidence_retriever")
    workflow.add_edge("evidence_retriever", "vlm_extractor")
    workflow.add_edge("vlm_extractor", "evidence_grader")
    workflow.add_edge("evidence_grader", "verdict_agent")
    workflow.add_edge("verdict_agent", "safety_checker")
    workflow.add_edge("safety_checker", END)
    
    return workflow.compile()


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim.
    
    Args:
        claim: The health claim to verify
        
    Returns:
        Final state with verdict and evidence
    """
    workflow = create_workflow()
    
    initial_state: FactCheckState = {
        "claim": claim,
        "pico": None,
        "sub_claims": [],
        "entities": {},
        "retrieval_plan": {},
        "evidence": [],
        "extracted_figures": [],
        "evidence_quality": {},
        "verdict": "",
        "confidence": 0.0,
        "explanation": "",
        "safety_flags": [],
        "is_dangerous": False,
        "agent_trace": [],
        "total_cost_usd": 0.0,
        "total_duration_seconds": 0.0,
    }
    
    result = await workflow.ainvoke(initial_state)
    return result
