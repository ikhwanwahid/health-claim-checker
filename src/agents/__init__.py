"""Agents (ReAct with tool use) for the health claim verification pipeline.

Agents use an LLM reasoning loop with access to tools. They can make
multiple tool calls, observe results, and decide the next action.

Modules:
    retrieval_planner: Decides which retrieval methods per sub-claim
    evidence_retriever: Orchestrates multi-method evidence retrieval
    vlm_extractor: Extracts data from medical figures via Claude Vision
    evidence_grader: Assesses evidence quality using GRADE framework
    verdict_agent: Synthesizes evidence into a 9-level verdict
"""

from src.agents import (
    retrieval_planner,
    evidence_retriever,
    vlm_extractor,
    evidence_grader,
    verdict_agent,
)

__all__ = [
    "retrieval_planner",
    "evidence_retriever",
    "vlm_extractor",
    "evidence_grader",
    "verdict_agent",
]
