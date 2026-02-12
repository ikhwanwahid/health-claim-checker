"""Evidence Retriever — Agent (ReAct).

Executes the retrieval plan produced by the Retrieval Planner.
Uses a multi-step reasoning loop to orchestrate searches across multiple
sources, re-rank results, and optionally perform deep full-text search.

Type: Agent (ReAct with tool use)
Model: Claude Sonnet

Tools available to this agent:
- pubmed_search: search PubMed via E-utilities API
- semantic_scholar_search: search Semantic Scholar API
- cochrane_search: search Cochrane systematic reviews
- clinical_trials_search: search ClinicalTrials.gov
- drugbank_lookup: look up drug info and interactions via OpenFDA/RxNorm
- cross_encoder_rerank: re-rank abstracts with PubMedBERT cross-encoder
- deep_search: on-the-fly full-text embedding search (~500 chunks/claim)
- guideline_search: search pre-indexed guideline vector store

Input (from state):
- retrieval_plan: per-sub-claim method list from the planner
- sub_claims: the sub-claims to find evidence for
- pico: PICO elements for query construction
- entities: medical entities for query terms

Output (to state):
- evidence: list of Evidence objects with source, content, quality_score
"""

from src.graph.state import FactCheckState


async def run_evidence_retriever(state: FactCheckState) -> FactCheckState:
    """Run the evidence retriever agent.

    Follows the retrieval plan to search across PubMed, Semantic Scholar,
    Cochrane, ClinicalTrials.gov, DrugBank, and guidelines. Re-ranks
    results and optionally performs deep full-text search for quantitative
    claims.

    Args:
        state: Pipeline state with retrieval_plan, sub_claims, pico,
               and entities populated.

    Returns:
        Updated state with evidence list populated.
    """
    # TODO: Implement ReAct agent with tool-use loop
    raise NotImplementedError("evidence_retriever not yet implemented")
