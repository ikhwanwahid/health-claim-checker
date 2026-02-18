"""Verdict Agent — Agent (ReAct).

Synthesizes all evidence, quality scores, and figure extractions into a
final verdict. Uses a reasoning loop to weigh evidence for and against
each sub-claim, reconcile conflicting evidence, and produce a nuanced
9-level verdict with confidence score and human-readable explanation.

Type: Agent (ReAct with tool use)
Model: Claude Sonnet

Tools available to this agent:
- weigh_evidence: aggregate quality-weighted evidence for/against a sub-claim
- reconcile_conflicts: reason about contradictory evidence
- compute_confidence: calculate confidence based on evidence strength/agreement
- format_explanation: produce structured human-readable explanation

Verdict Taxonomy (9 levels):
- SUPPORTED: strong evidence confirms the claim
- SUPPORTED_WITH_CAVEATS: true but needs context
- OVERSTATED: kernel of truth, exaggerated
- MISLEADING: technically true, wrong impression
- PRELIMINARY: some evidence, too early to confirm
- OUTDATED: was true, evidence has changed
- NOT_SUPPORTED: no credible evidence found
- REFUTED: directly contradicted by evidence
- DANGEROUS: could cause harm if believed

Input (from state):
- sub_claims: decomposed sub-claims
- evidence: retrieved evidence
- evidence_quality: per-evidence quality scores
- extracted_figures: VLM-extracted data

Output (to state):
- verdict: one of 9 verdict levels
- confidence: 0.0-1.0 confidence score
- explanation: human-readable explanation with citations
- sub_claims: updated with per-sub-claim verdicts
"""

from src.models import FactCheckState


async def run_verdict_agent(state: FactCheckState) -> FactCheckState:
    """Run the verdict agent.

    Weighs all evidence (text + figures) using quality scores and the
    evidence hierarchy, reconciles conflicts, and produces a final
    9-level verdict with confidence and explanation.

    Args:
        state: Pipeline state with evidence, evidence_quality, and
               extracted_figures populated.

    Returns:
        Updated state with verdict, confidence, explanation, and
        per-sub-claim verdicts populated.
    """
    # TODO: Implement ReAct agent with tool-use loop
    raise NotImplementedError("verdict_agent not yet implemented")
