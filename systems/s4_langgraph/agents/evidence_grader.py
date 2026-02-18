"""Evidence Grader — Agent (ReAct).

Assesses the quality and relevance of each piece of retrieved evidence.
Uses a reasoning loop to classify study types, apply the evidence hierarchy,
evaluate methodological quality, and produce per-evidence quality scores
following the GRADE framework.

Type: Agent (ReAct with tool use)
Model: Claude Sonnet

Tools available to this agent:
- classify_study_type: determine study type (RCT, cohort, case-control, etc.)
- assess_methodology: evaluate sample size, blinding, randomization
- check_relevance: score how directly evidence addresses the sub-claim
- apply_grade: apply GRADE framework criteria (risk of bias, inconsistency,
  indirectness, imprecision, publication bias)

Evidence Hierarchy Weights:
- guideline: 1.0, systematic_review: 0.9, rct: 0.8, cohort: 0.6
- case_control: 0.5, case_report: 0.3, in_vitro: 0.2, expert_opinion: 0.1

Input (from state):
- evidence: list of Evidence objects from retriever
- extracted_figures: VLM-extracted data from figures
- sub_claims: sub-claims for relevance scoring

Output (to state):
- evidence_quality: dict with per-evidence quality scores, study type
  classifications, and overall evidence strength per sub-claim
"""

from src.models import FactCheckState


async def run_evidence_grader(state: FactCheckState) -> FactCheckState:
    """Run the evidence grader agent.

    Evaluates each piece of evidence for study type, methodological quality,
    and relevance to the sub-claims. Applies the GRADE framework and evidence
    hierarchy weights to produce quality scores.

    Args:
        state: Pipeline state with evidence and extracted_figures populated.

    Returns:
        Updated state with evidence_quality populated.
    """
    # TODO: Implement ReAct agent with tool-use loop
    raise NotImplementedError("evidence_grader not yet implemented")
