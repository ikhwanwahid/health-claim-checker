"""VLM Extractor — Agent (ReAct).

Extracts structured data from medical figures (forest plots, Kaplan-Meier
curves, dose-response charts) found in retrieved papers. Uses Claude Vision
in a reasoning loop to identify figure types, extract numerical data, and
assess whether figures support or contradict the claim.

Type: Agent (ReAct with tool use)
Model: Claude Vision (Sonnet with vision)

Tools available to this agent:
- fetch_pdf_figures: extract figures/images from a PDF
- classify_figure: identify figure type (forest plot, KM curve, etc.)
- extract_forest_plot: parse hazard ratios, CIs from forest plots
- extract_kaplan_meier: parse survival curves and endpoints
- extract_table: parse tabular data from figure images

Input (from state):
- evidence: retrieved evidence (some with PMC full-text / PDF links)
- sub_claims: sub-claims to check figures against

Output (to state):
- extracted_figures: list of dicts with figure type, extracted data,
  source paper, and relevance to sub-claims
"""

from src.graph.state import FactCheckState


async def run_vlm_extractor(state: FactCheckState) -> FactCheckState:
    """Run the VLM extractor agent.

    Scans retrieved evidence for papers with available figures/PDFs,
    extracts and interprets medical visualizations using Claude Vision,
    and structures the extracted data for the evidence grader.

    Args:
        state: Pipeline state with evidence list populated.

    Returns:
        Updated state with extracted_figures populated.
    """
    # TODO: Implement ReAct agent with Claude Vision tool-use loop
    raise NotImplementedError("vlm_extractor not yet implemented")
