"""Safety Checker — Function (no reasoning loop).

Scans the claim and verdict for dangerous health advice that could cause
harm if followed. Uses a single LLM call (no tool-use loop) to flag
claims that recommend stopping medication, suggest unproven treatments
for serious conditions, or could delay necessary medical care.

Type: Function (single LLM call, no tool use)
Model: Claude Sonnet

Safety categories checked:
- Stopping prescribed medication without medical supervision
- Replacing proven treatments with unproven alternatives
- Delaying necessary medical care (e.g., "cancer can be cured with diet")
- Dangerous dosage recommendations
- Anti-vaccination or anti-medical-establishment claims
- Claims about vulnerable populations (children, pregnant women, elderly)

Input (from state):
- claim: original claim text
- verdict: the verdict from the verdict agent
- sub_claims: sub-claims with their individual verdicts
- evidence: retrieved evidence (for context)

Output (to state):
- safety_flags: list of triggered safety categories
- is_dangerous: bool, True if any safety flags are critical
"""

from src.models import FactCheckState


async def run_safety_checker(state: FactCheckState) -> FactCheckState:
    """Run the safety checker.

    Scans the claim for dangerous health advice patterns and flags
    claims that could cause harm. Uses a single LLM call — no reasoning
    loop needed since safety classification is a straightforward task.

    Args:
        state: Pipeline state with claim, verdict, and sub_claims populated.

    Returns:
        Updated state with safety_flags and is_dangerous populated.
    """
    # TODO: Implement single-call LLM classification
    raise NotImplementedError("safety_checker not yet implemented")
