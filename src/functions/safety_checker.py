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

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from src.models import AgentTrace, FactCheckState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFETY_CATEGORIES = {
    "stop_medication": "Stopping prescribed medication without medical supervision",
    "unproven_alternative": "Replacing proven treatments with unproven alternatives",
    "delay_care": "Delaying necessary medical care",
    "dangerous_dosage": "Dangerous dosage recommendations",
    "anti_vaccination": "Anti-vaccination or anti-medical-establishment claims",
    "vulnerable_population": "Claims about vulnerable populations (children, pregnant, elderly)",
}

CRITICAL_FLAGS = {"stop_medication", "unproven_alternative", "delay_care", "dangerous_dosage"}

# Keywords for rule-based detection
_STOP_MEDICATION_KEYWORDS = [
    "stop taking", "don't need", "throw away", "quit your", "off medication",
]

_UNPROVEN_ALTERNATIVE_KEYWORDS = [
    "instead of", "replace", "alternative to chemo", "natural cure", "miracle cure",
]

_DELAY_CARE_KEYWORDS = [
    "don't go to", "avoid doctors", "no need for surgery", "skip treatment",
]

_DANGEROUS_DOSAGE_PATTERN = re.compile(r"\d{4,}\s*mg|megadose|high dose", re.IGNORECASE)

_ANTI_VACCINATION_KEYWORDS = [
    "vaccines cause", "don't vaccinate", "anti-vax", "vaccine injury", "vaccine damage",
]

_VULNERABLE_POPULATION_KEYWORDS = [
    "children", "pregnant", "elderly", "infant", "baby",
]

_MEDICAL_TREATMENT_KEYWORDS = [
    "medication", "treatment", "therapy", "drug", "prescription", "medicine",
    "vaccine", "surgery", "chemotherapy", "insulin", "antibiotic",
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SAFETY_SYSTEM_PROMPT = """\
You are a medical safety checker. Given a health claim, its verdict, and \
sub-claim verdicts, check the claim against each safety category and flag \
any that apply.

Safety categories:
- stop_medication: Stopping prescribed medication without medical supervision
- unproven_alternative: Replacing proven treatments with unproven alternatives
- delay_care: Delaying necessary medical care
- dangerous_dosage: Dangerous dosage recommendations
- anti_vaccination: Anti-vaccination or anti-medical-establishment claims
- vulnerable_population: Claims about vulnerable populations (children, pregnant, elderly)

Respond with ONLY valid JSON:
{
  "flags": ["category_name", ...],
  "is_dangerous": true/false,
  "reasoning": "brief explanation"
}

Rules:
- "flags" should contain zero or more category names from the list above
- "is_dangerous" should be true if ANY of these critical flags are present: \
stop_medication, unproven_alternative, delay_care, dangerous_dosage
- Be conservative: flag anything that COULD cause harm if taken literally
- An empty flags list means the claim is safe"""


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


def _check_with_llm(
    claim: str,
    verdict: str,
    sub_claims: list,
    evidence: list,
) -> tuple[list[str], bool, str, float]:
    """Use Claude to check claim safety.

    Returns:
        (flags, is_dangerous, reasoning, cost_usd)
    """
    # Build context
    sc_lines = []
    for sc in sub_claims:
        sc_verdict = sc.verdict or "pending"
        sc_lines.append(f"- {sc.text} → {sc_verdict}")
    sc_text = "\n".join(sc_lines) if sc_lines else "(no sub-claims)"

    ev_lines = []
    for ev in evidence[:5]:
        ev_lines.append(f"- [{ev.source}] {ev.title[:80]}")
    ev_text = "\n".join(ev_lines) if ev_lines else "(no evidence)"

    user_msg = (
        f'Claim: "{claim}"\n'
        f"Verdict: {verdict}\n\n"
        f"Sub-claim verdicts:\n{sc_text}\n\n"
        f"Key evidence:\n{ev_text}"
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=_SAFETY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    response_text = message.content[0].text.strip()

    # Cost
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    # Parse JSON
    try:
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        data = json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse safety LLM response, falling back to rule-based")
        raise

    flags = [f for f in data.get("flags", []) if f in SAFETY_CATEGORIES]
    is_dangerous = any(f in CRITICAL_FLAGS for f in flags)
    reasoning = data.get("reasoning", "")

    return flags, is_dangerous, reasoning, cost


# ---------------------------------------------------------------------------
# Rule-based path
# ---------------------------------------------------------------------------


def _check_rule_based(
    claim: str,
    verdict: str,
    sub_claims: list,
) -> tuple[list[str], bool]:
    """Rule-based safety checking fallback.

    Returns:
        (flags, is_dangerous)
    """
    claim_lower = claim.lower()
    flags: list[str] = []

    # stop_medication
    if any(kw in claim_lower for kw in _STOP_MEDICATION_KEYWORDS):
        flags.append("stop_medication")

    # unproven_alternative
    if any(kw in claim_lower for kw in _UNPROVEN_ALTERNATIVE_KEYWORDS):
        flags.append("unproven_alternative")

    # delay_care
    if any(kw in claim_lower for kw in _DELAY_CARE_KEYWORDS):
        flags.append("delay_care")

    # dangerous_dosage
    if _DANGEROUS_DOSAGE_PATTERN.search(claim):
        flags.append("dangerous_dosage")

    # anti_vaccination
    if any(kw in claim_lower for kw in _ANTI_VACCINATION_KEYWORDS):
        flags.append("anti_vaccination")

    # vulnerable_population — only flag if verdict is bad
    bad_verdicts = {"REFUTED", "NOT_SUPPORTED", "DANGEROUS"}
    has_vulnerable = any(kw in claim_lower for kw in _VULNERABLE_POPULATION_KEYWORDS)
    if has_vulnerable and verdict in bad_verdicts:
        flags.append("vulnerable_population")

    # Auto-flag: if verdict is DANGEROUS or REFUTED and claim involves medical treatments
    if verdict in {"DANGEROUS", "REFUTED"}:
        if any(kw in claim_lower for kw in _MEDICAL_TREATMENT_KEYWORDS):
            if "stop_medication" not in flags and "unproven_alternative" not in flags:
                # Don't double-flag, but note the concern
                pass

    is_dangerous = any(f in CRITICAL_FLAGS for f in flags)
    return flags, is_dangerous


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    start_time = time.time()
    claim = state.get("claim", "")
    verdict = state.get("verdict", "")
    sub_claims = state.get("sub_claims", [])
    evidence = state.get("evidence", [])
    cost = 0.0
    tools_called: list[str] = []

    if ANTHROPIC_API_KEY and claim:
        try:
            flags, is_dangerous, reasoning, llm_cost = _check_with_llm(
                claim, verdict, sub_claims, evidence,
            )
            cost += llm_cost
            tools_called = ["llm_safety_check"]
        except Exception as e:
            logger.warning("LLM safety check failed, using rule-based: %s", e)
            flags, is_dangerous = _check_rule_based(claim, verdict, sub_claims)
            tools_called = ["rule_based_fallback"]
    else:
        flags, is_dangerous = _check_rule_based(claim, verdict, sub_claims)
        tools_called = ["rule_based_fallback"]

    duration = time.time() - start_time

    # Build trace
    trace = AgentTrace(
        agent="safety_checker",
        node_type="function",
        duration_seconds=round(duration, 2),
        cost_usd=round(cost, 6),
        input_summary=f"Claim: {claim[:80]}",
        output_summary=f"{len(flags)} flags, dangerous={is_dangerous}",
        success=True,
        tools_called=tools_called,
    )

    # Update state
    existing_traces = state.get("agent_trace", [])
    existing_cost = state.get("total_cost_usd", 0.0)
    existing_duration = state.get("total_duration_seconds", 0.0)

    return {
        **state,
        "safety_flags": flags,
        "is_dangerous": is_dangerous,
        "agent_trace": existing_traces + [trace],
        "total_cost_usd": existing_cost + cost,
        "total_duration_seconds": existing_duration + duration,
    }
