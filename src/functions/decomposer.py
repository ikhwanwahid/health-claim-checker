"""Claim Decomposer — Function (no reasoning loop).

Takes a raw health claim and produces:
1. Medical entities (via scispaCy NER)
2. PICO extraction (LLM or rule-based)
3. Atomic sub-claims that can each be independently verified

Type: Function (fixed steps, single LLM call for decomposition)
Model: Claude Sonnet
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from src.models import AgentTrace, FactCheckState, PICO, SubClaim
from src.medical_nlp.medical_ner import extract_entities, MedicalEntities
from src.medical_nlp.pico_extractor import extract_pico

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a medical claim analyst. Given a health claim, its PICO elements, \
and extracted medical entities, decompose it into atomic sub-claims.

Each sub-claim should be:
- A single, independently verifiable statement
- Specific enough to search for evidence
- Faithful to the original claim (don't add or remove meaning)

Also assign PICO elements to each sub-claim where applicable.

Respond with ONLY a JSON array:
[
  {
    "text": "the sub-claim text",
    "pico": {
      "population": "...",
      "intervention": "...",
      "comparison": "...",
      "outcome": "..."
    }
  }
]

Guidelines:
- Simple claims (single intervention, single outcome) may produce just 1 sub-claim
- Complex claims should be split (e.g., "X treats Y and prevents Z" → 2 sub-claims)
- Quantitative claims should preserve the numbers (e.g., "reduces risk by 50%")
- If the claim compares two things, include the comparison in the sub-claim
- Use null for PICO elements that don't apply to a specific sub-claim"""


def _decompose_with_llm(
    claim: str,
    pico: PICO,
    entities: MedicalEntities,
) -> tuple[list[SubClaim], float]:
    """Use Claude to decompose a claim into sub-claims."""
    entity_list = entities.all_entities()
    entity_str = f"\nEntities: {', '.join(entity_list)}" if entity_list else ""

    pico_str = (
        f"\nPICO: P={pico.population}, I={pico.intervention}, "
        f"C={pico.comparison}, O={pico.outcome}"
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=_DECOMPOSE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f'Claim: "{claim}"{pico_str}{entity_str}',
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Parse JSON
    try:
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        items = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse decomposition response: %s", response_text)
        # Fallback: treat entire claim as a single sub-claim
        return [SubClaim(id="sc-1", text=claim, pico=pico)]

    sub_claims = []
    for i, item in enumerate(items):
        sc_pico = None
        if item.get("pico"):
            p = item["pico"]
            sc_pico = PICO(
                population=p.get("population"),
                intervention=p.get("intervention"),
                comparison=p.get("comparison"),
                outcome=p.get("outcome"),
            )

        sub_claims.append(SubClaim(
            id=f"sc-{i + 1}",
            text=item.get("text", claim),
            pico=sc_pico or pico,
        ))

    # Extract cost info from the response
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000  # Sonnet pricing

    return sub_claims, cost


def _decompose_rule_based(
    claim: str,
    pico: PICO,
    entities: MedicalEntities,
) -> list[SubClaim]:
    """Rule-based decomposition fallback.

    Splits on conjunctions and common multi-claim patterns.
    """
    claim_lower = claim.lower()

    # Split on "and" between independent clauses
    splitters = [" and also ", " and ", " as well as ", " while also "]
    parts = [claim]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            if splitter in part.lower():
                idx = part.lower().find(splitter)
                left = part[:idx].strip()
                right = part[idx + len(splitter):].strip()
                if left:
                    new_parts.append(left)
                if right:
                    new_parts.append(right)
            else:
                new_parts.append(part)
        parts = new_parts

    sub_claims = []
    for i, part in enumerate(parts):
        sub_claims.append(SubClaim(
            id=f"sc-{i + 1}",
            text=part.strip().rstrip("."),
            pico=pico,
        ))

    return sub_claims


async def run_decomposer(state: FactCheckState) -> FactCheckState:
    """Run the decomposer agent.

    Takes the raw claim from state and produces:
    - pico: PICO extraction
    - sub_claims: list of atomic sub-claims
    - entities: extracted medical entities
    - agent_trace: updated with decomposer trace

    Args:
        state: Current pipeline state with 'claim' populated.

    Returns:
        Updated state.
    """
    start_time = time.time()
    claim = state["claim"]
    cost = 0.0

    # Step 1: Extract entities via NER
    entities = extract_entities(claim)
    entities_dict = {
        "drugs": entities.drugs,
        "conditions": entities.conditions,
        "genes": entities.genes,
        "organisms": entities.organisms,
        "procedures": entities.procedures,
        "anatomical": entities.anatomical,
    }

    # Step 2: Extract PICO
    use_llm = bool(ANTHROPIC_API_KEY)
    pico, _ = extract_pico(claim, use_llm=use_llm)

    # Step 3: Decompose into sub-claims
    if ANTHROPIC_API_KEY:
        try:
            sub_claims, llm_cost = _decompose_with_llm(claim, pico, entities)
            cost += llm_cost
        except Exception as e:
            logger.warning("LLM decomposition failed, using rule-based: %s", e)
            sub_claims = _decompose_rule_based(claim, pico, entities)
    else:
        sub_claims = _decompose_rule_based(claim, pico, entities)

    duration = time.time() - start_time

    # Build trace
    trace = AgentTrace(
        agent="decomposer",
        node_type="function",
        duration_seconds=round(duration, 2),
        cost_usd=round(cost, 6),
        input_summary=f"Claim: {claim[:80]}",
        output_summary=f"{len(sub_claims)} sub-claims, PICO extracted",
        success=True,
        tools_called=["extract_entities", "extract_pico", "decompose_claim"],
    )

    # Update state
    existing_traces = state.get("agent_trace", [])
    existing_cost = state.get("total_cost_usd", 0.0)
    existing_duration = state.get("total_duration_seconds", 0.0)

    return {
        **state,
        "pico": pico,
        "sub_claims": sub_claims,
        "entities": entities_dict,
        "agent_trace": existing_traces + [trace],
        "total_cost_usd": existing_cost + cost,
        "total_duration_seconds": existing_duration + duration,
    }
