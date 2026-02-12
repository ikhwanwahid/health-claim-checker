"""PICO element extraction from health claims.

Uses scispaCy NER for entity detection and Claude for structured PICO framing.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from src.graph.state import PICO
from src.medical_nlp.medical_ner import MedicalEntities, extract_entities

logger = logging.getLogger(__name__)

_PICO_SYSTEM_PROMPT = """\
You are a medical research assistant. Given a health claim and its extracted \
medical entities, identify the PICO elements.

PICO framework:
- P (Population): The patient group or population the claim is about.
- I (Intervention): The treatment, exposure, or action being discussed.
- C (Comparison): What the intervention is being compared to, ONLY if an \
explicit comparator is stated in the claim (e.g., "more effective than X", \
"as good as Y", "compared to Z"). If no comparison is mentioned, use null.
- O (Outcome): The health outcome or effect being claimed.

Rules:
- Extract only what is stated or directly implied. Do not fabricate.
- If an element is truly absent and cannot be inferred, use null.
- Keep each element concise (a short phrase, not a full sentence).

Respond with ONLY a JSON object:
{
  "population": "...",
  "intervention": "...",
  "comparison": "...",
  "outcome": "..."
}"""


def extract_pico_with_llm(
    claim: str,
    entities: Optional[MedicalEntities] = None,
) -> PICO:
    """Extract PICO elements using Claude.

    Args:
        claim: The health claim text.
        entities: Pre-extracted medical entities (extracted if not provided).

    Returns:
        PICO object with populated fields.
    """
    if entities is None:
        entities = extract_entities(claim)

    entity_context = ""
    if entities.all_entities():
        entity_context = f"\n\nExtracted medical entities: {', '.join(entities.all_entities())}"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=_PICO_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Health claim: \"{claim}\"{entity_context}",
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        data = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse PICO response: %s", response_text)
        return PICO()

    return PICO(
        population=data.get("population"),
        intervention=data.get("intervention"),
        comparison=data.get("comparison"),
        outcome=data.get("outcome"),
    )


def extract_pico_rule_based(claim: str, entities: MedicalEntities) -> PICO:
    """Lightweight rule-based PICO extraction (no LLM call).

    Uses sentence structure (verb as pivot) and intervention heuristics
    to assign PICO roles. The key insight: health claims typically follow
    the pattern "[Intervention] [effect verb] [Outcome] in [Population]".
    """
    claim_lower = claim.lower()

    # --- Step 1: Find the effect verb (pivot point) ---
    effect_verbs = [
        "reverses", "prevents", "reduces", "causes", "treats", "cures",
        "improves", "increases", "decreases", "protects", "lowers",
        "raises", "inhibits", "promotes", "alleviates", "mitigates",
        "is as effective as", "is more effective than", "is better than",
        "helps", "boosts", "enhances", "worsens", "triggers",
    ]

    verb = None
    verb_pos = -1
    for v in effect_verbs:
        idx = claim_lower.find(v)
        if idx != -1:
            verb = v
            verb_pos = idx
            break

    # --- Step 2: Split claim around the verb ---
    before_verb = claim[:verb_pos].strip() if verb_pos > 0 else ""
    after_verb = claim[verb_pos + len(verb):].strip().rstrip(".") if verb else ""

    # --- Step 3: Identify intervention ---
    # Heuristics: what comes before the verb is typically the intervention.
    # Also check known intervention patterns.
    intervention = None
    comparison = None

    _INTERVENTION_HINTS = {
        "fasting", "intermittent fasting", "exercise", "walking", "running",
        "yoga", "meditation", "diet", "keto", "ketogenic", "vegan",
        "vegetarian", "paleo", "mediterranean", "low-carb", "caloric restriction",
        "sleep", "sunlight", "cold exposure", "sauna", "acupuncture",
        "massage", "drinking", "supplementation", "smoking cessation",
    }

    _SUPPLEMENT_HINTS = {
        "vitamin", "zinc", "magnesium", "iron", "calcium", "omega",
        "probiotic", "prebiotic", "fiber", "protein", "collagen",
        "melatonin", "creatine", "turmeric", "curcumin", "ginger",
        "garlic", "green tea", "fish oil", "ashwagandha", "elderberry",
    }

    if before_verb:
        intervention = before_verb
    elif entities.drugs:
        intervention = entities.drugs[0]

    # --- Step 4: Handle comparison claims ("X is as effective as Y") ---
    comparison_patterns = [
        "is as effective as", "is more effective than", "is better than",
        "works better than", "is safer than",
    ]
    for pattern in comparison_patterns:
        if pattern in claim_lower:
            parts = claim_lower.split(pattern)
            if len(parts) == 2:
                intervention = claim[:claim_lower.find(pattern)].strip()
                rest = claim[claim_lower.find(pattern) + len(pattern):].strip()
                # Split "ibuprofen for arthritis pain" into comparison + context
                for_idx = rest.lower().find(" for ")
                if for_idx != -1:
                    comparison = rest[:for_idx].strip()
                    after_verb = rest[for_idx + 5:].strip().rstrip(".")
                else:
                    comparison = rest.rstrip(".")
                    after_verb = ""
            break

    # --- Step 5: Extract population from "in [population]" pattern ---
    population = None
    pop_patterns = [" in ", " among ", " for ", " of "]

    if after_verb:
        for pat in pop_patterns:
            idx = after_verb.lower().rfind(pat)
            if idx != -1:
                population = after_verb[idx + len(pat):].strip().rstrip(".")
                after_verb = after_verb[:idx].strip()
                break

    # --- Step 6: What remains after the verb (minus population) is the outcome ---
    outcome = after_verb if after_verb else None

    # --- Step 7: Refine — check if intervention was misclassified ---
    # If intervention looks like a disease/condition, it might actually be population
    if intervention and not population:
        int_lower = intervention.lower()
        is_likely_intervention = (
            any(hint in int_lower for hint in _INTERVENTION_HINTS)
            or any(hint in int_lower for hint in _SUPPLEMENT_HINTS)
            or bool(entities.drugs)
        )
        if not is_likely_intervention:
            # Check if it looks more like a population
            _POPULATION_HINTS = {"patients", "adults", "children", "women", "men", "people", "elderly"}
            if any(hint in int_lower for hint in _POPULATION_HINTS):
                population = intervention
                intervention = None

    return PICO(
        population=population,
        intervention=intervention,
        comparison=comparison,
        outcome=outcome,
    )


def extract_pico(
    claim: str,
    use_llm: bool = True,
) -> tuple[PICO, MedicalEntities]:
    """Main entry point: extract PICO elements from a health claim.

    Args:
        claim: The health claim text.
        use_llm: Whether to use Claude for PICO extraction (True) or
                 fall back to rule-based extraction (False).

    Returns:
        Tuple of (PICO, MedicalEntities).
    """
    entities = extract_entities(claim)

    if use_llm:
        try:
            pico = extract_pico_with_llm(claim, entities)
        except Exception as e:
            logger.warning("LLM PICO extraction failed, falling back to rules: %s", e)
            pico = extract_pico_rule_based(claim, entities)
    else:
        pico = extract_pico_rule_based(claim, entities)

    return pico, entities
