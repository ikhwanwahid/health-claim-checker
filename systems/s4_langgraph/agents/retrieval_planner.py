"""Retrieval Planner — Agent (ReAct).

Decides *which* retrieval methods to use for each sub-claim.
Uses a reasoning loop (LLM + tools) to inspect sub-claim characteristics,
check guideline coverage, and assign methods. Falls back to rule-based
planning when no API key is available.

Type: Agent (ReAct with tool use)
Model: Claude Sonnet

Tools available to this agent:
- analyze_claim_characteristics: detect claim features via regex/keywords
- check_guideline_coverage: check if guidelines cover the topic
- assign_methods: commit retrieval methods for a sub-claim

Input (from state):
- sub_claims: list of atomic sub-claims from the decomposer
- entities: extracted medical entities dict
- pico: PICO extraction

Output (to state):
- retrieval_plan: dict mapping each sub-claim ID to a list of methods
  e.g. {"sc-1": ["pubmed_api", "semantic_scholar", "cross_encoder"],
        "sc-2": ["guideline_store", "cochrane_api", "cross_encoder"]}
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, GUIDELINES_DIR
from src.models import AgentTrace, FactCheckState, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_METHODS = {
    "pubmed_api",
    "semantic_scholar",
    "cochrane_api",
    "clinical_trials",
    "drugbank_api",
    "cross_encoder",
    "deep_search",
    "guideline_store",
}

DEFAULT_METHODS = ["pubmed_api", "semantic_scholar", "cross_encoder"]

MAX_REACT_STEPS = 10

# Keyword sets for claim characteristic detection
_DRUG_INTERACTION_KEYWORDS = {
    "interact", "interaction", "interactions", "combined with",
    "co-administer", "concomitant", "contraindicated", "potentiate",
    "antagonize", "synergistic",
}

_RECOMMENDATION_KEYWORDS = {
    "should", "recommend", "recommended", "guideline", "guidelines",
    "advised", "advisory", "consensus", "standard of care", "first-line",
    "second-line", "approved for",
}

_TREATMENT_KEYWORDS = {
    "effective", "treats", "treatment", "cure", "cures", "therapeutic",
    "therapy", "efficacy", "efficacious", "beneficial",
}

_COMPARISON_KEYWORDS = {
    "compared to", "versus", "vs", "better than", "worse than",
    "superior", "inferior", "non-inferior", "equivalent",
    "more effective", "less effective",
}

_SAFETY_KEYWORDS = {
    "safe", "safety", "side effect", "side effects", "adverse",
    "adverse event", "toxicity", "harmful", "harm", "risk",
    "dangerous", "death", "mortality",
}

# Regex for quantitative claims (numbers, percentages, fold-changes)
_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|fold|times|mg|ml|kg|g/d[Ll]|mmol)"
    r"|\breduce[sd]?\s+(?:by\s+)?\d+"
    r"|\bincrease[sd]?\s+(?:by\s+)?\d+"
    r"|\b\d+\s*-\s*\d+\s*%"
)

# Guideline topic keywords (mapped from directory names / common topics)
_GUIDELINE_TOPICS: dict[str, list[str]] = {
    "who": [
        "vaccination", "vaccine", "immunization", "malaria", "tuberculosis",
        "hiv", "aids", "maternal", "child health", "nutrition", "obesity",
        "diabetes", "hypertension", "cardiovascular", "cancer screening",
        "antimicrobial", "antibiotic resistance", "mental health",
        "physical activity", "breastfeeding", "sugar", "sodium", "salt",
    ],
    "nih": [
        "cancer", "diabetes", "cardiovascular", "heart", "stroke",
        "alzheimer", "dementia", "asthma", "copd", "arthritis",
        "depression", "anxiety", "substance abuse", "opioid",
        "cholesterol", "blood pressure", "supplement", "vitamin",
        "clinical trial", "genomic", "precision medicine",
    ],
    "moh_singapore": [
        "diabetes", "hypertension", "lipid", "cholesterol", "copd",
        "asthma", "stroke", "dementia", "depression", "obesity",
        "chronic kidney", "osteoporosis", "cancer screening",
    ],
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_analyze_claim(
    sub_claim_text: str,
    entities: dict,
) -> dict[str, Any]:
    """Detect claim characteristics via regex/keyword matching.

    Returns a dict of boolean features and extracted details.
    """
    text_lower = sub_claim_text.lower()

    # Drug-related checks
    has_drugs = bool(entities.get("drugs"))
    drug_names = entities.get("drugs", [])
    has_drug_interaction = has_drugs and any(
        kw in text_lower for kw in _DRUG_INTERACTION_KEYWORDS
    )

    # Quantitative
    number_matches = _NUMBER_PATTERN.findall(text_lower)
    has_numbers = bool(number_matches)

    # Recommendation / guideline
    asks_recommendation = any(kw in text_lower for kw in _RECOMMENDATION_KEYWORDS)

    # Treatment effectiveness
    asks_effectiveness = any(kw in text_lower for kw in _TREATMENT_KEYWORDS)

    # Comparison
    has_comparison = any(kw in text_lower for kw in _COMPARISON_KEYWORDS)

    # Safety / adverse events
    asks_safety = any(kw in text_lower for kw in _SAFETY_KEYWORDS)

    # Conditions mentioned
    conditions = entities.get("conditions", [])

    return {
        "sub_claim": sub_claim_text,
        "has_drugs": has_drugs,
        "drug_names": drug_names,
        "has_drug_interaction": has_drug_interaction,
        "has_numbers": has_numbers,
        "number_matches": number_matches,
        "asks_recommendation": asks_recommendation,
        "asks_effectiveness": asks_effectiveness,
        "has_comparison": has_comparison,
        "asks_safety": asks_safety,
        "conditions": conditions,
        "entity_count": sum(len(v) for v in entities.values() if isinstance(v, list)),
    }


def _tool_check_guidelines(
    sub_claim_text: str,
    entities: dict,
) -> dict[str, Any]:
    """Check if pre-indexed guidelines cover the topic.

    Uses keyword matching against guideline directory topics.
    Returns which guideline sources are relevant.
    """
    text_lower = sub_claim_text.lower()

    # Combine text + entity names for matching
    search_terms = [text_lower]
    for key in ("drugs", "conditions", "procedures"):
        for term in entities.get(key, []):
            search_terms.append(term.lower())
    search_text = " ".join(search_terms)

    matches: dict[str, list[str]] = {}
    for source, topics in _GUIDELINE_TOPICS.items():
        matched_topics = [t for t in topics if t in search_text]
        if matched_topics:
            matches[source] = matched_topics

    # Check if guideline PDFs actually exist on disk
    sources_with_files: list[str] = []
    for source in matches:
        source_dir = GUIDELINES_DIR / source
        if source_dir.exists():
            pdf_files = list(source_dir.glob("*.pdf"))
            if pdf_files:
                sources_with_files.append(source)

    has_coverage = bool(matches)
    has_files = bool(sources_with_files)

    return {
        "has_guideline_coverage": has_coverage,
        "matching_sources": matches,
        "sources_with_indexed_files": sources_with_files,
        "recommendation": (
            "guideline_store is recommended — matching topics found"
            if has_coverage
            else "no guideline coverage detected for this topic"
        ),
        "note": (
            "Guideline PDFs are available on disk for: "
            + ", ".join(sources_with_files)
            if has_files
            else "No guideline PDFs indexed yet — guideline_store will "
            "have no results until scripts/index_guidelines.py is run"
        ),
    }


def _tool_assign_methods(
    sub_claim_id: str,
    methods: list[str],
    reasoning: str,
    assignments: dict[str, list[str]],
    all_sub_claim_ids: set[str],
) -> dict[str, Any]:
    """Commit retrieval methods for a sub-claim.

    Validates method names, tracks assignments, reports remaining sub-claims.
    """
    # Validate methods
    invalid = [m for m in methods if m not in VALID_METHODS]
    if invalid:
        return {
            "success": False,
            "error": f"Invalid method(s): {invalid}. Valid: {sorted(VALID_METHODS)}",
            "assigned": False,
        }

    if not methods:
        return {
            "success": False,
            "error": "Must assign at least one method.",
            "assigned": False,
        }

    # Validate sub-claim ID
    if sub_claim_id not in all_sub_claim_ids:
        return {
            "success": False,
            "error": (
                f"Unknown sub-claim ID '{sub_claim_id}'. "
                f"Valid IDs: {sorted(all_sub_claim_ids)}"
            ),
            "assigned": False,
        }

    # Record assignment
    assignments[sub_claim_id] = methods

    remaining = all_sub_claim_ids - set(assignments.keys())
    return {
        "success": True,
        "assigned": True,
        "sub_claim_id": sub_claim_id,
        "methods": methods,
        "reasoning": reasoning,
        "remaining_sub_claims": sorted(remaining),
        "all_assigned": len(remaining) == 0,
    }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    entities: dict,
    assignments: dict[str, list[str]],
    all_sub_claim_ids: set[str],
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate function."""
    if tool_name == "analyze_claim_characteristics":
        return _tool_analyze_claim(
            sub_claim_text=tool_input["sub_claim_text"],
            entities=entities,
        )
    elif tool_name == "check_guideline_coverage":
        return _tool_check_guidelines(
            sub_claim_text=tool_input["sub_claim_text"],
            entities=entities,
        )
    elif tool_name == "assign_methods":
        return _tool_assign_methods(
            sub_claim_id=tool_input["sub_claim_id"],
            methods=tool_input["methods"],
            reasoning=tool_input.get("reasoning", ""),
            assignments=assignments,
            all_sub_claim_ids=all_sub_claim_ids,
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Anthropic tool schema definitions
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "name": "analyze_claim_characteristics",
        "description": (
            "Analyze a sub-claim's characteristics using regex and keyword "
            "matching. Detects: drug interactions, quantitative claims, "
            "recommendations, treatment effectiveness, comparisons, safety "
            "concerns. Use this FIRST for each sub-claim before assigning methods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_text": {
                    "type": "string",
                    "description": "The text of the sub-claim to analyze.",
                },
            },
            "required": ["sub_claim_text"],
        },
    },
    {
        "name": "check_guideline_coverage",
        "description": (
            "Check if pre-indexed clinical guidelines (WHO, NIH, MOH Singapore) "
            "cover the topic of a sub-claim. Returns which guideline sources "
            "match and whether indexed PDFs are available. Use this to decide "
            "whether to include 'guideline_store' in the retrieval plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_text": {
                    "type": "string",
                    "description": "The text of the sub-claim to check.",
                },
            },
            "required": ["sub_claim_text"],
        },
    },
    {
        "name": "assign_methods",
        "description": (
            "Commit the retrieval methods for a specific sub-claim. Call this "
            "once per sub-claim after analyzing its characteristics. Valid "
            "methods: pubmed_api, semantic_scholar, cochrane_api, "
            "clinical_trials, drugbank_api, cross_encoder, deep_search, "
            "guideline_store. Always include cross_encoder for re-ranking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim (e.g., 'sc-1').",
                },
                "methods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of retrieval methods to assign.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why these methods were chosen.",
                },
            },
            "required": ["sub_claim_id", "methods", "reasoning"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt + user message builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a retrieval planning agent for a medical fact-checking system. Your job \
is to decide which retrieval methods to use for each sub-claim.

## Available Retrieval Methods

| Method | What it does | When to use |
|--------|-------------|-------------|
| pubmed_api | Search PubMed for biomedical literature | Almost always — primary literature source |
| semantic_scholar | Search Semantic Scholar for papers | Broad academic search, good complement to PubMed |
| cross_encoder | Re-rank retrieved abstracts by relevance | Almost always — improves precision after API search |
| cochrane_api | Search Cochrane Library for systematic reviews | ONLY when claim explicitly compares treatments or asks "does X treat/prevent Y?" |
| clinical_trials | Search ClinicalTrials.gov for trials | ONLY when claim is about a specific drug/intervention treating a specific condition |
| drugbank_api | Look up drug interactions and mechanisms | ONLY when claim explicitly mentions drug-drug interaction, combination, or co-administration |
| deep_search | Embed and search full-text chunks with PubMedBERT | ONLY when claim contains specific numbers, percentages, or dosages |
| guideline_store | Search pre-indexed clinical guidelines (WHO/NIH/MOH) | ONLY when claim is about what doctors should prescribe or official recommendations |

## Decision Rules

1. **Always include**: pubmed_api + semantic_scholar + cross_encoder (baseline for every claim)
2. **Drug interaction claims** → add drugbank_api. ONLY if the claim mentions two drugs interacting, \
being combined, or being co-administered. A claim merely mentioning a drug does NOT qualify.
3. **Treatment comparison/efficacy** → add cochrane_api + clinical_trials. ONLY if the claim says \
something "treats", "prevents", "is effective for", or compares two treatments. General health \
claims (e.g. "water prevents kidney stones") do NOT qualify — use only the baseline.
4. **Official recommendations** → add guideline_store. ONLY if the claim uses words like "should", \
"recommended", "first-line", "standard of care". Do NOT add for general health claims.
5. **Quantitative claims** → add deep_search. ONLY if the claim contains specific numbers or percentages.
6. **Safety/adverse events** → add clinical_trials. ONLY if the claim mentions side effects, \
adverse events, bleeding risk, toxicity, or mortality — not merely "risk" in a general sense.

## IMPORTANT: Be selective, not comprehensive

Each method has a cost. The goal is to use the MINIMUM set of methods needed. \
Many claims need only the 3 baseline methods. Do NOT add cochrane_api or clinical_trials \
"just in case" — only add them when the claim characteristics clearly warrant it. \
If analyze_claim_characteristics returns no special flags, assign ONLY the baseline.

## Process

For EACH sub-claim:
1. Call analyze_claim_characteristics to detect features
2. Call check_guideline_coverage ONLY if asks_recommendation is true
3. Call assign_methods — let the detected flags drive your choices, do not add extras

Work through all sub-claims systematically. After assigning methods to every \
sub-claim, stop — your work is done."""


def _build_user_message(state: FactCheckState) -> str:
    """Build the user message with sub-claims, entities, and PICO."""
    sub_claims = state.get("sub_claims", [])
    entities = state.get("entities", {})
    pico = state.get("pico")

    parts: list[str] = []
    parts.append(f"Original claim: \"{state['claim']}\"")
    parts.append("")

    # PICO
    if pico:
        parts.append("PICO extraction:")
        parts.append(f"  Population: {pico.population or 'N/A'}")
        parts.append(f"  Intervention: {pico.intervention or 'N/A'}")
        parts.append(f"  Comparison: {pico.comparison or 'N/A'}")
        parts.append(f"  Outcome: {pico.outcome or 'N/A'}")
        parts.append("")

    # Entities
    if entities:
        parts.append("Extracted entities:")
        for etype, elist in entities.items():
            if elist:
                parts.append(f"  {etype}: {', '.join(elist)}")
        parts.append("")

    # Sub-claims
    parts.append(f"Sub-claims to plan retrieval for ({len(sub_claims)} total):")
    for sc in sub_claims:
        sc_pico = ""
        if sc.pico:
            sc_pico = (
                f" [P={sc.pico.population}, I={sc.pico.intervention}, "
                f"C={sc.pico.comparison}, O={sc.pico.outcome}]"
            )
        parts.append(f"  {sc.id}: \"{sc.text}\"{sc_pico}")

    parts.append("")
    parts.append(
        "Please analyze each sub-claim and assign retrieval methods. "
        "Use the tools provided to analyze characteristics, check guideline "
        "coverage, and assign methods for each sub-claim."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def _plan_with_react(state: FactCheckState) -> tuple[dict[str, list[str]], float, list[ToolCall], int]:
    """Run the ReAct loop to produce a retrieval plan.

    Returns:
        (assignments, cost_usd, tool_calls, reasoning_steps)
    """
    entities = state.get("entities", {})
    sub_claims = state.get("sub_claims", [])
    all_sub_claim_ids = {sc.id for sc in sub_claims}

    # Shared mutable state for assignments
    assignments: dict[str, list[str]] = {}
    tool_call_records: list[ToolCall] = []
    total_cost = 0.0
    reasoning_steps = 0

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_message(state)},
    ]

    for step in range(MAX_REACT_STEPS):
        reasoning_steps += 1

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=_SYSTEM_PROMPT,
            tools=_TOOL_SCHEMAS,
            messages=messages,
        )

        # Track cost (Sonnet pricing)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_cost += (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

        # Check if the model wants to use tools
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # Model is done (end_turn or just text) — break
            break

        # Build assistant message with all content blocks
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and build tool results
        tool_results: list[dict[str, Any]] = []
        for tool_block in tool_use_blocks:
            tc_start = time.time()
            result = _execute_tool(
                tool_name=tool_block.name,
                tool_input=tool_block.input,
                entities=entities,
                assignments=assignments,
                all_sub_claim_ids=all_sub_claim_ids,
            )
            tc_duration = time.time() - tc_start

            tool_call_records.append(ToolCall(
                tool=tool_block.name,
                input_summary=json.dumps(tool_block.input)[:200],
                output_summary=json.dumps(result)[:200],
                duration_seconds=round(tc_duration, 4),
                success="error" not in result,
            ))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

        # Check if all sub-claims have been assigned
        if all_sub_claim_ids <= set(assignments.keys()):
            logger.info("All sub-claims assigned after %d steps", step + 1)
            break
    else:
        logger.warning("ReAct loop hit max steps (%d)", MAX_REACT_STEPS)

    return assignments, total_cost, tool_call_records, reasoning_steps


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _plan_rule_based(state: FactCheckState) -> dict[str, list[str]]:
    """Deterministic keyword-based retrieval planning (no API key needed).

    Applies the same detection logic as analyze_claim_characteristics
    and maps features to methods.
    """
    sub_claims = state.get("sub_claims", [])
    entities = state.get("entities", {})
    plan: dict[str, list[str]] = {}

    for sc in sub_claims:
        characteristics = _tool_analyze_claim(sc.text, entities)
        guideline_info = _tool_check_guidelines(sc.text, entities)

        # Start with defaults
        methods = list(DEFAULT_METHODS)

        # Drug interaction → drugbank
        if characteristics["has_drug_interaction"]:
            methods.append("drugbank_api")

        # Recommendation → guidelines + cochrane
        if characteristics["asks_recommendation"]:
            if "guideline_store" not in methods:
                methods.append("guideline_store")
            if "cochrane_api" not in methods:
                methods.append("cochrane_api")

        # Treatment effectiveness → clinical_trials + cochrane
        if characteristics["asks_effectiveness"]:
            if "clinical_trials" not in methods:
                methods.append("clinical_trials")
            if "cochrane_api" not in methods:
                methods.append("cochrane_api")

        # Quantitative → deep_search
        if characteristics["has_numbers"]:
            if "deep_search" not in methods:
                methods.append("deep_search")

        # Safety → clinical_trials + cochrane
        if characteristics["asks_safety"]:
            if "clinical_trials" not in methods:
                methods.append("clinical_trials")
            if "cochrane_api" not in methods:
                methods.append("cochrane_api")

        # Guideline coverage detected → guideline_store
        if guideline_info["has_guideline_coverage"]:
            if "guideline_store" not in methods:
                methods.append("guideline_store")

        plan[sc.id] = methods

    return plan


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_retrieval_planner(state: FactCheckState) -> FactCheckState:
    """Run the retrieval planner agent.

    Examines each sub-claim's characteristics (entities, PICO, claim type)
    and decides which retrieval methods to invoke. Uses a ReAct loop with
    tool use when an API key is available, otherwise falls back to
    rule-based planning.

    Args:
        state: Pipeline state with sub_claims, entities, and pico populated.

    Returns:
        Updated state with retrieval_plan populated.
    """
    start_time = time.time()
    sub_claims = state.get("sub_claims", [])
    all_sub_claim_ids = {sc.id for sc in sub_claims}
    cost = 0.0
    tool_call_records: list[ToolCall] = []
    reasoning_steps = 0
    tools_called: list[str] = []

    if ANTHROPIC_API_KEY and sub_claims:
        try:
            assignments, cost, tool_call_records, reasoning_steps = (
                _plan_with_react(state)
            )
            tools_called = list({tc.tool for tc in tool_call_records})
        except Exception as e:
            logger.warning(
                "ReAct planning failed, using rule-based fallback: %s", e
            )
            assignments = _plan_rule_based(state)
            tools_called = ["rule_based_fallback"]
    else:
        assignments = _plan_rule_based(state)
        tools_called = ["rule_based_fallback"]

    # Assign defaults to any sub-claims not covered
    for sc_id in all_sub_claim_ids:
        if sc_id not in assignments:
            logger.info("Assigning defaults to unplanned sub-claim: %s", sc_id)
            assignments[sc_id] = list(DEFAULT_METHODS)

    duration = time.time() - start_time

    # Build trace
    trace = AgentTrace(
        agent="retrieval_planner",
        node_type="agent",
        duration_seconds=round(duration, 2),
        cost_usd=round(cost, 6),
        input_summary=(
            f"{len(sub_claims)} sub-claims from: "
            f"{state['claim'][:60]}"
        ),
        output_summary=(
            f"Planned {len(assignments)} sub-claims, "
            f"methods: {_summarize_methods(assignments)}"
        ),
        success=True,
        tools_called=tools_called,
        tool_calls=tool_call_records,
        reasoning_steps=reasoning_steps,
    )

    # Update state
    existing_traces = state.get("agent_trace", [])
    existing_cost = state.get("total_cost_usd", 0.0)
    existing_duration = state.get("total_duration_seconds", 0.0)

    return {
        **state,
        "retrieval_plan": assignments,
        "agent_trace": existing_traces + [trace],
        "total_cost_usd": existing_cost + cost,
        "total_duration_seconds": existing_duration + duration,
    }


def _summarize_methods(assignments: dict[str, list[str]]) -> str:
    """Summarize unique methods across all assignments."""
    all_methods: set[str] = set()
    for methods in assignments.values():
        all_methods.update(methods)
    return ", ".join(sorted(all_methods))
