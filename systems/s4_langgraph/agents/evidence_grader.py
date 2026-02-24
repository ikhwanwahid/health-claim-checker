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

from __future__ import annotations

import json
import logging
import time
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, EVIDENCE_WEIGHTS
from src.models import AgentTrace, FactCheckState, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REACT_STEPS = 15

VALID_STUDY_TYPES = {
    "guideline", "systematic_review", "meta_analysis", "rct", "cohort",
    "case_control", "case_report", "in_vitro", "expert_opinion", "unknown",
}

METHODOLOGY_WEIGHTS = {
    "sample_size_rating": 0.3,
    "blinding": 0.25,
    "randomization": 0.25,
    "follow_up": 0.2,
}

RATING_SCORES = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}

VALID_METHODOLOGY_RATINGS = {"strong", "moderate", "weak", "not_applicable"}
VALID_OVERALL_RATINGS = {"high", "moderate", "low"}
OVERALL_RATING_SCORES = {"high": 1.0, "moderate": 0.6, "low": 0.3}

VALID_DIRECTIONS = {"supports", "opposes", "neutral"}

GRADE_PENALTIES = {
    "no_serious_concern": 0.0,
    "serious": -0.15,
    "very_serious": -0.30,
}

QUALITY_SCORES = {"high": 1.0, "moderate": 0.7, "low": 0.4, "very_low": 0.2}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_methodology_score(
    sample_size_rating: str,
    blinding: str,
    randomization: str,
    follow_up: str,
) -> float:
    """Compute a methodology score (0-1) from dimension ratings.

    Dimensions rated 'not_applicable' are excluded from the weighted average.
    """
    dimensions = {
        "sample_size_rating": sample_size_rating,
        "blinding": blinding,
        "randomization": randomization,
        "follow_up": follow_up,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, rating in dimensions.items():
        if rating == "not_applicable":
            continue
        score = RATING_SCORES.get(rating, 0.3)
        weight = METHODOLOGY_WEIGHTS[dim]
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.6  # default moderate if all not_applicable
    return round(weighted_sum / total_weight, 4)


def compute_evidence_strength(
    hierarchy_weight: float,
    methodology_score: float,
    relevance_score: float,
) -> float:
    """Compute final evidence strength: weighted combination, clamped 0-1."""
    raw = hierarchy_weight * 0.4 + methodology_score * 0.3 + relevance_score * 0.3
    return round(max(0.0, min(1.0, raw)), 4)


def compute_grade_penalty(
    risk_of_bias: str,
    inconsistency: str,
    indirectness: str,
    imprecision: str,
    publication_bias: str,
) -> float:
    """Compute total GRADE penalty (negative number or 0)."""
    total = 0.0
    for criterion in [risk_of_bias, inconsistency, indirectness, imprecision, publication_bias]:
        total += GRADE_PENALTIES.get(criterion, 0.0)
    return round(total, 4)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_classify_study_type(
    evidence_id: str,
    study_type: str,
    confidence: float,
    grading_results: dict[str, dict],
) -> dict[str, Any]:
    """Record study type classification for an evidence item."""
    if study_type not in VALID_STUDY_TYPES:
        return {
            "success": False,
            "error": (
                f"Invalid study_type '{study_type}'. "
                f"Valid types: {sorted(VALID_STUDY_TYPES)}"
            ),
        }

    hierarchy_weight = EVIDENCE_WEIGHTS.get(study_type, 0.1)

    record = grading_results.setdefault(evidence_id, {})
    record["study_type"] = study_type
    record["hierarchy_weight"] = hierarchy_weight
    record["classification_confidence"] = round(max(0.0, min(1.0, confidence)), 4)

    return {
        "success": True,
        "evidence_id": evidence_id,
        "study_type": study_type,
        "hierarchy_weight": hierarchy_weight,
        "confidence": record["classification_confidence"],
    }


def _tool_assess_methodology(
    evidence_id: str,
    sample_size_rating: str,
    blinding: str,
    randomization: str,
    follow_up: str,
    overall_rating: str,
    grading_results: dict[str, dict],
) -> dict[str, Any]:
    """Record methodology assessment for an evidence item."""
    for dim_name, dim_val in [
        ("sample_size_rating", sample_size_rating),
        ("blinding", blinding),
        ("randomization", randomization),
        ("follow_up", follow_up),
    ]:
        if dim_val not in VALID_METHODOLOGY_RATINGS:
            return {
                "success": False,
                "error": (
                    f"Invalid rating '{dim_val}' for {dim_name}. "
                    f"Valid: {sorted(VALID_METHODOLOGY_RATINGS)}"
                ),
            }

    if overall_rating not in VALID_OVERALL_RATINGS:
        return {
            "success": False,
            "error": (
                f"Invalid overall_rating '{overall_rating}'. "
                f"Valid: {sorted(VALID_OVERALL_RATINGS)}"
            ),
        }

    methodology_score = compute_methodology_score(
        sample_size_rating, blinding, randomization, follow_up,
    )

    record = grading_results.setdefault(evidence_id, {})
    record["methodology"] = {
        "sample_size_rating": sample_size_rating,
        "blinding": blinding,
        "randomization": randomization,
        "follow_up": follow_up,
        "overall_rating": overall_rating,
    }
    record["methodology_score"] = methodology_score

    return {
        "success": True,
        "evidence_id": evidence_id,
        "methodology_score": methodology_score,
        "overall_rating": overall_rating,
    }


def _tool_check_relevance(
    evidence_id: str,
    sub_claim_id: str,
    relevance_score: float,
    direction: str,
    key_finding: str,
    grading_results: dict[str, dict],
) -> dict[str, Any]:
    """Record relevance of evidence to a specific sub-claim."""
    if direction not in VALID_DIRECTIONS:
        return {
            "success": False,
            "error": (
                f"Invalid direction '{direction}'. "
                f"Valid: {sorted(VALID_DIRECTIONS)}"
            ),
        }

    clamped_score = round(max(0.0, min(1.0, relevance_score)), 4)

    record = grading_results.setdefault(evidence_id, {})
    relevance = record.setdefault("relevance", {})
    relevance[sub_claim_id] = {
        "score": clamped_score,
        "direction": direction,
        "key_finding": key_finding,
    }

    return {
        "success": True,
        "evidence_id": evidence_id,
        "sub_claim_id": sub_claim_id,
        "relevance_score": clamped_score,
        "direction": direction,
    }


def _tool_apply_grade(
    evidence_id: str,
    risk_of_bias: str,
    inconsistency: str,
    indirectness: str,
    imprecision: str,
    publication_bias: str,
    overall_quality: str,
    grading_results: dict[str, dict],
    graded_evidence: set[str],
) -> dict[str, Any]:
    """Apply GRADE criteria and compute final evidence strength."""
    for crit_name, crit_val in [
        ("risk_of_bias", risk_of_bias),
        ("inconsistency", inconsistency),
        ("indirectness", indirectness),
        ("imprecision", imprecision),
        ("publication_bias", publication_bias),
    ]:
        if crit_val not in GRADE_PENALTIES:
            return {
                "success": False,
                "error": (
                    f"Invalid GRADE value '{crit_val}' for {crit_name}. "
                    f"Valid: {sorted(GRADE_PENALTIES.keys())}"
                ),
            }

    if overall_quality not in QUALITY_SCORES:
        return {
            "success": False,
            "error": (
                f"Invalid overall_quality '{overall_quality}'. "
                f"Valid: {sorted(QUALITY_SCORES.keys())}"
            ),
        }

    record = grading_results.setdefault(evidence_id, {})
    record["grade"] = {
        "risk_of_bias": risk_of_bias,
        "inconsistency": inconsistency,
        "indirectness": indirectness,
        "imprecision": imprecision,
        "publication_bias": publication_bias,
        "overall_quality": overall_quality,
    }

    # Compute final evidence strength
    hierarchy_weight = record.get("hierarchy_weight", 0.1)
    methodology_score = record.get("methodology_score", 0.6)

    # Average relevance across all sub-claims
    relevance_data = record.get("relevance", {})
    if relevance_data:
        avg_relevance = sum(r["score"] for r in relevance_data.values()) / len(relevance_data)
    else:
        avg_relevance = 0.5

    evidence_strength = compute_evidence_strength(
        hierarchy_weight, methodology_score, avg_relevance,
    )
    record["evidence_strength"] = evidence_strength

    graded_evidence.add(evidence_id)

    # Check for missing prior assessments
    warnings = []
    if "study_type" not in record:
        warnings.append("classify_study_type was not called before apply_grade")
    if "methodology" not in record:
        warnings.append("assess_methodology was not called before apply_grade")
    if not relevance_data:
        warnings.append("check_relevance was not called before apply_grade")

    result: dict[str, Any] = {
        "success": True,
        "evidence_id": evidence_id,
        "evidence_strength": evidence_strength,
        "hierarchy_weight": hierarchy_weight,
        "methodology_score": methodology_score,
        "avg_relevance": round(avg_relevance, 4),
        "overall_quality": overall_quality,
    }
    if warnings:
        result["warnings"] = warnings

    return result


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    grading_results: dict[str, dict],
    graded_evidence: set[str],
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate function."""
    if tool_name == "classify_study_type":
        return _tool_classify_study_type(
            evidence_id=tool_input["evidence_id"],
            study_type=tool_input["study_type"],
            confidence=tool_input.get("confidence", 0.8),
            grading_results=grading_results,
        )
    elif tool_name == "assess_methodology":
        return _tool_assess_methodology(
            evidence_id=tool_input["evidence_id"],
            sample_size_rating=tool_input["sample_size_rating"],
            blinding=tool_input["blinding"],
            randomization=tool_input["randomization"],
            follow_up=tool_input["follow_up"],
            overall_rating=tool_input["overall_rating"],
            grading_results=grading_results,
        )
    elif tool_name == "check_relevance":
        return _tool_check_relevance(
            evidence_id=tool_input["evidence_id"],
            sub_claim_id=tool_input["sub_claim_id"],
            relevance_score=tool_input["relevance_score"],
            direction=tool_input["direction"],
            key_finding=tool_input.get("key_finding", ""),
            grading_results=grading_results,
        )
    elif tool_name == "apply_grade":
        return _tool_apply_grade(
            evidence_id=tool_input["evidence_id"],
            risk_of_bias=tool_input["risk_of_bias"],
            inconsistency=tool_input["inconsistency"],
            indirectness=tool_input["indirectness"],
            imprecision=tool_input["imprecision"],
            publication_bias=tool_input["publication_bias"],
            overall_quality=tool_input["overall_quality"],
            grading_results=grading_results,
            graded_evidence=graded_evidence,
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Anthropic tool schema definitions
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "name": "classify_study_type",
        "description": (
            "Classify the study type of an evidence item based on its content. "
            "Valid types: guideline, systematic_review, meta_analysis, rct, cohort, "
            "case_control, case_report, in_vitro, expert_opinion, unknown. "
            "This determines the evidence hierarchy weight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_id": {
                    "type": "string",
                    "description": "The ID of the evidence item.",
                },
                "study_type": {
                    "type": "string",
                    "description": (
                        "The study type classification. One of: guideline, "
                        "systematic_review, meta_analysis, rct, cohort, "
                        "case_control, case_report, in_vitro, expert_opinion, unknown."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the classification (0.0-1.0).",
                },
            },
            "required": ["evidence_id", "study_type", "confidence"],
        },
    },
    {
        "name": "assess_methodology",
        "description": (
            "Assess the methodological quality of an evidence item. "
            "Rate each dimension as 'strong', 'moderate', 'weak', or "
            "'not_applicable'. Provide an overall rating of 'high', "
            "'moderate', or 'low'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_id": {
                    "type": "string",
                    "description": "The ID of the evidence item.",
                },
                "sample_size_rating": {
                    "type": "string",
                    "description": "Sample size adequacy: strong, moderate, weak, or not_applicable.",
                },
                "blinding": {
                    "type": "string",
                    "description": "Blinding quality: strong, moderate, weak, or not_applicable.",
                },
                "randomization": {
                    "type": "string",
                    "description": "Randomization quality: strong, moderate, weak, or not_applicable.",
                },
                "follow_up": {
                    "type": "string",
                    "description": "Follow-up adequacy: strong, moderate, weak, or not_applicable.",
                },
                "overall_rating": {
                    "type": "string",
                    "description": "Overall methodology rating: high, moderate, or low.",
                },
            },
            "required": [
                "evidence_id", "sample_size_rating", "blinding",
                "randomization", "follow_up", "overall_rating",
            ],
        },
    },
    {
        "name": "check_relevance",
        "description": (
            "Score how directly a piece of evidence addresses a specific sub-claim. "
            "Record the relevance score (0.0-1.0), direction (supports/opposes/neutral), "
            "and a brief key finding. Call this for EACH sub-claim that the evidence "
            "is relevant to."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_id": {
                    "type": "string",
                    "description": "The ID of the evidence item.",
                },
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim this evidence relates to.",
                },
                "relevance_score": {
                    "type": "number",
                    "description": "How directly the evidence addresses the sub-claim (0.0-1.0).",
                },
                "direction": {
                    "type": "string",
                    "description": "Whether the evidence supports, opposes, or is neutral toward the sub-claim.",
                },
                "key_finding": {
                    "type": "string",
                    "description": "Brief summary of what the evidence says about the sub-claim.",
                },
            },
            "required": ["evidence_id", "sub_claim_id", "relevance_score", "direction", "key_finding"],
        },
    },
    {
        "name": "apply_grade",
        "description": (
            "Apply the GRADE framework to an evidence item and compute its final "
            "evidence strength score. Each criterion is rated as 'no_serious_concern', "
            "'serious', or 'very_serious'. Overall quality is 'high', 'moderate', "
            "'low', or 'very_low'. Call this LAST for each evidence item, after "
            "classify_study_type, assess_methodology, and check_relevance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_id": {
                    "type": "string",
                    "description": "The ID of the evidence item.",
                },
                "risk_of_bias": {
                    "type": "string",
                    "description": "Risk of bias: no_serious_concern, serious, or very_serious.",
                },
                "inconsistency": {
                    "type": "string",
                    "description": "Inconsistency: no_serious_concern, serious, or very_serious.",
                },
                "indirectness": {
                    "type": "string",
                    "description": "Indirectness: no_serious_concern, serious, or very_serious.",
                },
                "imprecision": {
                    "type": "string",
                    "description": "Imprecision: no_serious_concern, serious, or very_serious.",
                },
                "publication_bias": {
                    "type": "string",
                    "description": "Publication bias: no_serious_concern, serious, or very_serious.",
                },
                "overall_quality": {
                    "type": "string",
                    "description": "Overall evidence quality: high, moderate, low, or very_low.",
                },
            },
            "required": [
                "evidence_id", "risk_of_bias", "inconsistency",
                "indirectness", "imprecision", "publication_bias",
                "overall_quality",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt + user message builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an evidence quality grader for a medical fact-checking system. Your job \
is to evaluate each piece of retrieved evidence for study type, methodological \
quality, and relevance to the sub-claims being verified.

## Available Tools

| Tool | When to use |
|------|-------------|
| classify_study_type | FIRST — classify what type of study the evidence is |
| assess_methodology | SECOND — assess methodological quality (sample size, blinding, etc.) |
| check_relevance | THIRD — score relevance to each linked sub-claim |
| apply_grade | LAST — apply GRADE framework and compute final strength |

## Evidence Hierarchy (highest to lowest)

1. Clinical guidelines (WHO, NIH, MOH) — weight 1.0
2. Systematic reviews / meta-analyses — weight 0.9
3. Randomized controlled trials (RCTs) — weight 0.8
4. Cohort studies — weight 0.6
5. Case-control studies — weight 0.5
6. Case reports — weight 0.3
7. In vitro studies — weight 0.2
8. Expert opinion / editorials — weight 0.1

## Process

For EACH evidence item:
1. Read its title, content/abstract, and source
2. Call classify_study_type to record the study type
3. Call assess_methodology to evaluate methodological quality
4. Call check_relevance for EACH sub-claim the evidence is linked to — \
the 'direction' field is CRITICAL: record whether the evidence 'supports', \
'opposes', or is 'neutral' toward each sub-claim
5. Call apply_grade to apply GRADE criteria and compute final strength

## Important

- The 'direction' field in check_relevance is critical for the verdict agent. \
Be precise: 'supports' means the evidence agrees with the sub-claim, 'opposes' \
means it contradicts it, 'neutral' means it's relevant but doesn't clearly \
support or oppose.
- Work through ALL evidence items, then stop.
- Base your assessment ONLY on the information provided in the evidence content."""


def _build_user_message(state: FactCheckState) -> str:
    """Build the user message with evidence items and sub-claims."""
    evidence = state.get("evidence", [])
    sub_claims = state.get("sub_claims", [])

    parts: list[str] = []
    parts.append(f"Original claim: \"{state['claim']}\"")
    parts.append("")

    # Sub-claims
    parts.append(f"Sub-claims ({len(sub_claims)}):")
    for sc in sub_claims:
        ev_ids = ", ".join(sc.evidence) if sc.evidence else "none"
        parts.append(f"  {sc.id}: \"{sc.text}\"")
        parts.append(f"    Linked evidence: {ev_ids}")
    parts.append("")

    # Evidence items
    parts.append(f"Evidence items to grade ({len(evidence)}):")
    for ev in evidence:
        parts.append(f"  [{ev.id}] ({ev.source}, study_type={ev.study_type or 'unknown'})")
        parts.append(f"    Title: {ev.title}")
        content_preview = ev.content[:500] if ev.content else "(no content)"
        parts.append(f"    Content: {content_preview}")
        if ev.quality_score > 0:
            parts.append(f"    Cross-encoder score: {ev.quality_score:.4f}")
        parts.append("")

    parts.append(
        "Please grade each evidence item. For each item: "
        "classify study type → assess methodology → check relevance "
        "to linked sub-claims → apply GRADE. Work through all items."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def _grade_with_react(
    state: FactCheckState,
) -> tuple[dict[str, dict], set[str], float, list[ToolCall], int]:
    """Run the ReAct loop to grade evidence.

    Returns:
        (grading_results, graded_evidence, cost_usd, tool_calls, reasoning_steps)
    """
    evidence = state.get("evidence", [])

    # Shared mutable state
    grading_results: dict[str, dict] = {}
    graded_evidence: set[str] = set()
    tool_call_records: list[ToolCall] = []
    total_cost = 0.0
    reasoning_steps = 0

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_message(state)},
    ]

    all_evidence_ids = {ev.id for ev in evidence}

    for step in range(MAX_REACT_STEPS):
        reasoning_steps += 1

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
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
            break

        # Build assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool
        tool_results: list[dict[str, Any]] = []
        for tool_block in tool_use_blocks:
            tc_start = time.time()
            result = _execute_tool(
                tool_name=tool_block.name,
                tool_input=tool_block.input,
                grading_results=grading_results,
                graded_evidence=graded_evidence,
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

        # Check if all evidence has been graded
        if all_evidence_ids <= graded_evidence:
            logger.info("All evidence graded after %d steps", step + 1)
            break
    else:
        logger.warning("ReAct loop hit max steps (%d)", MAX_REACT_STEPS)

    return grading_results, graded_evidence, total_cost, tool_call_records, reasoning_steps


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _grade_rule_based(state: FactCheckState) -> dict[str, dict]:
    """Deterministic evidence grading (no API key needed).

    Uses existing metadata on Evidence objects (study_type, quality_score)
    to produce baseline grades.
    """
    evidence = state.get("evidence", [])
    sub_claims = state.get("sub_claims", [])

    grading_results: dict[str, dict] = {}

    for ev in evidence:
        # Study type: use existing or default to expert_opinion
        study_type = ev.study_type if ev.study_type and ev.study_type != "unknown" else "expert_opinion"
        hierarchy_weight = EVIDENCE_WEIGHTS.get(study_type, 0.1)

        # Methodology: default moderate
        methodology_score = 0.6

        # Relevance: use cross-encoder quality_score if available, else 0.5
        base_relevance = ev.quality_score if ev.quality_score > 0 else 0.5

        # Build relevance per linked sub-claim
        relevance: dict[str, dict] = {}
        for sc in sub_claims:
            if ev.id in sc.evidence:
                relevance[sc.id] = {
                    "score": base_relevance,
                    "direction": "neutral",
                    "key_finding": "",
                }

        # If no sub-claims linked, add as generic relevance
        if not relevance:
            for sc in sub_claims:
                relevance[sc.id] = {
                    "score": base_relevance,
                    "direction": "neutral",
                    "key_finding": "",
                }

        # GRADE: all no_serious_concern as conservative default
        grade = {
            "risk_of_bias": "no_serious_concern",
            "inconsistency": "no_serious_concern",
            "indirectness": "no_serious_concern",
            "imprecision": "no_serious_concern",
            "publication_bias": "no_serious_concern",
            "overall_quality": "moderate",
        }

        # Final strength
        evidence_strength = compute_evidence_strength(
            hierarchy_weight, methodology_score, base_relevance,
        )

        grading_results[ev.id] = {
            "study_type": study_type,
            "hierarchy_weight": hierarchy_weight,
            "methodology_score": methodology_score,
            "relevance": relevance,
            "grade": grade,
            "evidence_strength": evidence_strength,
        }

    return grading_results


# ---------------------------------------------------------------------------
# Output structure builder
# ---------------------------------------------------------------------------


def _build_evidence_quality(
    grading_results: dict[str, dict],
    sub_claims: list,
) -> dict:
    """Build the evidence_quality output structure.

    Returns:
        Dict with 'per_evidence' and 'per_subclaim' sections.
    """
    per_evidence: dict[str, dict] = {}
    for ev_id, record in grading_results.items():
        per_evidence[ev_id] = {
            "study_type": record.get("study_type", "unknown"),
            "hierarchy_weight": record.get("hierarchy_weight", 0.1),
            "methodology_score": record.get("methodology_score", 0.6),
            "relevance": record.get("relevance", {}),
            "grade": record.get("grade", {}),
            "evidence_strength": record.get("evidence_strength", 0.0),
        }

    # Per sub-claim aggregation
    per_subclaim: dict[str, dict] = {}
    for sc in sub_claims:
        sc_evidence_ids: list[str] = []
        strengths: list[float] = []
        direction_counts = {"supports": 0, "opposes": 0, "neutral": 0}

        for ev_id, record in grading_results.items():
            relevance = record.get("relevance", {})
            if sc.id in relevance:
                sc_evidence_ids.append(ev_id)
                strengths.append(record.get("evidence_strength", 0.0))
                direction = relevance[sc.id].get("direction", "neutral")
                if direction in direction_counts:
                    direction_counts[direction] += 1

        avg_strength = round(sum(strengths) / len(strengths), 4) if strengths else 0.0

        # Top evidence by strength
        ev_with_strength = [
            (eid, grading_results[eid].get("evidence_strength", 0.0))
            for eid in sc_evidence_ids
        ]
        ev_with_strength.sort(key=lambda x: x[1], reverse=True)
        top_ids = [eid for eid, _ in ev_with_strength[:5]]

        per_subclaim[sc.id] = {
            "evidence_count": len(sc_evidence_ids),
            "avg_strength": avg_strength,
            "direction_summary": direction_counts,
            "top_evidence_ids": top_ids,
        }

    return {
        "per_evidence": per_evidence,
        "per_subclaim": per_subclaim,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    start_time = time.time()
    evidence = state.get("evidence", [])
    sub_claims = state.get("sub_claims", [])
    cost = 0.0
    tool_call_records: list[ToolCall] = []
    reasoning_steps = 0
    tools_called: list[str] = []

    if ANTHROPIC_API_KEY and evidence:
        try:
            grading_results, graded_ev, cost, tool_call_records, reasoning_steps = (
                _grade_with_react(state)
            )
            tools_called = list({tc.tool for tc in tool_call_records})
        except Exception as e:
            logger.warning(
                "ReAct grading failed, using rule-based fallback: %s", e
            )
            grading_results = _grade_rule_based(state)
            tools_called = ["rule_based_fallback"]
    else:
        grading_results = _grade_rule_based(state)
        tools_called = ["rule_based_fallback"]

    duration = time.time() - start_time

    # Build output structure
    evidence_quality = _build_evidence_quality(grading_results, sub_claims)

    # Build trace
    trace = AgentTrace(
        agent="evidence_grader",
        node_type="agent",
        duration_seconds=round(duration, 2),
        cost_usd=round(cost, 6),
        input_summary=(
            f"{len(evidence)} evidence items, "
            f"{len(sub_claims)} sub-claims"
        ),
        output_summary=(
            f"Graded {len(grading_results)} evidence items, "
            f"{len(evidence_quality.get('per_subclaim', {}))} sub-claim summaries"
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
        "evidence_quality": evidence_quality,
        "agent_trace": existing_traces + [trace],
        "total_cost_usd": existing_cost + cost,
        "total_duration_seconds": existing_duration + duration,
    }
