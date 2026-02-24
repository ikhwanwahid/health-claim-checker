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
- assign_subclaim_verdict: record verdict for a specific sub-claim
- synthesize_overall: produce final verdict after all sub-claims assessed

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
- extracted_figures: VLM-extracted data from figures

Output (to state):
- verdict: one of 9 verdict levels
- confidence: 0.0-1.0 confidence score
- explanation: human-readable explanation with citations
- sub_claims: updated with per-sub-claim verdicts
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, VERDICTS
from src.models import AgentTrace, FactCheckState, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REACT_STEPS = 12

CRITICAL_VERDICTS = {"REFUTED", "NOT_SUPPORTED", "DANGEROUS"}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_weigh_evidence(
    sub_claim_id: str,
    supporting_ids: list[str],
    opposing_ids: list[str],
    neutral_ids: list[str],
    evidence_quality: dict,
) -> dict[str, Any]:
    """Compute weighted support/opposition score using evidence strength."""
    per_evidence = evidence_quality.get("per_evidence", {})

    def _sum_strength(ids: list[str]) -> float:
        total = 0.0
        for eid in ids:
            ev_info = per_evidence.get(eid, {})
            total += ev_info.get("evidence_strength", 0.0)
        return round(total, 4)

    support_score = _sum_strength(supporting_ids)
    oppose_score = _sum_strength(opposing_ids)
    neutral_score = _sum_strength(neutral_ids)

    if support_score > oppose_score * 1.2:
        balance = "supports"
    elif oppose_score > support_score * 1.2:
        balance = "opposes"
    else:
        balance = "balanced"

    return {
        "success": True,
        "sub_claim_id": sub_claim_id,
        "support_score": support_score,
        "oppose_score": oppose_score,
        "neutral_score": neutral_score,
        "balance": balance,
        "evidence_summary": (
            f"{len(supporting_ids)} supporting (strength={support_score:.2f}), "
            f"{len(opposing_ids)} opposing (strength={oppose_score:.2f}), "
            f"{len(neutral_ids)} neutral (strength={neutral_score:.2f})"
        ),
    }


def _tool_reconcile_conflicts(
    sub_claim_id: str,
    conflicts_description: str,
    resolution: str,
    resolved_direction: str,
    conflict_records: list[dict],
) -> dict[str, Any]:
    """Record reasoning about why conflicting evidence favors one direction."""
    if resolved_direction not in {"supports", "opposes", "neutral", "inconclusive"}:
        return {
            "success": False,
            "error": (
                f"Invalid resolved_direction '{resolved_direction}'. "
                f"Valid: supports, opposes, neutral, inconclusive"
            ),
        }

    conflict_records.append({
        "sub_claim_id": sub_claim_id,
        "conflicts_description": conflicts_description,
        "resolution": resolution,
        "resolved_direction": resolved_direction,
    })

    return {
        "success": True,
        "sub_claim_id": sub_claim_id,
        "resolved_direction": resolved_direction,
    }


def _tool_assign_subclaim_verdict(
    sub_claim_id: str,
    verdict: str,
    confidence: float,
    reasoning: str,
    subclaim_verdicts: dict[str, dict],
    all_subclaim_ids: set[str],
) -> dict[str, Any]:
    """Record verdict for a specific sub-claim."""
    if verdict not in VERDICTS:
        return {
            "success": False,
            "error": (
                f"Invalid verdict '{verdict}'. "
                f"Valid: {VERDICTS}"
            ),
        }

    clamped_confidence = round(max(0.0, min(1.0, confidence)), 4)

    subclaim_verdicts[sub_claim_id] = {
        "verdict": verdict,
        "confidence": clamped_confidence,
        "reasoning": reasoning,
    }

    remaining = all_subclaim_ids - set(subclaim_verdicts.keys())

    return {
        "success": True,
        "sub_claim_id": sub_claim_id,
        "verdict": verdict,
        "confidence": clamped_confidence,
        "remaining_subclaims": sorted(remaining),
    }


def _tool_synthesize_overall(
    overall_verdict: str,
    confidence: float,
    explanation: str,
    overall_result: dict,
) -> dict[str, Any]:
    """Record the final overall verdict."""
    if overall_verdict not in VERDICTS:
        return {
            "success": False,
            "error": (
                f"Invalid verdict '{overall_verdict}'. "
                f"Valid: {VERDICTS}"
            ),
        }

    clamped_confidence = round(max(0.0, min(1.0, confidence)), 4)

    overall_result["verdict"] = overall_verdict
    overall_result["confidence"] = clamped_confidence
    overall_result["explanation"] = explanation
    overall_result["complete"] = True

    return {
        "success": True,
        "verdict": overall_verdict,
        "confidence": clamped_confidence,
    }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    evidence_quality: dict,
    subclaim_verdicts: dict[str, dict],
    all_subclaim_ids: set[str],
    overall_result: dict,
    conflict_records: list[dict],
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate function."""
    if tool_name == "weigh_evidence":
        return _tool_weigh_evidence(
            sub_claim_id=tool_input["sub_claim_id"],
            supporting_ids=tool_input.get("supporting_ids", []),
            opposing_ids=tool_input.get("opposing_ids", []),
            neutral_ids=tool_input.get("neutral_ids", []),
            evidence_quality=evidence_quality,
        )
    elif tool_name == "reconcile_conflicts":
        return _tool_reconcile_conflicts(
            sub_claim_id=tool_input["sub_claim_id"],
            conflicts_description=tool_input.get("conflicts_description", ""),
            resolution=tool_input.get("resolution", ""),
            resolved_direction=tool_input.get("resolved_direction", "inconclusive"),
            conflict_records=conflict_records,
        )
    elif tool_name == "assign_subclaim_verdict":
        return _tool_assign_subclaim_verdict(
            sub_claim_id=tool_input["sub_claim_id"],
            verdict=tool_input["verdict"],
            confidence=tool_input.get("confidence", 0.5),
            reasoning=tool_input.get("reasoning", ""),
            subclaim_verdicts=subclaim_verdicts,
            all_subclaim_ids=all_subclaim_ids,
        )
    elif tool_name == "synthesize_overall":
        return _tool_synthesize_overall(
            overall_verdict=tool_input["overall_verdict"],
            confidence=tool_input.get("confidence", 0.5),
            explanation=tool_input.get("explanation", ""),
            overall_result=overall_result,
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Anthropic tool schema definitions
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "name": "weigh_evidence",
        "description": (
            "Aggregate quality-weighted evidence for/against a sub-claim. "
            "Provide the IDs of supporting, opposing, and neutral evidence items. "
            "Returns weighted scores and a balance assessment. Call this FIRST "
            "for each sub-claim to understand the evidence landscape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim to weigh evidence for.",
                },
                "supporting_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence IDs that support the sub-claim.",
                },
                "opposing_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence IDs that oppose the sub-claim.",
                },
                "neutral_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence IDs that are neutral toward the sub-claim.",
                },
            },
            "required": ["sub_claim_id", "supporting_ids", "opposing_ids", "neutral_ids"],
        },
    },
    {
        "name": "reconcile_conflicts",
        "description": (
            "Record your reasoning about conflicting evidence for a sub-claim. "
            "Use this when supporting and opposing evidence both have significant "
            "strength and you need to explain why one direction is favored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim with conflicting evidence.",
                },
                "conflicts_description": {
                    "type": "string",
                    "description": "Description of the conflicting evidence.",
                },
                "resolution": {
                    "type": "string",
                    "description": "Your reasoning for how the conflict is resolved.",
                },
                "resolved_direction": {
                    "type": "string",
                    "description": "The resolved direction: supports, opposes, neutral, or inconclusive.",
                },
            },
            "required": ["sub_claim_id", "conflicts_description", "resolution", "resolved_direction"],
        },
    },
    {
        "name": "assign_subclaim_verdict",
        "description": (
            "Assign a verdict to a specific sub-claim after weighing evidence "
            "and reconciling any conflicts. Valid verdicts: SUPPORTED, "
            "SUPPORTED_WITH_CAVEATS, OVERSTATED, MISLEADING, PRELIMINARY, "
            "OUTDATED, NOT_SUPPORTED, REFUTED, DANGEROUS."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim to assign a verdict to.",
                },
                "verdict": {
                    "type": "string",
                    "description": "The verdict for this sub-claim.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the verdict (0.0-1.0).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for the verdict.",
                },
            },
            "required": ["sub_claim_id", "verdict", "confidence", "reasoning"],
        },
    },
    {
        "name": "synthesize_overall",
        "description": (
            "Produce the final overall verdict after ALL sub-claims have been "
            "assessed. Synthesize sub-claim verdicts into a single overall "
            "verdict weighted by evidence strength and confidence. Call this "
            "LAST, after all sub-claims have been assigned verdicts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "overall_verdict": {
                    "type": "string",
                    "description": "The overall verdict for the claim.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Overall confidence (0.0-1.0).",
                },
                "explanation": {
                    "type": "string",
                    "description": (
                        "Human-readable explanation of the verdict with "
                        "citations to key evidence."
                    ),
                },
            },
            "required": ["overall_verdict", "confidence", "explanation"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt + user message builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a verdict synthesis agent for a medical fact-checking system. Your job \
is to weigh all evidence for each sub-claim, reconcile conflicts, assign \
per-sub-claim verdicts, and synthesize an overall verdict.

## Process

For EACH sub-claim:
1. Review the evidence quality scores and directions from the grader
2. Call weigh_evidence to aggregate supporting vs opposing evidence
3. If there are strong conflicts, call reconcile_conflicts to explain your reasoning
4. Call assign_subclaim_verdict with a verdict and confidence

After ALL sub-claims are assessed:
5. Call synthesize_overall with the final verdict, confidence, and explanation

## Verdict Taxonomy

| Verdict | When to use |
|---------|-------------|
| SUPPORTED | Strong, consistent evidence confirms the claim |
| SUPPORTED_WITH_CAVEATS | True but needs important context or qualifications |
| OVERSTATED | Kernel of truth but exaggerated in scope or certainty |
| MISLEADING | Technically true but creates a wrong impression |
| PRELIMINARY | Some evidence but too early or too weak to confirm |
| OUTDATED | Was true but evidence has since changed |
| NOT_SUPPORTED | No credible evidence found to support the claim |
| REFUTED | Directly contradicted by strong evidence |
| DANGEROUS | Could cause harm if believed and acted upon |

## Decision Guidelines

- When support >> opposition AND strength > 0.7: SUPPORTED or SUPPORTED_WITH_CAVEATS
- When opposition >> support AND strength > 0.6: REFUTED or NOT_SUPPORTED
- When mixed strong evidence on both sides: PRELIMINARY or MISLEADING
- When weak evidence in either direction: NOT_SUPPORTED or PRELIMINARY
- Overall verdict is NOT a simple majority vote — weight by evidence strength
- Use evidence direction and strength from the grader output; don't re-assess quality

## Important

- Work through ALL sub-claims before synthesizing the overall verdict
- Be precise about the distinction between REFUTED (strong contradicting evidence) \
and NOT_SUPPORTED (absence of evidence)
- SUPPORTED_WITH_CAVEATS is appropriate when claims are broadly true but miss \
important nuance"""


def _build_user_message(state: FactCheckState) -> str:
    """Build the user message with evidence quality and sub-claims."""
    sub_claims = state.get("sub_claims", [])
    evidence = state.get("evidence", [])
    evidence_quality = state.get("evidence_quality", {})
    per_evidence = evidence_quality.get("per_evidence", {})
    per_subclaim = evidence_quality.get("per_subclaim", {})

    parts: list[str] = []
    parts.append(f'Original claim: "{state["claim"]}"')
    parts.append("")

    # Sub-claims with evidence summaries
    parts.append(f"Sub-claims ({len(sub_claims)}):")
    for sc in sub_claims:
        parts.append(f"  [{sc.id}] \"{sc.text}\"")

        # Evidence linked to this sub-claim
        sc_info = per_subclaim.get(sc.id, {})
        ev_count = sc_info.get("evidence_count", 0)
        avg_str = sc_info.get("avg_strength", 0.0)
        direction = sc_info.get("direction_summary", {})
        parts.append(
            f"    Evidence: {ev_count} items, avg_strength={avg_str:.3f}"
        )
        parts.append(
            f"    Directions: supports={direction.get('supports', 0)}, "
            f"opposes={direction.get('opposes', 0)}, "
            f"neutral={direction.get('neutral', 0)}"
        )

        # Top evidence details
        top_ids = sc_info.get("top_evidence_ids", [])
        for eid in top_ids[:3]:
            ev_info = per_evidence.get(eid, {})
            ev_obj = next((e for e in evidence if e.id == eid), None)
            title = ev_obj.title[:60] if ev_obj else eid
            strength = ev_info.get("evidence_strength", 0.0)
            study_type = ev_info.get("study_type", "unknown")

            # Get direction for this sub-claim
            rel = ev_info.get("relevance", {}).get(sc.id, {})
            ev_direction = rel.get("direction", "unknown")
            key_finding = rel.get("key_finding", "")

            parts.append(
                f"    [{eid}] {study_type} | strength={strength:.3f} | "
                f"direction={ev_direction}"
            )
            parts.append(f"      Title: {title}")
            if key_finding:
                parts.append(f"      Finding: {key_finding[:100]}")
        parts.append("")

    parts.append(
        "Please weigh evidence for each sub-claim, reconcile any conflicts, "
        "assign per-sub-claim verdicts, and then synthesize an overall verdict."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def _verdict_with_react(
    state: FactCheckState,
) -> tuple[dict[str, dict], dict, float, list[ToolCall], int]:
    """Run the ReAct loop to produce verdicts.

    Returns:
        (subclaim_verdicts, overall_result, cost_usd, tool_calls, reasoning_steps)
    """
    sub_claims = state.get("sub_claims", [])
    evidence_quality = state.get("evidence_quality", {})

    # Shared mutable state
    subclaim_verdicts: dict[str, dict] = {}
    overall_result: dict = {}
    conflict_records: list[dict] = []
    tool_call_records: list[ToolCall] = []
    total_cost = 0.0
    reasoning_steps = 0

    all_subclaim_ids = {sc.id for sc in sub_claims}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_message(state)},
    ]

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
                evidence_quality=evidence_quality,
                subclaim_verdicts=subclaim_verdicts,
                all_subclaim_ids=all_subclaim_ids,
                overall_result=overall_result,
                conflict_records=conflict_records,
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

        # Check if overall verdict has been synthesized
        if overall_result.get("complete"):
            logger.info("Verdict synthesized after %d steps", step + 1)
            break
    else:
        logger.warning("ReAct loop hit max steps (%d)", MAX_REACT_STEPS)

    return subclaim_verdicts, overall_result, total_cost, tool_call_records, reasoning_steps


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _verdict_rule_based(state: FactCheckState) -> tuple[dict[str, dict], dict]:
    """Deterministic verdict assignment (no API key needed).

    Returns:
        (subclaim_verdicts, overall_result)
    """
    sub_claims = state.get("sub_claims", [])
    evidence_quality = state.get("evidence_quality", {})
    evidence = state.get("evidence", [])
    per_subclaim = evidence_quality.get("per_subclaim", {})

    subclaim_verdicts: dict[str, dict] = {}

    for sc in sub_claims:
        sc_info = per_subclaim.get(sc.id, {})
        direction = sc_info.get("direction_summary", {})
        avg_strength = sc_info.get("avg_strength", 0.0)
        supports = direction.get("supports", 0)
        opposes = direction.get("opposes", 0)
        neutral = direction.get("neutral", 0)
        total = supports + opposes + neutral

        if total == 0:
            verdict = "NOT_SUPPORTED"
            confidence = 0.2
        elif opposes > supports and avg_strength > 0.6:
            verdict = "REFUTED"
            confidence = avg_strength * (opposes / max(total, 1))
        elif opposes > supports and avg_strength <= 0.6:
            verdict = "NOT_SUPPORTED"
            confidence = avg_strength * (opposes / max(total, 1))
        elif supports > opposes and avg_strength > 0.7:
            verdict = "SUPPORTED"
            confidence = avg_strength * (supports / max(total, 1))
        elif supports > opposes and avg_strength > 0.4:
            verdict = "SUPPORTED_WITH_CAVEATS"
            confidence = avg_strength * (supports / max(total, 1))
        elif supports > opposes and avg_strength <= 0.4:
            verdict = "PRELIMINARY"
            confidence = avg_strength * (supports / max(total, 1))
        elif supports == opposes:
            verdict = "PRELIMINARY"
            confidence = avg_strength * 0.5
        else:
            verdict = "NOT_SUPPORTED"
            confidence = 0.3

        confidence = round(max(0.0, min(1.0, confidence)), 4)

        subclaim_verdicts[sc.id] = {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": (
                f"Based on {total} evidence items: "
                f"{supports} supporting, {opposes} opposing, {neutral} neutral. "
                f"Average strength: {avg_strength:.3f}."
            ),
        }

    # Overall verdict: weighted mode of sub-claim verdicts
    if not subclaim_verdicts:
        overall_result = {
            "verdict": "NOT_SUPPORTED",
            "confidence": 0.0,
            "explanation": "No sub-claims to assess.",
            "complete": True,
        }
    else:
        # Weight each verdict by its confidence
        verdict_weights: dict[str, float] = {}
        for sc_id, info in subclaim_verdicts.items():
            v = info["verdict"]
            verdict_weights[v] = verdict_weights.get(v, 0.0) + info["confidence"]

        overall_verdict = max(verdict_weights, key=verdict_weights.get)
        avg_confidence = round(
            sum(info["confidence"] for info in subclaim_verdicts.values())
            / len(subclaim_verdicts),
            4,
        )

        n_evidence = len(evidence)
        n_subclaims = len(subclaim_verdicts)
        verdict_summary = Counter(
            info["verdict"] for info in subclaim_verdicts.values()
        )
        summary_parts = [f"{v}: {c}" for v, c in verdict_summary.most_common()]

        overall_result = {
            "verdict": overall_verdict,
            "confidence": avg_confidence,
            "explanation": (
                f"Based on {n_evidence} evidence items across {n_subclaims} "
                f"sub-claims: {', '.join(summary_parts)}."
            ),
            "complete": True,
        }

    return subclaim_verdicts, overall_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    start_time = time.time()
    sub_claims = state.get("sub_claims", [])
    evidence = state.get("evidence", [])
    cost = 0.0
    tool_call_records: list[ToolCall] = []
    reasoning_steps = 0
    tools_called: list[str] = []

    if ANTHROPIC_API_KEY and evidence:
        try:
            subclaim_verdicts, overall_result, cost, tool_call_records, reasoning_steps = (
                _verdict_with_react(state)
            )
            tools_called = list({tc.tool for tc in tool_call_records})
        except Exception as e:
            logger.warning(
                "ReAct verdict failed, using rule-based fallback: %s", e
            )
            subclaim_verdicts, overall_result = _verdict_rule_based(state)
            tools_called = ["rule_based_fallback"]
    else:
        subclaim_verdicts, overall_result = _verdict_rule_based(state)
        tools_called = ["rule_based_fallback"]

    duration = time.time() - start_time

    # Update sub-claims with verdicts
    updated_sub_claims = []
    for sc in sub_claims:
        sc_info = subclaim_verdicts.get(sc.id, {})
        updated_sc = sc.model_copy(update={
            "verdict": sc_info.get("verdict", sc.verdict),
            "confidence": sc_info.get("confidence", sc.confidence),
        })
        updated_sub_claims.append(updated_sc)

    # Build trace
    trace = AgentTrace(
        agent="verdict_agent",
        node_type="agent",
        duration_seconds=round(duration, 2),
        cost_usd=round(cost, 6),
        input_summary=(
            f"{len(evidence)} evidence items, "
            f"{len(sub_claims)} sub-claims"
        ),
        output_summary=(
            f"Verdict: {overall_result.get('verdict', 'unknown')}, "
            f"confidence={overall_result.get('confidence', 0.0):.2f}"
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
        "verdict": overall_result.get("verdict", "NOT_SUPPORTED"),
        "confidence": overall_result.get("confidence", 0.0),
        "explanation": overall_result.get("explanation", ""),
        "sub_claims": updated_sub_claims,
        "agent_trace": existing_traces + [trace],
        "total_cost_usd": existing_cost + cost,
        "total_duration_seconds": existing_duration + duration,
    }
