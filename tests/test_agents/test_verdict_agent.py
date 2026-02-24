"""Tests for the Verdict Agent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from systems.s4_langgraph.agents.verdict_agent import (
    MAX_REACT_STEPS,
    _build_user_message,
    _execute_tool,
    _tool_assign_subclaim_verdict,
    _tool_reconcile_conflicts,
    _tool_synthesize_overall,
    _tool_weigh_evidence,
    _verdict_rule_based,
    _verdict_with_react,
    run_verdict_agent,
)
from src.config import VERDICTS
from src.models import AgentTrace, Evidence, PICO, SubClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    ev_id: str = "ev-pm-12345",
    source: str = "pubmed",
    title: str = "Test Article",
    content: str = "This RCT studied vaccine efficacy in children.",
    study_type: str = "rct",
    quality_score: float = 0.85,
) -> Evidence:
    return Evidence(
        id=ev_id,
        source=source,
        retrieval_method="api",
        title=title,
        content=content,
        study_type=study_type,
        quality_score=quality_score,
    )


def _make_evidence_quality(
    ev_ids: list[str] | None = None,
    sc_ids: list[str] | None = None,
    strengths: list[float] | None = None,
    directions: list[str] | None = None,
) -> dict:
    """Build a minimal evidence_quality dict for testing."""
    if ev_ids is None:
        ev_ids = ["ev-1"]
    if sc_ids is None:
        sc_ids = ["sc-1"]
    if strengths is None:
        strengths = [0.8] * len(ev_ids)
    if directions is None:
        directions = ["supports"] * len(ev_ids)

    per_evidence = {}
    for i, ev_id in enumerate(ev_ids):
        relevance = {}
        for sc_id in sc_ids:
            relevance[sc_id] = {
                "score": 0.8,
                "direction": directions[i] if i < len(directions) else "neutral",
                "key_finding": f"Finding for {ev_id}",
            }
        per_evidence[ev_id] = {
            "study_type": "rct",
            "hierarchy_weight": 0.8,
            "methodology_score": 0.7,
            "relevance": relevance,
            "grade": {},
            "evidence_strength": strengths[i] if i < len(strengths) else 0.5,
        }

    per_subclaim = {}
    for sc_id in sc_ids:
        dir_counts = {"supports": 0, "opposes": 0, "neutral": 0}
        sc_strengths = []
        for i, ev_id in enumerate(ev_ids):
            d = directions[i] if i < len(directions) else "neutral"
            if d in dir_counts:
                dir_counts[d] += 1
            sc_strengths.append(strengths[i] if i < len(strengths) else 0.5)

        avg_str = sum(sc_strengths) / len(sc_strengths) if sc_strengths else 0.0

        per_subclaim[sc_id] = {
            "evidence_count": len(ev_ids),
            "avg_strength": round(avg_str, 4),
            "direction_summary": dir_counts,
            "top_evidence_ids": ev_ids[:5],
        }

    return {"per_evidence": per_evidence, "per_subclaim": per_subclaim}


def _make_state(
    claim: str = "The MMR vaccine causes autism in children",
    sub_claims: list[SubClaim] | None = None,
    evidence: list[Evidence] | None = None,
    evidence_quality: dict | None = None,
) -> dict:
    """Build a minimal FactCheckState dict for testing."""
    if sub_claims is None:
        sub_claims = [
            SubClaim(id="sc-1", text="MMR vaccine causes autism", evidence=["ev-1"]),
        ]
    if evidence is None:
        evidence = [_make_evidence(ev_id="ev-1")]
    if evidence_quality is None:
        evidence_quality = _make_evidence_quality(
            ev_ids=["ev-1"],
            sc_ids=["sc-1"],
            strengths=[0.8],
            directions=["opposes"],
        )

    return {
        "claim": claim,
        "sub_claims": sub_claims,
        "entities": {},
        "pico": PICO(),
        "retrieval_plan": {},
        "evidence": evidence,
        "extracted_figures": [],
        "evidence_quality": evidence_quality,
        "verdict": "",
        "confidence": 0.0,
        "explanation": "",
        "safety_flags": [],
        "is_dangerous": False,
        "agent_trace": [],
        "total_cost_usd": 0.0,
        "total_duration_seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# Tests: weigh_evidence tool
# ---------------------------------------------------------------------------


class TestWeighEvidence:
    """Tests for the weigh_evidence tool."""

    def test_all_supporting(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2"],
            strengths=[0.8, 0.7],
            directions=["supports", "supports"],
        )
        result = _tool_weigh_evidence(
            "sc-1", ["ev-1", "ev-2"], [], [], eq,
        )
        assert result["success"] is True
        assert result["support_score"] == round(0.8 + 0.7, 4)
        assert result["oppose_score"] == 0.0
        assert result["balance"] == "supports"

    def test_mixed_directions(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2", "ev-3"],
            strengths=[0.5, 0.9, 0.3],
            directions=["supports", "opposes", "neutral"],
        )
        result = _tool_weigh_evidence(
            "sc-1", ["ev-1"], ["ev-2"], ["ev-3"], eq,
        )
        assert result["success"] is True
        assert result["support_score"] == 0.5
        assert result["oppose_score"] == 0.9
        # 0.9 > 0.5 * 1.2 (= 0.6), so opposes wins
        assert result["balance"] == "opposes"

    def test_empty_evidence(self):
        eq = _make_evidence_quality()
        result = _tool_weigh_evidence("sc-1", [], [], [], eq)
        assert result["success"] is True
        assert result["support_score"] == 0.0
        assert result["oppose_score"] == 0.0
        assert result["balance"] == "balanced"

    def test_strength_lookup_from_grader(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-special"],
            strengths=[0.95],
        )
        result = _tool_weigh_evidence("sc-1", ["ev-special"], [], [], eq)
        assert result["support_score"] == 0.95


# ---------------------------------------------------------------------------
# Tests: reconcile_conflicts tool
# ---------------------------------------------------------------------------


class TestReconcileConflicts:
    """Tests for the reconcile_conflicts tool."""

    def test_records_resolution(self):
        records: list[dict] = []
        result = _tool_reconcile_conflicts(
            "sc-1",
            "RCT says X but meta-analysis says Y",
            "Meta-analysis has larger sample, favoring Y",
            "opposes",
            records,
        )
        assert result["success"] is True
        assert result["resolved_direction"] == "opposes"
        assert len(records) == 1
        assert records[0]["sub_claim_id"] == "sc-1"

    def test_valid_directions(self):
        for direction in ["supports", "opposes", "neutral", "inconclusive"]:
            records: list[dict] = []
            result = _tool_reconcile_conflicts(
                "sc-1", "desc", "resolution", direction, records,
            )
            assert result["success"] is True

    def test_invalid_direction(self):
        records: list[dict] = []
        result = _tool_reconcile_conflicts(
            "sc-1", "desc", "resolution", "maybe", records,
        )
        assert result["success"] is False
        assert "maybe" in result["error"]


# ---------------------------------------------------------------------------
# Tests: assign_subclaim_verdict tool
# ---------------------------------------------------------------------------


class TestAssignSubclaimVerdict:
    """Tests for the assign_subclaim_verdict tool."""

    def test_valid_verdict(self):
        verdicts: dict[str, dict] = {}
        all_ids = {"sc-1", "sc-2"}
        result = _tool_assign_subclaim_verdict(
            "sc-1", "REFUTED", 0.85, "Strong evidence contradicts claim",
            verdicts, all_ids,
        )
        assert result["success"] is True
        assert result["verdict"] == "REFUTED"
        assert result["confidence"] == 0.85
        assert "sc-2" in result["remaining_subclaims"]
        assert verdicts["sc-1"]["verdict"] == "REFUTED"

    def test_invalid_verdict(self):
        verdicts: dict[str, dict] = {}
        result = _tool_assign_subclaim_verdict(
            "sc-1", "INVALID_VERDICT", 0.5, "reason",
            verdicts, {"sc-1"},
        )
        assert result["success"] is False
        assert "INVALID_VERDICT" in result["error"]

    def test_tracks_remaining_subclaims(self):
        verdicts: dict[str, dict] = {}
        all_ids = {"sc-1", "sc-2", "sc-3"}

        result1 = _tool_assign_subclaim_verdict(
            "sc-1", "SUPPORTED", 0.9, "reason", verdicts, all_ids,
        )
        assert set(result1["remaining_subclaims"]) == {"sc-2", "sc-3"}

        result2 = _tool_assign_subclaim_verdict(
            "sc-2", "REFUTED", 0.8, "reason", verdicts, all_ids,
        )
        assert set(result2["remaining_subclaims"]) == {"sc-3"}

    def test_confidence_clamping(self):
        verdicts: dict[str, dict] = {}
        result = _tool_assign_subclaim_verdict(
            "sc-1", "SUPPORTED", 1.5, "reason", verdicts, {"sc-1"},
        )
        assert result["confidence"] == 1.0

        verdicts2: dict[str, dict] = {}
        result2 = _tool_assign_subclaim_verdict(
            "sc-1", "SUPPORTED", -0.5, "reason", verdicts2, {"sc-1"},
        )
        assert result2["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Tests: synthesize_overall tool
# ---------------------------------------------------------------------------


class TestSynthesizeOverall:
    """Tests for the synthesize_overall tool."""

    def test_valid_verdict_and_confidence(self):
        overall: dict = {}
        result = _tool_synthesize_overall(
            "REFUTED", 0.85,
            "Multiple RCTs refute this claim.",
            overall,
        )
        assert result["success"] is True
        assert result["verdict"] == "REFUTED"
        assert result["confidence"] == 0.85
        assert overall["complete"] is True

    def test_invalid_verdict(self):
        overall: dict = {}
        result = _tool_synthesize_overall(
            "WRONG_VERDICT", 0.5, "explanation", overall,
        )
        assert result["success"] is False
        assert "WRONG_VERDICT" in result["error"]

    def test_confidence_clamping(self):
        overall: dict = {}
        result = _tool_synthesize_overall(
            "SUPPORTED", 1.5, "explanation", overall,
        )
        assert result["confidence"] == 1.0

        overall2: dict = {}
        result2 = _tool_synthesize_overall(
            "SUPPORTED", -0.3, "explanation", overall2,
        )
        assert result2["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Tests for the tool dispatcher."""

    def _base_args(self):
        return {
            "evidence_quality": _make_evidence_quality(),
            "subclaim_verdicts": {},
            "all_subclaim_ids": {"sc-1"},
            "overall_result": {},
            "conflict_records": [],
        }

    def test_dispatches_weigh_evidence(self):
        args = self._base_args()
        result = _execute_tool(
            "weigh_evidence",
            {"sub_claim_id": "sc-1", "supporting_ids": ["ev-1"], "opposing_ids": [], "neutral_ids": []},
            **args,
        )
        assert result["success"] is True

    def test_dispatches_reconcile_conflicts(self):
        args = self._base_args()
        result = _execute_tool(
            "reconcile_conflicts",
            {
                "sub_claim_id": "sc-1",
                "conflicts_description": "conflict",
                "resolution": "resolved",
                "resolved_direction": "supports",
            },
            **args,
        )
        assert result["success"] is True

    def test_dispatches_assign_verdict(self):
        args = self._base_args()
        result = _execute_tool(
            "assign_subclaim_verdict",
            {"sub_claim_id": "sc-1", "verdict": "REFUTED", "confidence": 0.8, "reasoning": "reason"},
            **args,
        )
        assert result["success"] is True

    def test_dispatches_synthesize_overall(self):
        args = self._base_args()
        result = _execute_tool(
            "synthesize_overall",
            {"overall_verdict": "REFUTED", "confidence": 0.85, "explanation": "Explanation"},
            **args,
        )
        assert result["success"] is True

    def test_unknown_tool_error(self):
        args = self._base_args()
        result = _execute_tool("nonexistent_tool", {}, **args)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: Rule-based fallback
# ---------------------------------------------------------------------------


class TestRuleBasedFallback:
    """Tests for _verdict_rule_based."""

    def test_strong_support_gives_supported(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2", "ev-3"],
            sc_ids=["sc-1"],
            strengths=[0.85, 0.8, 0.75],
            directions=["supports", "supports", "supports"],
        )
        sc = SubClaim(id="sc-1", text="Test claim", evidence=["ev-1", "ev-2", "ev-3"])
        state = _make_state(sub_claims=[sc], evidence_quality=eq)

        subclaim_verdicts, overall_result = _verdict_rule_based(state)

        assert subclaim_verdicts["sc-1"]["verdict"] == "SUPPORTED"
        assert overall_result["verdict"] == "SUPPORTED"

    def test_strong_opposition_gives_refuted(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2"],
            sc_ids=["sc-1"],
            strengths=[0.85, 0.8],
            directions=["opposes", "opposes"],
        )
        sc = SubClaim(id="sc-1", text="Test claim", evidence=["ev-1", "ev-2"])
        state = _make_state(sub_claims=[sc], evidence_quality=eq)

        subclaim_verdicts, overall_result = _verdict_rule_based(state)

        assert subclaim_verdicts["sc-1"]["verdict"] == "REFUTED"

    def test_weak_evidence_gives_preliminary(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1"],
            sc_ids=["sc-1"],
            strengths=[0.3],
            directions=["supports"],
        )
        # Manually set the avg_strength to 0.3
        eq["per_subclaim"]["sc-1"]["avg_strength"] = 0.3

        sc = SubClaim(id="sc-1", text="Test claim", evidence=["ev-1"])
        state = _make_state(sub_claims=[sc], evidence_quality=eq)

        subclaim_verdicts, _ = _verdict_rule_based(state)

        assert subclaim_verdicts["sc-1"]["verdict"] == "PRELIMINARY"

    def test_mixed_balanced_gives_preliminary(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2"],
            sc_ids=["sc-1"],
            strengths=[0.7, 0.7],
            directions=["supports", "opposes"],
        )
        sc = SubClaim(id="sc-1", text="Test claim", evidence=["ev-1", "ev-2"])
        state = _make_state(sub_claims=[sc], evidence_quality=eq)

        subclaim_verdicts, _ = _verdict_rule_based(state)

        assert subclaim_verdicts["sc-1"]["verdict"] == "PRELIMINARY"

    def test_no_evidence_gives_not_supported(self):
        eq = {"per_evidence": {}, "per_subclaim": {}}
        sc = SubClaim(id="sc-1", text="Test claim")
        state = _make_state(sub_claims=[sc], evidence=[], evidence_quality=eq)

        subclaim_verdicts, overall_result = _verdict_rule_based(state)

        assert subclaim_verdicts["sc-1"]["verdict"] == "NOT_SUPPORTED"
        assert subclaim_verdicts["sc-1"]["confidence"] == 0.2

    def test_confidence_formula(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2", "ev-3"],
            sc_ids=["sc-1"],
            strengths=[0.9, 0.8, 0.7],
            directions=["opposes", "opposes", "neutral"],
        )
        sc = SubClaim(id="sc-1", text="Test claim")
        state = _make_state(sub_claims=[sc], evidence_quality=eq)

        subclaim_verdicts, _ = _verdict_rule_based(state)

        # avg_strength * (max(supports, opposes) / total)
        avg_str = eq["per_subclaim"]["sc-1"]["avg_strength"]
        confidence = subclaim_verdicts["sc-1"]["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_empty_subclaims(self):
        state = _make_state(sub_claims=[], evidence_quality={"per_evidence": {}, "per_subclaim": {}})

        subclaim_verdicts, overall_result = _verdict_rule_based(state)

        assert subclaim_verdicts == {}
        assert overall_result["verdict"] == "NOT_SUPPORTED"
        assert overall_result["confidence"] == 0.0

    def test_multi_subclaim_overall(self):
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2"],
            sc_ids=["sc-1", "sc-2"],
            strengths=[0.85, 0.8],
            directions=["opposes", "opposes"],
        )
        sc1 = SubClaim(id="sc-1", text="Sub-claim 1")
        sc2 = SubClaim(id="sc-2", text="Sub-claim 2")
        state = _make_state(sub_claims=[sc1, sc2], evidence_quality=eq)

        subclaim_verdicts, overall_result = _verdict_rule_based(state)

        assert "sc-1" in subclaim_verdicts
        assert "sc-2" in subclaim_verdicts
        assert overall_result["verdict"] in VERDICTS
        assert overall_result.get("complete") is True


# ---------------------------------------------------------------------------
# Tests: Build user message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for _build_user_message."""

    def test_formats_evidence_quality_and_subclaims(self):
        state = _make_state()
        msg = _build_user_message(state)
        assert "MMR vaccine causes autism" in msg
        assert "sc-1" in msg
        assert "avg_strength" in msg

    def test_handles_empty_evidence(self):
        state = _make_state(
            evidence=[],
            evidence_quality={"per_evidence": {}, "per_subclaim": {}},
        )
        msg = _build_user_message(state)
        assert "Sub-claims (1)" in msg


# ---------------------------------------------------------------------------
# Tests: Entry point
# ---------------------------------------------------------------------------


class TestRunVerdictAgent:
    """Tests for the main entry point."""

    def test_no_api_key_uses_rule_based(self):
        state = _make_state()

        with patch("systems.s4_langgraph.agents.verdict_agent.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_verdict_agent(state))

        assert result["verdict"] in VERDICTS
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["explanation"] != ""
        assert result["total_cost_usd"] == 0.0
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0].agent == "verdict_agent"
        assert result["agent_trace"][0].node_type == "agent"

    def test_trace_accumulation(self):
        prior_trace = AgentTrace(
            agent="evidence_grader",
            node_type="agent",
            duration_seconds=2.0,
            cost_usd=0.01,
            input_summary="test",
            output_summary="test",
            success=True,
        )
        state = _make_state()
        state["agent_trace"] = [prior_trace]
        state["total_cost_usd"] = 0.01
        state["total_duration_seconds"] = 2.0

        with patch("systems.s4_langgraph.agents.verdict_agent.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_verdict_agent(state))

        assert len(result["agent_trace"]) == 2
        assert result["agent_trace"][0].agent == "evidence_grader"
        assert result["agent_trace"][1].agent == "verdict_agent"
        assert result["total_cost_usd"] >= 0.01

    def test_fallback_on_exception(self):
        state = _make_state()

        with patch("systems.s4_langgraph.agents.verdict_agent.ANTHROPIC_API_KEY", "test-key"):
            with patch(
                "systems.s4_langgraph.agents.verdict_agent._verdict_with_react",
                side_effect=Exception("API error"),
            ):
                result = asyncio.run(run_verdict_agent(state))

        assert result["verdict"] in VERDICTS
        assert result["total_cost_usd"] == 0.0

    def test_empty_evidence(self):
        state = _make_state(
            evidence=[],
            evidence_quality={"per_evidence": {}, "per_subclaim": {}},
        )

        with patch("systems.s4_langgraph.agents.verdict_agent.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_verdict_agent(state))

        assert result["verdict"] == "NOT_SUPPORTED"
        assert len(result["agent_trace"]) == 1

    def test_subclaims_updated_with_verdicts(self):
        state = _make_state()

        with patch("systems.s4_langgraph.agents.verdict_agent.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_verdict_agent(state))

        updated_sc = result["sub_claims"][0]
        assert updated_sc.verdict is not None
        assert updated_sc.verdict in VERDICTS
        assert updated_sc.confidence > 0.0


# ---------------------------------------------------------------------------
# Tests: ReAct loop (mocked Anthropic)
# ---------------------------------------------------------------------------


class TestReActLoop:
    """Tests for _verdict_with_react with mocked Anthropic API."""

    def test_weigh_assign_synthesize_sequence(self):
        state = _make_state()

        # Step 1: weigh_evidence
        block1 = MagicMock()
        block1.type = "tool_use"
        block1.name = "weigh_evidence"
        block1.input = {
            "sub_claim_id": "sc-1",
            "supporting_ids": [],
            "opposing_ids": ["ev-1"],
            "neutral_ids": [],
        }
        block1.id = "c1"

        # Step 2: assign_subclaim_verdict
        block2 = MagicMock()
        block2.type = "tool_use"
        block2.name = "assign_subclaim_verdict"
        block2.input = {
            "sub_claim_id": "sc-1",
            "verdict": "REFUTED",
            "confidence": 0.85,
            "reasoning": "Strong opposing evidence from RCTs",
        }
        block2.id = "c2"

        # Step 3: synthesize_overall
        block3 = MagicMock()
        block3.type = "tool_use"
        block3.name = "synthesize_overall"
        block3.input = {
            "overall_verdict": "REFUTED",
            "confidence": 0.85,
            "explanation": "Multiple large RCTs found no link between MMR and autism.",
        }
        block3.id = "c3"

        resps = []
        for block in [block1, block2, block3]:
            r = MagicMock()
            r.content = [block]
            r.usage = MagicMock(input_tokens=500, output_tokens=100)
            resps.append(r)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = resps

        with patch("systems.s4_langgraph.agents.verdict_agent.anthropic.Anthropic", return_value=mock_client):
            subclaim_verdicts, overall_result, cost, tool_calls, steps = _verdict_with_react(state)

        assert subclaim_verdicts["sc-1"]["verdict"] == "REFUTED"
        assert overall_result["verdict"] == "REFUTED"
        assert overall_result["complete"] is True
        assert cost > 0
        assert len(tool_calls) == 3
        assert steps == 3  # breaks after synthesize_overall (complete=True)

    def test_multi_subclaim(self):
        sc1 = SubClaim(id="sc-1", text="Sub-claim 1", evidence=["ev-1"])
        sc2 = SubClaim(id="sc-2", text="Sub-claim 2", evidence=["ev-2"])
        eq = _make_evidence_quality(
            ev_ids=["ev-1", "ev-2"],
            sc_ids=["sc-1", "sc-2"],
            strengths=[0.8, 0.7],
            directions=["opposes", "supports"],
        )
        ev1 = _make_evidence(ev_id="ev-1")
        ev2 = _make_evidence(ev_id="ev-2")
        state = _make_state(sub_claims=[sc1, sc2], evidence=[ev1, ev2], evidence_quality=eq)

        # Assign verdicts for both, then synthesize
        blocks = []
        for sc_id, verdict in [("sc-1", "REFUTED"), ("sc-2", "SUPPORTED")]:
            b = MagicMock()
            b.type = "tool_use"
            b.name = "assign_subclaim_verdict"
            b.input = {
                "sub_claim_id": sc_id,
                "verdict": verdict,
                "confidence": 0.8,
                "reasoning": "reason",
            }
            b.id = f"assign_{sc_id}"
            blocks.append(b)

        # Synthesize
        b_synth = MagicMock()
        b_synth.type = "tool_use"
        b_synth.name = "synthesize_overall"
        b_synth.input = {
            "overall_verdict": "MISLEADING",
            "confidence": 0.7,
            "explanation": "Mixed evidence across sub-claims.",
        }
        b_synth.id = "synth"
        blocks.append(b_synth)

        resps = []
        for block in blocks:
            r = MagicMock()
            r.content = [block]
            r.usage = MagicMock(input_tokens=500, output_tokens=100)
            resps.append(r)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = resps

        with patch("systems.s4_langgraph.agents.verdict_agent.anthropic.Anthropic", return_value=mock_client):
            subclaim_verdicts, overall_result, cost, _, _ = _verdict_with_react(state)

        assert "sc-1" in subclaim_verdicts
        assert "sc-2" in subclaim_verdicts
        assert overall_result["verdict"] == "MISLEADING"

    def test_cost_tracking(self):
        state = _make_state()

        block = MagicMock()
        block.type = "tool_use"
        block.name = "synthesize_overall"
        block.input = {
            "overall_verdict": "REFUTED",
            "confidence": 0.85,
            "explanation": "Explanation",
        }
        block.id = "c1"

        resp = MagicMock()
        resp.content = [block]
        resp.usage = MagicMock(input_tokens=1000, output_tokens=200)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp]

        with patch("systems.s4_langgraph.agents.verdict_agent.anthropic.Anthropic", return_value=mock_client):
            _, _, cost, _, _ = _verdict_with_react(state)

        expected_cost = (1000 * 3.0 + 200 * 15.0) / 1_000_000
        assert abs(cost - expected_cost) < 1e-9
