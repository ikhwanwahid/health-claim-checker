"""Tests for the Evidence Grader agent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from systems.s4_langgraph.agents.evidence_grader import (
    GRADE_PENALTIES,
    MAX_REACT_STEPS,
    METHODOLOGY_WEIGHTS,
    QUALITY_SCORES,
    RATING_SCORES,
    VALID_DIRECTIONS,
    VALID_STUDY_TYPES,
    _build_evidence_quality,
    _build_user_message,
    _execute_tool,
    _grade_rule_based,
    _grade_with_react,
    _tool_apply_grade,
    _tool_assess_methodology,
    _tool_check_relevance,
    _tool_classify_study_type,
    compute_evidence_strength,
    compute_grade_penalty,
    compute_methodology_score,
    run_evidence_grader,
)
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
    pmid: str | None = "12345",
) -> Evidence:
    return Evidence(
        id=ev_id,
        source=source,
        retrieval_method="api",
        title=title,
        content=content,
        study_type=study_type,
        quality_score=quality_score,
        pmid=pmid,
    )


def _make_state(
    claim: str = "The MMR vaccine causes autism in children",
    sub_claims: list[SubClaim] | None = None,
    evidence: list[Evidence] | None = None,
    entities: dict | None = None,
) -> dict:
    """Build a minimal FactCheckState dict for testing."""
    if sub_claims is None:
        sub_claims = [
            SubClaim(id="sc-1", text="MMR vaccine causes autism", evidence=["ev-pm-12345"]),
        ]
    if evidence is None:
        evidence = [_make_evidence()]
    if entities is None:
        entities = {"drugs": [], "conditions": ["autism"], "genes": []}

    return {
        "claim": claim,
        "sub_claims": sub_claims,
        "entities": entities,
        "pico": PICO(),
        "retrieval_plan": {},
        "evidence": evidence,
        "extracted_figures": [],
        "evidence_quality": {},
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
# Tests: classify_study_type tool
# ---------------------------------------------------------------------------


class TestClassifyStudyType:
    """Tests for the classify_study_type tool."""

    def test_valid_rct(self):
        results: dict[str, dict] = {}
        result = _tool_classify_study_type("ev-1", "rct", 0.9, results)
        assert result["success"] is True
        assert result["study_type"] == "rct"
        assert result["hierarchy_weight"] == 0.8
        assert results["ev-1"]["study_type"] == "rct"

    def test_valid_guideline(self):
        results: dict[str, dict] = {}
        result = _tool_classify_study_type("ev-1", "guideline", 0.95, results)
        assert result["success"] is True
        assert result["hierarchy_weight"] == 1.0

    def test_unknown_type_gets_default_weight(self):
        results: dict[str, dict] = {}
        result = _tool_classify_study_type("ev-1", "unknown", 0.5, results)
        assert result["success"] is True
        assert result["hierarchy_weight"] == 0.1

    def test_invalid_type_returns_error(self):
        results: dict[str, dict] = {}
        result = _tool_classify_study_type("ev-1", "blog_post", 0.5, results)
        assert result["success"] is False
        assert "blog_post" in result["error"]

    def test_confidence_clamping(self):
        results: dict[str, dict] = {}
        result = _tool_classify_study_type("ev-1", "rct", 1.5, results)
        assert result["confidence"] == 1.0

        result2 = _tool_classify_study_type("ev-2", "rct", -0.5, results)
        assert result2["confidence"] == 0.0

    def test_all_valid_types_accepted(self):
        for study_type in VALID_STUDY_TYPES:
            results: dict[str, dict] = {}
            result = _tool_classify_study_type("ev-1", study_type, 0.8, results)
            assert result["success"] is True, f"Failed for type: {study_type}"


# ---------------------------------------------------------------------------
# Tests: assess_methodology tool
# ---------------------------------------------------------------------------


class TestAssessMethodology:
    """Tests for the assess_methodology tool."""

    def test_all_strong(self):
        results: dict[str, dict] = {}
        result = _tool_assess_methodology(
            "ev-1", "strong", "strong", "strong", "strong", "high", results,
        )
        assert result["success"] is True
        assert result["methodology_score"] == 1.0
        assert result["overall_rating"] == "high"

    def test_mixed_ratings(self):
        results: dict[str, dict] = {}
        result = _tool_assess_methodology(
            "ev-1", "strong", "moderate", "weak", "moderate", "moderate", results,
        )
        assert result["success"] is True
        score = result["methodology_score"]
        assert 0.0 < score < 1.0

    def test_not_applicable_excluded(self):
        results: dict[str, dict] = {}
        result = _tool_assess_methodology(
            "ev-1", "strong", "not_applicable", "not_applicable", "strong", "high", results,
        )
        assert result["success"] is True
        # Only sample_size (0.3 weight) and follow_up (0.2 weight) count
        # Both strong (1.0), so score should be 1.0
        assert result["methodology_score"] == 1.0

    def test_all_not_applicable_defaults(self):
        results: dict[str, dict] = {}
        result = _tool_assess_methodology(
            "ev-1", "not_applicable", "not_applicable", "not_applicable",
            "not_applicable", "moderate", results,
        )
        assert result["success"] is True
        assert result["methodology_score"] == 0.6  # default moderate

    def test_invalid_rating_returns_error(self):
        results: dict[str, dict] = {}
        result = _tool_assess_methodology(
            "ev-1", "excellent", "strong", "strong", "strong", "high", results,
        )
        assert result["success"] is False
        assert "excellent" in result["error"]

    def test_invalid_overall_rating_returns_error(self):
        results: dict[str, dict] = {}
        result = _tool_assess_methodology(
            "ev-1", "strong", "strong", "strong", "strong", "excellent", results,
        )
        assert result["success"] is False
        assert "excellent" in result["error"]


# ---------------------------------------------------------------------------
# Tests: check_relevance tool
# ---------------------------------------------------------------------------


class TestCheckRelevance:
    """Tests for the check_relevance tool."""

    def test_valid_supports(self):
        results: dict[str, dict] = {}
        result = _tool_check_relevance(
            "ev-1", "sc-1", 0.9, "supports", "Study found strong efficacy", results,
        )
        assert result["success"] is True
        assert result["direction"] == "supports"
        assert result["relevance_score"] == 0.9
        assert results["ev-1"]["relevance"]["sc-1"]["direction"] == "supports"

    def test_valid_opposes(self):
        results: dict[str, dict] = {}
        result = _tool_check_relevance(
            "ev-1", "sc-1", 0.8, "opposes", "No significant effect found", results,
        )
        assert result["success"] is True
        assert result["direction"] == "opposes"

    def test_valid_neutral(self):
        results: dict[str, dict] = {}
        result = _tool_check_relevance(
            "ev-1", "sc-1", 0.5, "neutral", "Inconclusive results", results,
        )
        assert result["success"] is True
        assert result["direction"] == "neutral"

    def test_score_clamped_high(self):
        results: dict[str, dict] = {}
        result = _tool_check_relevance(
            "ev-1", "sc-1", 1.5, "supports", "Finding", results,
        )
        assert result["relevance_score"] == 1.0

    def test_score_clamped_low(self):
        results: dict[str, dict] = {}
        result = _tool_check_relevance(
            "ev-1", "sc-1", -0.3, "neutral", "Finding", results,
        )
        assert result["relevance_score"] == 0.0

    def test_invalid_direction(self):
        results: dict[str, dict] = {}
        result = _tool_check_relevance(
            "ev-1", "sc-1", 0.5, "maybe", "Finding", results,
        )
        assert result["success"] is False
        assert "maybe" in result["error"]

    def test_multiple_subclaims_per_evidence(self):
        results: dict[str, dict] = {}
        _tool_check_relevance("ev-1", "sc-1", 0.9, "supports", "Finding 1", results)
        _tool_check_relevance("ev-1", "sc-2", 0.3, "neutral", "Finding 2", results)
        assert "sc-1" in results["ev-1"]["relevance"]
        assert "sc-2" in results["ev-1"]["relevance"]


# ---------------------------------------------------------------------------
# Tests: apply_grade tool
# ---------------------------------------------------------------------------


class TestApplyGrade:
    """Tests for the apply_grade tool."""

    def test_no_concerns_high_quality(self):
        results: dict[str, dict] = {
            "ev-1": {
                "study_type": "rct",
                "hierarchy_weight": 0.8,
                "methodology": {"overall_rating": "high"},
                "methodology_score": 1.0,
                "relevance": {"sc-1": {"score": 0.9, "direction": "opposes", "key_finding": "..."}},
            },
        }
        graded: set[str] = set()
        result = _tool_apply_grade(
            "ev-1",
            "no_serious_concern", "no_serious_concern", "no_serious_concern",
            "no_serious_concern", "no_serious_concern", "high",
            results, graded,
        )
        assert result["success"] is True
        assert result["evidence_strength"] > 0.0
        assert "ev-1" in graded
        assert not result.get("warnings")

    def test_serious_penalties(self):
        results: dict[str, dict] = {
            "ev-1": {
                "hierarchy_weight": 0.8,
                "methodology_score": 0.6,
                "relevance": {"sc-1": {"score": 0.5, "direction": "neutral", "key_finding": ""}},
            },
        }
        graded: set[str] = set()
        result = _tool_apply_grade(
            "ev-1",
            "serious", "serious", "no_serious_concern",
            "no_serious_concern", "no_serious_concern", "moderate",
            results, graded,
        )
        assert result["success"] is True
        assert "ev-1" in graded

    def test_strength_formula(self):
        """Verify the exact strength formula: hierarchy*0.4 + method*0.3 + relevance*0.3."""
        results: dict[str, dict] = {
            "ev-1": {
                "hierarchy_weight": 0.8,
                "methodology_score": 0.7,
                "relevance": {"sc-1": {"score": 0.9, "direction": "supports", "key_finding": ""}},
            },
        }
        graded: set[str] = set()
        result = _tool_apply_grade(
            "ev-1",
            "no_serious_concern", "no_serious_concern", "no_serious_concern",
            "no_serious_concern", "no_serious_concern", "high",
            results, graded,
        )
        expected = round(0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.3, 4)
        assert result["evidence_strength"] == expected

    def test_missing_prior_assessments_warning(self):
        results: dict[str, dict] = {"ev-1": {}}
        graded: set[str] = set()
        result = _tool_apply_grade(
            "ev-1",
            "no_serious_concern", "no_serious_concern", "no_serious_concern",
            "no_serious_concern", "no_serious_concern", "moderate",
            results, graded,
        )
        assert result["success"] is True
        assert "warnings" in result
        assert len(result["warnings"]) == 3  # study_type, methodology, relevance

    def test_invalid_grade_criterion(self):
        results: dict[str, dict] = {}
        graded: set[str] = set()
        result = _tool_apply_grade(
            "ev-1",
            "very_bad", "no_serious_concern", "no_serious_concern",
            "no_serious_concern", "no_serious_concern", "high",
            results, graded,
        )
        assert result["success"] is False
        assert "very_bad" in result["error"]

    def test_invalid_overall_quality(self):
        results: dict[str, dict] = {}
        graded: set[str] = set()
        result = _tool_apply_grade(
            "ev-1",
            "no_serious_concern", "no_serious_concern", "no_serious_concern",
            "no_serious_concern", "no_serious_concern", "excellent",
            results, graded,
        )
        assert result["success"] is False
        assert "excellent" in result["error"]


# ---------------------------------------------------------------------------
# Tests: Tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Tests for the tool dispatcher."""

    def test_dispatches_classify(self):
        results: dict[str, dict] = {}
        graded: set[str] = set()
        result = _execute_tool(
            "classify_study_type",
            {"evidence_id": "ev-1", "study_type": "rct", "confidence": 0.9},
            results, graded,
        )
        assert result["success"] is True
        assert result["study_type"] == "rct"

    def test_dispatches_assess(self):
        results: dict[str, dict] = {}
        graded: set[str] = set()
        result = _execute_tool(
            "assess_methodology",
            {
                "evidence_id": "ev-1",
                "sample_size_rating": "strong",
                "blinding": "moderate",
                "randomization": "strong",
                "follow_up": "moderate",
                "overall_rating": "moderate",
            },
            results, graded,
        )
        assert result["success"] is True

    def test_dispatches_relevance(self):
        results: dict[str, dict] = {}
        graded: set[str] = set()
        result = _execute_tool(
            "check_relevance",
            {
                "evidence_id": "ev-1",
                "sub_claim_id": "sc-1",
                "relevance_score": 0.8,
                "direction": "supports",
                "key_finding": "Strong positive effect",
            },
            results, graded,
        )
        assert result["success"] is True

    def test_dispatches_apply_grade(self):
        results: dict[str, dict] = {"ev-1": {"hierarchy_weight": 0.8, "methodology_score": 0.6}}
        graded: set[str] = set()
        result = _execute_tool(
            "apply_grade",
            {
                "evidence_id": "ev-1",
                "risk_of_bias": "no_serious_concern",
                "inconsistency": "no_serious_concern",
                "indirectness": "no_serious_concern",
                "imprecision": "no_serious_concern",
                "publication_bias": "no_serious_concern",
                "overall_quality": "high",
            },
            results, graded,
        )
        assert result["success"] is True

    def test_unknown_tool_error(self):
        result = _execute_tool("nonexistent_tool", {}, {}, set())
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: Rule-based fallback
# ---------------------------------------------------------------------------


class TestRuleBasedGrading:
    """Tests for _grade_rule_based."""

    def test_uses_existing_study_type(self):
        ev = _make_evidence(study_type="systematic_review")
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        assert results["ev-pm-12345"]["study_type"] == "systematic_review"
        assert results["ev-pm-12345"]["hierarchy_weight"] == 0.9

    def test_unknown_defaults_to_expert_opinion(self):
        ev = _make_evidence(study_type="unknown")
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        assert results["ev-pm-12345"]["study_type"] == "expert_opinion"
        assert results["ev-pm-12345"]["hierarchy_weight"] == 0.1

    def test_uses_quality_score_as_relevance(self):
        ev = _make_evidence(quality_score=0.92)
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        relevance = results["ev-pm-12345"]["relevance"]
        assert relevance["sc-1"]["score"] == 0.92

    def test_zero_quality_score_defaults_to_half(self):
        ev = _make_evidence(quality_score=0.0)
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        relevance = results["ev-pm-12345"]["relevance"]
        assert relevance["sc-1"]["score"] == 0.5

    def test_direction_is_neutral(self):
        ev = _make_evidence()
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        assert results["ev-pm-12345"]["relevance"]["sc-1"]["direction"] == "neutral"

    def test_per_subclaim_aggregation(self):
        ev1 = _make_evidence(ev_id="ev-1", study_type="rct", quality_score=0.9)
        ev2 = _make_evidence(ev_id="ev-2", study_type="cohort", quality_score=0.7)
        sc = SubClaim(id="sc-1", text="Claim text", evidence=["ev-1", "ev-2"])
        state = _make_state(sub_claims=[sc], evidence=[ev1, ev2])
        results = _grade_rule_based(state)
        assert "ev-1" in results
        assert "ev-2" in results

    def test_empty_evidence(self):
        state = _make_state(evidence=[])
        results = _grade_rule_based(state)
        assert results == {}

    def test_methodology_default_moderate(self):
        ev = _make_evidence()
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        assert results["ev-pm-12345"]["methodology_score"] == 0.6

    def test_grade_defaults_no_concern(self):
        ev = _make_evidence()
        state = _make_state(evidence=[ev])
        results = _grade_rule_based(state)
        grade = results["ev-pm-12345"]["grade"]
        assert grade["risk_of_bias"] == "no_serious_concern"
        assert grade["inconsistency"] == "no_serious_concern"


# ---------------------------------------------------------------------------
# Tests: Build user message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for _build_user_message."""

    def test_includes_claim(self):
        state = _make_state()
        msg = _build_user_message(state)
        assert "MMR vaccine causes autism" in msg

    def test_includes_evidence_details(self):
        state = _make_state()
        msg = _build_user_message(state)
        assert "ev-pm-12345" in msg
        assert "Test Article" in msg

    def test_includes_subclaim_ids(self):
        state = _make_state()
        msg = _build_user_message(state)
        assert "sc-1" in msg

    def test_handles_empty_evidence(self):
        state = _make_state(evidence=[])
        msg = _build_user_message(state)
        assert "Evidence items to grade (0)" in msg


# ---------------------------------------------------------------------------
# Tests: Entry point
# ---------------------------------------------------------------------------


class TestRunEvidenceGrader:
    """Tests for the main entry point."""

    def test_no_api_key_uses_rule_based(self):
        state = _make_state()

        with patch("systems.s4_langgraph.agents.evidence_grader.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_grader(state))

        assert "per_evidence" in result["evidence_quality"]
        assert "per_subclaim" in result["evidence_quality"]
        assert result["total_cost_usd"] == 0.0
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0].agent == "evidence_grader"
        assert result["agent_trace"][0].node_type == "agent"

    def test_trace_accumulation(self):
        prior_trace = AgentTrace(
            agent="evidence_retriever",
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

        with patch("systems.s4_langgraph.agents.evidence_grader.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_grader(state))

        assert len(result["agent_trace"]) == 2
        assert result["agent_trace"][0].agent == "evidence_retriever"
        assert result["agent_trace"][1].agent == "evidence_grader"
        assert result["total_cost_usd"] >= 0.01

    def test_fallback_on_exception(self):
        state = _make_state()

        with patch("systems.s4_langgraph.agents.evidence_grader.ANTHROPIC_API_KEY", "test-key"):
            with patch(
                "systems.s4_langgraph.agents.evidence_grader._grade_with_react",
                side_effect=Exception("API error"),
            ):
                result = asyncio.run(run_evidence_grader(state))

        assert "per_evidence" in result["evidence_quality"]
        assert result["total_cost_usd"] == 0.0

    def test_empty_evidence(self):
        state = _make_state(evidence=[], sub_claims=[SubClaim(id="sc-1", text="Test")])

        with patch("systems.s4_langgraph.agents.evidence_grader.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_grader(state))

        assert result["evidence_quality"]["per_evidence"] == {}
        assert len(result["agent_trace"]) == 1

    def test_evidence_quality_structure(self):
        ev1 = _make_evidence(ev_id="ev-1", study_type="rct", quality_score=0.9)
        ev2 = _make_evidence(ev_id="ev-2", study_type="cohort", quality_score=0.6)
        sc = SubClaim(id="sc-1", text="Test claim", evidence=["ev-1", "ev-2"])
        state = _make_state(sub_claims=[sc], evidence=[ev1, ev2])

        with patch("systems.s4_langgraph.agents.evidence_grader.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_grader(state))

        eq = result["evidence_quality"]
        assert "ev-1" in eq["per_evidence"]
        assert "ev-2" in eq["per_evidence"]
        assert "sc-1" in eq["per_subclaim"]
        assert eq["per_subclaim"]["sc-1"]["evidence_count"] == 2
        assert eq["per_subclaim"]["sc-1"]["avg_strength"] > 0
        assert "direction_summary" in eq["per_subclaim"]["sc-1"]


# ---------------------------------------------------------------------------
# Tests: ReAct loop (mocked Anthropic)
# ---------------------------------------------------------------------------


class TestReActLoop:
    """Tests for _grade_with_react with mocked Anthropic API."""

    def test_classify_assess_relevance_grade_sequence(self):
        ev = _make_evidence()
        state = _make_state(evidence=[ev])

        # Step 1: classify
        block1 = MagicMock()
        block1.type = "tool_use"
        block1.name = "classify_study_type"
        block1.input = {"evidence_id": "ev-pm-12345", "study_type": "rct", "confidence": 0.9}
        block1.id = "c1"

        # Step 2: assess
        block2 = MagicMock()
        block2.type = "tool_use"
        block2.name = "assess_methodology"
        block2.input = {
            "evidence_id": "ev-pm-12345",
            "sample_size_rating": "strong",
            "blinding": "strong",
            "randomization": "strong",
            "follow_up": "moderate",
            "overall_rating": "high",
        }
        block2.id = "c2"

        # Step 3: relevance
        block3 = MagicMock()
        block3.type = "tool_use"
        block3.name = "check_relevance"
        block3.input = {
            "evidence_id": "ev-pm-12345",
            "sub_claim_id": "sc-1",
            "relevance_score": 0.9,
            "direction": "opposes",
            "key_finding": "No link between MMR and autism",
        }
        block3.id = "c3"

        # Step 4: grade
        block4 = MagicMock()
        block4.type = "tool_use"
        block4.name = "apply_grade"
        block4.input = {
            "evidence_id": "ev-pm-12345",
            "risk_of_bias": "no_serious_concern",
            "inconsistency": "no_serious_concern",
            "indirectness": "no_serious_concern",
            "imprecision": "no_serious_concern",
            "publication_bias": "no_serious_concern",
            "overall_quality": "high",
        }
        block4.id = "c4"

        resps = []
        for block in [block1, block2, block3, block4]:
            r = MagicMock()
            r.content = [block]
            r.usage = MagicMock(input_tokens=500, output_tokens=100)
            resps.append(r)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = resps

        with patch("systems.s4_langgraph.agents.evidence_grader.anthropic.Anthropic", return_value=mock_client):
            grading_results, graded_ev, cost, tool_calls, steps = _grade_with_react(state)

        assert "ev-pm-12345" in grading_results
        assert "ev-pm-12345" in graded_ev
        assert grading_results["ev-pm-12345"]["study_type"] == "rct"
        assert grading_results["ev-pm-12345"]["evidence_strength"] > 0
        assert cost > 0
        assert len(tool_calls) == 4
        assert steps == 4  # breaks after apply_grade (all evidence graded)

    def test_multi_evidence(self):
        ev1 = _make_evidence(ev_id="ev-1")
        ev2 = _make_evidence(ev_id="ev-2")
        sc = SubClaim(id="sc-1", text="Test claim", evidence=["ev-1", "ev-2"])
        state = _make_state(sub_claims=[sc], evidence=[ev1, ev2])

        # Simplified: grade each with classify + apply_grade
        blocks = []
        for ev_id in ["ev-1", "ev-2"]:
            b_classify = MagicMock()
            b_classify.type = "tool_use"
            b_classify.name = "classify_study_type"
            b_classify.input = {"evidence_id": ev_id, "study_type": "rct", "confidence": 0.9}
            b_classify.id = f"classify_{ev_id}"
            blocks.append(b_classify)

            b_grade = MagicMock()
            b_grade.type = "tool_use"
            b_grade.name = "apply_grade"
            b_grade.input = {
                "evidence_id": ev_id,
                "risk_of_bias": "no_serious_concern",
                "inconsistency": "no_serious_concern",
                "indirectness": "no_serious_concern",
                "imprecision": "no_serious_concern",
                "publication_bias": "no_serious_concern",
                "overall_quality": "high",
            }
            b_grade.id = f"grade_{ev_id}"
            blocks.append(b_grade)

        resps = []
        for block in blocks:
            r = MagicMock()
            r.content = [block]
            r.usage = MagicMock(input_tokens=500, output_tokens=100)
            resps.append(r)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = resps

        with patch("systems.s4_langgraph.agents.evidence_grader.anthropic.Anthropic", return_value=mock_client):
            grading_results, graded_ev, cost, tool_calls, steps = _grade_with_react(state)

        assert "ev-1" in graded_ev
        assert "ev-2" in graded_ev

    def test_cost_tracking(self):
        ev = _make_evidence()
        state = _make_state(evidence=[ev])

        block = MagicMock()
        block.type = "tool_use"
        block.name = "apply_grade"
        block.input = {
            "evidence_id": "ev-pm-12345",
            "risk_of_bias": "no_serious_concern",
            "inconsistency": "no_serious_concern",
            "indirectness": "no_serious_concern",
            "imprecision": "no_serious_concern",
            "publication_bias": "no_serious_concern",
            "overall_quality": "high",
        }
        block.id = "c1"

        resp1 = MagicMock()
        resp1.content = [block]
        resp1.usage = MagicMock(input_tokens=1000, output_tokens=200)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp1]

        with patch("systems.s4_langgraph.agents.evidence_grader.anthropic.Anthropic", return_value=mock_client):
            _, _, cost, _, _ = _grade_with_react(state)

        expected_cost = (1000 * 3.0 + 200 * 15.0) / 1_000_000
        assert abs(cost - expected_cost) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Scoring helpers
# ---------------------------------------------------------------------------


class TestScoringHelpers:
    """Tests for scoring utility functions."""

    def test_methodology_score_all_strong(self):
        score = compute_methodology_score("strong", "strong", "strong", "strong")
        assert score == 1.0

    def test_methodology_score_all_weak(self):
        score = compute_methodology_score("weak", "weak", "weak", "weak")
        assert score == 0.3

    def test_methodology_score_mixed(self):
        score = compute_methodology_score("strong", "moderate", "weak", "strong")
        # strong: 1.0*0.3 + moderate: 0.6*0.25 + weak: 0.3*0.25 + strong: 1.0*0.2
        expected = (1.0 * 0.3 + 0.6 * 0.25 + 0.3 * 0.25 + 1.0 * 0.2) / 1.0
        assert abs(score - round(expected, 4)) < 1e-4

    def test_evidence_strength_formula(self):
        strength = compute_evidence_strength(0.8, 0.7, 0.9)
        expected = round(0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.3, 4)
        assert strength == expected

    def test_evidence_strength_clamped(self):
        # Max possible: 1.0 * 0.4 + 1.0 * 0.3 + 1.0 * 0.3 = 1.0
        assert compute_evidence_strength(1.0, 1.0, 1.0) == 1.0
        # With very high inputs (shouldn't happen but test clamping)
        assert compute_evidence_strength(2.0, 2.0, 2.0) == 1.0

    def test_grade_penalty_no_concerns(self):
        penalty = compute_grade_penalty(
            "no_serious_concern", "no_serious_concern", "no_serious_concern",
            "no_serious_concern", "no_serious_concern",
        )
        assert penalty == 0.0

    def test_grade_penalty_all_serious(self):
        penalty = compute_grade_penalty(
            "serious", "serious", "serious", "serious", "serious",
        )
        assert penalty == round(-0.15 * 5, 4)

    def test_grade_penalty_mixed(self):
        penalty = compute_grade_penalty(
            "serious", "no_serious_concern", "very_serious",
            "no_serious_concern", "serious",
        )
        expected = round(-0.15 + 0.0 + -0.30 + 0.0 + -0.15, 4)
        assert penalty == expected


# ---------------------------------------------------------------------------
# Tests: Build evidence quality output
# ---------------------------------------------------------------------------


class TestBuildEvidenceQuality:
    """Tests for _build_evidence_quality."""

    def test_basic_structure(self):
        grading_results = {
            "ev-1": {
                "study_type": "rct",
                "hierarchy_weight": 0.8,
                "methodology_score": 0.7,
                "relevance": {"sc-1": {"score": 0.9, "direction": "opposes", "key_finding": "No link"}},
                "grade": {"risk_of_bias": "no_serious_concern"},
                "evidence_strength": 0.78,
            },
        }
        sub_claims = [SubClaim(id="sc-1", text="Test claim")]
        result = _build_evidence_quality(grading_results, sub_claims)

        assert "per_evidence" in result
        assert "per_subclaim" in result
        assert result["per_evidence"]["ev-1"]["study_type"] == "rct"
        assert result["per_subclaim"]["sc-1"]["evidence_count"] == 1
        assert result["per_subclaim"]["sc-1"]["direction_summary"]["opposes"] == 1

    def test_multiple_evidence_per_subclaim(self):
        grading_results = {
            "ev-1": {
                "study_type": "rct",
                "hierarchy_weight": 0.8,
                "methodology_score": 0.7,
                "relevance": {"sc-1": {"score": 0.9, "direction": "supports", "key_finding": ""}},
                "grade": {},
                "evidence_strength": 0.8,
            },
            "ev-2": {
                "study_type": "cohort",
                "hierarchy_weight": 0.6,
                "methodology_score": 0.5,
                "relevance": {"sc-1": {"score": 0.7, "direction": "opposes", "key_finding": ""}},
                "grade": {},
                "evidence_strength": 0.6,
            },
        }
        sub_claims = [SubClaim(id="sc-1", text="Test claim")]
        result = _build_evidence_quality(grading_results, sub_claims)

        sc1 = result["per_subclaim"]["sc-1"]
        assert sc1["evidence_count"] == 2
        assert sc1["avg_strength"] == round((0.8 + 0.6) / 2, 4)
        assert sc1["direction_summary"]["supports"] == 1
        assert sc1["direction_summary"]["opposes"] == 1
        assert len(sc1["top_evidence_ids"]) == 2
        assert sc1["top_evidence_ids"][0] == "ev-1"  # higher strength

    def test_empty_results(self):
        result = _build_evidence_quality({}, [SubClaim(id="sc-1", text="Test")])
        assert result["per_evidence"] == {}
        assert result["per_subclaim"]["sc-1"]["evidence_count"] == 0
        assert result["per_subclaim"]["sc-1"]["avg_strength"] == 0.0
