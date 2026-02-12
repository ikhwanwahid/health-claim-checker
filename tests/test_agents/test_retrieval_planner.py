"""Tests for the Retrieval Planner agent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.retrieval_planner import (
    DEFAULT_METHODS,
    VALID_METHODS,
    _build_user_message,
    _plan_rule_based,
    _tool_analyze_claim,
    _tool_assign_methods,
    _tool_check_guidelines,
    _execute_tool,
    run_retrieval_planner,
)
from src.graph.state import PICO, SubClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    claim: str = "Test claim",
    sub_claims: list[SubClaim] | None = None,
    entities: dict | None = None,
    pico: PICO | None = None,
) -> dict:
    """Build a minimal FactCheckState dict for testing."""
    if sub_claims is None:
        sub_claims = [SubClaim(id="sc-1", text=claim)]
    if entities is None:
        entities = {"drugs": [], "conditions": [], "genes": [], "organisms": [], "procedures": [], "anatomical": []}
    if pico is None:
        pico = PICO()

    return {
        "claim": claim,
        "sub_claims": sub_claims,
        "entities": entities,
        "pico": pico,
        "retrieval_plan": {},
        "evidence": [],
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
# Tests: _tool_analyze_claim
# ---------------------------------------------------------------------------


class TestAnalyzeClaimCharacteristics:
    """Tests for the analyze_claim_characteristics tool."""

    def test_detects_drug_interaction(self):
        result = _tool_analyze_claim(
            "Aspirin interacts with warfarin",
            {"drugs": ["aspirin", "warfarin"], "conditions": []},
        )
        assert result["has_drugs"] is True
        assert result["has_drug_interaction"] is True
        assert result["drug_names"] == ["aspirin", "warfarin"]

    def test_no_drug_interaction_without_drugs(self):
        result = _tool_analyze_claim(
            "Exercise interacts with mood",
            {"drugs": [], "conditions": ["mood disorder"]},
        )
        assert result["has_drugs"] is False
        assert result["has_drug_interaction"] is False

    def test_detects_quantitative_claim(self):
        result = _tool_analyze_claim(
            "Reduces blood pressure by 10%",
            {"drugs": [], "conditions": ["hypertension"]},
        )
        assert result["has_numbers"] is True
        assert len(result["number_matches"]) > 0

    def test_detects_percentage_range(self):
        result = _tool_analyze_claim(
            "Risk reduced by 20-30%",
            {"drugs": [], "conditions": []},
        )
        assert result["has_numbers"] is True

    def test_no_numbers_in_plain_claim(self):
        result = _tool_analyze_claim(
            "Vitamin D prevents cancer",
            {"drugs": [], "conditions": ["cancer"]},
        )
        assert result["has_numbers"] is False

    def test_detects_recommendation(self):
        result = _tool_analyze_claim(
            "Patients should take metformin as first-line treatment",
            {"drugs": ["metformin"], "conditions": []},
        )
        assert result["asks_recommendation"] is True

    def test_detects_treatment_effectiveness(self):
        result = _tool_analyze_claim(
            "Metformin is effective for diabetes",
            {"drugs": ["metformin"], "conditions": ["diabetes"]},
        )
        assert result["asks_effectiveness"] is True

    def test_detects_comparison(self):
        result = _tool_analyze_claim(
            "Drug A is superior to Drug B for hypertension",
            {"drugs": ["Drug A", "Drug B"], "conditions": ["hypertension"]},
        )
        assert result["has_comparison"] is True

    def test_detects_safety_concern(self):
        result = _tool_analyze_claim(
            "Long-term aspirin use increases risk of bleeding",
            {"drugs": ["aspirin"], "conditions": []},
        )
        assert result["asks_safety"] is True

    def test_entity_count(self):
        result = _tool_analyze_claim(
            "Test claim",
            {"drugs": ["a", "b"], "conditions": ["c"], "genes": ["d"]},
        )
        assert result["entity_count"] == 4


# ---------------------------------------------------------------------------
# Tests: _tool_check_guidelines
# ---------------------------------------------------------------------------


class TestCheckGuidelineCoverage:
    """Tests for the check_guideline_coverage tool."""

    def test_matches_who_topic(self):
        result = _tool_check_guidelines(
            "Vaccination schedule for children",
            {"drugs": [], "conditions": []},
        )
        assert result["has_guideline_coverage"] is True
        assert "who" in result["matching_sources"]

    def test_matches_nih_topic(self):
        result = _tool_check_guidelines(
            "Cholesterol management in cardiovascular disease",
            {"drugs": [], "conditions": ["cardiovascular disease"]},
        )
        assert result["has_guideline_coverage"] is True
        assert "nih" in result["matching_sources"]

    def test_matches_via_entity(self):
        """Entities should contribute to matching, not just claim text."""
        result = _tool_check_guidelines(
            "This drug treats a condition",
            {"drugs": [], "conditions": ["diabetes"]},
        )
        assert result["has_guideline_coverage"] is True

    def test_no_coverage_for_obscure_topic(self):
        result = _tool_check_guidelines(
            "Quantum entanglement in photosynthesis",
            {"drugs": [], "conditions": []},
        )
        assert result["has_guideline_coverage"] is False
        assert result["matching_sources"] == {}

    def test_files_not_indexed_when_dirs_empty(self):
        """With empty guideline dirs, sources_with_indexed_files should be empty."""
        result = _tool_check_guidelines(
            "Diabetes treatment guidelines",
            {"drugs": [], "conditions": ["diabetes"]},
        )
        # Dirs exist but are empty in dev setup
        assert result["sources_with_indexed_files"] == []


# ---------------------------------------------------------------------------
# Tests: _tool_assign_methods
# ---------------------------------------------------------------------------


class TestAssignMethods:
    """Tests for the assign_methods tool."""

    def test_valid_assignment(self):
        assignments: dict[str, list[str]] = {}
        result = _tool_assign_methods(
            sub_claim_id="sc-1",
            methods=["pubmed_api", "semantic_scholar", "cross_encoder"],
            reasoning="Standard search",
            assignments=assignments,
            all_sub_claim_ids={"sc-1", "sc-2"},
        )
        assert result["success"] is True
        assert result["assigned"] is True
        assert assignments["sc-1"] == ["pubmed_api", "semantic_scholar", "cross_encoder"]
        assert result["remaining_sub_claims"] == ["sc-2"]

    def test_all_assigned_flag(self):
        assignments: dict[str, list[str]] = {"sc-1": ["pubmed_api"]}
        result = _tool_assign_methods(
            sub_claim_id="sc-2",
            methods=["pubmed_api"],
            reasoning="Final",
            assignments=assignments,
            all_sub_claim_ids={"sc-1", "sc-2"},
        )
        assert result["all_assigned"] is True
        assert result["remaining_sub_claims"] == []

    def test_rejects_invalid_method(self):
        assignments: dict[str, list[str]] = {}
        result = _tool_assign_methods(
            sub_claim_id="sc-1",
            methods=["pubmed_api", "google_search"],
            reasoning="Test",
            assignments=assignments,
            all_sub_claim_ids={"sc-1"},
        )
        assert result["success"] is False
        assert "google_search" in result["error"]
        assert "sc-1" not in assignments

    def test_rejects_empty_methods(self):
        assignments: dict[str, list[str]] = {}
        result = _tool_assign_methods(
            sub_claim_id="sc-1",
            methods=[],
            reasoning="Test",
            assignments=assignments,
            all_sub_claim_ids={"sc-1"},
        )
        assert result["success"] is False

    def test_rejects_unknown_subclaim(self):
        assignments: dict[str, list[str]] = {}
        result = _tool_assign_methods(
            sub_claim_id="sc-99",
            methods=["pubmed_api"],
            reasoning="Test",
            assignments=assignments,
            all_sub_claim_ids={"sc-1"},
        )
        assert result["success"] is False
        assert "sc-99" in result["error"]


# ---------------------------------------------------------------------------
# Tests: _execute_tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Tests for the tool dispatcher."""

    def test_dispatches_analyze(self):
        result = _execute_tool(
            "analyze_claim_characteristics",
            {"sub_claim_text": "Aspirin treats pain"},
            {"drugs": ["aspirin"], "conditions": ["pain"]},
            {},
            {"sc-1"},
        )
        assert "has_drugs" in result

    def test_dispatches_guidelines(self):
        result = _execute_tool(
            "check_guideline_coverage",
            {"sub_claim_text": "Diabetes treatment"},
            {"drugs": [], "conditions": ["diabetes"]},
            {},
            {"sc-1"},
        )
        assert "has_guideline_coverage" in result

    def test_dispatches_assign(self):
        assignments: dict[str, list[str]] = {}
        result = _execute_tool(
            "assign_methods",
            {"sub_claim_id": "sc-1", "methods": ["pubmed_api"], "reasoning": "test"},
            {},
            assignments,
            {"sc-1"},
        )
        assert result["success"] is True

    def test_unknown_tool(self):
        result = _execute_tool("nonexistent_tool", {}, {}, {}, set())
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: Rule-based fallback
# ---------------------------------------------------------------------------


class TestRuleBasedPlanner:
    """Tests for _plan_rule_based."""

    def test_basic_claim_gets_defaults(self):
        state = _make_state("Vitamin C prevents colds")
        plan = _plan_rule_based(state)
        assert "sc-1" in plan
        for method in DEFAULT_METHODS:
            assert method in plan["sc-1"]

    def test_drug_interaction_adds_drugbank(self):
        state = _make_state(
            "Aspirin interacts with warfarin",
            entities={"drugs": ["aspirin", "warfarin"], "conditions": []},
        )
        plan = _plan_rule_based(state)
        assert "drugbank_api" in plan["sc-1"]

    def test_recommendation_adds_guidelines(self):
        state = _make_state("Patients should take statins")
        plan = _plan_rule_based(state)
        assert "guideline_store" in plan["sc-1"]
        assert "cochrane_api" in plan["sc-1"]

    def test_effectiveness_adds_trials(self):
        state = _make_state("Metformin is effective for diabetes")
        plan = _plan_rule_based(state)
        assert "clinical_trials" in plan["sc-1"]
        assert "cochrane_api" in plan["sc-1"]

    def test_quantitative_adds_deep_search(self):
        state = _make_state("Reduces risk by 50%")
        plan = _plan_rule_based(state)
        assert "deep_search" in plan["sc-1"]

    def test_safety_adds_trials_and_cochrane(self):
        state = _make_state("Aspirin has adverse effects on bleeding")
        plan = _plan_rule_based(state)
        assert "clinical_trials" in plan["sc-1"]
        assert "cochrane_api" in plan["sc-1"]

    def test_guideline_topic_adds_guideline_store(self):
        state = _make_state(
            "Diabetes management",
            entities={"drugs": [], "conditions": ["diabetes"]},
        )
        plan = _plan_rule_based(state)
        assert "guideline_store" in plan["sc-1"]

    def test_multiple_subclaims(self):
        state = _make_state(
            "Drug A treats X and Drug B interacts with C",
            sub_claims=[
                SubClaim(id="sc-1", text="Drug A treats condition X"),
                SubClaim(id="sc-2", text="Drug B interacts with drug C"),
            ],
            entities={"drugs": ["Drug A", "Drug B", "drug C"], "conditions": ["condition X"]},
        )
        plan = _plan_rule_based(state)
        assert "sc-1" in plan
        assert "sc-2" in plan
        assert "drugbank_api" in plan["sc-2"]

    def test_all_methods_are_valid(self):
        """Ensure rule-based planner only assigns valid methods."""
        state = _make_state(
            "Patients should take metformin which is effective and reduces HbA1c by 1.5%",
            entities={"drugs": ["metformin"], "conditions": ["diabetes"]},
        )
        plan = _plan_rule_based(state)
        for methods in plan.values():
            for m in methods:
                assert m in VALID_METHODS, f"Invalid method: {m}"


# ---------------------------------------------------------------------------
# Tests: User message builder
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for _build_user_message."""

    def test_includes_claim(self):
        state = _make_state("Vitamin D prevents cancer")
        msg = _build_user_message(state)
        assert "Vitamin D prevents cancer" in msg

    def test_includes_subclaim_ids(self):
        state = _make_state(
            "Test",
            sub_claims=[
                SubClaim(id="sc-1", text="First sub-claim"),
                SubClaim(id="sc-2", text="Second sub-claim"),
            ],
        )
        msg = _build_user_message(state)
        assert "sc-1" in msg
        assert "sc-2" in msg

    def test_includes_entities(self):
        state = _make_state(
            "Aspirin treats pain",
            entities={"drugs": ["aspirin"], "conditions": ["pain"]},
        )
        msg = _build_user_message(state)
        assert "aspirin" in msg
        assert "pain" in msg

    def test_includes_pico(self):
        state = _make_state(
            "Test",
            pico=PICO(
                population="adults",
                intervention="exercise",
                comparison="sedentary",
                outcome="weight loss",
            ),
        )
        msg = _build_user_message(state)
        assert "adults" in msg
        assert "exercise" in msg


# ---------------------------------------------------------------------------
# Tests: run_retrieval_planner (integration-level, mocked API)
# ---------------------------------------------------------------------------


class TestRunRetrievalPlanner:
    """Tests for the main entry point."""

    def test_rule_based_when_no_api_key(self):
        """Without API key, should use rule-based fallback."""
        state = _make_state("Vitamin D prevents cancer")

        with patch("src.agents.retrieval_planner.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_retrieval_planner(state))

        assert "sc-1" in result["retrieval_plan"]
        assert result["total_cost_usd"] == 0.0
        # Trace should be recorded
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0].agent == "retrieval_planner"
        assert result["agent_trace"][0].node_type == "agent"

    def test_unassigned_subclaims_get_defaults(self):
        """Sub-claims not covered by the planner get default methods."""
        state = _make_state(
            "Complex claim",
            sub_claims=[
                SubClaim(id="sc-1", text="First"),
                SubClaim(id="sc-2", text="Second"),
            ],
        )

        # Mock ReAct to only assign sc-1
        mock_result = (
            {"sc-1": ["pubmed_api"]},
            0.001,
            [],
            1,
        )
        with patch("src.agents.retrieval_planner.ANTHROPIC_API_KEY", "test-key"):
            with patch(
                "src.agents.retrieval_planner._plan_with_react",
                return_value=mock_result,
            ):
                result = asyncio.run(run_retrieval_planner(state))

        # sc-2 should get defaults
        assert result["retrieval_plan"]["sc-1"] == ["pubmed_api"]
        assert result["retrieval_plan"]["sc-2"] == list(DEFAULT_METHODS)

    def test_falls_back_on_react_failure(self):
        """If ReAct raises an exception, should fall back to rule-based."""
        state = _make_state("Aspirin treats pain")

        with patch("src.agents.retrieval_planner.ANTHROPIC_API_KEY", "test-key"):
            with patch(
                "src.agents.retrieval_planner._plan_with_react",
                side_effect=Exception("API error"),
            ):
                result = asyncio.run(run_retrieval_planner(state))

        # Should still have a plan via fallback
        assert "sc-1" in result["retrieval_plan"]
        assert result["total_cost_usd"] == 0.0

    def test_cost_and_trace_accumulation(self):
        """Cost and traces should accumulate from prior state."""
        from src.graph.state import AgentTrace

        prior_trace = AgentTrace(
            agent="decomposer",
            node_type="function",
            duration_seconds=1.0,
            cost_usd=0.005,
            input_summary="test",
            output_summary="test",
            success=True,
        )
        state = _make_state("Test claim")
        state["agent_trace"] = [prior_trace]
        state["total_cost_usd"] = 0.005
        state["total_duration_seconds"] = 1.0

        with patch("src.agents.retrieval_planner.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_retrieval_planner(state))

        assert len(result["agent_trace"]) == 2
        assert result["agent_trace"][0].agent == "decomposer"
        assert result["agent_trace"][1].agent == "retrieval_planner"
        assert result["total_cost_usd"] >= 0.005

    def test_empty_subclaims(self):
        """Should handle empty sub-claims list gracefully."""
        state = _make_state("Empty test", sub_claims=[])

        with patch("src.agents.retrieval_planner.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_retrieval_planner(state))

        assert result["retrieval_plan"] == {}


# ---------------------------------------------------------------------------
# Tests: ReAct loop (mocked Anthropic client)
# ---------------------------------------------------------------------------


class TestReActLoop:
    """Tests for _plan_with_react with mocked Anthropic API."""

    def test_react_processes_tool_calls(self):
        """ReAct loop should process tool calls and build assignments."""
        from src.agents.retrieval_planner import _plan_with_react

        state = _make_state(
            "Aspirin treats headaches",
            entities={"drugs": ["aspirin"], "conditions": ["headaches"]},
        )

        # Build mock responses: first call returns tool_use, second returns end_turn
        mock_analyze_block = MagicMock()
        mock_analyze_block.type = "tool_use"
        mock_analyze_block.name = "analyze_claim_characteristics"
        mock_analyze_block.input = {"sub_claim_text": "Aspirin treats headaches"}
        mock_analyze_block.id = "call_1"

        mock_assign_block = MagicMock()
        mock_assign_block.type = "tool_use"
        mock_assign_block.name = "assign_methods"
        mock_assign_block.input = {
            "sub_claim_id": "sc-1",
            "methods": ["pubmed_api", "semantic_scholar", "cross_encoder"],
            "reasoning": "Standard search for treatment claim",
        }
        mock_assign_block.id = "call_2"

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "All sub-claims have been assigned."

        # First response: analyze tool call
        resp1 = MagicMock()
        resp1.content = [mock_analyze_block]
        resp1.usage = MagicMock(input_tokens=500, output_tokens=100)

        # Second response: assign tool call
        resp2 = MagicMock()
        resp2.content = [mock_assign_block]
        resp2.usage = MagicMock(input_tokens=600, output_tokens=120)

        # Third response: text only (end turn)
        resp3 = MagicMock()
        resp3.content = [mock_text_block]
        resp3.usage = MagicMock(input_tokens=700, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp1, resp2, resp3]

        with patch("src.agents.retrieval_planner.anthropic.Anthropic", return_value=mock_client):
            assignments, cost, tool_calls, steps = _plan_with_react(state)

        assert "sc-1" in assignments
        assert assignments["sc-1"] == ["pubmed_api", "semantic_scholar", "cross_encoder"]
        assert cost > 0
        assert len(tool_calls) == 2  # analyze + assign
        assert steps == 2  # loop breaks after assign (all sub-claims assigned)

    def test_react_stops_when_all_assigned(self):
        """ReAct loop should break early when all sub-claims are assigned."""
        from src.agents.retrieval_planner import _plan_with_react

        state = _make_state("Test claim")

        mock_assign_block = MagicMock()
        mock_assign_block.type = "tool_use"
        mock_assign_block.name = "assign_methods"
        mock_assign_block.input = {
            "sub_claim_id": "sc-1",
            "methods": ["pubmed_api"],
            "reasoning": "Quick",
        }
        mock_assign_block.id = "call_1"

        resp1 = MagicMock()
        resp1.content = [mock_assign_block]
        resp1.usage = MagicMock(input_tokens=500, output_tokens=100)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp1]

        with patch("src.agents.retrieval_planner.anthropic.Anthropic", return_value=mock_client):
            assignments, cost, tool_calls, steps = _plan_with_react(state)

        assert assignments["sc-1"] == ["pubmed_api"]
        # Should have only made 1 API call since all assigned
        assert mock_client.messages.create.call_count == 1
