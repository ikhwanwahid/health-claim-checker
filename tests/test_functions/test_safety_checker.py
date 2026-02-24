"""Tests for the Safety Checker function node."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.functions.safety_checker import (
    CRITICAL_FLAGS,
    SAFETY_CATEGORIES,
    _check_rule_based,
    _check_with_llm,
    _DANGEROUS_DOSAGE_PATTERN,
    run_safety_checker,
)
from src.models import AgentTrace, Evidence, PICO, SubClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    claim: str = "Vitamin D is good for bone health",
    verdict: str = "SUPPORTED",
    sub_claims: list[SubClaim] | None = None,
    evidence: list[Evidence] | None = None,
) -> dict:
    """Build a minimal FactCheckState dict for testing."""
    if sub_claims is None:
        sub_claims = [
            SubClaim(id="sc-1", text="Vitamin D supports bone health", verdict="SUPPORTED"),
        ]
    if evidence is None:
        evidence = []

    return {
        "claim": claim,
        "sub_claims": sub_claims,
        "entities": {},
        "pico": PICO(),
        "retrieval_plan": {},
        "evidence": evidence,
        "extracted_figures": [],
        "evidence_quality": {},
        "verdict": verdict,
        "confidence": 0.8,
        "explanation": "Evidence supports this claim.",
        "safety_flags": [],
        "is_dangerous": False,
        "agent_trace": [],
        "total_cost_usd": 0.0,
        "total_duration_seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# Tests: Rule-based detection
# ---------------------------------------------------------------------------


class TestRuleBasedDetection:
    """Tests for _check_rule_based."""

    def test_stop_medication_keywords(self):
        flags, is_dangerous = _check_rule_based(
            "You should stop taking your blood pressure medication",
            "REFUTED", [],
        )
        assert "stop_medication" in flags
        assert is_dangerous is True

    def test_anti_vaccination_keywords(self):
        flags, is_dangerous = _check_rule_based(
            "Vaccines cause autism in children",
            "REFUTED", [],
        )
        assert "anti_vaccination" in flags
        assert is_dangerous is False  # anti_vaccination is not critical

    def test_vulnerable_population_with_bad_verdict(self):
        flags, _ = _check_rule_based(
            "This supplement is safe for pregnant women",
            "REFUTED", [],
        )
        assert "vulnerable_population" in flags

    def test_vulnerable_population_with_good_verdict(self):
        flags, _ = _check_rule_based(
            "This supplement is safe for pregnant women",
            "SUPPORTED", [],
        )
        assert "vulnerable_population" not in flags

    def test_dangerous_dosage_regex(self):
        flags, is_dangerous = _check_rule_based(
            "Take 50000 mg of vitamin C daily",
            "NOT_SUPPORTED", [],
        )
        assert "dangerous_dosage" in flags
        assert is_dangerous is True

    def test_dangerous_dosage_megadose(self):
        flags, is_dangerous = _check_rule_based(
            "A megadose of zinc will cure your cold",
            "NOT_SUPPORTED", [],
        )
        assert "dangerous_dosage" in flags
        assert is_dangerous is True

    def test_no_flags_benign_claim(self):
        flags, is_dangerous = _check_rule_based(
            "Regular exercise improves cardiovascular health",
            "SUPPORTED", [],
        )
        assert flags == []
        assert is_dangerous is False

    def test_unproven_alternative(self):
        flags, is_dangerous = _check_rule_based(
            "Use this natural cure instead of chemotherapy",
            "REFUTED", [],
        )
        assert "unproven_alternative" in flags
        assert is_dangerous is True

    def test_delay_care(self):
        flags, is_dangerous = _check_rule_based(
            "You don't go to the doctor for this condition",
            "NOT_SUPPORTED", [],
        )
        assert "delay_care" in flags
        assert is_dangerous is True

    def test_multiple_flags(self):
        flags, is_dangerous = _check_rule_based(
            "Don't vaccinate your children, use a natural cure instead of vaccines",
            "REFUTED", [],
        )
        assert "anti_vaccination" in flags
        assert "unproven_alternative" in flags


# ---------------------------------------------------------------------------
# Tests: LLM path (mocked)
# ---------------------------------------------------------------------------


class TestLLMPath:
    """Tests for _check_with_llm with mocked Anthropic."""

    def test_returns_parsed_flags(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "flags": ["anti_vaccination"],
            "is_dangerous": False,
            "reasoning": "Claim questions vaccine safety",
        }))]
        mock_response.usage = MagicMock(input_tokens=200, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("src.functions.safety_checker.anthropic.Anthropic", return_value=mock_client):
            flags, is_dangerous, reasoning, cost = _check_with_llm(
                "Vaccines cause autism", "REFUTED", [], [],
            )

        assert flags == ["anti_vaccination"]
        assert is_dangerous is False  # anti_vaccination is not critical
        assert cost > 0

    def test_handles_json_parse_failure(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON at all")]
        mock_response.usage = MagicMock(input_tokens=200, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("src.functions.safety_checker.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(json.JSONDecodeError):
                _check_with_llm("Test claim", "SUPPORTED", [], [])

    def test_handles_api_exception(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        with patch("src.functions.safety_checker.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(Exception, match="API error"):
                _check_with_llm("Test claim", "SUPPORTED", [], [])


# ---------------------------------------------------------------------------
# Tests: Entry point
# ---------------------------------------------------------------------------


class TestRunSafetyChecker:
    """Tests for the main entry point."""

    def test_no_api_key_uses_rule_based(self):
        state = _make_state(
            claim="Vaccines cause autism in children",
            verdict="REFUTED",
        )

        with patch("src.functions.safety_checker.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_safety_checker(state))

        assert "anti_vaccination" in result["safety_flags"]
        assert result["is_dangerous"] is False
        assert result["total_cost_usd"] == 0.0
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0].agent == "safety_checker"
        assert result["agent_trace"][0].node_type == "function"

    def test_trace_accumulation(self):
        prior_trace = AgentTrace(
            agent="verdict_agent",
            node_type="agent",
            duration_seconds=3.0,
            cost_usd=0.02,
            input_summary="test",
            output_summary="test",
            success=True,
        )
        state = _make_state()
        state["agent_trace"] = [prior_trace]
        state["total_cost_usd"] = 0.02
        state["total_duration_seconds"] = 3.0

        with patch("src.functions.safety_checker.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_safety_checker(state))

        assert len(result["agent_trace"]) == 2
        assert result["agent_trace"][0].agent == "verdict_agent"
        assert result["agent_trace"][1].agent == "safety_checker"
        assert result["total_cost_usd"] >= 0.02

    def test_is_dangerous_critical_flag(self):
        state = _make_state(
            claim="Stop taking your heart medication immediately",
            verdict="REFUTED",
        )

        with patch("src.functions.safety_checker.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_safety_checker(state))

        assert "stop_medication" in result["safety_flags"]
        assert result["is_dangerous"] is True

    def test_is_dangerous_non_critical_flag(self):
        state = _make_state(
            claim="Vaccines cause problems in children",
            verdict="REFUTED",
        )

        with patch("src.functions.safety_checker.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_safety_checker(state))

        # "vaccines cause" matches anti_vaccination, "children" + REFUTED matches vulnerable_population
        # Neither is a critical flag
        assert result["is_dangerous"] is False

    def test_empty_claim(self):
        state = _make_state(claim="", verdict="")

        with patch("src.functions.safety_checker.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_safety_checker(state))

        assert result["safety_flags"] == []
        assert result["is_dangerous"] is False

    def test_fallback_on_llm_exception(self):
        state = _make_state(
            claim="Vaccines cause autism in children",
            verdict="REFUTED",
        )

        with patch("src.functions.safety_checker.ANTHROPIC_API_KEY", "test-key"):
            with patch(
                "src.functions.safety_checker._check_with_llm",
                side_effect=Exception("API error"),
            ):
                result = asyncio.run(run_safety_checker(state))

        # Should fall back to rule-based
        assert "anti_vaccination" in result["safety_flags"]
        assert len(result["agent_trace"]) == 1
