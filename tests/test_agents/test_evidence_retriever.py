"""Tests for the Evidence Retriever agent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch, call

import pytest

from systems.s4_langgraph.agents.evidence_retriever import (
    MAX_REACT_STEPS,
    MAX_RESULTS_PER_SOURCE,
    RERANK_TOP_K,
    _build_query_for_subclaim,
    _build_user_message,
    _cochrane_to_evidence,
    _deduplicate_evidence,
    _drug_to_evidence,
    _EvidenceWrapper,
    _execute_tool,
    _infer_study_type,
    _normalize_title,
    _pubmed_to_evidence,
    _retrieve_rule_based,
    _retrieve_with_react,
    _s2_to_evidence,
    _tool_lookup_drug_info,
    _tool_mark_complete,
    _tool_rerank_evidence,
    _tool_search_clinical_trials,
    _tool_search_cochrane,
    _tool_search_pubmed,
    _tool_search_semantic_scholar,
    _trial_to_evidence,
    run_evidence_retriever,
)
from src.models import AgentTrace, Evidence, PICO, SubClaim
from src.retrieval.pubmed_client import PubMedArticle
from src.retrieval.semantic_scholar import S2Paper
from src.retrieval.cochrane_client import CochraneReview
from src.retrieval.clinical_trials import ClinicalTrial
from src.retrieval.drugbank_client import DrugInfo, DrugInteraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    claim: str = "Test claim",
    sub_claims: list[SubClaim] | None = None,
    entities: dict | None = None,
    pico: PICO | None = None,
    retrieval_plan: dict | None = None,
) -> dict:
    """Build a minimal FactCheckState dict for testing."""
    if sub_claims is None:
        sub_claims = [SubClaim(id="sc-1", text=claim)]
    if entities is None:
        entities = {"drugs": [], "conditions": [], "genes": [], "organisms": [], "procedures": [], "anatomical": []}
    if pico is None:
        pico = PICO()
    if retrieval_plan is None:
        retrieval_plan = {sc.id: ["pubmed_api", "semantic_scholar", "cross_encoder"] for sc in sub_claims}

    return {
        "claim": claim,
        "sub_claims": sub_claims,
        "entities": entities,
        "pico": pico,
        "retrieval_plan": retrieval_plan,
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


def _make_pubmed_article(
    pmid: str = "12345",
    title: str = "Test Article",
    abstract: str = "Test abstract content",
    publication_types: list[str] | None = None,
) -> PubMedArticle:
    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        publication_types=publication_types or [],
    )


def _make_s2_paper(
    paper_id: str = "abc123def456",
    title: str = "S2 Test Paper",
    abstract: str = "S2 abstract",
    pmid: str | None = None,
) -> S2Paper:
    external_ids = {}
    if pmid:
        external_ids["PubMed"] = pmid
    return S2Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        external_ids=external_ids,
    )


def _make_cochrane_review(
    doi: str = "10.1002/14651858.CD000001",
    title: str = "Cochrane Review",
    abstract: str = "Systematic review abstract",
) -> CochraneReview:
    return CochraneReview(doi=doi, title=title, abstract=abstract)


def _make_clinical_trial(
    nct_id: str = "NCT00000001",
    title: str = "Clinical Trial",
    brief_summary: str = "Trial summary",
    study_type: str = "INTERVENTIONAL",
) -> ClinicalTrial:
    return ClinicalTrial(
        nct_id=nct_id,
        title=title,
        brief_summary=brief_summary,
        study_type=study_type,
    )


# ---------------------------------------------------------------------------
# Tests: Query building
# ---------------------------------------------------------------------------


class TestQueryBuilding:
    """Tests for _build_query_for_subclaim."""

    def test_pubmed_uses_pico_when_available(self):
        sc = SubClaim(
            id="sc-1",
            text="Vaccine prevents measles",
            pico=PICO(population="children", intervention="measles vaccine"),
        )
        query = _build_query_for_subclaim(sc, {}, "pubmed")
        assert "children" in query
        assert "measles vaccine" in query

    def test_pubmed_falls_back_to_text(self):
        sc = SubClaim(id="sc-1", text="Vitamin D prevents cancer")
        query = _build_query_for_subclaim(sc, {}, "pubmed")
        assert query == "Vitamin D prevents cancer"

    def test_pubmed_falls_back_when_pico_incomplete(self):
        sc = SubClaim(
            id="sc-1",
            text="Some claim",
            pico=PICO(population="adults"),  # no intervention
        )
        query = _build_query_for_subclaim(sc, {}, "pubmed")
        assert query == "Some claim"

    def test_semantic_scholar_uses_text(self):
        sc = SubClaim(id="sc-1", text="Aspirin reduces heart attack risk")
        query = _build_query_for_subclaim(sc, {}, "semantic_scholar")
        assert query == "Aspirin reduces heart attack risk"

    def test_clinical_trials_uses_entities_fallback(self):
        sc = SubClaim(id="sc-1", text="Drug treats condition")
        entities = {"conditions": ["diabetes"], "drugs": ["metformin"]}
        query = _build_query_for_subclaim(sc, entities, "clinical_trials")
        assert "diabetes" in query
        assert "metformin" in query


# ---------------------------------------------------------------------------
# Tests: Evidence conversion
# ---------------------------------------------------------------------------


class TestEvidenceConversion:
    """Tests for source-specific evidence conversion."""

    def test_pubmed_to_evidence(self):
        articles = [_make_pubmed_article(pmid="99999", title="My Article")]
        evidence = _pubmed_to_evidence(articles, "sc-1")
        assert len(evidence) == 1
        assert evidence[0].id == "ev-pm-99999"
        assert evidence[0].source == "pubmed"
        assert evidence[0].pmid == "99999"
        assert evidence[0].retrieval_method == "api"

    def test_s2_to_evidence(self):
        papers = [_make_s2_paper(paper_id="abc123def456", pmid="11111")]
        evidence = _s2_to_evidence(papers, "sc-1")
        assert len(evidence) == 1
        assert evidence[0].source == "semantic_scholar"
        assert evidence[0].pmid == "11111"
        assert evidence[0].id.startswith("ev-s2-")

    def test_s2_to_evidence_no_pmid(self):
        papers = [_make_s2_paper(paper_id="xyz789")]
        evidence = _s2_to_evidence(papers, "sc-1")
        assert evidence[0].pmid is None

    def test_cochrane_to_evidence(self):
        reviews = [_make_cochrane_review()]
        evidence = _cochrane_to_evidence(reviews, "sc-1")
        assert len(evidence) == 1
        assert evidence[0].source == "cochrane"
        assert evidence[0].study_type == "systematic_review"

    def test_trial_to_evidence_rct(self):
        trials = [_make_clinical_trial(study_type="INTERVENTIONAL")]
        evidence = _trial_to_evidence(trials, "sc-1")
        assert evidence[0].study_type == "rct"

    def test_trial_to_evidence_observational(self):
        trials = [_make_clinical_trial(study_type="OBSERVATIONAL")]
        evidence = _trial_to_evidence(trials, "sc-1")
        assert evidence[0].study_type == "cohort"


# ---------------------------------------------------------------------------
# Tests: Study type inference
# ---------------------------------------------------------------------------


class TestStudyTypeInference:
    """Tests for _infer_study_type."""

    def test_rct(self):
        assert _infer_study_type(["Randomized Controlled Trial"]) == "rct"

    def test_meta_analysis(self):
        assert _infer_study_type(["Meta-Analysis"]) == "meta_analysis"

    def test_systematic_review(self):
        assert _infer_study_type(["Systematic Review"]) == "systematic_review"

    def test_guideline(self):
        assert _infer_study_type(["Practice Guideline"]) == "guideline"

    def test_unknown_type(self):
        assert _infer_study_type(["Journal Article"]) == "unknown"

    def test_multiple_types_picks_highest_priority(self):
        """When multiple types match, should pick the highest priority."""
        result = _infer_study_type(["Randomized Controlled Trial", "Meta-Analysis"])
        assert result == "meta_analysis"  # meta_analysis > rct in priority

    def test_empty_list(self):
        assert _infer_study_type([]) == "unknown"


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for _deduplicate_evidence."""

    def test_dedup_by_pmid(self):
        ev1 = Evidence(id="ev-1", source="pubmed", retrieval_method="api",
                       title="Title A", content="...", pmid="12345")
        ev2 = Evidence(id="ev-2", source="semantic_scholar", retrieval_method="api",
                       title="Title A from S2", content="...", pmid="12345")
        result = _deduplicate_evidence([ev1, ev2])
        assert len(result) == 1
        assert result[0].source == "pubmed"  # prefer pubmed

    def test_dedup_by_title(self):
        ev1 = Evidence(id="ev-1", source="pubmed", retrieval_method="api",
                       title="My Research Paper", content="...")
        ev2 = Evidence(id="ev-2", source="semantic_scholar", retrieval_method="api",
                       title="My Research Paper", content="...")
        result = _deduplicate_evidence([ev1, ev2])
        assert len(result) == 1

    def test_no_false_dedup(self):
        ev1 = Evidence(id="ev-1", source="pubmed", retrieval_method="api",
                       title="Paper A", content="...", pmid="111")
        ev2 = Evidence(id="ev-2", source="pubmed", retrieval_method="api",
                       title="Paper B", content="...", pmid="222")
        result = _deduplicate_evidence([ev1, ev2])
        assert len(result) == 2

    def test_source_priority(self):
        """PubMed should be preferred over S2 for same PMID."""
        ev_s2 = Evidence(id="ev-s2", source="semantic_scholar", retrieval_method="api",
                         title="Same Paper", content="short", pmid="999")
        ev_pm = Evidence(id="ev-pm", source="pubmed", retrieval_method="api",
                         title="Same Paper", content="longer abstract with details", pmid="999")
        # S2 first in list
        result = _deduplicate_evidence([ev_s2, ev_pm])
        assert len(result) == 1
        assert result[0].source == "pubmed"


# ---------------------------------------------------------------------------
# Tests: Tool functions
# ---------------------------------------------------------------------------


class TestToolFunctions:
    """Tests for individual tool implementations."""

    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_search_pubmed(self, mock_search):
        mock_search.return_value = [_make_pubmed_article()]
        sc = SubClaim(id="sc-1", text="Test claim")
        collected: dict[str, list[Evidence]] = {}
        result = _tool_search_pubmed(sc, {}, collected)
        assert result["success"] is True
        assert result["results_found"] == 1
        assert len(collected["sc-1"]) == 1

    @patch("systems.s4_langgraph.agents.evidence_retriever.s2_search")
    def test_search_semantic_scholar(self, mock_search):
        mock_search.return_value = [_make_s2_paper()]
        sc = SubClaim(id="sc-1", text="Test claim")
        collected: dict[str, list[Evidence]] = {}
        result = _tool_search_semantic_scholar(sc, {}, collected)
        assert result["success"] is True
        assert result["results_found"] == 1

    @patch("systems.s4_langgraph.agents.evidence_retriever.search_cochrane")
    def test_search_cochrane(self, mock_search):
        mock_search.return_value = [_make_cochrane_review()]
        sc = SubClaim(id="sc-1", text="Test claim")
        collected: dict[str, list[Evidence]] = {}
        result = _tool_search_cochrane(sc, {}, collected)
        assert result["success"] is True
        assert result["evidence_added"] == 1

    @patch("systems.s4_langgraph.agents.evidence_retriever.ct_search")
    def test_search_clinical_trials(self, mock_search):
        mock_search.return_value = [_make_clinical_trial()]
        sc = SubClaim(id="sc-1", text="Test claim")
        collected: dict[str, list[Evidence]] = {}
        result = _tool_search_clinical_trials(sc, {}, collected)
        assert result["success"] is True
        assert result["results_found"] == 1

    @patch("systems.s4_langgraph.agents.evidence_retriever.ct_search_pico")
    def test_search_clinical_trials_with_pico(self, mock_search):
        mock_search.return_value = [_make_clinical_trial()]
        sc = SubClaim(
            id="sc-1", text="Test claim",
            pico=PICO(population="adults", intervention="drug A"),
        )
        collected: dict[str, list[Evidence]] = {}
        result = _tool_search_clinical_trials(sc, {}, collected)
        assert result["success"] is True
        mock_search.assert_called_once()

    @patch("systems.s4_langgraph.agents.evidence_retriever.get_interactions")
    @patch("systems.s4_langgraph.agents.evidence_retriever.search_drug_label")
    def test_lookup_drug_info(self, mock_label, mock_interactions):
        mock_label.return_value = [DrugInfo(name="aspirin", indications="Pain relief")]
        mock_interactions.return_value = [
            DrugInteraction(drug_a="aspirin", drug_b="warfarin", description="Increased bleeding risk")
        ]
        collected: dict[str, list[Evidence]] = {}
        result = _tool_lookup_drug_info("aspirin", collected, "sc-1")
        assert result["success"] is True
        assert result["labels_found"] == 1
        assert result["interactions_found"] == 1
        assert len(collected["sc-1"]) == 2

    @patch("systems.s4_langgraph.agents.evidence_retriever.rerank_papers")
    def test_rerank_evidence(self, mock_rerank):
        ev1 = Evidence(id="ev-1", source="pubmed", retrieval_method="api",
                       title="Paper A", content="Abstract A")
        ev2 = Evidence(id="ev-2", source="pubmed", retrieval_method="api",
                       title="Paper B", content="Abstract B")

        # Mock returns wrappers with scores
        mock_rerank.return_value = [
            (_EvidenceWrapper(ev2), 0.95),
            (_EvidenceWrapper(ev1), 0.60),
        ]

        sc = SubClaim(id="sc-1", text="Test query")
        collected = {"sc-1": [ev1, ev2]}
        result = _tool_rerank_evidence(sc, collected)
        assert result["success"] is True
        assert result["reranked"] == 2
        assert collected["sc-1"][0].quality_score == 0.95

    def test_mark_retrieval_complete(self):
        ev = Evidence(id="ev-1", source="pubmed", retrieval_method="api",
                      title="Paper", content="...")
        collected = {"sc-1": [ev]}
        completed: set[str] = set()
        result = _tool_mark_complete("sc-1", collected, completed, {"sc-1", "sc-2"})
        assert result["success"] is True
        assert "sc-1" in completed
        assert result["remaining_sub_claims"] == ["sc-2"]
        assert result["all_complete"] is False


# ---------------------------------------------------------------------------
# Tests: Tool dispatcher
# ---------------------------------------------------------------------------


class TestToolDispatcher:
    """Tests for _execute_tool."""

    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_dispatches_search_pubmed(self, mock_search):
        mock_search.return_value = []
        sc = SubClaim(id="sc-1", text="Test")
        result = _execute_tool(
            "search_pubmed",
            {"sub_claim_id": "sc-1"},
            {"sc-1": sc}, {}, {}, set(), {"sc-1"},
        )
        assert result["success"] is True

    def test_unknown_tool(self):
        result = _execute_tool(
            "nonexistent", {}, {}, {}, {}, set(), set(),
        )
        assert "error" in result

    def test_unknown_subclaim_id(self):
        result = _execute_tool(
            "search_pubmed",
            {"sub_claim_id": "sc-99"},
            {}, {}, {}, set(), {"sc-1"},
        )
        assert result["success"] is False

    def test_dispatches_mark_complete(self):
        collected = {"sc-1": []}
        completed: set[str] = set()
        result = _execute_tool(
            "mark_retrieval_complete",
            {"sub_claim_id": "sc-1"},
            {}, {}, collected, completed, {"sc-1"},
        )
        assert result["success"] is True

    def test_mark_complete_unknown_id(self):
        result = _execute_tool(
            "mark_retrieval_complete",
            {"sub_claim_id": "sc-99"},
            {}, {}, {}, set(), {"sc-1"},
        )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Tests: Rule-based fallback
# ---------------------------------------------------------------------------


class TestRuleBasedRetrieval:
    """Tests for _retrieve_rule_based."""

    @patch("systems.s4_langgraph.agents.evidence_retriever.s2_search")
    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_calls_planned_methods(self, mock_pubmed, mock_s2):
        mock_pubmed.return_value = [_make_pubmed_article()]
        mock_s2.return_value = [_make_s2_paper()]

        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api", "semantic_scholar"]},
        )
        collected = _retrieve_rule_based(state)
        assert "sc-1" in collected
        assert len(collected["sc-1"]) == 2
        mock_pubmed.assert_called_once()
        mock_s2.assert_called_once()

    def test_skips_deep_search_with_warning(self, caplog):
        state = _make_state(
            retrieval_plan={"sc-1": ["deep_search"]},
        )
        import logging
        with caplog.at_level(logging.WARNING):
            collected = _retrieve_rule_based(state)
        assert "not yet implemented" in caplog.text
        assert collected["sc-1"] == []

    def test_skips_guideline_store_with_warning(self, caplog):
        state = _make_state(
            retrieval_plan={"sc-1": ["guideline_store"]},
        )
        import logging
        with caplog.at_level(logging.WARNING):
            collected = _retrieve_rule_based(state)
        assert "not yet implemented" in caplog.text

    @patch("systems.s4_langgraph.agents.evidence_retriever.rerank_papers")
    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_applies_cross_encoder(self, mock_pubmed, mock_rerank):
        article = _make_pubmed_article()
        mock_pubmed.return_value = [article]
        # rerank_papers returns (wrapper, score) tuples
        mock_rerank.side_effect = lambda q, papers, top_k=None: [
            (p, 0.9) for p in papers
        ]

        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api", "cross_encoder"]},
        )
        collected = _retrieve_rule_based(state)
        assert len(collected["sc-1"]) == 1
        assert collected["sc-1"][0].quality_score == 0.9
        mock_rerank.assert_called_once()

    @patch("systems.s4_langgraph.agents.evidence_retriever.s2_search")
    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_deduplicates_results(self, mock_pubmed, mock_s2):
        # Same paper from both sources
        mock_pubmed.return_value = [_make_pubmed_article(pmid="11111", title="Same Paper")]
        mock_s2.return_value = [_make_s2_paper(pmid="11111", title="Same Paper")]

        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api", "semantic_scholar"]},
        )
        collected = _retrieve_rule_based(state)
        assert len(collected["sc-1"]) == 1
        assert collected["sc-1"][0].source == "pubmed"

    @patch("systems.s4_langgraph.agents.evidence_retriever.get_interactions")
    @patch("systems.s4_langgraph.agents.evidence_retriever.search_drug_label")
    def test_drugbank_calls_for_each_drug(self, mock_label, mock_interactions):
        mock_label.return_value = [DrugInfo(name="test")]
        mock_interactions.return_value = []

        state = _make_state(
            entities={"drugs": ["aspirin", "warfarin"], "conditions": []},
            retrieval_plan={"sc-1": ["drugbank_api"]},
        )
        collected = _retrieve_rule_based(state)
        assert mock_label.call_count == 2


# ---------------------------------------------------------------------------
# Tests: User message builder
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    """Tests for _build_user_message."""

    def test_includes_claim(self):
        state = _make_state("Vitamin D prevents cancer")
        msg = _build_user_message(state)
        assert "Vitamin D prevents cancer" in msg

    def test_includes_retrieval_plan(self):
        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api", "semantic_scholar"]},
        )
        msg = _build_user_message(state)
        assert "pubmed_api" in msg
        assert "semantic_scholar" in msg

    def test_includes_pico(self):
        state = _make_state(
            pico=PICO(population="adults", intervention="exercise"),
        )
        msg = _build_user_message(state)
        assert "adults" in msg
        assert "exercise" in msg


# ---------------------------------------------------------------------------
# Tests: Evidence wrapper
# ---------------------------------------------------------------------------


class TestEvidenceWrapper:
    """Tests for _EvidenceWrapper."""

    def test_maps_content_to_abstract(self):
        ev = Evidence(id="ev-1", source="pubmed", retrieval_method="api",
                      title="Title", content="My content")
        wrapper = _EvidenceWrapper(ev)
        assert wrapper.title == "Title"
        assert wrapper.abstract == "My content"
        assert wrapper.evidence is ev


# ---------------------------------------------------------------------------
# Tests: Entry point
# ---------------------------------------------------------------------------


class TestRunEvidenceRetriever:
    """Tests for the main entry point."""

    @patch("systems.s4_langgraph.agents.evidence_retriever.s2_search")
    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_no_api_key_uses_rule_based(self, mock_pubmed, mock_s2):
        mock_pubmed.return_value = [_make_pubmed_article()]
        mock_s2.return_value = []

        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api", "semantic_scholar"]},
        )

        with patch("systems.s4_langgraph.agents.evidence_retriever.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_retriever(state))

        assert len(result["evidence"]) >= 1
        assert result["total_cost_usd"] == 0.0
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0].agent == "evidence_retriever"
        assert result["agent_trace"][0].node_type == "agent"

    @patch("systems.s4_langgraph.agents.evidence_retriever.s2_search")
    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_trace_accumulation(self, mock_pubmed, mock_s2):
        mock_pubmed.return_value = []
        mock_s2.return_value = []

        prior_trace = AgentTrace(
            agent="retrieval_planner",
            node_type="agent",
            duration_seconds=1.0,
            cost_usd=0.005,
            input_summary="test",
            output_summary="test",
            success=True,
        )
        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api"]},
        )
        state["agent_trace"] = [prior_trace]
        state["total_cost_usd"] = 0.005
        state["total_duration_seconds"] = 1.0

        with patch("systems.s4_langgraph.agents.evidence_retriever.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_retriever(state))

        assert len(result["agent_trace"]) == 2
        assert result["agent_trace"][0].agent == "retrieval_planner"
        assert result["agent_trace"][1].agent == "evidence_retriever"
        assert result["total_cost_usd"] >= 0.005

    def test_fallback_on_react_exception(self):
        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api"]},
        )

        with patch("systems.s4_langgraph.agents.evidence_retriever.ANTHROPIC_API_KEY", "test-key"):
            with patch(
                "systems.s4_langgraph.agents.evidence_retriever._retrieve_with_react",
                side_effect=Exception("API error"),
            ):
                with patch(
                    "systems.s4_langgraph.agents.evidence_retriever._retrieve_rule_based",
                    return_value={"sc-1": []},
                ):
                    result = asyncio.run(run_evidence_retriever(state))

        assert result["total_cost_usd"] == 0.0

    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_empty_plan(self, mock_pubmed):
        mock_pubmed.return_value = []
        state = _make_state(sub_claims=[], retrieval_plan={})

        with patch("systems.s4_langgraph.agents.evidence_retriever.ANTHROPIC_API_KEY", None):
            result = asyncio.run(run_evidence_retriever(state))

        assert result["evidence"] == []
        assert len(result["agent_trace"]) == 1

    @patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch")
    def test_evidence_linking_to_subclaims(self, mock_pubmed):
        mock_pubmed.return_value = [_make_pubmed_article(pmid="111")]

        state = _make_state(
            sub_claims=[
                SubClaim(id="sc-1", text="First claim"),
                SubClaim(id="sc-2", text="Second claim"),
            ],
            retrieval_plan={
                "sc-1": ["pubmed_api"],
                "sc-2": [],
            },
        )

        with patch("systems.s4_langgraph.agents.evidence_retriever.ANTHROPIC_API_KEY", None):
            with patch("systems.s4_langgraph.agents.evidence_retriever.s2_search", return_value=[]):
                result = asyncio.run(run_evidence_retriever(state))

        # sc-1 should have evidence IDs, sc-2 should have none
        sc1 = [sc for sc in result["sub_claims"] if sc.id == "sc-1"][0]
        sc2 = [sc for sc in result["sub_claims"] if sc.id == "sc-2"][0]
        assert len(sc1.evidence) >= 1
        assert sc2.evidence == []


# ---------------------------------------------------------------------------
# Tests: ReAct loop (mocked Anthropic)
# ---------------------------------------------------------------------------


class TestReActLoop:
    """Tests for _retrieve_with_react with mocked Anthropic API."""

    def test_search_rerank_complete_sequence(self):
        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api", "cross_encoder"]},
        )

        # Step 1: LLM calls search_pubmed
        mock_search_block = MagicMock()
        mock_search_block.type = "tool_use"
        mock_search_block.name = "search_pubmed"
        mock_search_block.input = {"sub_claim_id": "sc-1"}
        mock_search_block.id = "call_1"

        # Step 2: LLM calls rerank_evidence
        mock_rerank_block = MagicMock()
        mock_rerank_block.type = "tool_use"
        mock_rerank_block.name = "rerank_evidence"
        mock_rerank_block.input = {"sub_claim_id": "sc-1"}
        mock_rerank_block.id = "call_2"

        # Step 3: LLM calls mark_retrieval_complete
        mock_complete_block = MagicMock()
        mock_complete_block.type = "tool_use"
        mock_complete_block.name = "mark_retrieval_complete"
        mock_complete_block.input = {"sub_claim_id": "sc-1"}
        mock_complete_block.id = "call_3"

        resp1 = MagicMock()
        resp1.content = [mock_search_block]
        resp1.usage = MagicMock(input_tokens=500, output_tokens=100)

        resp2 = MagicMock()
        resp2.content = [mock_rerank_block]
        resp2.usage = MagicMock(input_tokens=600, output_tokens=100)

        resp3 = MagicMock()
        resp3.content = [mock_complete_block]
        resp3.usage = MagicMock(input_tokens=700, output_tokens=100)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp1, resp2, resp3]

        with patch("systems.s4_langgraph.agents.evidence_retriever.anthropic.Anthropic", return_value=mock_client):
            with patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch", return_value=[_make_pubmed_article()]):
                with patch("systems.s4_langgraph.agents.evidence_retriever.rerank_papers") as mock_rr:
                    mock_rr.side_effect = lambda q, papers, top_k=None: [(p, 0.9) for p in papers]
                    collected, cost, tool_calls, steps = _retrieve_with_react(state)

        assert "sc-1" in collected
        assert cost > 0
        assert len(tool_calls) == 3
        assert steps == 3  # breaks after mark_complete (all complete)

    def test_multi_subclaim(self):
        state = _make_state(
            sub_claims=[
                SubClaim(id="sc-1", text="Claim A"),
                SubClaim(id="sc-2", text="Claim B"),
            ],
            retrieval_plan={
                "sc-1": ["pubmed_api"],
                "sc-2": ["pubmed_api"],
            },
        )

        # Search sc-1
        block1 = MagicMock()
        block1.type = "tool_use"
        block1.name = "search_pubmed"
        block1.input = {"sub_claim_id": "sc-1"}
        block1.id = "c1"
        # Complete sc-1
        block2 = MagicMock()
        block2.type = "tool_use"
        block2.name = "mark_retrieval_complete"
        block2.input = {"sub_claim_id": "sc-1"}
        block2.id = "c2"
        # Search sc-2
        block3 = MagicMock()
        block3.type = "tool_use"
        block3.name = "search_pubmed"
        block3.input = {"sub_claim_id": "sc-2"}
        block3.id = "c3"
        # Complete sc-2
        block4 = MagicMock()
        block4.type = "tool_use"
        block4.name = "mark_retrieval_complete"
        block4.input = {"sub_claim_id": "sc-2"}
        block4.id = "c4"

        resps = []
        for block in [block1, block2, block3, block4]:
            r = MagicMock()
            r.content = [block]
            r.usage = MagicMock(input_tokens=500, output_tokens=100)
            resps.append(r)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = resps

        with patch("systems.s4_langgraph.agents.evidence_retriever.anthropic.Anthropic", return_value=mock_client):
            with patch("systems.s4_langgraph.agents.evidence_retriever.pubmed_search_and_fetch", return_value=[_make_pubmed_article()]):
                collected, cost, tool_calls, steps = _retrieve_with_react(state)

        assert "sc-1" in collected
        assert "sc-2" in collected
        assert steps == 4

    def test_cost_tracking(self):
        state = _make_state(
            retrieval_plan={"sc-1": ["pubmed_api"]},
        )

        # One tool call, then text (done)
        block = MagicMock()
        block.type = "tool_use"
        block.name = "mark_retrieval_complete"
        block.input = {"sub_claim_id": "sc-1"}
        block.id = "c1"
        resp1 = MagicMock()
        resp1.content = [block]
        resp1.usage = MagicMock(input_tokens=1000, output_tokens=200)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp1]

        with patch("systems.s4_langgraph.agents.evidence_retriever.anthropic.Anthropic", return_value=mock_client):
            collected, cost, tool_calls, steps = _retrieve_with_react(state)

        # Cost = (1000 * 3.0 + 200 * 15.0) / 1_000_000 = 0.006
        expected_cost = (1000 * 3.0 + 200 * 15.0) / 1_000_000
        assert abs(cost - expected_cost) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Drug evidence conversion
# ---------------------------------------------------------------------------


class TestDrugToEvidence:
    """Tests for _drug_to_evidence."""

    def test_drug_label_evidence(self):
        drugs = [DrugInfo(
            name="aspirin",
            generic_name="acetylsalicylic acid",
            indications="Pain relief",
            contraindications="Bleeding disorders",
        )]
        evidence = _drug_to_evidence(drugs, [], "aspirin")
        assert len(evidence) == 1
        assert evidence[0].source == "drugbank"
        assert "Pain relief" in evidence[0].content
        assert "Bleeding disorders" in evidence[0].content

    def test_drug_interaction_evidence(self):
        interactions = [DrugInteraction(
            drug_a="aspirin",
            drug_b="warfarin",
            description="Increased bleeding risk",
            severity="high",
        )]
        evidence = _drug_to_evidence([], interactions, "aspirin")
        assert len(evidence) == 1
        assert "Increased bleeding risk" in evidence[0].content
        assert "high" in evidence[0].content

    def test_combined_drug_and_interaction(self):
        drugs = [DrugInfo(name="aspirin")]
        interactions = [DrugInteraction(
            drug_a="aspirin", drug_b="warfarin", description="Risk")]
        evidence = _drug_to_evidence(drugs, interactions, "aspirin")
        assert len(evidence) == 2


# ---------------------------------------------------------------------------
# Tests: Title normalization
# ---------------------------------------------------------------------------


class TestNormalizeTitle:
    """Tests for _normalize_title."""

    def test_strips_punctuation(self):
        assert _normalize_title("Hello, World!") == "helloworld"

    def test_lowercase(self):
        assert _normalize_title("UPPERCASE") == "uppercase"

    def test_handles_empty(self):
        assert _normalize_title("") == ""
