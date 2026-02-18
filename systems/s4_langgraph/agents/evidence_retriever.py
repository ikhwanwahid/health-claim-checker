"""Evidence Retriever — Agent (ReAct).

Executes the retrieval plan produced by the Retrieval Planner.
Uses a multi-step reasoning loop to orchestrate searches across multiple
sources, re-rank results, and produce Evidence objects.

Type: Agent (ReAct with tool use)
Model: Claude Sonnet

Tools available to this agent:
- search_pubmed: search PubMed via E-utilities API
- search_semantic_scholar: search Semantic Scholar API
- search_cochrane: search Cochrane systematic reviews
- search_clinical_trials: search ClinicalTrials.gov
- lookup_drug_info: look up drug info and interactions via OpenFDA/RxNorm
- rerank_evidence: re-rank collected evidence with cross-encoder
- mark_retrieval_complete: mark a sub-claim's retrieval as done

Input (from state):
- retrieval_plan: per-sub-claim method list from the planner
- sub_claims: the sub-claims to find evidence for
- pico: PICO elements for query construction
- entities: medical entities for query terms

Output (to state):
- evidence: list of Evidence objects with source, content, quality_score
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Optional

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from src.models import AgentTrace, Evidence, FactCheckState, SubClaim, ToolCall
from src.retrieval.pubmed_client import (
    PubMedArticle,
    build_query_from_pico,
    search_and_fetch as pubmed_search_and_fetch,
)
from src.retrieval.semantic_scholar import S2Paper, search as s2_search
from src.retrieval.cochrane_client import CochraneReview, search_cochrane
from src.retrieval.clinical_trials import (
    ClinicalTrial,
    search as ct_search,
    search_by_condition_and_intervention as ct_search_pico,
)
from src.retrieval.drugbank_client import (
    DrugInfo,
    DrugInteraction,
    search_drug_label,
    get_interactions,
)
from src.retrieval.cross_encoder import rerank_papers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REACT_STEPS = 15
MAX_RESULTS_PER_SOURCE = 15
RERANK_TOP_K = 10

# Map PubMed publication type strings to evidence hierarchy keys
_PUBMED_TYPE_MAP: dict[str, str] = {
    "Randomized Controlled Trial": "rct",
    "Clinical Trial": "rct",
    "Clinical Trial, Phase I": "rct",
    "Clinical Trial, Phase II": "rct",
    "Clinical Trial, Phase III": "rct",
    "Clinical Trial, Phase IV": "rct",
    "Controlled Clinical Trial": "rct",
    "Pragmatic Clinical Trial": "rct",
    "Meta-Analysis": "meta_analysis",
    "Systematic Review": "systematic_review",
    "Review": "systematic_review",
    "Observational Study": "cohort",
    "Cohort Study": "cohort",
    "Case-Control Study": "case_control",
    "Case Reports": "case_report",
    "Practice Guideline": "guideline",
    "Guideline": "guideline",
    "In Vitro Techniques": "in_vitro",
    "Editorial": "expert_opinion",
    "Comment": "expert_opinion",
    "Letter": "expert_opinion",
}


# ---------------------------------------------------------------------------
# Query building helpers
# ---------------------------------------------------------------------------


def _build_query_for_subclaim(
    sub_claim: SubClaim,
    entities: dict,
    source: str,
) -> str:
    """Build a search query from sub-claim context for a given source.

    Args:
        sub_claim: The sub-claim to build a query for.
        entities: Extracted medical entities.
        source: The retrieval source (pubmed, semantic_scholar, cochrane, clinical_trials).

    Returns:
        A query string appropriate for the source.
    """
    pico = sub_claim.pico

    if source == "pubmed":
        # Use PICO-based query if we have population + intervention
        if pico and pico.population and pico.intervention:
            query = build_query_from_pico(
                population=pico.population,
                intervention=pico.intervention,
                comparison=pico.comparison,
                outcome=pico.outcome,
            )
            if query:
                return query
        # Fall back to sub-claim text
        return sub_claim.text

    elif source == "semantic_scholar":
        # Natural language works best for S2
        return sub_claim.text

    elif source == "cochrane":
        # Pass population + intervention if available
        if pico and pico.population and pico.intervention:
            return f"{pico.intervention} {pico.population}"
        return sub_claim.text

    elif source == "clinical_trials":
        # Use condition + intervention from PICO/entities
        if pico and pico.population and pico.intervention:
            return f"{pico.population} {pico.intervention}"
        conditions = entities.get("conditions", [])
        drugs = entities.get("drugs", [])
        if conditions and drugs:
            return f"{conditions[0]} {drugs[0]}"
        return sub_claim.text

    return sub_claim.text


# ---------------------------------------------------------------------------
# Evidence conversion — one function per source
# ---------------------------------------------------------------------------


def _infer_study_type(publication_types: list[str]) -> str:
    """Map PubMed publication type strings to evidence hierarchy keys.

    Returns the highest-priority study type found, or 'unknown'.
    """
    # Priority order: guideline > systematic_review > meta_analysis > rct > ...
    priority = [
        "guideline", "systematic_review", "meta_analysis", "rct",
        "cohort", "case_control", "case_report", "in_vitro",
        "expert_opinion",
    ]
    found_types: set[str] = set()
    for pt in publication_types:
        mapped = _PUBMED_TYPE_MAP.get(pt)
        if mapped:
            found_types.add(mapped)

    for p in priority:
        if p in found_types:
            return p

    return "unknown"


def _pubmed_to_evidence(
    articles: list[PubMedArticle],
    sub_claim_id: str,
) -> list[Evidence]:
    """Convert PubMed articles to Evidence objects."""
    results = []
    for article in articles:
        study_type = _infer_study_type(article.publication_types)
        content = article.abstract or article.title
        results.append(Evidence(
            id=f"ev-pm-{article.pmid}",
            source="pubmed",
            retrieval_method="api",
            title=article.title,
            content=content,
            url=article.url,
            study_type=study_type,
            pmid=article.pmid,
        ))
    return results


def _s2_to_evidence(
    papers: list[S2Paper],
    sub_claim_id: str,
) -> list[Evidence]:
    """Convert Semantic Scholar papers to Evidence objects."""
    results = []
    for paper in papers:
        content = paper.abstract or paper.tldr or paper.title
        pmid = paper.pmid
        ev_id = f"ev-s2-{paper.paper_id[:12]}"
        results.append(Evidence(
            id=ev_id,
            source="semantic_scholar",
            retrieval_method="api",
            title=paper.title,
            content=content,
            url=paper.url,
            study_type="unknown",
            pmid=pmid,
        ))
    return results


def _cochrane_to_evidence(
    reviews: list[CochraneReview],
    sub_claim_id: str,
) -> list[Evidence]:
    """Convert Cochrane reviews to Evidence objects."""
    results = []
    for review in reviews:
        content = review.abstract or review.title
        ev_id = f"ev-co-{review.doi.replace('/', '-')[:20]}" if review.doi else f"ev-co-{uuid.uuid4().hex[:8]}"
        results.append(Evidence(
            id=ev_id,
            source="cochrane",
            retrieval_method="api",
            title=review.title,
            content=content,
            url=review.url if review.doi else None,
            study_type="systematic_review",
        ))
    return results


def _trial_to_evidence(
    trials: list[ClinicalTrial],
    sub_claim_id: str,
) -> list[Evidence]:
    """Convert clinical trials to Evidence objects."""
    results = []
    for trial in trials:
        content = trial.brief_summary or trial.detailed_description or trial.title
        study_type = "rct" if trial.is_rct else "cohort"
        results.append(Evidence(
            id=f"ev-ct-{trial.nct_id}",
            source="clinical_trials",
            retrieval_method="api",
            title=trial.title,
            content=content,
            url=trial.url,
            study_type=study_type,
        ))
    return results


def _drug_to_evidence(
    drugs: list[DrugInfo],
    interactions: list[DrugInteraction],
    drug_name: str,
) -> list[Evidence]:
    """Convert drug info and interactions to Evidence objects."""
    results = []
    for drug in drugs:
        sections = []
        if drug.indications:
            sections.append(f"Indications: {drug.indications}")
        if drug.contraindications:
            sections.append(f"Contraindications: {drug.contraindications}")
        if drug.warnings:
            sections.append(f"Warnings: {drug.warnings}")
        if drug.adverse_reactions:
            sections.append(f"Adverse reactions: {drug.adverse_reactions}")
        if drug.drug_interactions:
            sections.append(f"Drug interactions: {drug.drug_interactions}")
        if drug.mechanism_of_action:
            sections.append(f"Mechanism: {drug.mechanism_of_action}")

        content = " | ".join(sections) if sections else f"Drug label for {drug.name}"
        ev_id = f"ev-drug-{drug_name.lower().replace(' ', '-')[:20]}"
        results.append(Evidence(
            id=ev_id,
            source="drugbank",
            retrieval_method="api",
            title=f"FDA Label: {drug.generic_name or drug.name}",
            content=content,
            study_type="guideline",
        ))

    for interaction in interactions:
        results.append(Evidence(
            id=f"ev-dint-{interaction.drug_a[:10]}-{interaction.drug_b[:10]}".lower().replace(" ", "-"),
            source="drugbank",
            retrieval_method="api",
            title=f"Interaction: {interaction.drug_a} + {interaction.drug_b}",
            content=f"{interaction.description} (severity: {interaction.severity})" if interaction.severity else interaction.description,
            study_type="guideline",
        ))

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Normalize a title for dedup comparison."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _deduplicate_evidence(evidence_list: list[Evidence]) -> list[Evidence]:
    """Deduplicate evidence by PMID first, then by normalized title.

    When duplicates found, prefer the source with richer metadata (PubMed > S2).
    """
    source_priority = {
        "pubmed": 0,
        "cochrane": 1,
        "clinical_trials": 2,
        "drugbank": 3,
        "semantic_scholar": 4,
    }

    # Pass 1: Group by PMID
    by_pmid: dict[str, list[Evidence]] = {}
    no_pmid: list[Evidence] = []
    for ev in evidence_list:
        if ev.pmid:
            by_pmid.setdefault(ev.pmid, []).append(ev)
        else:
            no_pmid.append(ev)

    deduped: list[Evidence] = []
    for pmid, group in by_pmid.items():
        # Pick the one with highest priority (lowest number)
        best = min(group, key=lambda e: source_priority.get(e.source, 99))
        deduped.append(best)

    # Pass 2: Dedup remaining by normalized title
    seen_titles: set[str] = set()
    # Add titles of PMID-deduped entries
    for ev in deduped:
        seen_titles.add(_normalize_title(ev.title))

    for ev in no_pmid:
        norm = _normalize_title(ev.title)
        if norm and norm not in seen_titles:
            seen_titles.add(norm)
            deduped.append(ev)

    return deduped


# ---------------------------------------------------------------------------
# Cross-encoder adapter
# ---------------------------------------------------------------------------


class _EvidenceWrapper:
    """Wraps Evidence to provide .abstract attribute for rerank_papers()."""

    def __init__(self, evidence: Evidence):
        self._evidence = evidence
        self.title = evidence.title
        self.abstract = evidence.content

    @property
    def evidence(self) -> Evidence:
        return self._evidence


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_search_pubmed(
    sub_claim: SubClaim,
    entities: dict,
    collected: dict[str, list[Evidence]],
    max_results: int = MAX_RESULTS_PER_SOURCE,
) -> dict[str, Any]:
    """Search PubMed for a sub-claim."""
    query = _build_query_for_subclaim(sub_claim, entities, "pubmed")
    logger.info("PubMed search for %s: %s", sub_claim.id, query)

    try:
        articles = pubmed_search_and_fetch(query, max_results=max_results)
        evidence = _pubmed_to_evidence(articles, sub_claim.id)
        collected.setdefault(sub_claim.id, []).extend(evidence)
        return {
            "success": True,
            "source": "pubmed",
            "query": query,
            "results_found": len(articles),
            "evidence_added": len(evidence),
        }
    except Exception as e:
        logger.error("PubMed search failed for %s: %s", sub_claim.id, e)
        return {"success": False, "error": str(e)}


def _tool_search_semantic_scholar(
    sub_claim: SubClaim,
    entities: dict,
    collected: dict[str, list[Evidence]],
    max_results: int = MAX_RESULTS_PER_SOURCE,
) -> dict[str, Any]:
    """Search Semantic Scholar for a sub-claim."""
    query = _build_query_for_subclaim(sub_claim, entities, "semantic_scholar")
    logger.info("S2 search for %s: %s", sub_claim.id, query)

    try:
        papers = s2_search(query, max_results=max_results)
        evidence = _s2_to_evidence(papers, sub_claim.id)
        collected.setdefault(sub_claim.id, []).extend(evidence)
        return {
            "success": True,
            "source": "semantic_scholar",
            "query": query,
            "results_found": len(papers),
            "evidence_added": len(evidence),
        }
    except Exception as e:
        logger.error("S2 search failed for %s: %s", sub_claim.id, e)
        return {"success": False, "error": str(e)}


def _tool_search_cochrane(
    sub_claim: SubClaim,
    entities: dict,
    collected: dict[str, list[Evidence]],
    max_results: int = MAX_RESULTS_PER_SOURCE,
) -> dict[str, Any]:
    """Search Cochrane for systematic reviews."""
    pico = sub_claim.pico
    population = pico.population if pico else None
    intervention = pico.intervention if pico else None
    query = _build_query_for_subclaim(sub_claim, entities, "cochrane")
    logger.info("Cochrane search for %s: %s", sub_claim.id, query)

    try:
        reviews = search_cochrane(
            query,
            max_results=max_results,
            population=population,
            intervention=intervention,
        )
        evidence = _cochrane_to_evidence(reviews, sub_claim.id)
        collected.setdefault(sub_claim.id, []).extend(evidence)
        return {
            "success": True,
            "source": "cochrane",
            "query": query,
            "results_found": len(reviews),
            "evidence_added": len(evidence),
        }
    except Exception as e:
        logger.error("Cochrane search failed for %s: %s", sub_claim.id, e)
        return {"success": False, "error": str(e)}


def _tool_search_clinical_trials(
    sub_claim: SubClaim,
    entities: dict,
    collected: dict[str, list[Evidence]],
    max_results: int = MAX_RESULTS_PER_SOURCE,
) -> dict[str, Any]:
    """Search ClinicalTrials.gov for trials."""
    pico = sub_claim.pico
    logger.info("ClinicalTrials search for %s", sub_claim.id)

    try:
        if pico and pico.population and pico.intervention:
            trials = ct_search_pico(
                condition=pico.population,
                intervention=pico.intervention,
                max_results=max_results,
            )
        else:
            query = _build_query_for_subclaim(sub_claim, entities, "clinical_trials")
            trials = ct_search(query, max_results=max_results)

        evidence = _trial_to_evidence(trials, sub_claim.id)
        collected.setdefault(sub_claim.id, []).extend(evidence)
        return {
            "success": True,
            "source": "clinical_trials",
            "results_found": len(trials),
            "evidence_added": len(evidence),
        }
    except Exception as e:
        logger.error("ClinicalTrials search failed for %s: %s", sub_claim.id, e)
        return {"success": False, "error": str(e)}


def _tool_lookup_drug_info(
    drug_name: str,
    collected: dict[str, list[Evidence]],
    sub_claim_id: str,
) -> dict[str, Any]:
    """Look up drug label info and interactions."""
    logger.info("Drug lookup for '%s' (sub-claim %s)", drug_name, sub_claim_id)

    try:
        drugs = search_drug_label(drug_name)
        interactions = get_interactions(drug_name)
        evidence = _drug_to_evidence(drugs, interactions, drug_name)
        collected.setdefault(sub_claim_id, []).extend(evidence)
        return {
            "success": True,
            "drug_name": drug_name,
            "labels_found": len(drugs),
            "interactions_found": len(interactions),
            "evidence_added": len(evidence),
        }
    except Exception as e:
        logger.error("Drug lookup failed for '%s': %s", drug_name, e)
        return {"success": False, "error": str(e)}


def _tool_rerank_evidence(
    sub_claim: SubClaim,
    collected: dict[str, list[Evidence]],
    top_k: int = RERANK_TOP_K,
) -> dict[str, Any]:
    """Re-rank collected evidence for a sub-claim using cross-encoder."""
    sc_evidence = collected.get(sub_claim.id, [])
    if not sc_evidence:
        return {"success": True, "reranked": 0, "message": "No evidence to rerank"}

    wrappers = [_EvidenceWrapper(ev) for ev in sc_evidence]

    try:
        ranked = rerank_papers(sub_claim.text, wrappers, top_k=top_k)
        reranked_evidence = []
        for wrapper, score in ranked:
            ev = wrapper.evidence
            ev_with_score = Evidence(
                id=ev.id,
                source=ev.source,
                retrieval_method=ev.retrieval_method,
                title=ev.title,
                content=ev.content,
                url=ev.url,
                study_type=ev.study_type,
                quality_score=round(score, 4),
                pmid=ev.pmid,
            )
            reranked_evidence.append(ev_with_score)

        collected[sub_claim.id] = reranked_evidence
        return {
            "success": True,
            "reranked": len(reranked_evidence),
            "top_score": round(ranked[0][1], 4) if ranked else 0.0,
        }
    except Exception as e:
        logger.error("Reranking failed for %s: %s", sub_claim.id, e)
        return {"success": False, "error": str(e)}


def _tool_mark_complete(
    sub_claim_id: str,
    collected: dict[str, list[Evidence]],
    completed: set[str],
    all_sub_claim_ids: set[str],
) -> dict[str, Any]:
    """Mark a sub-claim's retrieval as complete."""
    # Deduplicate evidence for this sub-claim
    sc_evidence = collected.get(sub_claim_id, [])
    collected[sub_claim_id] = _deduplicate_evidence(sc_evidence)

    completed.add(sub_claim_id)
    remaining = all_sub_claim_ids - completed
    return {
        "success": True,
        "sub_claim_id": sub_claim_id,
        "evidence_count": len(collected.get(sub_claim_id, [])),
        "remaining_sub_claims": sorted(remaining),
        "all_complete": len(remaining) == 0,
    }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    sub_claims_by_id: dict[str, SubClaim],
    entities: dict,
    collected: dict[str, list[Evidence]],
    completed: set[str],
    all_sub_claim_ids: set[str],
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate function."""
    sc_id = tool_input.get("sub_claim_id", "")
    sub_claim = sub_claims_by_id.get(sc_id)

    if tool_name == "search_pubmed":
        if not sub_claim:
            return {"success": False, "error": f"Unknown sub-claim ID: {sc_id}"}
        max_results = tool_input.get("max_results", MAX_RESULTS_PER_SOURCE)
        return _tool_search_pubmed(sub_claim, entities, collected, max_results)

    elif tool_name == "search_semantic_scholar":
        if not sub_claim:
            return {"success": False, "error": f"Unknown sub-claim ID: {sc_id}"}
        max_results = tool_input.get("max_results", MAX_RESULTS_PER_SOURCE)
        return _tool_search_semantic_scholar(sub_claim, entities, collected, max_results)

    elif tool_name == "search_cochrane":
        if not sub_claim:
            return {"success": False, "error": f"Unknown sub-claim ID: {sc_id}"}
        max_results = tool_input.get("max_results", MAX_RESULTS_PER_SOURCE)
        return _tool_search_cochrane(sub_claim, entities, collected, max_results)

    elif tool_name == "search_clinical_trials":
        if not sub_claim:
            return {"success": False, "error": f"Unknown sub-claim ID: {sc_id}"}
        max_results = tool_input.get("max_results", MAX_RESULTS_PER_SOURCE)
        return _tool_search_clinical_trials(sub_claim, entities, collected, max_results)

    elif tool_name == "lookup_drug_info":
        drug_name = tool_input.get("drug_name", "")
        if not drug_name:
            return {"success": False, "error": "drug_name is required"}
        # Use the first sub_claim_id if none provided for drug lookup
        lookup_sc_id = sc_id or (sorted(all_sub_claim_ids)[0] if all_sub_claim_ids else "")
        return _tool_lookup_drug_info(drug_name, collected, lookup_sc_id)

    elif tool_name == "rerank_evidence":
        if not sub_claim:
            return {"success": False, "error": f"Unknown sub-claim ID: {sc_id}"}
        top_k = tool_input.get("top_k", RERANK_TOP_K)
        return _tool_rerank_evidence(sub_claim, collected, top_k)

    elif tool_name == "mark_retrieval_complete":
        if sc_id not in all_sub_claim_ids:
            return {"success": False, "error": f"Unknown sub-claim ID: {sc_id}"}
        return _tool_mark_complete(sc_id, collected, completed, all_sub_claim_ids)

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Anthropic tool schema definitions
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "name": "search_pubmed",
        "description": (
            "Search PubMed for biomedical literature relevant to a sub-claim. "
            "Builds a PICO-based query if PICO elements are available, otherwise "
            "uses the sub-claim text. Returns PubMed articles as evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim (e.g., 'sc-1').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to retrieve (default 15).",
                },
            },
            "required": ["sub_claim_id"],
        },
    },
    {
        "name": "search_semantic_scholar",
        "description": (
            "Search Semantic Scholar for academic papers relevant to a sub-claim. "
            "Uses the sub-claim text as a natural language query. "
            "Good complement to PubMed for broader academic coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim (e.g., 'sc-1').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to retrieve (default 15).",
                },
            },
            "required": ["sub_claim_id"],
        },
    },
    {
        "name": "search_cochrane",
        "description": (
            "Search for Cochrane systematic reviews. Use ONLY when the retrieval "
            "plan includes 'cochrane_api' for this sub-claim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim (e.g., 'sc-1').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to retrieve (default 15).",
                },
            },
            "required": ["sub_claim_id"],
        },
    },
    {
        "name": "search_clinical_trials",
        "description": (
            "Search ClinicalTrials.gov for registered clinical trials. "
            "Use ONLY when the retrieval plan includes 'clinical_trials' for "
            "this sub-claim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim (e.g., 'sc-1').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to retrieve (default 15).",
                },
            },
            "required": ["sub_claim_id"],
        },
    },
    {
        "name": "lookup_drug_info",
        "description": (
            "Look up FDA drug label information and known interactions for a "
            "specific drug. Use when the retrieval plan includes 'drugbank_api'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "The drug name to look up.",
                },
                "sub_claim_id": {
                    "type": "string",
                    "description": "The sub-claim ID to associate results with.",
                },
            },
            "required": ["drug_name"],
        },
    },
    {
        "name": "rerank_evidence",
        "description": (
            "Re-rank all collected evidence for a sub-claim using a cross-encoder "
            "model. Use when the retrieval plan includes 'cross_encoder' and "
            "AFTER search tools have been called for this sub-claim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim to re-rank evidence for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to keep (default 10).",
                },
            },
            "required": ["sub_claim_id"],
        },
    },
    {
        "name": "mark_retrieval_complete",
        "description": (
            "Mark a sub-claim's evidence retrieval as complete. This triggers "
            "deduplication and reports remaining sub-claims. Call this AFTER "
            "all search and rerank tools for a sub-claim are done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_claim_id": {
                    "type": "string",
                    "description": "The ID of the sub-claim to mark complete.",
                },
            },
            "required": ["sub_claim_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt + user message builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an evidence retrieval agent for a medical fact-checking system. Your job \
is to execute a retrieval plan by searching multiple biomedical sources and \
collecting evidence for each sub-claim.

## Available Tools

| Tool | When to use |
|------|-------------|
| search_pubmed | When plan includes 'pubmed_api' — primary biomedical literature |
| search_semantic_scholar | When plan includes 'semantic_scholar' — broad academic papers |
| search_cochrane | When plan includes 'cochrane_api' — systematic reviews |
| search_clinical_trials | When plan includes 'clinical_trials' — registered trials |
| lookup_drug_info | When plan includes 'drugbank_api' — drug labels and interactions |
| rerank_evidence | When plan includes 'cross_encoder' — re-rank AFTER searching |
| mark_retrieval_complete | After all searches + reranking for a sub-claim |

## Process

For EACH sub-claim:
1. Check its retrieval plan to see which methods are assigned
2. Call the corresponding search tools (pubmed, semantic_scholar, etc.)
3. If 'cross_encoder' is in the plan, call rerank_evidence AFTER all searches
4. Call mark_retrieval_complete when done with this sub-claim
5. Move on to the next sub-claim

## Important Rules

- Skip 'deep_search' and 'guideline_store' methods — they are not yet implemented
- Do NOT invent or fabricate evidence — only report what the tools return
- Work through ALL sub-claims systematically, then stop
- The tools handle query construction internally — just pass the sub_claim_id"""


def _build_user_message(state: FactCheckState) -> str:
    """Build the user message with sub-claims, plan, and context."""
    sub_claims = state.get("sub_claims", [])
    retrieval_plan = state.get("retrieval_plan", {})
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

    # Sub-claims with their retrieval plans
    parts.append(f"Sub-claims to retrieve evidence for ({len(sub_claims)} total):")
    for sc in sub_claims:
        methods = retrieval_plan.get(sc.id, [])
        methods_str = ", ".join(methods) if methods else "none assigned"
        sc_pico = ""
        if sc.pico:
            sc_pico = (
                f" [P={sc.pico.population}, I={sc.pico.intervention}, "
                f"C={sc.pico.comparison}, O={sc.pico.outcome}]"
            )
        parts.append(f"  {sc.id}: \"{sc.text}\"{sc_pico}")
        parts.append(f"    Retrieval plan: {methods_str}")

    parts.append("")
    parts.append(
        "Please execute the retrieval plan for each sub-claim. "
        "Use the assigned methods, rerank if specified, then mark complete."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def _retrieve_with_react(
    state: FactCheckState,
) -> tuple[dict[str, list[Evidence]], float, list[ToolCall], int]:
    """Run the ReAct loop to retrieve evidence.

    Returns:
        (collected_evidence, cost_usd, tool_calls, reasoning_steps)
    """
    sub_claims = state.get("sub_claims", [])
    entities = state.get("entities", {})
    sub_claims_by_id = {sc.id: sc for sc in sub_claims}
    all_sub_claim_ids = set(sub_claims_by_id.keys())

    # Shared mutable state
    collected: dict[str, list[Evidence]] = {}
    completed: set[str] = set()
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
            max_tokens=2048,
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
                sub_claims_by_id=sub_claims_by_id,
                entities=entities,
                collected=collected,
                completed=completed,
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

        # Check if all sub-claims complete
        if all_sub_claim_ids <= completed:
            logger.info("All sub-claims retrieved after %d steps", step + 1)
            break
    else:
        logger.warning("ReAct loop hit max steps (%d)", MAX_REACT_STEPS)

    return collected, total_cost, tool_call_records, reasoning_steps


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _retrieve_rule_based(state: FactCheckState) -> dict[str, list[Evidence]]:
    """Deterministic retrieval following the plan (no API key needed).

    For each sub-claim, iterates its planned methods and calls the
    corresponding client directly.
    """
    sub_claims = state.get("sub_claims", [])
    retrieval_plan = state.get("retrieval_plan", {})
    entities = state.get("entities", {})
    collected: dict[str, list[Evidence]] = {}

    for sc in sub_claims:
        methods = retrieval_plan.get(sc.id, [])
        sc_evidence: list[Evidence] = []

        for method in methods:
            if method == "pubmed_api":
                query = _build_query_for_subclaim(sc, entities, "pubmed")
                try:
                    articles = pubmed_search_and_fetch(query, max_results=MAX_RESULTS_PER_SOURCE)
                    sc_evidence.extend(_pubmed_to_evidence(articles, sc.id))
                except Exception as e:
                    logger.warning("PubMed search failed for %s: %s", sc.id, e)

            elif method == "semantic_scholar":
                query = _build_query_for_subclaim(sc, entities, "semantic_scholar")
                try:
                    papers = s2_search(query, max_results=MAX_RESULTS_PER_SOURCE)
                    sc_evidence.extend(_s2_to_evidence(papers, sc.id))
                except Exception as e:
                    logger.warning("S2 search failed for %s: %s", sc.id, e)

            elif method == "cochrane_api":
                query = _build_query_for_subclaim(sc, entities, "cochrane")
                population = sc.pico.population if sc.pico else None
                intervention = sc.pico.intervention if sc.pico else None
                try:
                    reviews = search_cochrane(
                        query, max_results=MAX_RESULTS_PER_SOURCE,
                        population=population, intervention=intervention,
                    )
                    sc_evidence.extend(_cochrane_to_evidence(reviews, sc.id))
                except Exception as e:
                    logger.warning("Cochrane search failed for %s: %s", sc.id, e)

            elif method == "clinical_trials":
                try:
                    if sc.pico and sc.pico.population and sc.pico.intervention:
                        trials = ct_search_pico(
                            condition=sc.pico.population,
                            intervention=sc.pico.intervention,
                            max_results=MAX_RESULTS_PER_SOURCE,
                        )
                    else:
                        query = _build_query_for_subclaim(sc, entities, "clinical_trials")
                        trials = ct_search(query, max_results=MAX_RESULTS_PER_SOURCE)
                    sc_evidence.extend(_trial_to_evidence(trials, sc.id))
                except Exception as e:
                    logger.warning("ClinicalTrials search failed for %s: %s", sc.id, e)

            elif method == "drugbank_api":
                drugs_list = entities.get("drugs", [])
                for drug_name in drugs_list:
                    try:
                        drug_info = search_drug_label(drug_name)
                        drug_interactions = get_interactions(drug_name)
                        sc_evidence.extend(_drug_to_evidence(drug_info, drug_interactions, drug_name))
                    except Exception as e:
                        logger.warning("Drug lookup failed for '%s': %s", drug_name, e)

            elif method == "cross_encoder":
                # Apply cross-encoder to whatever we've collected so far
                if sc_evidence:
                    try:
                        wrappers = [_EvidenceWrapper(ev) for ev in sc_evidence]
                        ranked = rerank_papers(sc.text, wrappers, top_k=RERANK_TOP_K)
                        reranked = []
                        for wrapper, score in ranked:
                            ev = wrapper.evidence
                            reranked.append(Evidence(
                                id=ev.id,
                                source=ev.source,
                                retrieval_method=ev.retrieval_method,
                                title=ev.title,
                                content=ev.content,
                                url=ev.url,
                                study_type=ev.study_type,
                                quality_score=round(score, 4),
                                pmid=ev.pmid,
                            ))
                        sc_evidence = reranked
                    except Exception as e:
                        logger.warning("Reranking failed for %s: %s", sc.id, e)

            elif method in ("deep_search", "guideline_store"):
                logger.warning(
                    "Skipping '%s' for %s — not yet implemented", method, sc.id
                )

            else:
                logger.warning("Unknown method '%s' for %s", method, sc.id)

        # Deduplicate
        collected[sc.id] = _deduplicate_evidence(sc_evidence)

    return collected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_evidence_retriever(state: FactCheckState) -> FactCheckState:
    """Run the evidence retriever agent.

    Follows the retrieval plan to search across PubMed, Semantic Scholar,
    Cochrane, ClinicalTrials.gov, DrugBank. Re-ranks results with
    cross-encoder and deduplicates.

    Args:
        state: Pipeline state with retrieval_plan, sub_claims, pico,
               and entities populated.

    Returns:
        Updated state with evidence list populated.
    """
    start_time = time.time()
    sub_claims = state.get("sub_claims", [])
    cost = 0.0
    tool_call_records: list[ToolCall] = []
    reasoning_steps = 0
    tools_called: list[str] = []

    if ANTHROPIC_API_KEY and sub_claims:
        try:
            collected, cost, tool_call_records, reasoning_steps = (
                _retrieve_with_react(state)
            )
            tools_called = list({tc.tool for tc in tool_call_records})
        except Exception as e:
            logger.warning(
                "ReAct retrieval failed, using rule-based fallback: %s", e
            )
            collected = _retrieve_rule_based(state)
            tools_called = ["rule_based_fallback"]
    else:
        collected = _retrieve_rule_based(state)
        tools_called = ["rule_based_fallback"]

    duration = time.time() - start_time

    # Flatten all evidence into a single list
    all_evidence: list[Evidence] = []
    for ev_list in collected.values():
        all_evidence.extend(ev_list)

    # Build evidence ID mapping per sub-claim
    evidence_by_subclaim: dict[str, list[str]] = {}
    for sc_id, ev_list in collected.items():
        evidence_by_subclaim[sc_id] = [ev.id for ev in ev_list]

    # Update sub-claims with evidence IDs (create new objects, don't mutate)
    updated_sub_claims = []
    for sc in sub_claims:
        ev_ids = evidence_by_subclaim.get(sc.id, [])
        updated_sc = SubClaim(
            id=sc.id,
            text=sc.text,
            pico=sc.pico,
            verdict=sc.verdict,
            evidence=ev_ids,
            confidence=sc.confidence,
        )
        updated_sub_claims.append(updated_sc)

    # Build trace
    total_evidence = len(all_evidence)
    trace = AgentTrace(
        agent="evidence_retriever",
        node_type="agent",
        duration_seconds=round(duration, 2),
        cost_usd=round(cost, 6),
        input_summary=(
            f"{len(sub_claims)} sub-claims, "
            f"plan: {len(state.get('retrieval_plan', {}))} entries"
        ),
        output_summary=(
            f"Retrieved {total_evidence} evidence items "
            f"across {len(collected)} sub-claims"
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
        "sub_claims": updated_sub_claims,
        "evidence": all_evidence,
        "agent_trace": existing_traces + [trace],
        "total_cost_usd": existing_cost + cost,
        "total_duration_seconds": existing_duration + duration,
    }
