"""Cochrane Library search client for systematic reviews.

Cochrane reviews are the gold standard for evidence synthesis.
This client searches for Cochrane reviews via PubMed and Semantic Scholar,
since Cochrane doesn't offer a free public API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_MIN_INTERVAL = 1.0  # Be polite
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


@dataclass
class CochraneReview:
    """A Cochrane systematic review."""
    doi: str
    title: str
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    review_type: str = "intervention"
    source: str = "cochrane"

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}"


def search_via_pubmed(
    query: str,
    max_results: int = 10,
) -> list[CochraneReview]:
    """Search for Cochrane reviews via PubMed.

    Filters to Cochrane Database of Systematic Reviews journal.

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        List of CochraneReview objects.
    """
    from src.retrieval.pubmed_client import search, fetch_articles

    cochrane_query = f'({query}) AND "Cochrane Database Syst Rev"[Journal]'

    pmids = search(cochrane_query, max_results=max_results)
    if not pmids:
        return []

    articles = fetch_articles(pmids)
    reviews = []
    for article in articles:
        reviews.append(CochraneReview(
            doi=article.doi or "",
            title=article.title,
            abstract=article.abstract,
            authors=article.authors,
            year=int(article.pub_date[:4]) if article.pub_date and len(article.pub_date) >= 4 else None,
        ))

    logger.info("Cochrane/PubMed search '%s' returned %d reviews", query, len(reviews))
    return reviews


def search_via_semantic_scholar(
    query: str,
    max_results: int = 10,
) -> list[CochraneReview]:
    """Search for Cochrane reviews via Semantic Scholar.

    Uses venue filter to find Cochrane reviews.

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        List of CochraneReview objects.
    """
    _rate_limit()

    params = {
        "query": f"{query} Cochrane systematic review",
        "limit": min(max_results, 100),
        "fields": "paperId,title,abstract,year,authors,venue,externalIds",
        "venue": "The Cochrane database of systematic reviews",
    }

    try:
        resp = requests.get(_S2_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Cochrane/S2 search failed: %s", e)
        return []

    reviews = []
    for item in data.get("data", []):
        ext_ids = item.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        authors = [a.get("name", "") for a in (item.get("authors") or [])]

        reviews.append(CochraneReview(
            doi=doi,
            title=item.get("title", ""),
            abstract=item.get("abstract"),
            authors=authors,
            year=item.get("year"),
        ))

    logger.info("Cochrane/S2 search '%s' returned %d reviews", query, len(reviews))
    return reviews


def search_cochrane(
    query: str,
    max_results: int = 10,
    population: Optional[str] = None,
    intervention: Optional[str] = None,
) -> list[CochraneReview]:
    """Search for Cochrane reviews using multiple methods.

    Combines PubMed and Semantic Scholar results, deduplicating by DOI.
    Optionally accepts PICO elements for more targeted search.

    Args:
        query: Search query.
        max_results: Maximum total results.
        population: PICO population for targeted search.
        intervention: PICO intervention for targeted search.

    Returns:
        Deduplicated list of CochraneReview objects.
    """
    if population and intervention:
        pico_query = f"{intervention} {population}"
    else:
        pico_query = query

    pubmed_results = search_via_pubmed(pico_query, max_results=max_results)
    s2_results = search_via_semantic_scholar(pico_query, max_results=max_results)

    # Deduplicate by DOI
    seen_dois = set()
    combined = []
    for review in pubmed_results + s2_results:
        doi_key = review.doi.lower() if review.doi else review.title.lower()
        if doi_key not in seen_dois:
            seen_dois.add(doi_key)
            combined.append(review)

    return combined[:max_results]
