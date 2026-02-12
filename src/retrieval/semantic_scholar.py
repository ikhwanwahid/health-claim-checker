"""Semantic Scholar API client for paper search and retrieval."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from src.config import SEMANTIC_SCHOLAR_API_KEY

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SEARCH_URL = f"{_BASE_URL}/paper/search"
_PAPER_URL = f"{_BASE_URL}/paper"

# Rate limit: 1 req/sec without key, 10 req/sec with key
_MIN_INTERVAL = 0.1 if SEMANTIC_SCHOLAR_API_KEY else 1.0
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


@dataclass
class S2Paper:
    """A Semantic Scholar paper."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = 0
    influential_citation_count: int = 0
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    external_ids: dict = field(default_factory=dict)
    tldr: Optional[str] = None
    fields_of_study: list[str] = field(default_factory=list)
    is_open_access: bool = False
    open_access_pdf_url: Optional[str] = None

    @property
    def pmid(self) -> Optional[str]:
        return self.external_ids.get("PubMed")

    @property
    def doi(self) -> Optional[str]:
        return self.external_ids.get("DOI")

    @property
    def url(self) -> str:
        return f"https://www.semanticscholar.org/paper/{self.paper_id}"


_SEARCH_FIELDS = (
    "paperId,title,abstract,year,citationCount,influentialCitationCount,"
    "authors,venue,externalIds,tldr,fieldsOfStudy,isOpenAccess,openAccessPdf"
)


def search(
    query: str,
    max_results: int = 20,
    year_range: Optional[str] = None,
    fields_of_study: Optional[list[str]] = None,
    open_access_only: bool = False,
) -> list[S2Paper]:
    """Search Semantic Scholar for papers.

    Args:
        query: Natural language search query.
        max_results: Maximum results to return (max 100).
        year_range: Year filter, e.g. "2020-2024" or "2020-".
        fields_of_study: Filter by field, e.g. ["Medicine", "Biology"].
        open_access_only: Only return open-access papers.

    Returns:
        List of S2Paper objects.
    """
    _rate_limit()

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": _SEARCH_FIELDS,
    }
    if year_range:
        params["year"] = year_range
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    if open_access_only:
        params["openAccessPdf"] = ""

    try:
        resp = requests.get(_SEARCH_URL, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error("Semantic Scholar search failed (HTTP %s): %s", e.response.status_code, e)
        return []
    except Exception as e:
        logger.error("Semantic Scholar search failed: %s", e)
        return []

    papers = []
    for item in data.get("data", []):
        papers.append(_parse_paper(item))

    logger.info("S2 search '%s' returned %d results", query, len(papers))
    return papers


def get_paper(paper_id: str) -> Optional[S2Paper]:
    """Fetch a single paper by Semantic Scholar ID, DOI, or PMID.

    Args:
        paper_id: S2 paper ID, or prefixed ID like "DOI:10.xxx" or "PMID:12345".

    Returns:
        S2Paper or None.
    """
    _rate_limit()

    url = f"{_PAPER_URL}/{paper_id}"
    params = {"fields": _SEARCH_FIELDS}

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        return _parse_paper(resp.json())
    except Exception as e:
        logger.error("S2 paper fetch failed for '%s': %s", paper_id, e)
        return None


def get_citations(paper_id: str, max_results: int = 50) -> list[S2Paper]:
    """Get papers that cite a given paper.

    Args:
        paper_id: Semantic Scholar paper ID.
        max_results: Maximum citations to return.

    Returns:
        List of citing S2Paper objects.
    """
    _rate_limit()

    url = f"{_PAPER_URL}/{paper_id}/citations"
    params = {
        "fields": "paperId,title,abstract,year,citationCount,authors,venue,externalIds",
        "limit": min(max_results, 100),
    }

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("S2 citations fetch failed: %s", e)
        return []

    papers = []
    for item in data.get("data", []):
        citing = item.get("citingPaper", {})
        if citing.get("paperId"):
            papers.append(_parse_paper(citing))

    return papers


def _parse_paper(data: dict) -> S2Paper:
    """Parse an S2 API response into an S2Paper."""
    authors = []
    for author in data.get("authors", []) or []:
        name = author.get("name")
        if name:
            authors.append(name)

    tldr_data = data.get("tldr")
    tldr = tldr_data.get("text") if isinstance(tldr_data, dict) else None

    oa_pdf = data.get("openAccessPdf")
    oa_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None

    return S2Paper(
        paper_id=data.get("paperId", ""),
        title=data.get("title", ""),
        abstract=data.get("abstract"),
        year=data.get("year"),
        citation_count=data.get("citationCount", 0) or 0,
        influential_citation_count=data.get("influentialCitationCount", 0) or 0,
        authors=authors,
        venue=data.get("venue", "") or "",
        external_ids=data.get("externalIds", {}) or {},
        tldr=tldr,
        fields_of_study=data.get("fieldsOfStudy", []) or [],
        is_open_access=data.get("isOpenAccess", False) or False,
        open_access_pdf_url=oa_url,
    )
