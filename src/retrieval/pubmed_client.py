"""PubMed E-utilities client for searching and fetching biomedical literature."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from Bio import Entrez

from src.config import NCBI_API_KEY

logger = logging.getLogger(__name__)

# NCBI requires an email for E-utilities
Entrez.email = "health-checker@example.com"
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

# Rate limit: 3/sec without key, 10/sec with key
_MIN_INTERVAL = 0.1 if NCBI_API_KEY else 0.34
_last_request_time = 0.0


def _rate_limit() -> None:
    """Enforce rate limiting between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


@dataclass
class PubMedArticle:
    """A PubMed article summary."""
    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    doi: Optional[str] = None
    pmc_id: Optional[str] = None
    mesh_terms: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def full_text_url(self) -> Optional[str]:
        if self.pmc_id:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}/"
        return None


def search(
    query: str,
    max_results: int = 20,
    sort: str = "relevance",
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> list[str]:
    """Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query (supports boolean operators, MeSH terms, field tags).
        max_results: Maximum number of results to return.
        sort: Sort order — "relevance" or "date".
        min_date: Minimum publication date (YYYY/MM/DD).
        max_date: Maximum publication date (YYYY/MM/DD).

    Returns:
        List of PMID strings.
    """
    _rate_limit()

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": sort,
        "usehistory": "y",
    }
    if min_date:
        params["mindate"] = min_date
        params["datetype"] = "pdat"
    if max_date:
        params["maxdate"] = max_date
        params["datetype"] = "pdat"

    try:
        handle = Entrez.esearch(**params)
        results = Entrez.read(handle)
        handle.close()
        pmids = results.get("IdList", [])
        logger.info("PubMed search '%s' returned %d results", query, len(pmids))
        return pmids
    except Exception as e:
        logger.error("PubMed search failed: %s", e)
        return []


def fetch_articles(pmids: list[str]) -> list[PubMedArticle]:
    """Fetch article details for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs.

    Returns:
        List of PubMedArticle objects.
    """
    if not pmids:
        return []

    _rate_limit()

    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        logger.error("PubMed fetch failed: %s", e)
        return []

    articles = []
    for article_data in records.get("PubmedArticle", []):
        try:
            articles.append(_parse_article(article_data))
        except Exception as e:
            logger.warning("Failed to parse article: %s", e)

    return articles


def _parse_article(data: dict) -> PubMedArticle:
    """Parse a single PubMed article from Entrez XML result."""
    medline = data.get("MedlineCitation", {})
    article = medline.get("Article", {})

    # PMID
    pmid = str(medline.get("PMID", ""))

    # Title
    title = str(article.get("ArticleTitle", ""))

    # Abstract
    abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
    if abstract_parts:
        abstract = " ".join(str(part) for part in abstract_parts)
    else:
        abstract = ""

    # Authors
    authors = []
    author_list = article.get("AuthorList", [])
    for author in author_list:
        last = author.get("LastName", "")
        first = author.get("ForeName", "")
        if last:
            authors.append(f"{last} {first}".strip())

    # Journal
    journal_info = article.get("Journal", {})
    journal = str(journal_info.get("Title", ""))

    # Publication date
    pub_date_info = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = pub_date_info.get("Year", "")
    month = pub_date_info.get("Month", "")
    pub_date = f"{year} {month}".strip()

    # DOI
    doi = None
    for id_info in article.get("ELocationID", []):
        if str(id_info.attributes.get("EIdType", "")) == "doi":
            doi = str(id_info)
            break

    # PMC ID
    pmc_id = None
    for id_info in data.get("PubmedData", {}).get("ArticleIdList", []):
        if str(id_info.attributes.get("IdType", "")) == "pmc":
            pmc_id = str(id_info)
            break

    # MeSH terms
    mesh_terms = []
    for mesh in medline.get("MeshHeadingList", []):
        descriptor = mesh.get("DescriptorName")
        if descriptor:
            mesh_terms.append(str(descriptor))

    # Publication types
    pub_types = []
    for pt in article.get("PublicationTypeList", []):
        pub_types.append(str(pt))

    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        pub_date=pub_date,
        doi=doi,
        pmc_id=pmc_id,
        mesh_terms=mesh_terms,
        publication_types=pub_types,
    )


def search_and_fetch(
    query: str,
    max_results: int = 20,
    sort: str = "relevance",
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> list[PubMedArticle]:
    """Search PubMed and fetch full article details in one call.

    Args:
        query: PubMed search query.
        max_results: Maximum number of results.
        sort: Sort order.
        min_date: Minimum publication date.
        max_date: Maximum publication date.

    Returns:
        List of PubMedArticle objects.
    """
    pmids = search(query, max_results=max_results, sort=sort,
                   min_date=min_date, max_date=max_date)
    return fetch_articles(pmids)


def build_query_from_pico(
    population: Optional[str] = None,
    intervention: Optional[str] = None,
    comparison: Optional[str] = None,
    outcome: Optional[str] = None,
) -> str:
    """Build a PubMed query string from PICO elements.

    Combines non-None elements with AND.
    """
    parts = []
    if population:
        parts.append(f"({population})")
    if intervention:
        parts.append(f"({intervention})")
    if comparison:
        parts.append(f"({comparison})")
    if outcome:
        parts.append(f"({outcome})")

    return " AND ".join(parts) if parts else ""
