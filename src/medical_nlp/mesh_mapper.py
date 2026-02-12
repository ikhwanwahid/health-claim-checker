"""Map medical terms to MeSH (Medical Subject Headings) vocabulary.

MeSH terms are used by PubMed for indexing, so mapping free-text entities
to MeSH descriptors improves search recall and precision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# NCBI E-utilities endpoint for MeSH lookup
_MESH_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_MESH_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class MeSHTerm:
    """A MeSH descriptor."""
    uid: str
    name: str
    tree_numbers: list[str]
    scope_note: Optional[str] = None


def search_mesh(term: str, api_key: Optional[str] = None) -> list[str]:
    """Search MeSH database for a term, return matching UIDs.

    Args:
        term: Free-text medical term to look up.
        api_key: NCBI API key for higher rate limits.

    Returns:
        List of MeSH UIDs.
    """
    params = {
        "db": "mesh",
        "term": term,
        "retmode": "json",
        "retmax": 5,
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(_MESH_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning("MeSH search failed for '%s': %s", term, e)
        return []


def fetch_mesh_details(uids: list[str], api_key: Optional[str] = None) -> list[MeSHTerm]:
    """Fetch MeSH descriptor details by UIDs.

    The MeSH efetch endpoint returns plain text (not XML), so we parse
    each record block from the text response.

    Args:
        uids: List of MeSH UIDs from search_mesh.
        api_key: NCBI API key.

    Returns:
        List of MeSHTerm objects.
    """
    if not uids:
        return []

    params = {
        "db": "mesh",
        "id": ",".join(uids),
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(_MESH_FETCH_URL, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("MeSH fetch failed: %s", e)
        return []

    return _parse_mesh_text(resp.text, uids)


def _parse_mesh_text(text: str, uids: list[str]) -> list[MeSHTerm]:
    """Parse the plain-text response from MeSH efetch.

    Each record starts with a line like '1: Descriptor Name' and the
    scope note is the block of text before 'Year introduced' or 'Subheadings'.
    """
    # Split into per-record blocks (separated by blank lines between records)
    record_blocks = re.split(r'\n\n+(?=\d+:\s)', text.strip())

    terms = []
    for i, block in enumerate(record_blocks):
        if not block.strip():
            continue

        lines = block.strip().split("\n")

        # First line: "1: Descriptor Name"
        first_line = lines[0]
        match = re.match(r'\d+:\s*(.+)', first_line)
        if not match:
            continue
        name = match.group(1).strip()

        uid = uids[i] if i < len(uids) else ""

        # Collect scope note: lines between the name and "Year introduced" / "Subheadings" / "Tree Number"
        scope_lines = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("Year introduced") or stripped.startswith("Subheadings:") or stripped.startswith("Tree Number"):
                break
            if stripped:
                scope_lines.append(stripped)
        scope_note = " ".join(scope_lines) if scope_lines else None

        terms.append(MeSHTerm(
            uid=uid,
            name=name,
            tree_numbers=[],
            scope_note=scope_note,
        ))

    return terms


def map_term_to_mesh(
    term: str,
    api_key: Optional[str] = None,
) -> Optional[MeSHTerm]:
    """Map a single medical term to its best MeSH descriptor.

    Args:
        term: Free-text medical term.
        api_key: NCBI API key.

    Returns:
        Best matching MeSHTerm, or None if no match found.
    """
    uids = search_mesh(term, api_key=api_key)
    if not uids:
        return None

    details = fetch_mesh_details(uids[:1], api_key=api_key)
    return details[0] if details else None


def map_entities_to_mesh(
    entities: list[str],
    api_key: Optional[str] = None,
) -> dict[str, Optional[MeSHTerm]]:
    """Map a list of medical entities to MeSH terms.

    Args:
        entities: List of free-text medical terms.
        api_key: NCBI API key.

    Returns:
        Dict mapping each input term to its MeSHTerm (or None).
    """
    results = {}
    for entity in entities:
        results[entity] = map_term_to_mesh(entity, api_key=api_key)
    return results
