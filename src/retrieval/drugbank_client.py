"""Drug information client using OpenFDA + RxNorm APIs.

DrugBank's full API requires a commercial license, so this module uses:
1. OpenFDA Drug API (free, no key required) for drug labels and adverse events.
2. RxNorm API (free, NLM) for drug name normalization and interactions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
_RXNORM_URL = "https://rxnav.nlm.nih.gov/REST"

_MIN_INTERVAL = 0.5
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


@dataclass
class DrugInfo:
    """Drug information from FDA labels."""
    name: str
    generic_name: str = ""
    brand_names: list[str] = field(default_factory=list)
    indications: str = ""
    contraindications: str = ""
    warnings: str = ""
    adverse_reactions: str = ""
    drug_interactions: str = ""
    dosage: str = ""
    mechanism_of_action: str = ""
    rxcui: Optional[str] = None


@dataclass
class DrugInteraction:
    """A drug-drug interaction."""
    drug_a: str
    drug_b: str
    description: str
    severity: str = ""


def search_drug_label(drug_name: str, max_results: int = 3) -> list[DrugInfo]:
    """Search OpenFDA for drug label information.

    Args:
        drug_name: Drug name (generic or brand).
        max_results: Maximum results.

    Returns:
        List of DrugInfo objects.
    """
    _rate_limit()

    params = {
        "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
        "limit": max_results,
    }

    try:
        resp = requests.get(_OPENFDA_LABEL_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("OpenFDA label search failed for '%s': %s", drug_name, e)
        return []

    drugs = []
    for result in data.get("results", []):
        openfda = result.get("openfda", {})

        generic_names = openfda.get("generic_name", [])
        brand_names = openfda.get("brand_name", [])
        rxcuis = openfda.get("rxcui", [])

        drugs.append(DrugInfo(
            name=drug_name,
            generic_name=generic_names[0] if generic_names else "",
            brand_names=brand_names,
            indications=_join_sections(result.get("indications_and_usage", [])),
            contraindications=_join_sections(result.get("contraindications", [])),
            warnings=_join_sections(result.get("warnings", [])),
            adverse_reactions=_join_sections(result.get("adverse_reactions", [])),
            drug_interactions=_join_sections(result.get("drug_interactions", [])),
            dosage=_join_sections(result.get("dosage_and_administration", [])),
            mechanism_of_action=_join_sections(result.get("mechanism_of_action", [])),
            rxcui=rxcuis[0] if rxcuis else None,
        ))

    logger.info("OpenFDA label search '%s' returned %d results", drug_name, len(drugs))
    return drugs


def get_interactions(drug_name: str) -> list[DrugInteraction]:
    """Get known drug interactions from RxNorm.

    Args:
        drug_name: Drug name to check interactions for.

    Returns:
        List of DrugInteraction objects.
    """
    rxcui = _get_rxcui(drug_name)
    if not rxcui:
        logger.info("No RxCUI found for '%s', cannot check interactions", drug_name)
        return []

    _rate_limit()

    url = f"{_RXNORM_URL}/interaction/interaction.json"
    params = {"rxcui": rxcui}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("RxNorm interaction lookup failed: %s", e)
        return []

    interactions = []
    for group in data.get("interactionTypeGroup", []):
        for int_type in group.get("interactionType", []):
            for pair in int_type.get("interactionPair", []):
                concepts = pair.get("interactionConcept", [])
                if len(concepts) >= 2:
                    drug_a = concepts[0].get("minConceptItem", {}).get("name", "")
                    drug_b = concepts[1].get("minConceptItem", {}).get("name", "")
                    desc = pair.get("description", "")
                    severity = pair.get("severity", "")

                    interactions.append(DrugInteraction(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        description=desc,
                        severity=severity,
                    ))

    logger.info("Found %d interactions for '%s'", len(interactions), drug_name)
    return interactions


def _get_rxcui(drug_name: str) -> Optional[str]:
    """Resolve a drug name to an RxNorm Concept Unique Identifier (RxCUI)."""
    _rate_limit()

    url = f"{_RXNORM_URL}/rxcui.json"
    params = {"name": drug_name, "search": 2}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId", [])
        return rxnorm_ids[0] if rxnorm_ids else None
    except Exception as e:
        logger.warning("RxCUI lookup failed for '%s': %s", drug_name, e)
        return None


def _join_sections(sections: list) -> str:
    """Join FDA label section text."""
    if not sections:
        return ""
    return " ".join(str(s).strip() for s in sections if s)
