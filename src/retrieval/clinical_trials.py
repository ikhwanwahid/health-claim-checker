"""ClinicalTrials.gov API v2 client for searching registered clinical trials."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://clinicaltrials.gov/api/v2"
_SEARCH_URL = f"{_BASE_URL}/studies"

_MIN_INTERVAL = 0.5
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


@dataclass
class ClinicalTrial:
    """A registered clinical trial from ClinicalTrials.gov."""
    nct_id: str
    title: str
    brief_summary: str = ""
    detailed_description: str = ""
    status: str = ""  # e.g., RECRUITING, COMPLETED, TERMINATED
    phase: str = ""  # e.g., PHASE1, PHASE2, PHASE3, PHASE4
    study_type: str = ""  # INTERVENTIONAL, OBSERVATIONAL
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    primary_outcomes: list[str] = field(default_factory=list)
    enrollment: Optional[int] = None
    start_date: str = ""
    completion_date: str = ""
    sponsor: str = ""
    has_results: bool = False

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    @property
    def is_rct(self) -> bool:
        return self.study_type == "INTERVENTIONAL"


def search(
    query: str,
    max_results: int = 20,
    status: Optional[list[str]] = None,
    phase: Optional[list[str]] = None,
    study_type: Optional[str] = None,
) -> list[ClinicalTrial]:
    """Search ClinicalTrials.gov for studies.

    Args:
        query: Free-text search query.
        max_results: Maximum results (max 1000).
        status: Filter by status, e.g. ["COMPLETED", "RECRUITING"].
        phase: Filter by phase, e.g. ["PHASE3", "PHASE4"].
        study_type: Filter by type, "INTERVENTIONAL" or "OBSERVATIONAL".

    Returns:
        List of ClinicalTrial objects.
    """
    _rate_limit()

    params = {
        "query.term": query,
        "pageSize": min(max_results, 100),
        "format": "json",
    }
    if status:
        params["filter.overallStatus"] = ",".join(status)
    if phase:
        params["filter.phase"] = ",".join(phase)
    if study_type:
        params["filter.studyType"] = study_type

    try:
        resp = requests.get(_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("ClinicalTrials.gov search failed: %s", e)
        return []

    trials = []
    for study in data.get("studies", []):
        try:
            trials.append(_parse_study(study))
        except Exception as e:
            logger.warning("Failed to parse trial: %s", e)

    logger.info("ClinicalTrials.gov search '%s' returned %d trials", query, len(trials))
    return trials


def get_study(nct_id: str) -> Optional[ClinicalTrial]:
    """Fetch a single study by NCT ID.

    Args:
        nct_id: ClinicalTrials.gov NCT identifier.

    Returns:
        ClinicalTrial or None.
    """
    _rate_limit()

    url = f"{_SEARCH_URL}/{nct_id}"
    params = {"format": "json"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return _parse_study(resp.json())
    except Exception as e:
        logger.error("ClinicalTrials.gov fetch failed for '%s': %s", nct_id, e)
        return None


def _parse_study(data: dict) -> ClinicalTrial:
    """Parse a study from the ClinicalTrials.gov v2 API response."""
    proto = data.get("protocolSection", {})
    id_module = proto.get("identificationModule", {})
    status_module = proto.get("statusModule", {})
    design_module = proto.get("designModule", {})
    desc_module = proto.get("descriptionModule", {})
    conditions_module = proto.get("conditionsModule", {})
    arms_module = proto.get("armsInterventionsModule", {})
    outcomes_module = proto.get("outcomesModule", {})
    sponsor_module = proto.get("sponsorCollaboratorsModule", {})
    results_section = data.get("resultsSection")

    nct_id = id_module.get("nctId", "")
    title = id_module.get("officialTitle") or id_module.get("briefTitle", "")
    brief_summary = desc_module.get("briefSummary", "")
    detailed_description = desc_module.get("detailedDescription", "")
    status = status_module.get("overallStatus", "")

    phases = design_module.get("phases", [])
    phase = phases[0] if phases else ""
    study_type = design_module.get("studyType", "")

    conditions = conditions_module.get("conditions", [])

    interventions = []
    for arm in arms_module.get("interventions", []):
        name = arm.get("name", "")
        int_type = arm.get("type", "")
        if name:
            interventions.append(f"{int_type}: {name}" if int_type else name)

    primary_outcomes = []
    for outcome in outcomes_module.get("primaryOutcomes", []):
        measure = outcome.get("measure", "")
        if measure:
            primary_outcomes.append(measure)

    enrollment_info = design_module.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count")

    start_info = status_module.get("startDateStruct", {})
    start_date = start_info.get("date", "")
    completion_info = status_module.get("completionDateStruct", {})
    completion_date = completion_info.get("date", "")

    lead_sponsor = sponsor_module.get("leadSponsor", {})
    sponsor = lead_sponsor.get("name", "")

    return ClinicalTrial(
        nct_id=nct_id,
        title=title,
        brief_summary=brief_summary,
        detailed_description=detailed_description,
        status=status,
        phase=phase,
        study_type=study_type,
        conditions=conditions,
        interventions=interventions,
        primary_outcomes=primary_outcomes,
        enrollment=enrollment,
        start_date=start_date,
        completion_date=completion_date,
        sponsor=sponsor,
        has_results=results_section is not None,
    )


def search_by_condition_and_intervention(
    condition: str,
    intervention: str,
    max_results: int = 20,
    completed_only: bool = False,
) -> list[ClinicalTrial]:
    """Convenience search using PICO-style condition + intervention.

    Args:
        condition: Medical condition (maps to PICO population).
        intervention: Treatment/drug (maps to PICO intervention).
        max_results: Maximum results.
        completed_only: Only return completed studies.

    Returns:
        List of ClinicalTrial objects.
    """
    query = f"{condition} {intervention}"
    status = ["COMPLETED"] if completed_only else None
    return search(query, max_results=max_results, status=status)
