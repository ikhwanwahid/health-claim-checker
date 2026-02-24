"""Unified dataset loading for benchmark evaluation.

Normalizes SciFact, PUBHEALTH, HealthVer, and COVID-Fact into a common
``BenchmarkClaim`` format so all evaluation code can consume a single
interface regardless of dataset.

Usage::

    from src.evaluation.dataset_loader import load_dataset, filter_health_claims

    claims = load_dataset("scifact", split="dev")
    vaccine_claims = filter_health_claims(claims, keywords=["vaccine", "immunization"])
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmarks"


@dataclass
class BenchmarkClaim:
    """Normalized representation of a claim from any benchmark dataset."""

    id: str
    claim: str
    label: str  # Original dataset label
    dataset: str  # "scifact", "pubhealth", "healthver", "covidfact"
    evidence_text: str  # Concatenated evidence (if available)
    split: str  # "train", "dev", "test"


# ---------------------------------------------------------------------------
# SciFact
# ---------------------------------------------------------------------------

_SCIFACT_SPLIT_FILES = {
    "train": "claims_train.jsonl",
    "dev": "claims_dev.jsonl",
    "test": "claims_test.jsonl",
}


def load_scifact(split: str = "dev") -> list[BenchmarkClaim]:
    """Load SciFact claims from JSONL.

    SciFact labels are at the evidence level (SUPPORT/CONTRADICT). We derive
    a claim-level label:
      - Any evidence with SUPPORT → "SUPPORTS"
      - Any evidence with CONTRADICT → "REFUTES"
      - No evidence → "NEI"
    """
    filename = _SCIFACT_SPLIT_FILES.get(split)
    if filename is None:
        raise ValueError(f"Invalid split '{split}' for SciFact. Choose from: {list(_SCIFACT_SPLIT_FILES)}")

    path = DATA_DIR / "scifact" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"SciFact file not found: {path}. Run 'uv run python scripts/download_benchmarks.py --dataset scifact'"
        )

    claims: list[BenchmarkClaim] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)

        # Derive claim-level label from evidence
        evidence = entry.get("evidence", {})
        labels_found: set[str] = set()
        evidence_texts: list[str] = []

        for _doc_id, annotations in evidence.items():
            for ann in annotations:
                labels_found.add(ann.get("label", ""))
                # Sentence indices aren't text — evidence text not directly available in SciFact claims file
                # We leave evidence_text empty for SciFact (evidence is in corpus.jsonl)

        if "SUPPORT" in labels_found:
            label = "SUPPORTS"
        elif "CONTRADICT" in labels_found:
            label = "REFUTES"
        else:
            label = "NEI"

        claims.append(BenchmarkClaim(
            id=str(entry["id"]),
            claim=entry["claim"],
            label=label,
            dataset="scifact",
            evidence_text="",
            split=split,
        ))

    return claims


# ---------------------------------------------------------------------------
# PUBHEALTH
# ---------------------------------------------------------------------------

_PUBHEALTH_SPLIT_FILES = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
}


def load_pubhealth(split: str = "test") -> list[BenchmarkClaim]:
    """Load PUBHEALTH claims from TSV.

    Columns include claim_id, claim, label, explanation, main_text, sources.
    Labels: true, false, mixture, unproven.
    """
    filename = _PUBHEALTH_SPLIT_FILES.get(split)
    if filename is None:
        raise ValueError(f"Invalid split '{split}' for PUBHEALTH. Choose from: {list(_PUBHEALTH_SPLIT_FILES)}")

    path = DATA_DIR / "pubhealth" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"PUBHEALTH file not found: {path}. Run 'uv run python scripts/download_benchmarks.py --dataset pubhealth'"
        )

    claims: list[BenchmarkClaim] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            claim_text = row.get("claim", "").strip()
            label = row.get("label", "").strip().lower()
            if not claim_text or not label:
                continue
            # Skip rows with non-standard labels
            if label not in {"true", "false", "mixture", "unproven"}:
                continue

            claim_id = row.get("claim_id", str(i))
            evidence = row.get("main_text", "").strip()
            explanation = row.get("explanation", "").strip()
            evidence_text = explanation if explanation else evidence

            claims.append(BenchmarkClaim(
                id=str(claim_id),
                claim=claim_text,
                label=label,
                dataset="pubhealth",
                evidence_text=evidence_text[:2000],  # Truncate very long texts
                split=split,
            ))

    return claims


# ---------------------------------------------------------------------------
# HealthVer
# ---------------------------------------------------------------------------

_HEALTHVER_SPLIT_FILES = {
    "train": "healthver_train.csv",
    "dev": "healthver_dev.csv",
    "test": "healthver_test.csv",
}


def load_healthver(split: str = "dev") -> list[BenchmarkClaim]:
    """Load HealthVer claim-evidence pairs from CSV.

    Each row is a (claim, abstract, verdict) triple. Verdicts: SUPPORT, REFUTE, NEUTRAL.
    We normalize to SUPPORTS/REFUTES/NEI to match SciFact convention.
    """
    filename = _HEALTHVER_SPLIT_FILES.get(split)
    if filename is None:
        raise ValueError(f"Invalid split '{split}' for HealthVer. Choose from: {list(_HEALTHVER_SPLIT_FILES)}")

    path = DATA_DIR / "healthver" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"HealthVer file not found: {path}. Run 'uv run python scripts/download_benchmarks.py --dataset healthver'"
        )

    label_map = {
        "SUPPORT": "SUPPORTS",
        "REFUTE": "REFUTES",
        "NEUTRAL": "NEI",
    }

    claims: list[BenchmarkClaim] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            claim_text = row.get("claim", "").strip()
            raw_label = row.get("label", row.get("verdict", "")).strip().upper()
            label = label_map.get(raw_label, raw_label)

            if not claim_text:
                continue

            claim_id = row.get("claim_id", row.get("id", str(i)))
            evidence = row.get("evidence", row.get("abstract", "")).strip()

            claims.append(BenchmarkClaim(
                id=str(claim_id),
                claim=claim_text,
                label=label,
                dataset="healthver",
                evidence_text=evidence[:2000],
                split=split,
            ))

    return claims


# ---------------------------------------------------------------------------
# COVID-Fact
# ---------------------------------------------------------------------------


def load_covidfact() -> list[BenchmarkClaim]:
    """Load COVID-Fact claims from JSONL.

    Single file, no splits. Labels: SUPPORTED → "true", REFUTED → "false".
    """
    path = DATA_DIR / "covidfact" / "COVIDFACT_dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"COVID-Fact file not found: {path}. Run 'uv run python scripts/download_benchmarks.py --dataset covidfact'"
        )

    label_map = {
        "SUPPORTED": "true",
        "REFUTED": "false",
    }

    claims: list[BenchmarkClaim] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()):
        entry = json.loads(line)
        raw_label = entry.get("label", "").strip().upper()
        label = label_map.get(raw_label, raw_label.lower())

        evidence_parts = entry.get("evidence", [])
        if isinstance(evidence_parts, list):
            evidence_text = " ".join(evidence_parts)
        else:
            evidence_text = str(evidence_parts)

        claims.append(BenchmarkClaim(
            id=str(i),
            claim=entry["claim"],
            label=label,
            dataset="covidfact",
            evidence_text=evidence_text[:2000],
            split="all",
        ))

    return claims


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_LOADERS = {
    "scifact": load_scifact,
    "pubhealth": load_pubhealth,
    "healthver": load_healthver,
    "covidfact": load_covidfact,
}


def load_dataset(name: str, split: str | None = None) -> list[BenchmarkClaim]:
    """Load a benchmark dataset by name.

    Args:
        name: One of "scifact", "pubhealth", "healthver", "covidfact".
        split: Dataset split — "train", "dev", or "test". Ignored for covidfact
               (single file). If None, uses the dataset's default split.

    Returns:
        List of BenchmarkClaim objects.
    """
    name = name.lower()
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_LOADERS)}")

    if name == "covidfact":
        return loader()

    if split is None:
        # Defaults: scifact→dev, pubhealth→test, healthver→dev
        defaults = {"scifact": "dev", "pubhealth": "test", "healthver": "dev"}
        split = defaults[name]

    return loader(split=split)


def filter_health_claims(
    claims: list[BenchmarkClaim],
    keywords: list[str] | None = None,
) -> list[BenchmarkClaim]:
    """Filter claims to a subset matching health-related keywords.

    Args:
        claims: List of BenchmarkClaim to filter.
        keywords: Case-insensitive keywords to match against claim text.
                  Defaults to vaccine/COVID-related terms.

    Returns:
        Filtered list where claim text contains at least one keyword.
    """
    if keywords is None:
        keywords = [
            "vaccine", "vaccination", "immunization", "immunize",
            "covid", "coronavirus", "sars-cov",
            "mrna", "pfizer", "moderna", "astrazeneca", "johnson",
            "booster", "dose", "jab", "antibod",
        ]

    keywords_lower = [k.lower() for k in keywords]

    return [
        c for c in claims
        if any(kw in c.claim.lower() for kw in keywords_lower)
    ]
