"""Verdict mapping utilities for benchmark evaluation.

Our system produces 9-level nuanced verdicts. Benchmark datasets use coarser
label sets (2–4 labels). This module maps our verdicts to each benchmark's
label space so we can compute standard metrics (F1, accuracy) against their
ground truth.

Dual evaluation strategy:
  1. **Benchmark eval** — collapse verdicts to match dataset labels (lossy).
  2. **Nuance eval** — evaluate on our curated claims with full 9-level labels.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Our 9-level verdict taxonomy
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """Health-claim verdict produced by the Verdict Agent."""

    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_CAVEATS = "SUPPORTED_WITH_CAVEATS"
    OVERSTATED = "OVERSTATED"
    MISLEADING = "MISLEADING"
    PRELIMINARY = "PRELIMINARY"
    OUTDATED = "OUTDATED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    REFUTED = "REFUTED"
    DANGEROUS = "DANGEROUS"


# ---------------------------------------------------------------------------
# Benchmark label sets
# ---------------------------------------------------------------------------

# SciFact & HealthVer share the same 3-label set
ScifactLabel = Literal["SUPPORTS", "REFUTES", "NEI"]
HealthVerLabel = Literal["SUPPORTS", "REFUTES", "NEI"]

# PUBHEALTH uses 4 labels
PubhealthLabel = Literal["true", "false", "mixture", "unproven"]

# COVID-Fact uses 2 labels
CovidfactLabel = Literal["true", "false"]


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

SCIFACT_MAP: dict[Verdict, ScifactLabel] = {
    Verdict.SUPPORTED:              "SUPPORTS",
    Verdict.SUPPORTED_WITH_CAVEATS: "SUPPORTS",
    Verdict.OVERSTATED:             "NEI",       # partial truth — closest to NEI
    Verdict.MISLEADING:             "REFUTES",   # wrong impression ≈ refutation
    Verdict.PRELIMINARY:            "NEI",
    Verdict.OUTDATED:               "REFUTES",   # no longer true ≈ refuted
    Verdict.NOT_SUPPORTED:          "NEI",
    Verdict.REFUTED:                "REFUTES",
    Verdict.DANGEROUS:              "REFUTES",
}

# HealthVer uses the same schema as SciFact
HEALTHVER_MAP: dict[Verdict, HealthVerLabel] = SCIFACT_MAP.copy()

PUBHEALTH_MAP: dict[Verdict, PubhealthLabel] = {
    Verdict.SUPPORTED:              "true",
    Verdict.SUPPORTED_WITH_CAVEATS: "mixture",   # true but needs context
    Verdict.OVERSTATED:             "mixture",    # kernel of truth, exaggerated
    Verdict.MISLEADING:             "mixture",    # technically true, wrong impression
    Verdict.PRELIMINARY:            "unproven",   # some evidence, too early
    Verdict.OUTDATED:               "false",      # was true, no longer
    Verdict.NOT_SUPPORTED:          "false",
    Verdict.REFUTED:                "false",
    Verdict.DANGEROUS:              "false",
}

COVIDFACT_MAP: dict[Verdict, CovidfactLabel] = {
    Verdict.SUPPORTED:              "true",
    Verdict.SUPPORTED_WITH_CAVEATS: "true",
    Verdict.OVERSTATED:             "false",
    Verdict.MISLEADING:             "false",
    Verdict.PRELIMINARY:            "false",
    Verdict.OUTDATED:               "false",
    Verdict.NOT_SUPPORTED:          "false",
    Verdict.REFUTED:                "false",
    Verdict.DANGEROUS:              "false",
}

# Registry for easy lookup by dataset name
BENCHMARK_MAPS: dict[str, dict[Verdict, str]] = {
    "scifact":    SCIFACT_MAP,
    "healthver":  HEALTHVER_MAP,
    "pubhealth":  PUBHEALTH_MAP,
    "covidfact":  COVIDFACT_MAP,
}


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------

def map_verdict(verdict: Verdict | str, dataset: str) -> str:
    """Map a 9-level verdict to a benchmark's label set.

    Args:
        verdict: Our system's verdict (str or Verdict enum).
        dataset: Benchmark name — one of "scifact", "healthver",
                 "pubhealth", "covidfact".

    Returns:
        The corresponding label in the benchmark's label space.

    Raises:
        KeyError: If dataset or verdict is not recognized.
    """
    if isinstance(verdict, str):
        verdict = Verdict(verdict)

    mapping = BENCHMARK_MAPS[dataset.lower()]
    return mapping[verdict]


def map_verdicts(verdicts: list[Verdict | str], dataset: str) -> list[str]:
    """Map a list of verdicts to a benchmark's label set."""
    return [map_verdict(v, dataset) for v in verdicts]


# ---------------------------------------------------------------------------
# Reverse mapping: benchmark label → possible verdicts (for analysis)
# ---------------------------------------------------------------------------

def reverse_map(dataset: str) -> dict[str, list[str]]:
    """Show which of our verdicts collapse into each benchmark label.

    Useful for understanding information loss in the mapping.

    Returns:
        Dict mapping benchmark label → list of our verdict names.
    """
    mapping = BENCHMARK_MAPS[dataset.lower()]
    result: dict[str, list[str]] = {}
    for verdict, label in mapping.items():
        result.setdefault(label, []).append(verdict.value)
    return result


def info_loss_report(dataset: str) -> str:
    """Print a human-readable report of what nuance is lost per benchmark.

    Example output for scifact:
        SUPPORTS  ← SUPPORTED, SUPPORTED_WITH_CAVEATS  (2 verdicts collapsed)
        REFUTES   ← MISLEADING, OUTDATED, REFUTED, DANGEROUS  (4 verdicts collapsed)
        NEI       ← OVERSTATED, PRELIMINARY, NOT_SUPPORTED  (3 verdicts collapsed)
    """
    rev = reverse_map(dataset)
    lines = [f"Verdict mapping for {dataset.upper()}:", ""]
    for label, verdicts in sorted(rev.items()):
        collapsed = len(verdicts)
        verdict_str = ", ".join(verdicts)
        note = "" if collapsed == 1 else f"  ({collapsed} verdicts collapsed)"
        lines.append(f"  {label:<12} ← {verdict_str}{note}")
    return "\n".join(lines)
