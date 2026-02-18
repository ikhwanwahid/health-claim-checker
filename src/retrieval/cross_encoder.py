"""Cross-encoder re-ranker for scoring candidate abstracts against a query.

Uses a cross-encoder model that processes query+document pairs together
through the transformer for more accurate relevance scoring than bi-encoders.
Designed to re-rank ~30-50 API-discovered candidates per sub-claim.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from src.config import CROSS_ENCODER_MODEL

logger = logging.getLogger(__name__)

_model = None


@dataclass
class RankedResult:
    """A candidate text with its cross-encoder relevance score."""

    index: int    # position in original candidate list
    score: float  # cross-encoder relevance score (0-1)
    text: str     # the candidate text


def _get_model():
    """Lazy-load the CrossEncoder model (singleton).

    Returns:
        CrossEncoder instance, or None if loading fails.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Loaded cross-encoder model: %s", CROSS_ENCODER_MODEL)
        return _model
    except Exception as e:
        logger.warning("Failed to load cross-encoder model: %s", e)
        return None


def _normalize_scores(scores: list[float]) -> list[float]:
    """Sigmoid normalization to [0, 1] range.

    Args:
        scores: Raw logit scores from the cross-encoder.

    Returns:
        Scores normalized to [0, 1] via sigmoid.
    """
    return [1.0 / (1.0 + math.exp(-s)) for s in scores]


def rerank(
    query: str,
    candidates: list[str],
    top_k: int | None = None,
) -> list[RankedResult]:
    """Score and rank candidate texts against a query using a cross-encoder.

    Args:
        query: The search query (e.g. a sub-claim).
        candidates: List of candidate texts (e.g. abstracts).
        top_k: If set, return only the top-k results. Returns all if None.

    Returns:
        List of RankedResult sorted by score descending.
    """
    if not candidates:
        return []

    model = _get_model()

    if model is None:
        logger.warning("Cross-encoder unavailable, returning candidates in original order")
        return [
            RankedResult(index=i, score=0.0, text=text)
            for i, text in enumerate(candidates)
        ][:top_k]

    pairs = [[query, candidate] for candidate in candidates]
    raw_scores = model.predict(pairs).tolist()
    normalized = _normalize_scores(raw_scores)

    results = [
        RankedResult(index=i, score=score, text=text)
        for i, (score, text) in enumerate(zip(normalized, candidates))
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def rerank_papers(
    query: str,
    papers: list[Any],
    top_k: int | None = None,
) -> list[tuple[Any, float]]:
    """Re-rank paper objects by relevance to a query.

    Accepts any object with `.title` and `.abstract` attributes
    (PubMedArticle, S2Paper, CochraneReview, etc.).

    Args:
        query: The search query (e.g. a sub-claim).
        papers: List of paper objects with title and abstract attributes.
        top_k: If set, return only the top-k results.

    Returns:
        List of (paper, score) tuples sorted by score descending.
    """
    if not papers:
        return []

    candidates = []
    for paper in papers:
        title = getattr(paper, "title", "") or ""
        abstract = getattr(paper, "abstract", "") or ""
        candidates.append(f"{title} {abstract}".strip())

    ranked = rerank(query, candidates, top_k=top_k)

    return [(papers[r.index], r.score) for r in ranked]
