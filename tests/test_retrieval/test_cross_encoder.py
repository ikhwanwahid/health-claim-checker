"""Tests for cross-encoder re-ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.retrieval.cross_encoder import (
    RankedResult,
    _normalize_scores,
    rerank,
    rerank_papers,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePaper:
    """Minimal paper object with title and abstract."""
    title: str
    abstract: Optional[str] = None


def _make_mock_model(scores: list[float]):
    """Create a mock CrossEncoder that returns predetermined scores."""
    model = MagicMock()
    model.predict.return_value = np.array(scores)
    return model


# ---------------------------------------------------------------------------
# _normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def test_zero_maps_to_half(self):
        assert _normalize_scores([0.0]) == pytest.approx([0.5])

    def test_large_positive_maps_near_one(self):
        result = _normalize_scores([10.0])
        assert result[0] > 0.99

    def test_large_negative_maps_near_zero(self):
        result = _normalize_scores([-10.0])
        assert result[0] < 0.01

    def test_monotonic(self):
        scores = [-5.0, -1.0, 0.0, 1.0, 5.0]
        normalized = _normalize_scores(scores)
        for i in range(len(normalized) - 1):
            assert normalized[i] < normalized[i + 1]


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

class TestRerank:
    def test_empty_candidates(self):
        assert rerank("query", []) == []

    def test_returns_sorted_by_score_desc(self):
        # Raw scores: candidate 0 gets low score, candidate 1 gets high score
        mock_model = _make_mock_model([-2.0, 3.0, 0.5])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank("test query", ["low", "high", "mid"])

        assert len(results) == 3
        assert results[0].text == "high"
        assert results[1].text == "mid"
        assert results[2].text == "low"
        assert results[0].score > results[1].score > results[2].score

    def test_top_k_limits_results(self):
        mock_model = _make_mock_model([1.0, 3.0, 2.0, 0.5])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank("query", ["a", "b", "c", "d"], top_k=2)

        assert len(results) == 2
        assert results[0].text == "b"  # highest score
        assert results[1].text == "c"  # second highest

    def test_preserves_original_index(self):
        mock_model = _make_mock_model([0.0, 5.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank("query", ["first", "second"])

        # "second" (index 1) should be first in results
        assert results[0].index == 1
        assert results[1].index == 0

    def test_model_receives_correct_pairs(self):
        mock_model = _make_mock_model([1.0, 2.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            rerank("my query", ["doc A", "doc B"])

        mock_model.predict.assert_called_once_with(
            [["my query", "doc A"], ["my query", "doc B"]]
        )

    def test_graceful_fallback_on_model_failure(self):
        with patch("src.retrieval.cross_encoder._get_model", return_value=None):
            results = rerank("query", ["a", "b", "c"])

        assert len(results) == 3
        # Original order preserved
        assert results[0].text == "a"
        assert results[1].text == "b"
        assert results[2].text == "c"
        # All scores are 0.0
        assert all(r.score == 0.0 for r in results)

    def test_fallback_respects_top_k(self):
        with patch("src.retrieval.cross_encoder._get_model", return_value=None):
            results = rerank("query", ["a", "b", "c"], top_k=2)

        assert len(results) == 2

    def test_result_dataclass_fields(self):
        mock_model = _make_mock_model([1.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank("query", ["text"])

        r = results[0]
        assert isinstance(r, RankedResult)
        assert isinstance(r.index, int)
        assert isinstance(r.score, float)
        assert isinstance(r.text, str)


# ---------------------------------------------------------------------------
# rerank_papers
# ---------------------------------------------------------------------------

class TestRerankPapers:
    def test_empty_papers(self):
        assert rerank_papers("query", []) == []

    def test_with_paper_objects(self):
        papers = [
            FakePaper(title="Unrelated geology paper", abstract="Rocks and minerals."),
            FakePaper(title="Vitamin D and COVID", abstract="RCT of vitamin D supplementation."),
        ]
        # Second paper should score higher
        mock_model = _make_mock_model([-1.0, 3.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank_papers("Does vitamin D prevent COVID?", papers)

        assert len(results) == 2
        assert results[0][0].title == "Vitamin D and COVID"
        assert results[0][1] > results[1][1]

    def test_with_none_abstract(self):
        papers = [FakePaper(title="Title only", abstract=None)]
        mock_model = _make_mock_model([1.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank_papers("query", papers)

        assert len(results) == 1
        assert results[0][0].title == "Title only"

    def test_top_k(self):
        papers = [
            FakePaper(title="A", abstract="aaa"),
            FakePaper(title="B", abstract="bbb"),
            FakePaper(title="C", abstract="ccc"),
        ]
        mock_model = _make_mock_model([1.0, 3.0, 2.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank_papers("query", papers, top_k=1)

        assert len(results) == 1
        assert results[0][0].title == "B"

    def test_returns_tuples_of_paper_and_score(self):
        papers = [FakePaper(title="Test", abstract="Abstract")]
        mock_model = _make_mock_model([2.0])
        with patch("src.retrieval.cross_encoder._get_model", return_value=mock_model):
            results = rerank_papers("query", papers)

        paper, score = results[0]
        assert paper is papers[0]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
