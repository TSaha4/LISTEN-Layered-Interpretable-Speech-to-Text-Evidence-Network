"""Unit tests for the Explainable AI layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.models.schemas import RetrievalCandidate, RetrievalResult, SLPSegment
from app.xai.shap_explainer import _fallback_highlights, explain_retrieval


def _segments() -> dict[str, SLPSegment]:
    return {
        "s0": SLPSegment(
            segment_id="s0",
            start_time=0.0,
            end_time=1.0,
            text="budget increase approved",
            entities=[],
        )
    }


class TestSHAPExplainer:
    def test_fallback_highlights_uniform_scores(self) -> None:
        retrieval = RetrievalResult(
            question="q",
            candidates=[RetrievalCandidate(segment_id="s0", score=0.6)],
        )
        result = _fallback_highlights(retrieval, _segments())
        assert sum(h.score for h in result["s0"]) == pytest.approx(0.6)
        assert len(result["s0"]) == 3

    @patch("app.xai.shap_explainer._retrieval_score_fn", return_value=0.5)
    @patch("app.xai.shap_explainer.embed_texts")
    def test_single_word_segment(self, mock_embed, mock_score) -> None:
        retrieval = RetrievalResult(
            question="budget?",
            candidates=[RetrievalCandidate(segment_id="s0", score=0.9)],
        )
        segments = {
            "s0": SLPSegment(
                segment_id="s0",
                start_time=0.0,
                end_time=1.0,
                text="budget",
                entities=[],
            )
        }
        result = explain_retrieval("budget?", retrieval, segments)
        assert result["s0"][0].word == "budget"
