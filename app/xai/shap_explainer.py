"""SHAP-based explanations for retrieval relevance scores."""

from __future__ import annotations

import numpy as np

from app import config
from app.dl.embeddings import embed_texts
from app.models.schemas import RetrievalResult, SLPSegment, WordHighlight


def _tokenise(text: str) -> list[str]:
    """Simple whitespace tokenisation for SHAP feature groups."""
    return text.split()


def _retrieval_score_fn(
    question: str,
    segment_text: str,
    segment_embedding: np.ndarray | None = None,
) -> float:
    """Cosine similarity between question and segment (bi-encoder proxy)."""
    q_vec = embed_texts([question])[0]
    if segment_embedding is not None:
        s_vec = segment_embedding
    else:
        s_vec = embed_texts([segment_text])[0]
    return float(np.dot(q_vec, s_vec))


def explain_retrieval(
    question: str,
    retrieval: RetrievalResult,
    segments_by_id: dict[str, SLPSegment],
) -> dict[str, list[WordHighlight]]:
    """Compute SHAP word-level highlights for retrieved segments.

    Uses a masker over segment words and a similarity-based scoring function
    as a lightweight proxy for retrieval relevance.

    Args:
        question: User question.
        retrieval: Top-k retrieval output.
        segments_by_id: Segment lookup.

    Returns:
        Mapping ``segment_id -> [{word, score}, ...]`` per the API contract.
    """
    try:
        import shap
    except ImportError:
        return _fallback_highlights(retrieval, segments_by_id)

    highlights: dict[str, list[WordHighlight]] = {}

    for candidate in retrieval.candidates:
        seg = segments_by_id.get(candidate.segment_id)
        if seg is None:
            continue

        words = _tokenise(seg.text)
        if not words:
            highlights[candidate.segment_id] = []
            continue

        if len(words) == 1:
            highlights[candidate.segment_id] = [
                WordHighlight(word=words[0], score=candidate.score)
            ]
            continue

        def model_fn(masks: np.ndarray, seg_text: str = seg.text) -> np.ndarray:
            """Score masked segment texts."""
            scores = []
            tokens = _tokenise(seg_text)
            for mask_row in masks:
                masked_tokens = [
                    tok if keep else "[MASK]"
                    for tok, keep in zip(tokens, mask_row.astype(bool))
                ]
                masked_text = " ".join(masked_tokens)
                scores.append(_retrieval_score_fn(question, masked_text))
            return np.array(scores)

        masker = shap.maskers.Independent(np.zeros((1, len(words)), dtype=bool))
        explainer = shap.Explainer(
            model_fn,
            masker,
            algorithm="permutation",
        )
        try:
            shap_values = explainer(
                np.ones((1, len(words))),
                max_evals=min(config.SHAP_MAX_EVALS, 2 ** len(words)),
            )
            values = shap_values.values[0]
            word_scores = [
                WordHighlight(word=w, score=float(v))
                for w, v in zip(words, values)
            ]
        except Exception:
            word_scores = _uniform_highlights(words, candidate.score)

        highlights[candidate.segment_id] = word_scores

    return highlights


def _uniform_highlights(words: list[str], score: float) -> list[WordHighlight]:
    """Fallback equal attribution per word."""
    per_word = score / max(len(words), 1)
    return [WordHighlight(word=w, score=per_word) for w in words]


def _fallback_highlights(
    retrieval: RetrievalResult,
    segments_by_id: dict[str, SLPSegment],
) -> dict[str, list[WordHighlight]]:
    """Deterministic highlights when SHAP is unavailable."""
    result: dict[str, list[WordHighlight]] = {}
    for candidate in retrieval.candidates:
        seg = segments_by_id.get(candidate.segment_id)
        if seg is None:
            continue
        words = _tokenise(seg.text)
        result[candidate.segment_id] = _uniform_highlights(words, candidate.score)
    return result
