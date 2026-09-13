"""Bi-encoder top-k retrieval over indexed meeting segments."""

from __future__ import annotations

from app import config
from app.dl.embeddings import SegmentIndex
from app.models.schemas import RetrievalCandidate, RetrievalResult, SLPSegment


def retrieve_candidates(
    question: str,
    index: SegmentIndex,
    segments_by_id: dict[str, SLPSegment],
    top_k: int | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant transcript segments for a question.

    Args:
        question: User question in natural language.
        index: FAISS index built from meeting segments.
        segments_by_id: Lookup map for segment metadata (unused for scoring
            but kept for API symmetry and future filtering).
        top_k: Override for number of retrieved segments.

    Returns:
        ``RetrievalResult`` matching the DL retrieval contract.
    """
    _ = segments_by_id  # reserved for future metadata-aware filtering
    k = top_k or config.RETRIEVAL_TOP_K
    hits = index.search(question, top_k=k)

    candidates = [
        RetrievalCandidate(segment_id=seg_id, score=score)
        for seg_id, score in hits
    ]
    return RetrievalResult(question=question, candidates=candidates)
