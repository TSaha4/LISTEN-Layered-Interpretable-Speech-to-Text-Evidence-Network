"""
LISTEN Phase 1 — Top-k Segment Retrieval

Given a question and a list of meeting segments, retrieves the top-k most
relevant segments using bi-encoder cosine similarity.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import config
from schemas import MeetingSegment
from phase1_retrieval.bi_encoder import BiEncoder


class SegmentRetriever:
    """Retrieves the most relevant transcript segments for a given question.

    Uses the bi-encoder to embed question and segments, then ranks by
    cosine similarity and returns the top-k.
    """

    def __init__(self, bi_encoder: BiEncoder | None = None, top_k: int = config.TOP_K_RETRIEVAL):
        """
        Args:
            bi_encoder: Pre-initialized BiEncoder instance (shared across pipeline).
            top_k: Number of top segments to retrieve.
        """
        self.encoder = bi_encoder or BiEncoder()
        self.top_k = top_k

        # Cache for segment embeddings (avoid re-encoding same meeting)
        self._cached_meeting_id: str | None = None
        self._cached_embeddings: np.ndarray | None = None
        self._cached_segments: List[MeetingSegment] | None = None

    def retrieve(
        self,
        question: str,
        segments: List[MeetingSegment],
        top_k: int | None = None,
        meeting_id: str | None = None,
    ) -> Tuple[List[MeetingSegment], np.ndarray, np.ndarray]:
        """Retrieve top-k segments most relevant to the question.

        Args:
            question: The user's question string.
            segments: All transcript segments from Person A's JSON.
            top_k: Override default top_k for this call.
            meeting_id: If provided, caches segment embeddings for this meeting.

        Returns:
            Tuple of:
              - retrieved_segments: Top-k MeetingSegment objects (highest relevance first).
              - retrieval_scores: np.ndarray of shape (top_k,) — cosine similarity scores.
              - segment_embeddings: np.ndarray of shape (top_k, embedding_dim) — embeddings
                  of the retrieved segments (reused by Phase 2 for graph node features).
        """
        k = top_k or self.top_k
        k = min(k, len(segments))  # Can't retrieve more than available

        # Encode segments (use cache if same meeting)
        if meeting_id and meeting_id == self._cached_meeting_id and self._cached_embeddings is not None:
            all_embeddings = self._cached_embeddings
        else:
            texts = [seg.text for seg in segments]
            all_embeddings = self.encoder.encode(texts, show_progress=len(texts) > 100)
            if meeting_id:
                self._cached_meeting_id = meeting_id
                self._cached_embeddings = all_embeddings
                self._cached_segments = segments

        # Encode question
        question_emb = self.encoder.encode_single(question)

        # Compute similarities and get top-k indices
        similarities = self.encoder.cosine_similarity(question_emb, all_embeddings)
        top_indices = np.argsort(similarities)[::-1][:k]

        # Gather results
        retrieved_segments = [segments[i] for i in top_indices]
        retrieval_scores = similarities[top_indices]
        segment_embeddings = all_embeddings[top_indices]

        return retrieved_segments, retrieval_scores, segment_embeddings

    def get_question_embedding(self, question: str) -> np.ndarray:
        """Get the embedding for a question (used as the question node in the graph).

        Args:
            question: The question string.

        Returns:
            np.ndarray of shape (embedding_dim,).
        """
        return self.encoder.encode_single(question)
