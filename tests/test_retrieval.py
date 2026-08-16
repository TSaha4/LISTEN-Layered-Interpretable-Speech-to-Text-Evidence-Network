"""
Tests for Phase 1: Bi-Encoder Retrieval
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import MeetingSegment
from phase1_retrieval.bi_encoder import BiEncoder
from phase1_retrieval.retriever import SegmentRetriever


# ── Test Fixtures ──

SAMPLE_SEGMENTS = [
    MeetingSegment(
        segment_id="seg_001",
        text="We need to decide on the remote control material.",
        start_time=0.0, end_time=5.0, speaker="PM",
        entities=["remote control", "material"],
    ),
    MeetingSegment(
        segment_id="seg_002",
        text="Rubber is more comfortable to hold than plastic.",
        start_time=5.0, end_time=10.0, speaker="ID",
        entities=["rubber", "plastic"],
    ),
    MeetingSegment(
        segment_id="seg_003",
        text="The budget for next quarter's marketing campaign is finalized.",
        start_time=10.0, end_time=15.0, speaker="ME",
        entities=["budget", "marketing campaign"],
    ),
    MeetingSegment(
        segment_id="seg_004",
        text="Users prefer rubber because it has better grip.",
        start_time=15.0, end_time=20.0, speaker="ME",
        entities=["users", "rubber", "grip"],
    ),
    MeetingSegment(
        segment_id="seg_005",
        text="Let's schedule the product launch for March.",
        start_time=20.0, end_time=25.0, speaker="PM",
        entities=["product launch", "March"],
    ),
]


# ── BiEncoder Tests ──

class TestBiEncoder:
    """Test the bi-encoder embedding model."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return BiEncoder()

    def test_encode_single(self, encoder):
        """Single text encoding produces correct shape."""
        emb = encoder.encode_single("Hello world")
        assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"

    def test_encode_batch(self, encoder):
        """Batch encoding produces correct shape."""
        texts = ["Hello", "World", "Test"]
        embs = encoder.encode(texts)
        assert embs.shape == (3, 384), f"Expected (3, 384), got {embs.shape}"

    def test_embeddings_normalized(self, encoder):
        """Embeddings should be L2-normalized."""
        emb = encoder.encode_single("Test sentence")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Expected norm ≈ 1.0, got {norm}"

    def test_similar_texts_high_similarity(self, encoder):
        """Semantically similar texts should have high cosine similarity."""
        emb1 = encoder.encode_single("The remote control is made of rubber")
        emb2 = encoder.encode_single("The TV remote has a rubber casing")
        emb3 = encoder.encode_single("The weather forecast predicts rain tomorrow")

        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13, (
            f"Similar texts should have higher similarity: "
            f"sim(remote,remote)={sim_12:.3f} vs sim(remote,weather)={sim_13:.3f}"
        )

    def test_cosine_similarity(self, encoder):
        """Cosine similarity computation between query and corpus."""
        query_emb = encoder.encode_single("rubber material")
        corpus_embs = encoder.encode(["rubber is soft", "wood is hard", "plastic is cheap"])

        sims = encoder.cosine_similarity(query_emb, corpus_embs)
        assert sims.shape == (3,), f"Expected (3,), got {sims.shape}"
        assert sims[0] > sims[1], "Rubber should be more similar to rubber"

    def test_pairwise_similarity(self, encoder):
        """Pairwise similarity matrix is symmetric."""
        embs = encoder.encode(["Hello", "World", "Test"])
        sim_matrix = encoder.pairwise_similarity(embs)

        assert sim_matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T, decimal=5)


# ── Retriever Tests ──

class TestSegmentRetriever:
    """Test the segment retrieval pipeline."""

    @pytest.fixture(scope="class")
    def retriever(self):
        return SegmentRetriever(top_k=3)

    def test_retrieve_correct_count(self, retriever):
        """Should return exactly top_k segments."""
        segments, scores, embeddings = retriever.retrieve(
            question="What material was chosen for the remote?",
            segments=SAMPLE_SEGMENTS,
            top_k=3,
        )
        assert len(segments) == 3
        assert len(scores) == 3
        assert embeddings.shape == (3, 384)

    def test_retrieve_relevant_segments(self, retriever):
        """Relevant segments should rank higher than irrelevant ones."""
        segments, scores, _ = retriever.retrieve(
            question="What material was chosen for the remote control?",
            segments=SAMPLE_SEGMENTS,
            top_k=3,
        )

        retrieved_ids = [s.segment_id for s in segments]
        # seg_001, seg_002, seg_004 are about materials/remote — should appear
        # seg_003 (budget), seg_005 (launch) should not
        assert "seg_003" not in retrieved_ids or "seg_005" not in retrieved_ids, (
            f"Irrelevant segments should not dominate top-3: {retrieved_ids}"
        )

    def test_scores_descending(self, retriever):
        """Retrieval scores should be in descending order."""
        _, scores, _ = retriever.retrieve(
            question="What material was chosen?",
            segments=SAMPLE_SEGMENTS,
            top_k=3,
        )
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores should be descending: {scores}"
            )

    def test_retrieve_all_when_k_exceeds_segments(self, retriever):
        """When top_k > num_segments, return all segments."""
        segments, _, _ = retriever.retrieve(
            question="Anything",
            segments=SAMPLE_SEGMENTS,
            top_k=100,
        )
        assert len(segments) == len(SAMPLE_SEGMENTS)

    def test_caching(self, retriever):
        """Caching should return same results for same meeting."""
        segs1, _, _ = retriever.retrieve(
            "What material?", SAMPLE_SEGMENTS, meeting_id="test_meeting"
        )
        segs2, _, _ = retriever.retrieve(
            "What material?", SAMPLE_SEGMENTS, meeting_id="test_meeting"
        )
        assert [s.segment_id for s in segs1] == [s.segment_id for s in segs2]

    def test_question_embedding(self, retriever):
        """Question embedding should have correct shape."""
        emb = retriever.get_question_embedding("What was decided?")
        assert emb.shape == (384,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
