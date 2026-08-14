"""
Tests for Phase 2: Graph Construction + GAT Model
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import MeetingSegment
from phase1_retrieval.bi_encoder import BiEncoder
from phase2_graph.graph_builder import EvidenceGraphBuilder
from phase2_graph.gat_model import EvidenceGAT


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
        text="Users prefer rubber because it has better grip.",
        start_time=10.0, end_time=15.0, speaker="ME",
        entities=["users", "rubber", "grip"],
    ),
    MeetingSegment(
        segment_id="seg_004",
        text="The budget for marketing is finalized.",
        start_time=15.0, end_time=20.0, speaker="ME",
        entities=["budget", "marketing"],
    ),
]


@pytest.fixture(scope="module")
def encoder():
    return BiEncoder()


@pytest.fixture(scope="module")
def embeddings(encoder):
    texts = [seg.text for seg in SAMPLE_SEGMENTS]
    return encoder.encode(texts)


@pytest.fixture(scope="module")
def question_embedding(encoder):
    return encoder.encode_single("What material was chosen for the remote?")


# ── Graph Builder Tests ──

class TestGraphBuilder:
    """Test evidence graph construction."""

    @pytest.fixture
    def builder(self):
        return EvidenceGraphBuilder(similarity_threshold=0.3, min_shared_entities=1)

    def test_graph_has_correct_num_nodes(self, builder, question_embedding, embeddings):
        """Graph should have N+1 nodes (question + N segments)."""
        data, meta = builder.build(question_embedding, SAMPLE_SEGMENTS, embeddings)
        assert data.num_nodes == len(SAMPLE_SEGMENTS) + 1
        assert data.x.shape == (len(SAMPLE_SEGMENTS) + 1, 384)

    def test_question_edges_created(self, builder, question_embedding, embeddings):
        """Every segment should be connected to the question node."""
        data, meta = builder.build(question_embedding, SAMPLE_SEGMENTS, embeddings)
        # Should have 2 * N question edges (bidirectional)
        assert meta["num_question_edges"] == 2 * len(SAMPLE_SEGMENTS)

    def test_entity_edges_created(self, builder, question_embedding, embeddings):
        """Segments sharing entities should be connected."""
        data, meta = builder.build(question_embedding, SAMPLE_SEGMENTS, embeddings)
        # seg_002 and seg_003 share "rubber" → should have entity edges
        assert meta["num_entity_edges"] > 0, "Expected entity edges for segments sharing 'rubber'"

    def test_edge_types_valid(self, builder, question_embedding, embeddings):
        """All edge types should be 0, 1, or 2."""
        data, _ = builder.build(question_embedding, SAMPLE_SEGMENTS, embeddings)
        edge_types = data.edge_type.numpy()
        assert all(t in [0, 1, 2] for t in edge_types), f"Invalid edge types: {set(edge_types)}"

    def test_metadata_node_mapping(self, builder, question_embedding, embeddings):
        """Metadata should correctly map node indices to segment IDs."""
        _, meta = builder.build(question_embedding, SAMPLE_SEGMENTS, embeddings)
        node_map = meta["node_id_map"]
        assert node_map[0] == "__question__"
        assert node_map[1] == "seg_001"
        assert node_map[2] == "seg_002"

    def test_build_with_labels(self, builder, question_embedding, embeddings):
        """build_with_labels should attach binary labels to nodes."""
        data, _ = builder.build_with_labels(
            question_embedding, SAMPLE_SEGMENTS, embeddings,
            supporting_ids=["seg_002", "seg_003"],
        )
        assert hasattr(data, "y")
        labels = data.y.numpy()
        assert labels[0] == 0.0   # question node
        assert labels[1] == 0.0   # seg_001 (not supporting)
        assert labels[2] == 1.0   # seg_002 (supporting)
        assert labels[3] == 1.0   # seg_003 (supporting)
        assert labels[4] == 0.0   # seg_004 (not supporting)

    def test_empty_entities_no_crash(self, builder, question_embedding, embeddings):
        """Segments with no entities should not crash graph construction."""
        segs = [
            MeetingSegment(
                segment_id="seg_a", text="Hello", start_time=0, end_time=1,
                speaker="A", entities=[],
            ),
            MeetingSegment(
                segment_id="seg_b", text="World", start_time=1, end_time=2,
                speaker="B", entities=[],
            ),
        ]
        embs = np.random.randn(2, 384).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        q_emb = np.random.randn(384).astype(np.float32)

        data, meta = builder.build(q_emb, segs, embs)
        assert data.num_nodes == 3
        assert meta["num_entity_edges"] == 0


# ── GAT Model Tests ──

class TestEvidenceGAT:
    """Test the GAT model architecture."""

    @pytest.fixture
    def model(self):
        return EvidenceGAT(
            input_dim=384,
            hidden_dim=64,    # Small for testing
            output_dim=32,
            num_heads=2,
            num_layers=2,
            dropout=0.0,      # No dropout for deterministic tests
        )

    @pytest.fixture
    def sample_graph(self, question_embedding, embeddings):
        builder = EvidenceGraphBuilder()
        data, _ = builder.build(question_embedding, SAMPLE_SEGMENTS, embeddings)
        return data

    def test_forward_output_shape(self, model, sample_graph):
        """Forward pass should produce correct output shapes."""
        output = model(sample_graph)
        n = sample_graph.num_nodes

        assert output["node_weights"].shape == (n,), f"Expected ({n},), got {output['node_weights'].shape}"
        assert output["node_embeddings"].shape == (n, 32), f"Expected ({n}, 32), got {output['node_embeddings'].shape}"
        assert output["logits"].shape == (n,), f"Expected ({n},), got {output['logits'].shape}"

    def test_weights_in_valid_range(self, model, sample_graph):
        """Node weights should be in [0, 1] (sigmoid output)."""
        output = model(sample_graph)
        weights = output["node_weights"].detach().numpy()
        assert weights.min() >= 0.0, f"Min weight {weights.min()} < 0"
        assert weights.max() <= 1.0, f"Max weight {weights.max()} > 1"

    def test_gradient_flow(self, model, sample_graph):
        """Gradients should flow through the model."""
        sample_graph.y = torch.zeros(sample_graph.num_nodes)
        sample_graph.y[2] = 1.0  # Mark seg_002 as supporting

        output = model(sample_graph)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"], sample_graph.y
        )
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowing through the model"

    def test_model_parameter_count(self, model):
        """Model should have a reasonable number of parameters."""
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model has no parameters"
        assert num_params < 10_000_000, f"Model too large: {num_params:,} params"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
