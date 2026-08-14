"""Unit tests for the Deep Learning layer (retrieval, graph, GAT)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.dl.gat_model import EvidenceGAT, evidence_graph_to_pyg, gat_output_from_logits
from app.dl.graph_builder import build_evidence_graph
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import EvidenceGraph, GraphEdge, GraphNode, SLPSegment


def _mock_segments() -> dict[str, SLPSegment]:
    """Three segments with shared entities and temporal adjacency."""
    segments = [
        SLPSegment(
            segment_id="m_seg_0000",
            start_time=0.0,
            end_time=5.0,
            text="Alice discussed the project budget with the team.",
            entities=["Alice"],
        ),
        SLPSegment(
            segment_id="m_seg_0001",
            start_time=5.0,
            end_time=10.0,
            text="Bob agreed the budget should increase for hardware.",
            entities=["Bob"],
        ),
        SLPSegment(
            segment_id="m_seg_0002",
            start_time=10.0,
            end_time=15.0,
            text="Alice confirmed the deadline is next Friday.",
            entities=["Alice"],
        ),
    ]
    return {s.segment_id: s for s in segments}


def _fixed_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(384).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


class TestGraphBuilder:
    """Verify evidence graph node/edge construction from mock segments."""

    def test_builds_one_node_per_candidate(self) -> None:
        segments_by_id = _mock_segments()
        candidate_ids = list(segments_by_id.keys())
        embeddings = {sid: _fixed_embedding(i) for i, sid in enumerate(candidate_ids)}

        graph = build_evidence_graph(
            candidate_ids,
            segments_by_id,
            precomputed_embeddings=embeddings,
        )

        assert len(graph.nodes) == 3
        assert all(isinstance(n, GraphNode) for n in graph.nodes)
        assert all(len(n.embedding) == 384 for n in graph.nodes)

    def test_entity_overlap_creates_edges(self) -> None:
        segments_by_id = _mock_segments()
        candidate_ids = ["m_seg_0000", "m_seg_0002"]  # both mention Alice
        embeddings = {sid: _fixed_embedding(i) for i, sid in enumerate(candidate_ids)}

        graph = build_evidence_graph(
            candidate_ids,
            segments_by_id,
            precomputed_embeddings=embeddings,
        )

        assert len(graph.edges) >= 1
        edge = graph.edges[0]
        assert isinstance(edge, GraphEdge)
        assert edge.weight > 0.0
        pair = {edge.source, edge.target}
        assert pair == {"m_seg_0000", "m_seg_0002"}

    def test_temporal_adjacency_between_consecutive_segments(self) -> None:
        segments_by_id = _mock_segments()
        candidate_ids = ["m_seg_0000", "m_seg_0001"]
        # Identical embeddings → semantic component is high.
        emb = _fixed_embedding(0)
        embeddings = {sid: emb.copy() for sid in candidate_ids}

        graph = build_evidence_graph(
            candidate_ids,
            segments_by_id,
            precomputed_embeddings=embeddings,
        )

        assert len(graph.edges) >= 1


class TestRetrieval:
    """Bi-encoder retrieval over a mocked FAISS index."""

    def test_retrieve_returns_scored_candidates(self) -> None:
        segments_by_id = _mock_segments()
        mock_index = MagicMock()
        mock_index.search.return_value = [
            ("m_seg_0000", 0.92),
            ("m_seg_0001", 0.81),
        ]

        result = retrieve_candidates(
            "What was said about the budget?",
            mock_index,
            segments_by_id,
            top_k=2,
        )

        assert result.question == "What was said about the budget?"
        assert len(result.candidates) == 2
        assert result.candidates[0].segment_id == "m_seg_0000"
        assert result.candidates[0].score == pytest.approx(0.92)


class TestGATModel:
    """Verify GAT forward-pass tensor shapes on a synthetic graph."""

    def test_forward_output_shapes(self) -> None:
        model = EvidenceGAT(in_dim=384, hidden_dim=64, num_heads=2, num_layers=2)
        num_nodes = 4
        num_edges = 6

        x = torch.randn(num_nodes, 384)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
            dtype=torch.long,
        )
        edge_attr = torch.rand(num_edges)

        node_logits, edge_logits = model(x, edge_index, edge_attr)

        assert node_logits.shape == (num_nodes, 1)
        assert edge_logits.shape == (num_edges, 1)

    def test_empty_graph(self) -> None:
        model = EvidenceGAT(in_dim=384, hidden_dim=64, num_heads=2, num_layers=1)
        x = torch.zeros((0, 384))
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_logits, edge_logits = model(x, edge_index, None)
        assert node_logits.shape == (0, 1)
        assert edge_logits.shape == (0, 1)

    def test_evidence_graph_to_pyg_roundtrip(self) -> None:
        graph = EvidenceGraph(
            nodes=[
                GraphNode(segment_id="a", embedding=[1.0, 0.0]),
                GraphNode(segment_id="b", embedding=[0.0, 1.0]),
            ],
            edges=[GraphEdge(source="a", target="b", weight=0.8)],
        )
        data = evidence_graph_to_pyg(graph)
        assert data.x.shape == (2, 2)
        assert data.edge_index.shape[0] == 2
        # Undirected: one edge → two directed edges
        assert data.edge_index.shape[1] == 2
        assert data.segment_ids == ["a", "b"]

    def test_gat_output_contract(self) -> None:
        segment_ids = ["a", "b", "c"]
        node_logits = torch.tensor([[2.0], [0.5], [-1.0]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_logits = torch.tensor([[1.0]])

        output = gat_output_from_logits(
            segment_ids, node_logits, edge_index, edge_logits, top_k=2
        )

        assert set(output.node_scores.keys()) == set(segment_ids)
        assert len(output.top_evidence_ids) == 2
        assert output.top_evidence_ids[0] == "a"
        assert "a_b" in output.edge_scores

    def test_single_layer_gat(self) -> None:
        """Single GAT layer should still produce valid shapes."""
        model = EvidenceGAT(in_dim=384, hidden_dim=128, num_heads=4, num_layers=1)
        x = torch.randn(3, 384)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        node_logits, edge_logits = model(x, edge_index, None)
        assert node_logits.shape == (3, 1)
        assert edge_logits.shape == (2, 1)
