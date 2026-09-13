"""
LISTEN Phase 2 — Evidence Graph Construction

Builds a PyTorch Geometric (PyG) graph from retrieved transcript segments.

Graph structure (adapted from HGN §3.1 for spoken meeting segments):
  - Nodes: [question_node, segment_0, segment_1, ..., segment_k-1]
  - Edge type 0: question ↔ every segment
  - Edge type 1: segment ↔ segment if they share ≥1 entity
  - Edge type 2: segment ↔ segment if cosine similarity > threshold

This is a simplified version of HGN's 4-level hierarchy (Q → P → S → E),
collapsed to a flat segment graph because spoken transcript segments are
already atomic utterances (unlike multi-sentence Wikipedia paragraphs).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

import config
from schemas import MeetingSegment


class EvidenceGraphBuilder:
    """Constructs a PyG Data object from retrieved segments and their embeddings.

    The graph is the input to the GAT model for multi-hop reasoning.
    """

    def __init__(
        self,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        min_shared_entities: int = config.MIN_SHARED_ENTITIES,
    ):
        """
        Args:
            similarity_threshold: Cosine sim threshold for type-2 edges.
            min_shared_entities: Minimum shared entities for type-1 edges.
        """
        self.sim_threshold = similarity_threshold
        self.min_shared_entities = min_shared_entities

    def build(
        self,
        question_embedding: np.ndarray,
        segments: List[MeetingSegment],
        segment_embeddings: np.ndarray,
    ) -> Tuple[Data, dict]:
        """Build the evidence graph as a PyG Data object.

        Args:
            question_embedding: Shape (embedding_dim,) — question vector.
            segments: List of retrieved MeetingSegment objects.
            segment_embeddings: Shape (N, embedding_dim) — segment vectors.

        Returns:
            Tuple of:
              - data: PyG Data with x (node features), edge_index, edge_type, edge_attr.
              - metadata: Dict mapping node indices to segment IDs and edge info.
        """
        num_segments = len(segments)
        num_nodes = 1 + num_segments  # node 0 = question, nodes 1..N = segments

        # ── Node features ──
        # Stack question embedding + segment embeddings
        node_features = np.vstack([
            question_embedding.reshape(1, -1),
            segment_embeddings,
        ])
        x = torch.tensor(node_features, dtype=torch.float32)

        # ── Build edges ──
        edge_sources = []
        edge_targets = []
        edge_types = []
        edge_relations = []  # Human-readable edge descriptions

        # Type 0: Question ↔ every segment (bidirectional)
        for i in range(num_segments):
            seg_node = i + 1  # offset by 1 (node 0 = question)

            # Question → segment
            edge_sources.append(0)
            edge_targets.append(seg_node)
            edge_types.append(0)
            edge_relations.append("question_link")

            # Segment → question
            edge_sources.append(seg_node)
            edge_targets.append(0)
            edge_types.append(0)
            edge_relations.append("question_link")

        # Type 1: Segment ↔ segment via shared entities
        for i in range(num_segments):
            entities_i = set(e.lower() for e in segments[i].entities)
            if not entities_i:
                continue

            for j in range(i + 1, num_segments):
                entities_j = set(e.lower() for e in segments[j].entities)
                shared = entities_i & entities_j

                if len(shared) >= self.min_shared_entities:
                    node_i = i + 1
                    node_j = j + 1
                    shared_str = ";".join(sorted(shared))

                    # Bidirectional
                    edge_sources.extend([node_i, node_j])
                    edge_targets.extend([node_j, node_i])
                    edge_types.extend([1, 1])
                    edge_relations.extend([
                        f"shared_entity:{shared_str}",
                        f"shared_entity:{shared_str}",
                    ])

        # Type 2: Segment ↔ segment via cosine similarity
        if num_segments > 1:
            # Compute pairwise cosine similarity (embeddings are L2-normalized)
            sim_matrix = segment_embeddings @ segment_embeddings.T

            for i in range(num_segments):
                for j in range(i + 1, num_segments):
                    if sim_matrix[i, j] > self.sim_threshold:
                        node_i = i + 1
                        node_j = j + 1

                        # Don't add duplicate edge if already connected by entity
                        pair_key = (min(node_i, node_j), max(node_i, node_j))
                        existing_pairs = set()
                        for s, t, et in zip(edge_sources, edge_targets, edge_types):
                            if et == 1:
                                existing_pairs.add((min(s, t), max(s, t)))

                        if pair_key not in existing_pairs:
                            edge_sources.extend([node_i, node_j])
                            edge_targets.extend([node_j, node_i])
                            edge_types.extend([2, 2])
                            edge_relations.extend([
                                "semantic_similarity",
                                "semantic_similarity",
                            ])

        # ── Assemble PyG Data ──
        if edge_sources:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            # Fallback: no edges (isolated nodes). Add self-loops to avoid empty graph.
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
        )

        # ── Metadata for later use (XAI, visualization) ──
        node_id_map = {0: "__question__"}
        for i, seg in enumerate(segments):
            node_id_map[i + 1] = seg.segment_id

        metadata = {
            "node_id_map": node_id_map,           # node_index → segment_id
            "edge_relations": edge_relations,      # per-edge human-readable relation
            "num_question_edges": num_segments * 2,
            "num_entity_edges": edge_types.count(1) if edge_types else 0,
            "num_similarity_edges": edge_types.count(2) if edge_types else 0,
        }

        return data, metadata

    def build_with_labels(
        self,
        question_embedding: np.ndarray,
        segments: List[MeetingSegment],
        segment_embeddings: np.ndarray,
        supporting_ids: List[str],
    ) -> Tuple[Data, dict]:
        """Build the evidence graph with ground-truth labels (for training).

        Same as build(), but additionally attaches binary node labels indicating
        which segments are ground-truth supporting facts.

        Args:
            supporting_ids: List of segment_ids that are ground-truth supporting facts.

        Returns:
            Same as build(), but Data.y contains binary labels [0 or 1] per node.
        """
        data, metadata = self.build(question_embedding, segments, segment_embeddings)

        # Build labels: node 0 = question (label 0), nodes 1..N = segments
        labels = [0.0]  # question node is not a supporting fact
        for seg in segments:
            labels.append(1.0 if seg.segment_id in supporting_ids else 0.0)

        data.y = torch.tensor(labels, dtype=torch.float32)

        return data, metadata
