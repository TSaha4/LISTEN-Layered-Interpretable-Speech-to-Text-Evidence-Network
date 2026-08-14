"""Construct evidence graphs from retrieved meeting segments."""

from __future__ import annotations

import itertools

import numpy as np

from app import config
from app.dl.embeddings import SegmentIndex, embed_texts
from app.models.schemas import EvidenceGraph, GraphEdge, GraphNode, SLPSegment


def _entity_overlap_weight(entities_a: set[str], entities_b: set[str]) -> float:
    """Jaccard similarity over normalised entity strings."""
    if not entities_a or not entities_b:
        return 0.0
    norm_a = {e.lower() for e in entities_a}
    norm_b = {e.lower() for e in entities_b}
    intersection = len(norm_a & norm_b)
    union = len(norm_a | norm_b)
    return intersection / union if union else 0.0


def _temporal_adjacency(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return 1.0 when segments are adjacent in time, else decay by gap."""
    gap = max(0.0, max(start_a, start_b) - min(end_a, end_b))
    if gap == 0.0:
        return 1.0
    # Soft decay: neighbouring turns within 5 seconds still connect weakly.
    return max(0.0, 1.0 - gap / 5.0)


def build_evidence_graph(
    candidate_ids: list[str],
    segments_by_id: dict[str, SLPSegment],
    index: SegmentIndex | None = None,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> EvidenceGraph:
    """Build a single-level evidence graph over retrieved segments.

    Nodes represent transcript segments; edges connect segments that share
    entities, are temporally adjacent, or are semantically similar.

    Args:
        candidate_ids: Segment ids from retrieval (defines node set).
        segments_by_id: Full meeting segment lookup.
        index: Optional FAISS index to fetch stored embeddings.
        precomputed_embeddings: Optional id → vector map (overrides index).

    Returns:
        ``EvidenceGraph`` with node embeddings and weighted edges.
    """
    nodes: list[GraphNode] = []
    valid_ids: list[str] = []

    for seg_id in candidate_ids:
        seg = segments_by_id.get(seg_id)
        if seg is None:
            continue

        if precomputed_embeddings and seg_id in precomputed_embeddings:
            emb = precomputed_embeddings[seg_id]
        elif index is not None:
            emb = index.get_embedding(seg_id)
            if emb is None:
                emb = embed_texts([seg.text])[0]
        else:
            emb = embed_texts([seg.text])[0]

        nodes.append(
            GraphNode(segment_id=seg_id, embedding=emb.astype(float).tolist())
        )
        valid_ids.append(seg_id)

    edges: list[GraphEdge] = []
    edge_candidates: list[tuple[str, str, float]] = []

    for src_id, tgt_id in itertools.combinations(valid_ids, 2):
        src = segments_by_id[src_id]
        tgt = segments_by_id[tgt_id]

        entity_w = _entity_overlap_weight(set(src.entities), set(tgt.entities))
        temporal_w = _temporal_adjacency(
            src.start_time, src.end_time, tgt.start_time, tgt.end_time
        )

        semantic_w = 0.0
        src_emb = np.array(next(n.embedding for n in nodes if n.segment_id == src_id))
        tgt_emb = np.array(next(n.embedding for n in nodes if n.segment_id == tgt_id))
        semantic_w = float(np.dot(src_emb, tgt_emb))  # vectors are L2-normalised

        combined = (
            config.GRAPH_ENTITY_EDGE_WEIGHT * entity_w
            + config.GRAPH_TEMPORAL_EDGE_WEIGHT * temporal_w
            + config.GRAPH_SEMANTIC_EDGE_WEIGHT * semantic_w
        )

        if combined >= config.GRAPH_SEMANTIC_EDGE_THRESHOLD:
            edge_candidates.append((src_id, tgt_id, combined))

    # Keep top edges per node to limit graph density for small hosts.
    per_node: dict[str, list[tuple[str, str, float]]] = {sid: [] for sid in valid_ids}
    for src, tgt, w in edge_candidates:
        per_node[src].append((src, tgt, w))
        per_node[tgt].append((src, tgt, w))

    kept: set[tuple[str, str]] = set()
    for sid in valid_ids:
        ranked = sorted(per_node[sid], key=lambda x: x[2], reverse=True)
        for src, tgt, w in ranked[: config.GRAPH_MAX_EDGES_PER_NODE]:
            key = tuple(sorted((src, tgt)))
            if key not in kept:
                kept.add(key)
                edges.append(GraphEdge(source=src, target=tgt, weight=float(w)))

    return EvidenceGraph(nodes=nodes, edges=edges)
