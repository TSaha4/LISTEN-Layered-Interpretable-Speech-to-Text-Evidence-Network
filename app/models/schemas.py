"""Pydantic schemas enforcing strict data contracts across LISTEN layers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SLP layer
# ---------------------------------------------------------------------------


class SLPSegment(BaseModel):
    """Output of the Speech & Language Processing layer for one segment."""

    segment_id: str
    start_time: float
    end_time: float
    text: str
    entities: list[str]


# ---------------------------------------------------------------------------
# DL layer
# ---------------------------------------------------------------------------


class RetrievalCandidate(BaseModel):
    """One retrieved segment with similarity score."""

    segment_id: str
    score: float


class RetrievalResult(BaseModel):
    """Output of bi-encoder top-k retrieval."""

    question: str
    candidates: list[RetrievalCandidate]


class GraphNode(BaseModel):
    """Node in the evidence graph."""

    segment_id: str
    embedding: list[float]


class GraphEdge(BaseModel):
    """Weighted edge between two evidence segments."""

    source: str
    target: str
    weight: float


class EvidenceGraph(BaseModel):
    """Output of graph_builder."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GATOutput(BaseModel):
    """Output of GAT multi-hop reasoning."""

    node_scores: dict[str, float]
    edge_scores: dict[str, float]
    top_evidence_ids: list[str]


class WordHighlight(BaseModel):
    """SHAP word-level highlight for one token."""

    word: str
    score: float


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Response after uploading and processing a meeting recording."""

    meeting_id: str
    segment_count: int
    segments: list[SLPSegment]
    message: str = "Processing complete."


class QueryRequest(BaseModel):
    """Ask-a-question payload."""

    meeting_id: str
    question: str


class AudioRef(BaseModel):
    """Audio clip reference for a segment."""

    segment_id: str
    start_time: float
    end_time: float
    text: str
    url: str


class QueryResponse(BaseModel):
    """Final structured JSON returned to the frontend."""

    answer: str
    gat_output: GATOutput  # Changed from evidence_graph to match frontend
    shap_highlights: dict[str, list[WordHighlight]]
    audio_refs: list[AudioRef]  # Changed from dict to list to match frontend


class CounterfactualRequest(BaseModel):
    """Remove one evidence node and re-run reasoning."""

    meeting_id: str
    question: str
    node_id_to_remove: str
    original_answer: str
    original_evidence_graph: GATOutput


class CounterfactualResponse(BaseModel):
    """Diff after counterfactual node removal."""

    original_answer: str
    counterfactual_answer: str
    removed_node_id: str
    answer_changed: bool
    evidence_graph: GATOutput
    shap_highlights: dict[str, list[WordHighlight]]


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = "ok"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    code: str | None = None


# ---------------------------------------------------------------------------
# Internal / training helpers
# ---------------------------------------------------------------------------


class MeetingRecord(BaseModel):
    """Persisted meeting state after SLP processing."""

    meeting_id: str
    audio_path: str
    segments: list[SLPSegment]
    index_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
