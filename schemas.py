"""
LISTEN DL Component — Input/Output Schemas
Pydantic models defining the contracts between Person A → Person B → Person C.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Input Contract (from Person A / SLP Layer)
# ──────────────────────────────────────────────

class MeetingSegment(BaseModel):
    """A single transcribed segment from a meeting recording.
    Produced by Person A's ASR + NER pipeline."""

    segment_id: str = Field(..., description="Unique segment identifier, e.g. 'seg_001'")
    text: str = Field(..., description="Transcribed text of this segment")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    speaker: str = Field(default="UNKNOWN", description="Speaker identifier")
    entities: List[str] = Field(default_factory=list, description="Named entities extracted by NER")


class MeetingInput(BaseModel):
    """Complete meeting transcript input from Person A."""

    meeting_id: str = Field(..., description="Unique meeting identifier")
    segments: List[MeetingSegment] = Field(..., min_length=1, description="List of transcript segments")


# ──────────────────────────────────────────────
# Output Contract (to Person C / XAI + Frontend)
# ──────────────────────────────────────────────

class EvidenceNode(BaseModel):
    """A node in the evidence graph — represents a transcript segment
    with its GAT-derived importance weight."""

    id: str = Field(..., description="Segment ID matching MeetingSegment.segment_id")
    text: str = Field(..., description="Segment text")
    speaker: str = Field(default="UNKNOWN", description="Speaker identifier")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    weight: float = Field(..., ge=0, le=1, description="GAT attention-derived importance score (0-1)")
    entities: List[str] = Field(default_factory=list, description="Entities in this segment")


class EvidenceEdge(BaseModel):
    """An edge in the evidence graph — connects two segments
    that share entities or have high semantic similarity."""

    source: str = Field(..., description="Source segment ID")
    target: str = Field(..., description="Target segment ID")
    weight: float = Field(..., ge=0, le=1, description="Edge attention weight from GAT")
    relation: str = Field(
        ...,
        description="Edge type: 'question_link', 'shared_entity:<entity>', or 'semantic_similarity'"
    )


class EvidenceGraph(BaseModel):
    """The full evidence graph output from the GAT reasoning layer.
    Used by Person C for visualization and counterfactual analysis."""

    nodes: List[EvidenceNode] = Field(..., description="Segment nodes with importance weights")
    edges: List[EvidenceEdge] = Field(..., description="Edges connecting related segments")


class PipelineOutput(BaseModel):
    """Complete output from Person B's DL pipeline.
    This is the contract Person C consumes."""

    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="Synthesized answer from LLM")
    evidence_graph: EvidenceGraph = Field(..., description="Evidence graph with nodes, edges, and weights")
    top_segments: List[str] = Field(
        ...,
        description="Ordered list of segment IDs by importance (highest first)"
    )
    model_metadata: Optional[ModelMetadata] = Field(
        default=None,
        description="Optional metadata about the model run"
    )


class ModelMetadata(BaseModel):
    """Optional metadata about a pipeline run — useful for debugging and evaluation."""

    retrieval_scores: Optional[dict] = Field(
        default=None,
        description="Mapping of segment_id → retrieval cosine similarity score"
    )
    gat_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to the GAT checkpoint used for this inference"
    )
    num_segments_retrieved: int = Field(default=0, description="Number of segments retrieved in Phase 1")
    num_segments_in_graph: int = Field(default=0, description="Number of nodes in the evidence graph")


# Fix forward reference — PipelineOutput references ModelMetadata
# which is defined after it. Pydantic v2 handles this with model_rebuild.
PipelineOutput.model_rebuild()
