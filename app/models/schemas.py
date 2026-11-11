"""Pydantic schemas enforcing strict data contracts across LISTEN layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SLP layer
# ---------------------------------------------------------------------------


class WordToken(BaseModel):
    """One ASR word with its timing and recognition probability."""

    word: str
    start: float
    end: float
    probability: float


class SLPSegment(BaseModel):
    """Output of the Speech & Language Processing layer for one segment.

    The uncertainty fields default to "fully reliable" so meetings persisted
    before word-level ASR confidence was captured still load unchanged.
    """

    segment_id: str
    start_time: float
    end_time: float
    text: str
    entities: list[str]
    words: list[WordToken] = Field(default_factory=list)
    asr_confidence: float = 1.0       # mean word probability
    min_word_confidence: float = 1.0  # weakest word in the segment
    hesitation_score: float = 0.0     # 0 = fluent, 1 = heavily hesitant
    no_speech_prob: float = 0.0
    speaker: str | None = None


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
    """Output of GAT multi-hop reasoning.

    ``node_scores`` are reliability-gated when gating is on; the ungated
    values are kept in ``raw_node_scores`` so the effect can be inspected.
    """

    node_scores: dict[str, float]
    edge_scores: dict[str, float]
    top_evidence_ids: list[str]
    raw_node_scores: dict[str, float] = Field(default_factory=dict)
    node_reliability: dict[str, float] = Field(default_factory=dict)
    edge_confidence: dict[str, float] = Field(default_factory=dict)


class WordHighlight(BaseModel):
    """SHAP word-level highlight for one token."""

    word: str
    score: float


class UncertainWord(BaseModel):
    """A low-confidence ASR word that the answer's evidence depends on."""

    segment_id: str
    word: str
    start: float
    end: float
    asr_probability: float
    shap_influence: float = 0.0


class AnswerReliability(BaseModel):
    """Calibrated reliability of one generated answer."""

    score: float                      # probability-like value in [0, 1]
    level: str                        # "high" | "medium" | "low"
    components: dict[str, float]
    uncertain_words: list[UncertainWord] = Field(default_factory=list)
    warning: str | None = None


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Response after uploading and processing a meeting recording."""

    meeting_id: str
    segment_count: int
    segments: list[SLPSegment]
    message: str = "Processing complete."
    stage_timings: dict[str, float] | None = None


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
    answer_supported: bool = True
    gat_output: GATOutput  # Changed from evidence_graph to match frontend
    shap_highlights: dict[str, list[WordHighlight]]
    audio_refs: list[AudioRef]  # Changed from dict to list to match frontend
    reliability: AnswerReliability | None = None
    trace: dict[str, Any] | None = None


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


# ---------------------------------------------------------------------------
# Commitments — cross-meeting decision tracking
# ---------------------------------------------------------------------------


class Commitment(BaseModel):
    """A decision or promise stated in a meeting, with its provenance."""

    commitment_id: str
    meeting_id: str
    segment_id: str
    text: str
    kind: str                         # "decision" | "commitment" | "deadline"
    owner: str | None = None          # PERSON entity when found
    topic_entities: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)  # DATE/MONEY/CARDINAL...
    polarity: int = 1                 # +1 affirm, -1 negated
    meeting_time: float               # epoch seconds of the meeting
    offset_s: float                   # seconds into the meeting
    reliability: float = 1.0          # from the source segment
    status: str = "active"            # "active" | "superseded" | "merged" | "disputed"
    superseded_by: str | None = None  # the record this one resolved into


class CommitmentEdge(BaseModel):
    """Relation between two commitments on the same topic."""

    source: str
    target: str
    relation: str                     # "supersedes" | "reaffirms" | "same_topic"
    reason: str
    similarity: float


class CommitmentTimeline(BaseModel):
    """Answer to 'what is the current decision on X and how did it change?'."""

    topic: str
    current: Commitment | None
    history: list[Commitment]
    edges: list[CommitmentEdge]
    reliability: float


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
