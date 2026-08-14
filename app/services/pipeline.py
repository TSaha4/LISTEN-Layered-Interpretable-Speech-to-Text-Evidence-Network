"""Orchestration service connecting SLP, DL, and XAI layers."""

from __future__ import annotations

import numpy as np

from app import config
from app.dl.answer_generation import generate_answer
from app.dl.embeddings import embed_texts
from app.dl.graph_builder import EvidenceGraphBuilder
from app.dl.inference import get_gat_inference
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import (
    AudioRef,
    CounterfactualResponse,
    GATOutput,
    QueryResponse,
)
from app.services.meeting_store import get_meeting_store
from app.xai.counterfactual import run_counterfactual
from app.xai.shap_explainer import explain_retrieval


def run_query_pipeline(meeting_id: str, question: str) -> QueryResponse:
    """Execute the full DL + XAI pipeline for a question.

    Args:
        meeting_id: Previously processed meeting identifier.
        question: Natural-language question.

    Returns:
        ``QueryResponse`` matching the final API contract.

    Raises:
        ValueError: If the meeting is not found.
    """
    store = get_meeting_store()
    record = store.get(meeting_id)
    if record is None:
        raise ValueError(f"Meeting not found: {meeting_id}")

    index = store.load_index(meeting_id)
    if index is None:
        raise ValueError(f"Index not found for meeting: {meeting_id}")

    segments_by_id = store.segments_by_id(meeting_id)

    # 1. Retrieval
    retrieval = retrieve_candidates(question, index, segments_by_id)
    candidate_ids = [c.segment_id for c in retrieval.candidates]

    # 2. Graph Construction
    question_embedding = embed_texts([question])[0]
    segments = [segments_by_id[sid] for sid in candidate_ids]
    
    # Extract embeddings for the retrieved segments from the index
    segment_embeddings = np.array([index.get_embedding(sid) for sid in candidate_ids])

    builder = EvidenceGraphBuilder()
    data, metadata = builder.build(question_embedding, segments, segment_embeddings)
    
    # 3. GAT Inference
    gat_output = get_gat_inference().run(data, metadata)

    # 4. LLM Answer Generation
    answer = generate_answer(question, gat_output, segments_by_id)
    
    # 5. XAI Explainability
    shap_highlights = explain_retrieval(question, retrieval, segments_by_id)

    # Prepare audio references for the frontend
    audio_refs = {
        sid: AudioRef(
            start_time=seg.start_time,
            end_time=seg.end_time,
            url=f"{config.API_PREFIX}/meetings/{meeting_id}/audio?start={seg.start_time}&end={seg.end_time}",
        )
        for sid, seg in segments_by_id.items()
        if sid in gat_output.top_evidence_ids
    }

    return QueryResponse(
        answer=answer,
        evidence_graph=gat_output,
        shap_highlights=shap_highlights,
        audio_refs=audio_refs,
    )


def run_counterfactual_pipeline(
    meeting_id: str,
    question: str,
    node_id_to_remove: str,
    original_answer: str,
    original_evidence_graph: GATOutput,
) -> CounterfactualResponse:
    """Remove an evidence node and compare the new answer."""
    return run_counterfactual(
        meeting_id=meeting_id,
        question=question,
        node_id_to_remove=node_id_to_remove,
        original_answer=original_answer,
        original_evidence_graph=original_evidence_graph,
    )
