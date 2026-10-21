"""Orchestration service connecting SLP, DL, and XAI layers."""

from __future__ import annotations

import numpy as np

from app import config
from app.dl.answer_generation import generate_answer_result
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

    import json
    from pathlib import Path
    import time

    # 1. Retrieval
    t0_retrieval = time.perf_counter()
    retrieval = retrieve_candidates(question, index, segments_by_id)
    retrieval_ms = round((time.perf_counter() - t0_retrieval) * 1000, 2)
    candidate_ids = [c.segment_id for c in retrieval.candidates]

    # 2. Graph Construction
    t0_graph = time.perf_counter()
    question_embedding = embed_texts([question])[0]
    segments = [segments_by_id[sid] for sid in candidate_ids]
    
    # Extract embeddings for the retrieved segments from the index
    segment_embeddings = np.array([index.get_embedding(sid) for sid in candidate_ids])

    builder = EvidenceGraphBuilder()
    data, metadata = builder.build(question_embedding, segments, segment_embeddings)
    graph_ms = round((time.perf_counter() - t0_graph) * 1000, 2)
    
    # 3. GAT Inference
    t0_gat = time.perf_counter()
    gat_output = get_gat_inference().run(data, metadata)
    gat_ms = round((time.perf_counter() - t0_gat) * 1000, 2)

    # 4. LLM Answer Generation. Preserve the GAT graph as-is for XAI, but
    # protect the answer context from weak GAT ranking by also including the
    # strongest raw retrieval candidates (GAT order remains first).
    t0_llm = time.perf_counter()
    answer_context_ids = list(dict.fromkeys(
        gat_output.top_evidence_ids + candidate_ids[:5]
    ))
    answer_result = generate_answer_result(
        question, gat_output, segments_by_id, answer_context_ids
    )
    llm_ms = round((time.perf_counter() - t0_llm) * 1000, 2)
    
    # 5. XAI Explainability: do not manufacture evidence for an unsupported answer.
    t0_shap = time.perf_counter()
    shap_highlights = explain_retrieval(question, retrieval, segments_by_id) if answer_result.supported else {}
    shap_ms = round((time.perf_counter() - t0_shap) * 1000, 2)

    # Prepare audio references for the frontend (as list with segment_id embedded)
    audio_refs = [
        AudioRef(
            segment_id=sid,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=seg.text,
            url=f"{config.API_PREFIX}/meetings/{meeting_id}/audio?start={seg.start_time}&end={seg.end_time}",
        )
        for sid, seg in segments_by_id.items()
        if answer_result.supported and sid in gat_output.top_evidence_ids
    ]

    # Trace diagnostics for This-Recording Insights tab
    retrieval_candidates = [
        {"segment_id": c.segment_id, "score": round(float(c.score), 4), "rank": idx + 1}
        for idx, c in enumerate(retrieval.candidates)
    ]
    gat_scores = {sid: float(gat_output.node_scores.get(sid, 0.0)) for sid in candidate_ids}
    gat_sorted = sorted(candidate_ids, key=lambda sid: gat_scores[sid], reverse=True)
    gat_ranks = {sid: rank + 1 for rank, sid in enumerate(gat_sorted)}
    gat_candidates = [
        {"segment_id": sid, "score": round(gat_scores[sid], 4), "rank": gat_ranks[sid]}
        for sid in candidate_ids
    ]
    rank_shifts = [
        {
            "segment_id": sid,
            "retrieval_rank": idx + 1,
            "gat_rank": gat_ranks[sid],
            "rank_shift": (idx + 1) - gat_ranks[sid],
            "retrieval_score": round(float(retrieval.candidates[idx].score), 4),
            "gat_score": round(gat_scores[sid], 4),
        }
        for idx, sid in enumerate(candidate_ids)
    ]

    if hasattr(data, "edge_type") and data.edge_type is not None:
        et = data.edge_type
        edge_counts_by_type = {
            "question_link": int((et == 0).sum().item()),
            "shared_entity": int((et == 1).sum().item()),
            "semantic_similarity": int((et == 2).sum().item()),
        }
    else:
        edge_counts_by_type = {"question_link": 0, "shared_entity": 0, "semantic_similarity": 0}

    # Gold evidence overlay: only if aligned labels exist and query matches
    gold_segment_ids = None
    processed_dir = config.DATA_DIR / "processed"
    aligned_candidates = []
    if record.metadata and record.metadata.get("original_filename"):
        orig_stem = Path(record.metadata["original_filename"]).stem
        aligned_candidates.append(processed_dir / f"{orig_stem}_aligned_labels.json")
    aligned_candidates.append(processed_dir / f"{meeting_id}_aligned_labels.json")

    for cand_path in aligned_candidates:
        if cand_path.is_file():
            try:
                with open(cand_path, encoding="utf-8") as f:
                    align_json = json.load(f)
                for q_obj in align_json.get("queries", []):
                    if q_obj.get("query", "").strip().lower() == question.strip().lower():
                        gold_segment_ids = q_obj.get("evidence_segment_ids", [])
                        break
                if gold_segment_ids is not None:
                    break
            except Exception:
                pass

    trace = {
        "retrieval_candidates": retrieval_candidates,
        "gat_candidates": gat_candidates,
        "rank_shifts": rank_shifts,
        "node_count": int(data.num_nodes),
        "edge_count": int(data.edge_index.size(1)),
        "edge_counts_by_type": edge_counts_by_type,
        "context_segment_ids": answer_context_ids,
        "stage_latency_ms": {
            "retrieval_ms": retrieval_ms,
            "graph_ms": graph_ms,
            "gat_ms": gat_ms,
            "llm_ms": llm_ms,
            "shap_ms": shap_ms,
            "total_ms": round(retrieval_ms + graph_ms + gat_ms + llm_ms + shap_ms, 2),
        },
        "gold_segment_ids": gold_segment_ids,
    }

    return QueryResponse(
        answer=answer_result.answer,
        answer_supported=answer_result.supported,
        gat_output=gat_output,  # Changed from evidence_graph
        shap_highlights=shap_highlights,
        audio_refs=audio_refs,
        trace=trace,
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
