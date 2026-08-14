"""Counterfactual evidence validation by removing one graph node."""

from __future__ import annotations

from app.dl.answer_generation import generate_answer
from app.dl.graph_builder import build_evidence_graph
from app.dl.inference import get_gat_inference
from app.dl.retrieval import retrieve_candidates
from app.models.schemas import CounterfactualResponse, GATOutput
from app.services.meeting_store import get_meeting_store
from app.xai.shap_explainer import explain_retrieval


def run_counterfactual(
    meeting_id: str,
    question: str,
    node_id_to_remove: str,
    original_answer: str,
    original_evidence_graph: GATOutput,
) -> CounterfactualResponse:
    """Remove one evidence node, re-run DL pipeline, and diff answers.

    Args:
        meeting_id: Processed meeting identifier.
        question: Original user question.
        node_id_to_remove: Segment id to drop from the evidence graph.
        original_answer: Answer from the unmodified pipeline run.
        original_evidence_graph: GAT output before removal (for reference).

    Returns:
        ``CounterfactualResponse`` with old/new answers and updated graph.

    Raises:
        ValueError: If the meeting is not found.
    """
    _ = original_evidence_graph  # retained for API contract / future diff metrics

    store = get_meeting_store()
    record = store.get(meeting_id)
    if record is None:
        raise ValueError(f"Meeting not found: {meeting_id}")

    index = store.load_index(meeting_id)
    if index is None:
        raise ValueError(f"Index not found for meeting: {meeting_id}")

    segments_by_id = store.segments_by_id(meeting_id)

    retrieval = retrieve_candidates(question, index, segments_by_id)
    candidate_ids = [c.segment_id for c in retrieval.candidates]
    candidate_ids = [sid for sid in candidate_ids if sid != node_id_to_remove]

    graph = build_evidence_graph(candidate_ids, segments_by_id, index=index)
    gat_output = get_gat_inference().run(graph)

    counterfactual_answer = generate_answer(question, gat_output, segments_by_id)
    shap_highlights = explain_retrieval(question, retrieval, segments_by_id)

    return CounterfactualResponse(
        original_answer=original_answer,
        counterfactual_answer=counterfactual_answer,
        removed_node_id=node_id_to_remove,
        answer_changed=counterfactual_answer.strip() != original_answer.strip(),
        evidence_graph=gat_output,
        shap_highlights=shap_highlights,
    )
