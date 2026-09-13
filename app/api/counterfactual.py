"""Counterfactual evidence validation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import CounterfactualRequest, CounterfactualResponse
from app.services.pipeline import run_counterfactual_pipeline

router = APIRouter(prefix="/counterfactual", tags=["counterfactual"])


@router.post("/", response_model=CounterfactualResponse)
async def counterfactual_remove_node(payload: CounterfactualRequest) -> CounterfactualResponse:
    """Remove one evidence node and re-run reasoning to validate its impact.

    Args:
        payload: Meeting id, question, node to remove, and original outputs.

    Returns:
        Original vs counterfactual answers and updated evidence graph.
    """
    try:
        return run_counterfactual_pipeline(
            meeting_id=payload.meeting_id,
            question=payload.question,
            node_id_to_remove=payload.node_id_to_remove,
            original_answer=payload.original_answer,
            original_evidence_graph=payload.original_evidence_graph,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
