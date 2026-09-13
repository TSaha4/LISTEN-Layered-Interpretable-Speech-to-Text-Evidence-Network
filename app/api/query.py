"""Question-answering endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.services.pipeline import run_query_pipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def ask_question(payload: QueryRequest) -> QueryResponse:
    """Ask a question over a previously uploaded meeting.

    Args:
        payload: ``meeting_id`` and natural-language ``question``.

    Returns:
        Answer, evidence graph, SHAP highlights, and audio references.
    """
    try:
        return run_query_pipeline(payload.meeting_id, payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
