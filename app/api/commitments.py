"""Cross-meeting commitment tracking endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.commitments.extractor import extract_commitments
from app.commitments.graph import get_commitment_graph
from app.models.schemas import Commitment, CommitmentEdge, CommitmentTimeline
from app.services.meeting_store import get_meeting_store

router = APIRouter(prefix="/commitments", tags=["commitments"])


class IndexMeetingResponse(BaseModel):
    meeting_id: str
    commitments: list[Commitment]
    new_edges: list[CommitmentEdge]


def index_meeting_commitments(meeting_id: str, meeting_time: float | None = None) -> IndexMeetingResponse:
    """(Re)extract a meeting's commitments and merge them into the graph."""
    store = get_meeting_store()
    record = store.get(meeting_id)
    if record is None:
        raise ValueError(f"Meeting not found: {meeting_id}")
    meta = record.metadata or {}
    when = meeting_time if meeting_time is not None else float(meta.get("meeting_time", time.time()))

    graph = get_commitment_graph()
    graph.remove_meeting(meeting_id)
    commitments = extract_commitments(record.segments, meeting_id, when)
    new_edges = graph.add_many(commitments)
    graph.save()
    return IndexMeetingResponse(meeting_id=meeting_id, commitments=graph.for_meeting(meeting_id), new_edges=new_edges)


@router.post("/index/{meeting_id}", response_model=IndexMeetingResponse)
async def index_meeting(meeting_id: str, meeting_time: float | None = None) -> IndexMeetingResponse:
    """Extract commitments from a processed meeting and add them to the graph."""
    try:
        return index_meeting_commitments(meeting_id, meeting_time)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/timeline", response_model=CommitmentTimeline)
async def commitment_timeline(topic: str, top_k: int = 10) -> CommitmentTimeline:
    """What is the current decision on ``topic`` and how did it change?"""
    if not topic.strip():
        raise HTTPException(status_code=422, detail="topic is required")
    return get_commitment_graph().timeline(topic, top_k=top_k)


@router.get("/meeting/{meeting_id}", response_model=list[Commitment])
async def meeting_commitments(meeting_id: str) -> list[Commitment]:
    """Commitments stated in one meeting."""
    return get_commitment_graph().for_meeting(meeting_id)
