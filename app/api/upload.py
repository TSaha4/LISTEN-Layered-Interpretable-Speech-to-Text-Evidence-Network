"""Meeting upload and SLP processing endpoints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import config
from app.models.schemas import UploadResponse
from app.services.meeting_store import get_meeting_store
from app.slp.asr import transcribe_audio
from app.slp.audio_extraction import extract_audio_from_video, is_audio_file, is_video_file
from app.slp.entity_extraction import enrich_segments_with_entities

router = APIRouter(prefix="/meetings", tags=["upload"])


@router.get("/{meeting_id}/audio")
async def stream_audio_clip(meeting_id: str, start: float, end: float) -> StreamingResponse:
    """Stream a WAV clip covering the requested meeting timestamp range."""
    if start < 0 or end <= start:
        raise HTTPException(status_code=422, detail="Require 0 <= start < end.")

    record = get_meeting_store().get(meeting_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    audio_path = Path(record.audio_path)
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Meeting audio file not found.")

    command = [
        "ffmpeg", "-v", "error", "-ss", str(start), "-to", str(end),
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", str(config.FFMPEG_SAMPLE_RATE),
        "-f", "wav", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="ffmpeg is not installed or not on PATH.") from exc
    if result.returncode != 0:
        raise HTTPException(status_code=502, detail=f"Could not create audio clip: {result.stderr.decode(errors='replace').strip()}")
    return StreamingResponse(iter([result.stdout]), media_type="audio/wav")


@router.get("", response_model=list[dict[str, Any]])
async def list_available_meetings() -> list[dict[str, Any]]:
    """List all persisted meetings available on disk."""
    store = get_meeting_store()
    results: list[dict[str, Any]] = []
    for mid in store.list_meetings():
        rec = store.get(mid)
        if rec:
            results.append({
                "meeting_id": rec.meeting_id,
                "original_filename": rec.metadata.get("original_filename") if rec.metadata else None,
                "segment_count": len(rec.segments),
            })
    return results


@router.get("/{meeting_id}", response_model=UploadResponse)
async def get_meeting_by_id(meeting_id: str) -> UploadResponse:
    """Retrieve an existing meeting record and its transcript segments."""
    rec = get_meeting_store().get(meeting_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    return UploadResponse(
        meeting_id=rec.meeting_id,
        segment_count=len(rec.segments),
        segments=rec.segments,
        stage_timings=rec.metadata.get("stage_timings") if rec.metadata else None,
    )

@router.get("/{meeting_id}/trace")
async def get_meeting_trace(meeting_id: str) -> dict[str, Any]:
    """Return trace and diagnostic information for a specific meeting."""
    store = get_meeting_store()
    record = store.get(meeting_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting not found.")

    segments = record.segments
    duration_s = float(segments[-1].end_time) if segments else 0.0

    entities_list: list[str] = []
    seg_list: list[dict[str, Any]] = []
    for s in segments:
        entities_list.extend(s.entities)
        seg_list.append({
            "segment_id": s.segment_id,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "text": s.text,
        })

    top_10 = [{"entity": k, "count": v} for k, v in Counter(entities_list).most_common(10)]

    try:
        index = store.load_index(meeting_id)
        index_size = len(index.segment_ids) if index and hasattr(index, "segment_ids") else len(segments)
    except Exception:
        index_size = len(segments)

    stage_timings = record.metadata.get("stage_timings") if record.metadata else None

    return {
        "meeting_id": meeting_id,
        "original_filename": record.metadata.get("original_filename") if record.metadata else None,
        "duration_s": duration_s,
        "n_segments": len(segments),
        "n_entities": len(entities_list),
        "top_10_entities": top_10,
        "segments": seg_list,
        "embedding_dim": config.EMBEDDING_DIM,
        "index_size": index_size,
        "stage_timings": stage_timings,
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_meeting(file: UploadFile = File(...)) -> UploadResponse:
    """Upload audio/video, run SLP pipeline, index segments, and persist.

    Args:
        file: Audio or video recording of a meeting.

    Returns:
        ``UploadResponse`` with segment list and assigned ``meeting_id``.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in config.ALLOWED_AUDIO_EXTENSIONS | config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    t_start = time.perf_counter()
    meeting_id = str(uuid.uuid4())
    upload_path = config.UPLOAD_DIR / f"{meeting_id}{suffix}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    t0_audio = time.perf_counter()
    if is_video_file(upload_path):
        audio_path = extract_audio_from_video(upload_path)
    elif is_audio_file(upload_path):
        audio_path = upload_path
    else:
        raise HTTPException(status_code=400, detail="Could not determine media type.")
    audio_extract_s = round(time.perf_counter() - t0_audio, 3)

    t0_asr = time.perf_counter()
    segments = transcribe_audio(str(audio_path), meeting_id=meeting_id)
    asr_s = round(time.perf_counter() - t0_asr, 3)

    t0_ner = time.perf_counter()
    segments = enrich_segments_with_entities(segments)
    ner_s = round(time.perf_counter() - t0_ner, 3)

    t0_index = time.perf_counter()
    stage_timings = {
        "audio_extract_s": audio_extract_s,
        "asr_s": asr_s,
        "ner_s": ner_s,
        "embed_index_s": 0.0,
        "total_s": 0.0,
    }
    record = get_meeting_store().save(
        segments=segments,
        audio_path=str(audio_path),
        meeting_id=meeting_id,
        metadata={
            "original_filename": file.filename,
            "stage_timings": stage_timings,
        },
    )
    embed_index_s = round(time.perf_counter() - t0_index, 3)
    stage_timings["embed_index_s"] = embed_index_s
    stage_timings["total_s"] = round(time.perf_counter() - t_start, 3)

    # Persist updated stage_timings in record
    record.metadata["stage_timings"] = stage_timings
    get_meeting_store()._meeting_path(meeting_id).write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return UploadResponse(
        meeting_id=record.meeting_id,
        segment_count=len(segments),
        segments=segments,
        stage_timings=stage_timings,
    )
