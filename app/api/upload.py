"""Meeting upload and SLP processing endpoints."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

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

    meeting_id = str(uuid.uuid4())
    upload_path = config.UPLOAD_DIR / f"{meeting_id}{suffix}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    if is_video_file(upload_path):
        audio_path = extract_audio_from_video(upload_path)
    elif is_audio_file(upload_path):
        audio_path = upload_path
    else:
        raise HTTPException(status_code=400, detail="Could not determine media type.")

    segments = transcribe_audio(str(audio_path), meeting_id=meeting_id)
    segments = enrich_segments_with_entities(segments)

    record = get_meeting_store().save(
        segments=segments,
        audio_path=str(audio_path),
        meeting_id=meeting_id,
        metadata={"original_filename": file.filename},
    )

    return UploadResponse(
        meeting_id=record.meeting_id,
        segment_count=len(segments),
        segments=segments,
    )
