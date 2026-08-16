"""Automatic speech recognition with faster-whisper (lazy-loaded)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app import config
from app.models.schemas import SLPSegment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass
class _WhisperHolder:
    """Lazy singleton holder for the Whisper model."""

    model: WhisperModel | None = None


_whisper_holder = _WhisperHolder()


def _get_whisper_model() -> WhisperModel:
    """Load faster-whisper on first use to keep startup memory low."""
    if _whisper_holder.model is None:
        from faster_whisper import WhisperModel

        _whisper_holder.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_holder.model


def transcribe_audio(
    audio_path: str,
    meeting_id: str,
    language: str | None = "en",
) -> list[SLPSegment]:
    """Transcribe audio into timestamped segments.

    Args:
        audio_path: Path to a WAV/MP3 audio file.
        meeting_id: Prefix used to build stable segment identifiers.
        language: Optional ISO language code passed to Whisper.

    Returns:
        List of ``SLPSegment`` objects with empty ``entities`` (filled later
        by the NER step).
    """
    model = _get_whisper_model()
    segments_iter, _info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )

    segments: list[SLPSegment] = []
    for idx, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            SLPSegment(
                segment_id=f"{meeting_id}_seg_{idx:04d}",
                start_time=float(seg.start),
                end_time=float(seg.end),
                text=text,
                entities=[],
            )
        )
    return segments


def reset_whisper_model() -> None:
    """Release the cached Whisper model (useful in tests)."""
    _whisper_holder.model = None
