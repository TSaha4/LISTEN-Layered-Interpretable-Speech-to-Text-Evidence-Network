"""Automatic speech recognition with faster-whisper (lazy-loaded).

Word timestamps are requested so every word carries its recognition
probability; the uncertainty layer turns those into per-segment reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app import config
from app.models.schemas import SLPSegment, WordToken
from app.slp.uncertainty import annotate_segment

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


def _words_from_segment(seg: object) -> list[WordToken]:
    """Convert faster-whisper ``Word`` objects (if any) to ``WordToken``."""
    raw_words = getattr(seg, "words", None) or []
    tokens: list[WordToken] = []
    for w in raw_words:
        text = str(getattr(w, "word", "")).strip()
        if not text:
            continue
        tokens.append(
            WordToken(
                word=text,
                start=float(getattr(w, "start", 0.0)),
                end=float(getattr(w, "end", 0.0)),
                probability=float(getattr(w, "probability", 1.0)),
            )
        )
    return tokens


def transcribe_audio(
    audio_path: str,
    meeting_id: str,
    language: str | None = "en",
) -> list[SLPSegment]:
    """Transcribe audio into timestamped, uncertainty-annotated segments.

    Args:
        audio_path: Path to a WAV/MP3 audio file.
        meeting_id: Prefix used to build stable segment identifiers.
        language: Optional ISO language code passed to Whisper.

    Returns:
        List of ``SLPSegment`` objects with empty ``entities`` (filled later
        by the NER step) and populated word/uncertainty fields.
    """
    model = _get_whisper_model()
    segments_iter, _info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        word_timestamps=config.WHISPER_WORD_TIMESTAMPS,
    )

    segments: list[SLPSegment] = []
    for idx, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if not text:
            continue
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        segment = SLPSegment(
            segment_id=f"{meeting_id}_seg_{idx:04d}",
            start_time=float(seg.start),
            end_time=float(seg.end),
            text=text,
            entities=[],
            words=_words_from_segment(seg),
            no_speech_prob=float(no_speech) if isinstance(no_speech, (int, float)) else 0.0,
        )
        segments.append(annotate_segment(segment))
    return segments


def reset_whisper_model() -> None:
    """Release the cached Whisper model (useful in tests)."""
    _whisper_holder.model = None
