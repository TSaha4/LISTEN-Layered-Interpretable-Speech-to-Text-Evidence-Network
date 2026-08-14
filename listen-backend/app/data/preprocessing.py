"""Preprocessing helpers converting raw datasets into LISTEN segment format."""

from __future__ import annotations

from app.data.loaders import QMSumSample
from app.models.schemas import SLPSegment
from app.slp.entity_extraction import extract_entities_from_text


def qmsum_turn_to_segment(
    meeting_id: str,
    turn_idx: int,
    turn: dict[str, str],
) -> SLPSegment:
    """Convert one QMSum meeting turn into an ``SLPSegment``.

    Args:
        meeting_id: Stable meeting identifier.
        turn_idx: Zero-based turn index within the meeting.
        turn: Dict with ``speaker`` and ``content`` keys.

    Returns:
        Segment with NER entities attached.
    """
    speaker = turn.get("speaker", "Unknown")
    content = turn.get("content", "").strip()
    text = f"{speaker}: {content}" if content else speaker
    return SLPSegment(
        segment_id=f"{meeting_id}_turn_{turn_idx:04d}",
        start_time=float(turn_idx),
        end_time=float(turn_idx + 1),
        text=text,
        entities=extract_entities_from_text(text),
    )


def qmsum_sample_to_segments(sample: QMSumSample) -> list[SLPSegment]:
    """Materialise all transcript turns for a QMSum sample as segments."""
    segments: list[SLPSegment] = []
    for idx, turn in enumerate(sample.meeting_transcripts):
        segments.append(qmsum_turn_to_segment(sample.meeting_id, idx, turn))
    return segments


def ami_record_to_segments(
    meeting_id: str,
    record: dict,
) -> list[SLPSegment]:
    """Best-effort conversion of an AMI HF record to segments.

    AMI schema varies by subset; this handles common text/transcript fields.
    """
    segments: list[SLPSegment] = []

    transcript = record.get("text") or record.get("transcript") or record.get("words")
    if isinstance(transcript, str) and transcript.strip():
        segments.append(
            SLPSegment(
                segment_id=f"{meeting_id}_seg_0000",
                start_time=0.0,
                end_time=0.0,
                text=transcript.strip(),
                entities=extract_entities_from_text(transcript),
            )
        )
    elif isinstance(transcript, list):
        for idx, item in enumerate(transcript):
            if isinstance(item, dict):
                text = item.get("text") or item.get("word") or ""
                start = float(item.get("start", idx))
                end = float(item.get("end", start + 1))
            else:
                text = str(item)
                start, end = float(idx), float(idx + 1)
            if not str(text).strip():
                continue
            segments.append(
                SLPSegment(
                    segment_id=f"{meeting_id}_seg_{idx:04d}",
                    start_time=start,
                    end_time=end,
                    text=str(text).strip(),
                    entities=extract_entities_from_text(str(text)),
                )
            )
    return segments
