"""Persist and retrieve processed meeting records."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app import config
from app.dl.embeddings import SegmentIndex
from app.models.schemas import MeetingRecord, SLPSegment


class MeetingStore:
    """File-backed store for meeting segments and FAISS indices."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.MEETING_STORE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _meeting_path(self, meeting_id: str) -> Path:
        return self.root / f"{meeting_id}.json"

    def _index_path(self, meeting_id: str) -> Path:
        return config.INDEX_DIR / meeting_id

    def save(
        self,
        segments: list[SLPSegment],
        audio_path: str,
        meeting_id: str | None = None,
        metadata: dict | None = None,
    ) -> MeetingRecord:
        """Persist segments, build FAISS index, and save meeting metadata.

        Args:
            segments: SLP-enriched transcript segments.
            audio_path: Path to the meeting audio file.
            meeting_id: Optional id; generated if omitted.
            metadata: Optional extra metadata dict.

        Returns:
            Saved ``MeetingRecord``.
        """
        meeting_id = meeting_id or str(uuid.uuid4())

        index = SegmentIndex()
        index.build(segments)
        index_dir = self._index_path(meeting_id)
        index.save(index_dir)

        record = MeetingRecord(
            meeting_id=meeting_id,
            audio_path=audio_path,
            segments=segments,
            index_path=str(index_dir),
            metadata=metadata or {},
        )
        self._meeting_path(meeting_id).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return record

    def get(self, meeting_id: str) -> MeetingRecord | None:
        """Load a meeting record by id."""
        path = self._meeting_path(meeting_id)
        if not path.exists():
            return None
        return MeetingRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def load_index(self, meeting_id: str) -> SegmentIndex | None:
        """Load the FAISS index for a meeting."""
        record = self.get(meeting_id)
        if record is None or not record.index_path:
            return None
        return SegmentIndex.load(record.index_path)

    def segments_by_id(self, meeting_id: str) -> dict[str, SLPSegment]:
        """Return segment lookup for a meeting."""
        record = self.get(meeting_id)
        if record is None:
            return {}
        return {s.segment_id: s for s in record.segments}

    def list_meetings(self) -> list[str]:
        """Return all stored meeting ids."""
        return [p.stem for p in self.root.glob("*.json")]


_store: MeetingStore | None = None


def get_meeting_store() -> MeetingStore:
    """Process-wide meeting store singleton."""
    global _store
    if _store is None:
        _store = MeetingStore()
    return _store
