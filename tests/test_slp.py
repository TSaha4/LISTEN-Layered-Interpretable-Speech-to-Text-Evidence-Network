"""Unit tests for the Speech & Language Processing layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import SLPSegment
from app.slp.audio_extraction import is_audio_file, is_video_file
from app.slp.entity_extraction import enrich_segments_with_entities, extract_entities_from_text
from app.slp.asr import transcribe_audio


class TestMediaDetection:
    def test_video_extensions(self) -> None:
        assert is_video_file("meeting.mp4")
        assert not is_video_file("meeting.wav")

    def test_audio_extensions(self) -> None:
        assert is_audio_file("meeting.wav")
        assert not is_audio_file("meeting.mp4")


class TestEntityExtraction:
    @patch("app.slp.entity_extraction._get_nlp")
    def test_extract_entities_deduplicates(self, mock_get_nlp: MagicMock) -> None:
        mock_ent = MagicMock()
        mock_ent.text = "Alice"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent, mock_ent]
        mock_get_nlp.return_value = MagicMock(return_value=mock_doc)

        entities = extract_entities_from_text("Alice met Alice.")
        assert entities == ["Alice"]

    @patch("app.slp.entity_extraction.extract_entities_from_text")
    def test_enrich_segments(self, mock_ner: MagicMock) -> None:
        mock_ner.return_value = ["Bob"]
        segments = [
            SLPSegment(
                segment_id="s0",
                start_time=0.0,
                end_time=1.0,
                text="Bob spoke.",
                entities=[],
            )
        ]
        enriched = enrich_segments_with_entities(segments)
        assert enriched[0].entities == ["Bob"]


class TestASR:
    @patch("app.slp.asr._get_whisper_model")
    def test_transcribe_produces_segments(self, mock_whisper: MagicMock) -> None:
        seg1 = MagicMock(text=" Hello world ", start=0.0, end=2.5)
        seg2 = MagicMock(text="   ", start=2.5, end=3.0)
        mock_whisper.return_value.transcribe.return_value = ([seg1, seg2], MagicMock())

        segments = transcribe_audio("fake.wav", meeting_id="meet1")
        assert len(segments) == 1
        assert segments[0].segment_id == "meet1_seg_0000"
        assert segments[0].text == "Hello world"
