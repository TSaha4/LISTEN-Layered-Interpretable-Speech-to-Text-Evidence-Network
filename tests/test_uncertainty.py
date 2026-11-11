"""Unit tests for word-level uncertainty cues and reliability fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.dl.graph_builder import EvidenceGraphBuilder
from app.models.schemas import SLPSegment, WordToken
from app.slp.asr import transcribe_audio
from app.slp.uncertainty import (
    annotate_segment,
    compute_cues,
    entity_confidence,
    evidence_reliability,
    low_confidence_words,
)


def _words(spec: list[tuple[str, float]], gap: float = 0.1) -> list[WordToken]:
    out: list[WordToken] = []
    t = 0.0
    for word, p in spec:
        out.append(WordToken(word=word, start=t, end=t + 0.3, probability=p))
        t += 0.3 + gap
    return out


def _segment(words: list[WordToken], text: str | None = None, **kw) -> SLPSegment:
    return SLPSegment(
        segment_id="s0", start_time=0.0, end_time=5.0,
        text=text or " ".join(w.word for w in words), entities=kw.pop("entities", []),
        words=words, **kw,
    )


class TestCues:
    def test_confident_fluent_segment(self) -> None:
        seg = _segment(_words([("we", 0.95), ("ship", 0.9), ("friday", 0.97)]))
        cues = compute_cues(seg.words, seg.text)
        assert cues.asr_confidence == pytest.approx(0.94, abs=1e-2)
        assert cues.min_word_confidence == pytest.approx(0.9)
        assert cues.hesitation_score == 0.0

    def test_fillers_and_hedges_raise_hesitation(self) -> None:
        seg = _segment(_words([("um", 0.9), ("maybe", 0.9), ("uh", 0.9), ("friday", 0.9)]))
        cues = compute_cues(seg.words, seg.text)
        assert cues.filler_count == 2
        assert cues.hedge_count == 1
        assert cues.hesitation_score > 0.5

    def test_long_pause_and_repetition_detected(self) -> None:
        words = _words([("the", 0.9), ("the", 0.9), ("budget", 0.9)])
        words[2] = words[2].model_copy(update={"start": words[1].end + 1.5, "end": words[1].end + 1.8})
        cues = compute_cues(words, "the the budget")
        assert cues.repetition_count == 1
        assert cues.long_pause_count == 1

    def test_no_words_means_fully_confident(self) -> None:
        seg = SLPSegment(segment_id="s", start_time=0, end_time=1, text="hello", entities=[])
        assert evidence_reliability(seg) == 1.0


class TestReliability:
    def test_low_asr_confidence_lowers_reliability(self) -> None:
        good = annotate_segment(_segment(_words([("a", 0.95), ("b", 0.95)])))
        bad = annotate_segment(_segment(_words([("a", 0.3), ("b", 0.4)])))
        assert evidence_reliability(good) > evidence_reliability(bad)

    def test_hesitation_lowers_reliability_independently(self) -> None:
        fluent = annotate_segment(_segment(_words([("ship", 0.95), ("friday", 0.95)])))
        hedged = annotate_segment(_segment(_words([("um", 0.95), ("maybe", 0.95), ("friday", 0.95)])))
        assert fluent.asr_confidence == pytest.approx(hedged.asr_confidence)
        assert evidence_reliability(fluent) > evidence_reliability(hedged)

    def test_reliability_has_floor(self) -> None:
        seg = annotate_segment(_segment(_words([("x", 0.0)]), no_speech_prob=1.0))
        assert evidence_reliability(seg) > 0.0

    def test_low_confidence_words_listed(self) -> None:
        seg = _segment(_words([("alpha", 0.9), ("beta", 0.2)]))
        assert [w.word for w in low_confidence_words(seg)] == ["beta"]


class TestEntityConfidence:
    def test_multiword_entity_alignment(self) -> None:
        seg = _segment(_words([("new", 0.9), ("york", 0.5), ("office", 0.9)]))
        assert entity_confidence(seg, "New York") == pytest.approx(0.7)

    def test_unaligned_entity_falls_back_to_segment_mean(self) -> None:
        seg = annotate_segment(_segment(_words([("a", 0.8), ("b", 0.6)])))
        assert entity_confidence(seg, "Zebra") == pytest.approx(seg.asr_confidence)


class TestUncertaintyGraph:
    def _build(self, p_entity_a: float, p_entity_b: float):
        a = annotate_segment(_segment(_words([("acme", p_entity_a), ("agreed", 0.9)]), entities=["Acme"]))
        b = annotate_segment(_segment(_words([("acme", p_entity_b), ("delivers", 0.9)]), entities=["Acme"]))
        b = b.model_copy(update={"segment_id": "s1"})
        emb = np.eye(2, 384, dtype=np.float32)  # orthogonal → no similarity edge
        return EvidenceGraphBuilder().build(np.zeros(384, dtype=np.float32), [a, b], emb)

    def test_well_heard_entity_creates_edge(self) -> None:
        data, meta = self._build(0.9, 0.9)
        assert meta["num_entity_edges"] == 2
        assert meta["gated_entity_edges"] == 0
        assert data.edge_confidence.shape[0] == data.edge_index.shape[1]

    def test_badly_heard_entity_is_gated(self) -> None:
        data, meta = self._build(0.9, 0.2)
        assert meta["num_entity_edges"] == 0
        assert meta["gated_entity_edges"] == 1
        assert meta["node_reliability"]["s1"] < meta["node_reliability"]["s0"]
        assert data.node_reliability[0].item() == 1.0  # question node


class TestASRWordCapture:
    @patch("app.slp.asr._get_whisper_model")
    def test_words_and_confidence_are_captured(self, mock_whisper: MagicMock) -> None:
        w1 = MagicMock(word=" we", start=0.0, end=0.2, probability=0.9)
        w2 = MagicMock(word=" um", start=0.3, end=0.4, probability=0.4)
        seg = MagicMock(text=" we um ", start=0.0, end=1.0, words=[w1, w2], no_speech_prob=0.05)
        mock_whisper.return_value.transcribe.return_value = ([seg], MagicMock())

        segments = transcribe_audio("fake.wav", meeting_id="m")
        assert len(segments[0].words) == 2
        assert segments[0].asr_confidence == pytest.approx(0.65)
        assert segments[0].min_word_confidence == pytest.approx(0.4)
        assert segments[0].hesitation_score > 0
        assert segments[0].no_speech_prob == pytest.approx(0.05)
