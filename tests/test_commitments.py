"""Unit tests for commitment extraction and the cross-meeting graph."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.commitments.extractor import extract_from_segment
from app.commitments.graph import CommitmentGraph
from app.models.schemas import Commitment, SLPSegment, WordToken


def _seg(sid: str, text: str, conf: float = 0.95, entities: list[str] | None = None) -> SLPSegment:
    words = [WordToken(word=w, start=i * 0.3, end=i * 0.3 + 0.2, probability=conf) for i, w in enumerate(text.split())]
    return SLPSegment(segment_id=sid, start_time=10.0, end_time=12.0, text=text, entities=entities or [],
                      words=words, asr_confidence=conf, min_word_confidence=conf)


class TestExtractor:
    def test_decision_detected_with_owner_and_value(self) -> None:
        seg = _seg("s0", "We decided the launch is by Friday.")
        labelled = [("Friday", "DATE")]
        c = extract_from_segment(seg, "m1", 1000.0, labelled=labelled)
        assert c is not None
        assert c.kind == "decision"
        assert c.values == ["Friday"]
        assert c.polarity == 1
        assert c.reliability > 0.9

    def test_commitment_with_person_owner(self) -> None:
        seg = _seg("s0", "Alice will send the report tomorrow.")
        c = extract_from_segment(seg, "m1", 1000.0, labelled=[("Alice", "PERSON")])
        assert c is not None and c.kind == "commitment" and c.owner == "Alice"

    def test_negation_flips_polarity(self) -> None:
        seg = _seg("s0", "We decided we will not use Postgres.")
        c = extract_from_segment(seg, "m1", 1000.0, labelled=[("Postgres", "PRODUCT")])
        assert c is not None and c.polarity == -1 and c.topic_entities == ["Postgres"]

    def test_plain_statement_ignored(self) -> None:
        assert extract_from_segment(_seg("s0", "The weather was nice."), "m1", 1.0, labelled=[]) is None

    def test_unreliable_statement_ignored(self) -> None:
        seg = _seg("s0", "We decided to ship Friday.", conf=0.1)
        assert extract_from_segment(seg, "m1", 1.0, labelled=[]) is None


def _c(cid: str, meeting: str, t: float, text: str, polarity: int = 1, values=None, rel: float = 0.95) -> Commitment:
    return Commitment(commitment_id=cid, meeting_id=meeting, segment_id=cid, text=text, kind="decision",
                      topic_entities=["launch"], values=values or [], polarity=polarity,
                      meeting_time=t, offset_s=0.0, reliability=rel)


@pytest.fixture
def graph(tmp_path):
    # Deterministic "embeddings": every commitment text maps to the same unit
    # vector, so topic similarity is 1.0 and only conflict logic is tested.
    with patch("app.commitments.graph.embed_texts", side_effect=lambda texts: np.ones((len(texts), 4)) / 2.0):
        yield CommitmentGraph(path=tmp_path / "g.json")


class TestGraph:
    def test_later_conflicting_decision_supersedes(self, graph: CommitmentGraph) -> None:
        graph.add(_c("a", "m1", 100.0, "launch Friday", values=["Friday"]))
        edges = graph.add(_c("b", "m2", 200.0, "launch Monday", values=["Monday"]))
        assert edges[0].relation == "supersedes"
        assert graph.nodes["a"].status == "superseded"
        assert graph.nodes["a"].superseded_by == "b"
        assert graph.nodes["b"].status == "active"

    def test_same_stance_reaffirms(self, graph: CommitmentGraph) -> None:
        graph.add(_c("a", "m1", 100.0, "launch Friday", values=["Friday"]))
        edges = graph.add(_c("b", "m2", 200.0, "still launch Friday", values=["Friday"]))
        assert edges[0].relation == "reaffirms"
        assert graph.nodes["a"].status == "active"
        assert graph.nodes["b"].status == "merged"
        assert graph.nodes["b"].superseded_by == "a"

    def test_unreliable_later_statement_only_disputes(self, graph: CommitmentGraph) -> None:
        graph.add(_c("a", "m1", 100.0, "launch Friday", values=["Friday"], rel=0.95))
        edges = graph.add(_c("b", "m2", 200.0, "um maybe Monday", values=["Monday"], rel=0.3))
        assert edges[0].relation == "disputes"
        assert graph.nodes["a"].status == "active"
        assert graph.nodes["b"].status == "disputed"
        tl = graph.timeline("launch")
        assert tl.current is not None and tl.current.commitment_id == "a"

    def test_out_of_order_insert_respects_time(self, graph: CommitmentGraph) -> None:
        graph.add(_c("b", "m2", 200.0, "launch Monday", values=["Monday"]))
        graph.add(_c("a", "m1", 100.0, "launch Friday", values=["Friday"]))
        assert graph.nodes["a"].status == "superseded"
        assert graph.nodes["b"].status == "active"

    def test_timeline_and_persistence(self, graph: CommitmentGraph, tmp_path) -> None:
        graph.add(_c("a", "m1", 100.0, "launch Friday", values=["Friday"]))
        graph.add(_c("b", "m2", 200.0, "launch Monday", values=["Monday"]))
        tl = graph.timeline("launch date")
        assert tl.current is not None and tl.current.commitment_id == "b"
        assert [c.commitment_id for c in tl.history] == ["b", "a"]
        assert tl.edges[0].relation == "supersedes"

        graph.save()
        with patch("app.commitments.graph.embed_texts", side_effect=lambda texts: np.ones((len(texts), 4)) / 2.0):
            reloaded = CommitmentGraph(path=tmp_path / "g.json")
        assert set(reloaded.nodes) == {"a", "b"}
        assert reloaded.nodes["a"].status == "superseded"

    def test_remove_meeting(self, graph: CommitmentGraph) -> None:
        graph.add(_c("a", "m1", 100.0, "launch Friday", values=["Friday"]))
        graph.add(_c("b", "m2", 200.0, "launch Monday", values=["Monday"]))
        graph.remove_meeting("m2")
        assert set(graph.nodes) == {"a"}
        assert graph.edges == []
