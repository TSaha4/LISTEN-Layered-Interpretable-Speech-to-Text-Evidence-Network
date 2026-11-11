"""Unit tests for answer-level reliability scoring and the GAT gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.dl.gat_model import ReliabilityGate
from app.models.schemas import GATOutput, RetrievalCandidate, RetrievalResult, SLPSegment, WordHighlight, WordToken
from app.xai.reliability import (
    CalibrationParams,
    calibrate,
    evidence_agreement_component,
    retrieval_margin_component,
    score_answer,
    uncertain_words_in_evidence,
)


def _seg(sid: str, conf: float, text: str = "ship by friday") -> SLPSegment:
    words = [WordToken(word=w, start=i * 0.3, end=i * 0.3 + 0.2, probability=conf) for i, w in enumerate(text.split())]
    return SLPSegment(segment_id=sid, start_time=0, end_time=1, text=text, entities=[], words=words,
                      asr_confidence=conf, min_word_confidence=conf)


def _gat(ids: list[str], rel: dict[str, float]) -> GATOutput:
    return GATOutput(node_scores={i: 0.8 for i in ids}, edge_scores={}, top_evidence_ids=ids,
                     raw_node_scores={i: 0.8 for i in ids}, node_reliability=rel)


def _retrieval(ids: list[str], scores: list[float]) -> RetrievalResult:
    return RetrievalResult(question="q", candidates=[RetrievalCandidate(segment_id=i, score=s) for i, s in zip(ids, scores)])


class TestReliabilityGate:
    def test_gate_downweights_unreliable_nodes(self) -> None:
        gate = ReliabilityGate(node_beta=1.0, edge_beta=0.5)
        probs = torch.tensor([0.9, 0.9])
        rel = torch.tensor([1.0, 0.25])
        gated, _ = gate(probs, rel, torch.zeros(0), torch.zeros(0))
        assert gated[0].item() == pytest.approx(0.9)
        assert gated[1].item() == pytest.approx(0.225)

    def test_zero_beta_is_identity(self) -> None:
        gate = ReliabilityGate(node_beta=0.0, edge_beta=0.0)
        probs = torch.tensor([0.9, 0.3])
        edges = torch.tensor([0.5])
        gated_n, gated_e = gate(probs, torch.tensor([0.1, 0.1]), edges, torch.tensor([0.2]))
        assert torch.allclose(gated_n, probs)
        assert torch.allclose(gated_e, edges)


class TestComponents:
    def test_retrieval_margin_prefers_clear_winner(self) -> None:
        clear = _retrieval(list("abcd"), [0.9, 0.85, 0.8, 0.2])
        flat = _retrieval(list("abcd"), [0.5, 0.5, 0.5, 0.5])
        assert retrieval_margin_component(clear) > retrieval_margin_component(flat)

    def test_agreement_of_identical_vectors_is_one(self) -> None:
        v = np.ones((3, 4)) / 2.0
        assert evidence_agreement_component(v) == pytest.approx(1.0)
        assert evidence_agreement_component(None) == 0.5

    def test_uncertain_words_ranked_by_influence(self) -> None:
        segs = {"s0": _seg("s0", 0.3, "budget friday")}
        shap = {"s0": [WordHighlight(word="budget", score=0.0), WordHighlight(word="friday", score=0.4)]}
        words = uncertain_words_in_evidence(["s0"], segs, shap)
        assert [w.word for w in words] == ["friday", "budget"]
        assert words[0].shap_influence == pytest.approx(0.4)


class TestScoreAnswer:
    def test_reliable_evidence_scores_higher(self) -> None:
        ids = ["s0", "s1"]
        retrieval = _retrieval(ids + ["s2"], [0.9, 0.85, 0.1])
        good = score_answer(_gat(ids, {"s0": 0.95, "s1": 0.95}), retrieval,
                            {"s0": _seg("s0", 0.95), "s1": _seg("s1", 0.95)}, ids, supported=True,
                            params=CalibrationParams())
        bad = score_answer(_gat(ids, {"s0": 0.2, "s1": 0.2}), retrieval,
                           {"s0": _seg("s0", 0.2), "s1": _seg("s1", 0.2)}, ids, supported=True,
                           params=CalibrationParams())
        assert good.score > bad.score
        assert good.level in {"high", "medium"}
        assert bad.warning is None  # no SHAP influence passed → no "leans on" warning
        assert len(bad.uncertain_words) > 0

    def test_unsupported_answer_scores_low(self) -> None:
        ids = ["s0"]
        r = score_answer(_gat(ids, {"s0": 0.9}), _retrieval(ids, [0.9]), {"s0": _seg("s0", 0.9)}, ids,
                         supported=False, params=CalibrationParams())
        assert r.components["grounding_supported"] == 0.0
        assert r.level != "high"


class TestCalibration:
    def test_calibrate_learns_sign(self) -> None:
        rows = [{"evidence_reliability": x, "retrieval_margin": 0.5, "evidence_agreement": 0.5, "grounding_supported": 1.0}
                for x in np.linspace(0, 1, 20)]
        labels = [1 if x > 0.5 else 0 for x in np.linspace(0, 1, 20)]
        params = calibrate(rows, labels, epochs=300, lr=0.5)
        assert params.w_evidence > 0

    def test_roundtrip_save_load(self, tmp_path) -> None:
        p = CalibrationParams(bias=0.1, w_evidence=1, w_margin=2, w_agreement=3, w_supported=4)
        path = tmp_path / "cal.json"
        p.save(path)
        assert CalibrationParams.load(path) == p
