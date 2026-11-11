"""Answer-level reliability: a calibrated score plus the words it hangs on.

The score fuses four signals that are available for every answer:

1. **Evidence reliability** — GAT-weighted mean of per-segment reliability
   (ASR confidence × fluency) over the segments the answer used.
2. **Retrieval margin** — how clearly the top candidates stood out.
3. **Evidence agreement** — mean cosine similarity among the used segments
   (independent, mutually consistent evidence is worth more than one hit).
4. **Grounding verdict** — whether the LLM judged the evidence sufficient.

They are combined through a logistic model whose weights can be re-fit from
labelled questions (``calibrate``), so the score reads as a probability that
the answer is correct rather than an arbitrary heuristic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app import config
from app.models.schemas import (
    AnswerReliability,
    GATOutput,
    RetrievalResult,
    SLPSegment,
    UncertainWord,
    WordHighlight,
)
from app.slp.uncertainty import evidence_reliability, low_confidence_words


@dataclass
class CalibrationParams:
    """Weights of the logistic reliability model."""

    bias: float = -1.0
    w_evidence: float = 3.0
    w_margin: float = 1.5
    w_agreement: float = 1.0
    w_supported: float = 2.0

    @classmethod
    def load(cls, path: Path | None = None) -> "CalibrationParams":
        path = path or config.RELIABILITY_CALIBRATION_PATH
        if path.is_file():
            try:
                return cls(**json.loads(path.read_text(encoding="utf-8")))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or config.RELIABILITY_CALIBRATION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def evidence_reliability_component(
    gat_output: GATOutput,
    segments_by_id: dict[str, SLPSegment],
    context_ids: list[str],
) -> float:
    """GAT-score-weighted mean reliability of the answer context."""
    weights: list[float] = []
    values: list[float] = []
    for sid in context_ids:
        seg = segments_by_id.get(sid)
        if seg is None:
            continue
        w = max(gat_output.raw_node_scores.get(sid, gat_output.node_scores.get(sid, 0.0)), 1e-3)
        weights.append(w)
        values.append(gat_output.node_reliability.get(sid, evidence_reliability(seg)))
    if not values:
        return 0.0
    return float(np.average(values, weights=weights))


def retrieval_margin_component(retrieval: RetrievalResult, k: int = 3) -> float:
    """Normalised gap between the top-k mean and the rest of the candidates."""
    scores = [c.score for c in retrieval.candidates]
    if len(scores) < 2:
        return 1.0 if scores else 0.0
    head = float(np.mean(scores[:k]))
    tail = float(np.mean(scores[k:])) if len(scores) > k else 0.0
    return float(min(1.0, max(0.0, (head - tail) / max(abs(head), 1e-6))))


def evidence_agreement_component(embeddings: np.ndarray | None) -> float:
    """Mean pairwise cosine similarity among the used segments."""
    if embeddings is None or len(embeddings) < 2:
        return 0.5
    sims = embeddings @ embeddings.T
    n = len(embeddings)
    upper = sims[np.triu_indices(n, k=1)]
    return float(min(1.0, max(0.0, upper.mean())))


def uncertain_words_in_evidence(
    context_ids: list[str],
    segments_by_id: dict[str, SLPSegment],
    shap_highlights: dict[str, list[WordHighlight]],
) -> list[UncertainWord]:
    """Low-confidence ASR words that also carried positive SHAP influence.

    These are the words the answer *leans on* but the recogniser was unsure
    of — the most actionable thing to show a user.
    """
    out: list[UncertainWord] = []
    for sid in context_ids:
        seg = segments_by_id.get(sid)
        if seg is None:
            continue
        shap_by_word = {h.word.lower().strip(".,!?;:"): h.score for h in shap_highlights.get(sid, [])}
        for w in low_confidence_words(seg):
            key = w.word.lower().strip(".,!?;:")
            influence = shap_by_word.get(key, 0.0)
            out.append(
                UncertainWord(
                    segment_id=sid,
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    asr_probability=w.probability,
                    shap_influence=float(influence),
                )
            )
    out.sort(key=lambda u: (-(u.shap_influence > 0), u.asr_probability))
    return out[: config.MAX_UNCERTAIN_WORDS]


def _level(score: float) -> str:
    if score >= config.RELIABILITY_HIGH_THRESHOLD:
        return "high"
    if score >= config.RELIABILITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def score_answer(
    gat_output: GATOutput,
    retrieval: RetrievalResult,
    segments_by_id: dict[str, SLPSegment],
    context_ids: list[str],
    supported: bool,
    shap_highlights: dict[str, list[WordHighlight]] | None = None,
    context_embeddings: np.ndarray | None = None,
    params: CalibrationParams | None = None,
) -> AnswerReliability:
    """Compute the calibrated answer reliability and its explanation."""
    params = params or CalibrationParams.load()
    components = {
        "evidence_reliability": evidence_reliability_component(gat_output, segments_by_id, context_ids),
        "retrieval_margin": retrieval_margin_component(retrieval),
        "evidence_agreement": evidence_agreement_component(context_embeddings),
        "grounding_supported": 1.0 if supported else 0.0,
    }
    z = (
        params.bias
        + params.w_evidence * components["evidence_reliability"]
        + params.w_margin * components["retrieval_margin"]
        + params.w_agreement * components["evidence_agreement"]
        + params.w_supported * components["grounding_supported"]
    )
    score = _sigmoid(z)
    if not supported:
        # An answer the grounding check rejected can never read as reliable,
        # however clean the audio was.
        score = min(score, config.RELIABILITY_MEDIUM_THRESHOLD - 0.05)
    uncertain = uncertain_words_in_evidence(context_ids, segments_by_id, shap_highlights or {})
    shaky_in_use = [u for u in uncertain if u.shap_influence > 0]
    return AnswerReliability(
        score=round(score, 4),
        level=_level(score),
        components={k: round(v, 4) for k, v in components.items()},
        uncertain_words=uncertain,
        warning=(
            f"The answer relies on {len(shaky_in_use)} word(s) the transcriber was unsure about."
            if shaky_in_use else None
        ),
    )


def calibrate(
    component_rows: list[dict[str, float]],
    labels: list[int],
    epochs: int = 500,
    lr: float = 0.1,
) -> CalibrationParams:
    """Fit the logistic weights by gradient descent on labelled answers.

    Args:
        component_rows: One dict per answer with the four component values.
        labels: 1 if the answer was judged correct, else 0.
    """
    keys = ["evidence_reliability", "retrieval_margin", "evidence_agreement", "grounding_supported"]
    X = np.array([[row.get(k, 0.0) for k in keys] for row in component_rows], dtype=float)
    y = np.array(labels, dtype=float)
    if len(X) == 0:
        return CalibrationParams()
    w = np.zeros(len(keys))
    b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return CalibrationParams(bias=float(b), w_evidence=float(w[0]), w_margin=float(w[1]), w_agreement=float(w[2]), w_supported=float(w[3]))
