"""Speech-level uncertainty cues derived from word-timed ASR output.

Two independent signals are computed per transcript segment:

* **Recognition uncertainty** — how sure the ASR model is that it heard the
  words correctly (mean / min word probability, no-speech probability).
* **Speaker hesitation** — how sure the *speaker* sounded: filled pauses
  ("um", "uh"), hedges ("I think", "maybe"), long intra-segment silences and
  self-repairs ("the the").

Both are later fused into a single evidence reliability value that the
graph layer consumes. Everything here is pure Python so it is testable
without loading Whisper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app import config
from app.models.schemas import SLPSegment, WordToken

FILLER_WORDS: frozenset[str] = frozenset({
    "um", "uh", "umm", "uhh", "hmm", "mm", "er", "erm", "ah", "eh",
})

HEDGE_PHRASES: tuple[str, ...] = (
    "i think", "i guess", "maybe", "perhaps", "probably", "possibly",
    "not sure", "kind of", "sort of", "i believe", "i suppose", "might",
    "or something", "i don't know", "something like",
)

_WORD_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class UncertaintyCues:
    """Per-segment uncertainty summary."""

    asr_confidence: float
    min_word_confidence: float
    hesitation_score: float
    filler_count: int
    hedge_count: int
    long_pause_count: int
    repetition_count: int


def _normalise(word: str) -> str:
    return "".join(_WORD_RE.findall(word.lower()))


def recognition_confidence(words: list[WordToken]) -> tuple[float, float]:
    """Return ``(mean_probability, min_probability)`` over content words."""
    probs = [w.probability for w in words if _normalise(w.word)]
    if not probs:
        return 1.0, 1.0
    return float(sum(probs) / len(probs)), float(min(probs))


def hesitation_cues(
    words: list[WordToken],
    text: str,
    pause_threshold_s: float = config.HESITATION_PAUSE_SECONDS,
) -> tuple[int, int, int, int]:
    """Count fillers, hedges, long pauses and immediate repetitions."""
    tokens = [_normalise(w.word) for w in words]
    tokens = [t for t in tokens if t]

    filler_count = sum(1 for t in tokens if t in FILLER_WORDS)

    lowered = " ".join(_WORD_RE.findall(text.lower()))
    hedge_count = sum(lowered.count(phrase) for phrase in HEDGE_PHRASES)

    long_pause_count = 0
    for prev, nxt in zip(words, words[1:]):
        if nxt.start - prev.end > pause_threshold_s:
            long_pause_count += 1

    repetition_count = sum(
        1 for a, b in zip(tokens, tokens[1:]) if a == b and a not in FILLER_WORDS
    )
    return filler_count, hedge_count, long_pause_count, repetition_count


def hesitation_score(
    filler_count: int,
    hedge_count: int,
    long_pause_count: int,
    repetition_count: int,
    word_count: int,
) -> float:
    """Combine hesitation counts into a bounded ``[0, 1]`` score.

    Counts are normalised per 10 words so long segments are not penalised
    simply for containing more speech.
    """
    if word_count <= 0:
        return 0.0
    per_ten = 10.0 / max(word_count, 10)
    raw = (
        config.HESITATION_FILLER_WEIGHT * filler_count
        + config.HESITATION_HEDGE_WEIGHT * hedge_count
        + config.HESITATION_PAUSE_WEIGHT * long_pause_count
        + config.HESITATION_REPEAT_WEIGHT * repetition_count
    ) * per_ten
    return float(min(1.0, raw))


def compute_cues(words: list[WordToken], text: str) -> UncertaintyCues:
    """Compute every uncertainty cue for one segment."""
    mean_p, min_p = recognition_confidence(words)
    fillers, hedges, pauses, repeats = hesitation_cues(words, text)
    word_count = len([w for w in words if _normalise(w.word)]) or len(text.split())
    return UncertaintyCues(
        asr_confidence=mean_p,
        min_word_confidence=min_p,
        hesitation_score=hesitation_score(fillers, hedges, pauses, repeats, word_count),
        filler_count=fillers,
        hedge_count=hedges,
        long_pause_count=pauses,
        repetition_count=repeats,
    )


def annotate_segment(segment: SLPSegment) -> SLPSegment:
    """Return a copy of ``segment`` with uncertainty fields filled in."""
    cues = compute_cues(segment.words, segment.text)
    return segment.model_copy(update={
        "asr_confidence": cues.asr_confidence,
        "min_word_confidence": cues.min_word_confidence,
        "hesitation_score": cues.hesitation_score,
    })


def evidence_reliability(segment: SLPSegment) -> float:
    """Fuse recognition and hesitation cues into one ``[0, 1]`` reliability.

    The fusion is multiplicative: evidence is only as reliable as its
    weakest layer. A perfectly transcribed but heavily hedged statement, or
    a confident statement the ASR could not hear, both score low.
    """
    recog = (
        config.RELIABILITY_MEAN_CONF_WEIGHT * segment.asr_confidence
        + (1.0 - config.RELIABILITY_MEAN_CONF_WEIGHT) * segment.min_word_confidence
    )
    recog *= 1.0 - segment.no_speech_prob
    fluency = 1.0 - config.RELIABILITY_HESITATION_PENALTY * segment.hesitation_score
    value = recog * fluency
    return float(min(1.0, max(config.RELIABILITY_FLOOR, value)))


def low_confidence_words(
    segment: SLPSegment,
    threshold: float = config.LOW_CONFIDENCE_WORD_THRESHOLD,
) -> list[WordToken]:
    """Words the ASR was unsure about, in transcript order."""
    return [w for w in segment.words if w.probability < threshold and _normalise(w.word)]


def entity_confidence(segment: SLPSegment, entity: str) -> float:
    """Mean ASR probability of the words that make up ``entity``.

    Falls back to the segment mean when word alignment is unavailable, so
    legacy meetings behave exactly as before.
    """
    if not segment.words:
        return segment.asr_confidence
    target = [_normalise(t) for t in entity.split()]
    target = [t for t in target if t]
    if not target:
        return segment.asr_confidence
    tokens = [(_normalise(w.word), w.probability) for w in segment.words]
    n = len(target)
    best: float | None = None
    for i in range(len(tokens) - n + 1):
        window = tokens[i:i + n]
        if [t for t, _ in window] == target:
            mean_p = sum(p for _, p in window) / n
            best = mean_p if best is None else max(best, mean_p)
    return best if best is not None else segment.asr_confidence
