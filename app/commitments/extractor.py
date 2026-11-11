"""Extract decisions and commitments from uncertainty-annotated segments.

Pattern-based so it runs offline and deterministically; the LLM is not in
the loop here on purpose, so every commitment is traceable to a literal
transcript span. Each commitment inherits the reliability of the segment
it came from, which lets the graph layer prefer a clearly-heard decision
over a mumbled one when they conflict.
"""

from __future__ import annotations

import hashlib
import re

from app import config
from app.models.schemas import Commitment, SLPSegment
from app.slp.uncertainty import evidence_reliability

DECISION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwe(?:'ve| have)? (?:decided|agreed|settled on|concluded|are going with|will go with)\b",
        r"\b(?:the )?decision is\b",
        r"\blet'?s (?:go with|finalize|finalise|lock in)\b",
        r"\bit'?s (?:settled|final|decided|agreed)\b",
        r"\bwe(?:'re| are) (?:moving|going) (?:ahead|forward) with\b",
    )
)

COMMITMENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:i|we|he|she|they)(?:'ll| will| am going to| are going to|'m going to|'re going to) \w+",
        r"\b(?:i|we)(?:'ve| have)? committed to\b",
        r"\b(?:i|we) promise\b",
        r"\b(?:i|we) (?:can|will) (?:take|own|handle|deliver|send|finish|ship)\b",
        r"\baction item\b",
        # Named subject: "Alice will send ...", "The team is going to ship ..."
        r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)? (?:will|is going to|'ll) "
        r"(?:send|take|own|handle|deliver|finish|ship|write|do|prepare|follow|share|review|set up|update)\b",
    )
)

DEADLINE_PATTERN = re.compile(
    r"\b(?:by|before|due|deadline(?: is)?|no later than) (?:end of |eod |next |this )?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|quarter|"
    r"q[1-4]|january|february|march|april|may|june|july|august|september|october|"
    r"november|december|\d{1,2}(?:st|nd|rd|th)?|tomorrow|tonight)",
    re.IGNORECASE,
)

NEGATION_PATTERN = re.compile(
    r"\b(?:not|no longer|won'?t|will not|can'?t|cannot|never|don'?t|isn'?t|aren'?t|"
    r"scrap|cancel(?:led|ed)?|drop(?:ped)?|revert(?:ed)?|instead of)\b",
    re.IGNORECASE,
)

VALUE_LABELS: frozenset[str] = frozenset({"DATE", "TIME", "MONEY", "CARDINAL", "PERCENT", "QUANTITY", "ORDINAL"})
TOPIC_LABELS: frozenset[str] = frozenset({"ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "GPE", "LOC", "NORP", "FAC", "LAW"})


def _classify(text: str) -> str | None:
    if any(p.search(text) for p in DECISION_PATTERNS):
        return "decision"
    if DEADLINE_PATTERN.search(text):
        return "deadline"
    if any(p.search(text) for p in COMMITMENT_PATTERNS):
        return "commitment"
    return None


def _labelled_entities(text: str) -> list[tuple[str, str]]:
    """Return ``(surface, label)`` pairs using spaCy when available."""
    try:
        from app.slp.entity_extraction import _get_nlp

        doc = _get_nlp()(text)
        return [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
    except Exception:
        return []


def _commitment_id(meeting_id: str, segment_id: str) -> str:
    return hashlib.sha1(f"{meeting_id}:{segment_id}".encode("utf-8")).hexdigest()[:12]


def extract_from_segment(
    segment: SLPSegment,
    meeting_id: str,
    meeting_time: float,
    labelled: list[tuple[str, str]] | None = None,
) -> Commitment | None:
    """Return a ``Commitment`` if the segment states one, else ``None``."""
    kind = _classify(segment.text)
    if kind is None:
        return None

    reliability = evidence_reliability(segment)
    if reliability < config.COMMITMENT_MIN_RELIABILITY:
        return None

    labelled = labelled if labelled is not None else _labelled_entities(segment.text)
    owner = next((s for s, label in labelled if label == "PERSON"), None)
    values = [s for s, label in labelled if label in VALUE_LABELS]
    topics = [s for s, label in labelled if label in TOPIC_LABELS]
    if not topics:
        # Fall back to any entity that is not the owner or a value.
        topics = [e for e in segment.entities if e != owner and e not in values]

    return Commitment(
        commitment_id=_commitment_id(meeting_id, segment.segment_id),
        meeting_id=meeting_id,
        segment_id=segment.segment_id,
        text=segment.text,
        kind=kind,
        owner=owner,
        topic_entities=topics,
        values=values,
        polarity=-1 if NEGATION_PATTERN.search(segment.text) else 1,
        meeting_time=float(meeting_time),
        offset_s=float(segment.start_time),
        reliability=reliability,
    )


def extract_commitments(
    segments: list[SLPSegment],
    meeting_id: str,
    meeting_time: float,
) -> list[Commitment]:
    """Extract every commitment stated in a meeting, in transcript order."""
    out: list[Commitment] = []
    for seg in segments:
        c = extract_from_segment(seg, meeting_id, meeting_time)
        if c is not None:
            out.append(c)
    return out
