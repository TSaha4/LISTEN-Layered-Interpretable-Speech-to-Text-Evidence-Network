"""Named-entity recognition on transcript segments using spaCy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app import config
from app.models.schemas import SLPSegment

if TYPE_CHECKING:
    import spacy


_nlp: spacy.Language | None = None


def _get_nlp() -> spacy.Language:
    """Load the spaCy pipeline lazily."""
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load(config.SPACY_MODEL)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{config.SPACY_MODEL}' is not installed. "
                f"Run: python -m spacy download {config.SPACY_MODEL}"
            ) from exc
    return _nlp


def extract_entities_from_text(text: str) -> list[str]:
    """Run spaCy NER on a single text span.

    Args:
        text: Transcript text for one segment.

    Returns:
        Deduplicated list of entity surface forms (order preserved).
    """
    doc = _get_nlp()(text)
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        surface = ent.text.strip()
        key = surface.lower()
        if surface and key not in seen:
            seen.add(key)
            entities.append(surface)
    return entities


def enrich_segments_with_entities(segments: list[SLPSegment]) -> list[SLPSegment]:
    """Attach spaCy entities to each segment, preserving other fields.

    Args:
        segments: ASR segments with empty ``entities`` lists.

    Returns:
        New list of segments with populated ``entities``.
    """
    enriched: list[SLPSegment] = []
    for seg in segments:
        enriched.append(
            seg.model_copy(
                update={"entities": extract_entities_from_text(seg.text)}
            )
        )
    return enriched


def reset_nlp() -> None:
    """Release cached spaCy pipeline (useful in tests)."""
    global _nlp
    _nlp = None
