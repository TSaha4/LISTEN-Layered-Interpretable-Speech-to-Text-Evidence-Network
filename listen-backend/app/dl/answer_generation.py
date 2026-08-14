"""LLM-based answer generation from top-weighted evidence segments."""

from __future__ import annotations

import os

from app import config
from app.models.schemas import GATOutput, SLPSegment


def _build_evidence_context(
    gat_output: GATOutput,
    segments_by_id: dict[str, SLPSegment],
) -> str:
    """Format top evidence segments as numbered context for the LLM."""
    lines: list[str] = []
    for rank, seg_id in enumerate(gat_output.top_evidence_ids, start=1):
        seg = segments_by_id.get(seg_id)
        if seg is None:
            continue
        score = gat_output.node_scores.get(seg_id, 0.0)
        lines.append(
            f"[{rank}] (score={score:.3f}, t={seg.start_time:.1f}-{seg.end_time:.1f}s) "
            f"{seg.text}"
        )
    return "\n".join(lines)


def generate_answer(
    question: str,
    gat_output: GATOutput,
    segments_by_id: dict[str, SLPSegment],
) -> str:
    """Generate a natural-language answer using an external LLM.

    Args:
        question: User question.
        gat_output: GAT-ranked evidence with ``top_evidence_ids``.
        segments_by_id: Meeting segment lookup.

    Returns:
        Answer string from the configured LLM provider.

    Raises:
        RuntimeError: If the API key is missing or the provider call fails.
    """
    evidence = _build_evidence_context(gat_output, segments_by_id)
    if not evidence.strip():
        return "Insufficient evidence found in the meeting transcript to answer this question."

    if config.LLM_PROVIDER == "openai":
        return _generate_openai(question, evidence)
    raise RuntimeError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")


def _generate_openai(question: str, evidence: str) -> str:
    """Call the OpenAI chat completions API."""
    api_key = os.environ.get(config.OPENAI_API_KEY_ENV)
    if not api_key:
        return _fallback_answer(question, evidence)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meeting QA assistant. Answer ONLY using the "
                        "provided evidence segments from a real meeting transcript. "
                        "If evidence is insufficient, say so explicitly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Evidence segments:\n{evidence}\n\n"
                        "Provide a concise, factual answer citing segment numbers."
                    ),
                },
            ],
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_answer(question, evidence)


def _fallback_answer(question: str, evidence: str) -> str:
    """Deterministic fallback when no LLM API key is configured."""
    first_line = evidence.split("\n")[0] if evidence else ""
    return (
        f"[Offline mode — set {config.OPENAI_API_KEY_ENV} for LLM answers] "
        f"Based on top evidence for '{question}': {first_line}"
    )
