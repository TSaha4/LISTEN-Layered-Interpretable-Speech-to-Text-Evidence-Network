"""Gemini-only grounded answer generation for meeting questions."""

from __future__ import annotations

from app import config
from app.models.schemas import GATOutput, SLPSegment


def _build_evidence_context(gat_output: GATOutput, segments_by_id: dict[str, SLPSegment]) -> str:
    """Format ranked evidence segments for Gemini."""
    lines: list[str] = []
    for rank, segment_id in enumerate(gat_output.top_evidence_ids, start=1):
        segment = segments_by_id.get(segment_id)
        if segment is None:
            continue
        score = gat_output.node_scores.get(segment_id, 0.0)
        lines.append(f"[{rank}] (score={score:.3f}, t={segment.start_time:.1f}-{segment.end_time:.1f}s) {segment.text}")
    return "\n".join(lines)


def generate_answer(question: str, gat_output: GATOutput, segments_by_id: dict[str, SLPSegment]) -> str:
    """Generate a grounded answer using Gemini only, with no provider fallback."""
    if config.PRIMARY_LLM_PROVIDER != "gemini":
        raise RuntimeError("Only the Gemini provider is enabled.")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    evidence = _build_evidence_context(gat_output, segments_by_id)
    if not evidence:
        return "Insufficient evidence found in the meeting transcript to answer this question."

    from google import genai
    from google.genai import types

    prompt = (
        "You are a meeting QA assistant. Answer ONLY using the provided evidence "
        "segments from a real meeting transcript. If evidence is insufficient, say so explicitly.\n\n"
        f"Question: {question}\n\nEvidence segments:\n{evidence}\n\n"
        "Provide a concise, factual answer citing segment numbers."
    )
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty answer.")
    return response.text
