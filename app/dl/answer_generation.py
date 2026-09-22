"""Gemini-only grounded answer generation for meeting questions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app import config
from app.models.schemas import GATOutput, SLPSegment


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    supported: bool


def _build_evidence_context(
    gat_output: GATOutput,
    segments_by_id: dict[str, SLPSegment],
    context_segment_ids: list[str] | None = None,
) -> str:
    """Format the selected answer-context segments for Gemini.

    ``context_segment_ids`` is deliberately independent of GAT visualization
    output: the live pipeline may add retrieved candidates as an answer safety
    net without changing the graph shown to the user.
    """
    lines: list[str] = []
    evidence_ids = context_segment_ids if context_segment_ids is not None else gat_output.top_evidence_ids
    for rank, segment_id in enumerate(evidence_ids, start=1):
        segment = segments_by_id.get(segment_id)
        if segment is None:
            continue
        score = gat_output.node_scores.get(segment_id, 0.0)
        lines.append(f"[{rank}] (score={score:.3f}, t={segment.start_time:.1f}-{segment.end_time:.1f}s) {segment.text}")
    return "\n".join(lines)


def generate_answer_result(
    question: str,
    gat_output: GATOutput,
    segments_by_id: dict[str, SLPSegment],
    context_segment_ids: list[str] | None = None,
) -> GeneratedAnswer:
    """Generate a grounded answer using Gemini only, with no provider fallback."""
    if config.PRIMARY_LLM_PROVIDER != "gemini":
        raise RuntimeError("Only the Gemini provider is enabled.")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    evidence = _build_evidence_context(gat_output, segments_by_id, context_segment_ids)
    if not evidence:
        return GeneratedAnswer("The meeting transcript does not contain enough evidence to answer this question.", False)

    from google import genai
    from google.genai import types

    prompt = (
        "You are LISTEN, a grounded meeting QA assistant. Answer ONLY from the provided "
        "evidence segments. Decide whether they actually answer the question; relevant-sounding "
        "words alone are not enough. Return JSON only with exactly these fields: "
        "{\"answer\": string, \"supported\": boolean}. If unsupported, set supported to false and "
        "write a short message saying the meeting transcript does not contain the answer.\n\n"
        f"Question: {question}\n\nEvidence segments:\n{evidence}\n\n"
        "When supported is true, provide a concise factual answer and cite the supporting segment numbers."
    )
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty answer.")
    try:
        payload = json.loads(response.text)
        answer = str(payload["answer"]).strip()
        supported = bool(payload["supported"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini returned an invalid grounded-answer response.") from exc
    if not answer:
        raise RuntimeError("Gemini returned an empty grounded answer.")
    return GeneratedAnswer(answer, supported)


def generate_answer(question: str, gat_output: GATOutput, segments_by_id: dict[str, SLPSegment]) -> str:
    """Compatibility wrapper used by the counterfactual pipeline."""
    return generate_answer_result(question, gat_output, segments_by_id).answer
