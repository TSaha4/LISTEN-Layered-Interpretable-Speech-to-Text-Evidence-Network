"""LLM-based answer generation with multi-provider fallback support."""

from __future__ import annotations

import logging
from typing import Callable

from app import config
from app.models.schemas import GATOutput, SLPSegment

logger = logging.getLogger(__name__)


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


def _generate_openai(question: str, evidence: str) -> str:
    """Generate answer using OpenAI API."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing.")

    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
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


def _generate_gemini(question: str, evidence: str) -> str:
    """Generate answer using Google Gemini API."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    prompt = (
        "You are a meeting QA assistant. Answer ONLY using the provided evidence "
        "segments from a real meeting transcript. If evidence is insufficient, say so explicitly.\n\n"
        f"Question: {question}\n\n"
        f"Evidence segments:\n{evidence}\n\n"
        "Provide a concise, factual answer citing segment numbers."
    )

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        ),
    )
    return response.text or ""


def _generate_groq(question: str, evidence: str) -> str:
    """Generate answer using Groq API."""
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
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


def generate_answer(
    question: str,
    gat_output: GATOutput,
    segments_by_id: dict[str, SLPSegment],
) -> str:
    """Generate a natural-language answer using an external LLM.
    Supports fallback across configured providers (OpenAI, Gemini, Groq).
    """
    evidence = _build_evidence_context(gat_output, segments_by_id)
    if not evidence.strip():
        return "Insufficient evidence found in the meeting transcript to answer this question."

    providers: dict[str, Callable[[str, str], str]] = {
        "openai": _generate_openai,
        "gemini": _generate_gemini,
        "groq": _generate_groq,
    }

    # Order the fallback sequence based on the PRIMARY_LLM_PROVIDER
    primary = config.PRIMARY_LLM_PROVIDER
    sequence = [primary] + [p for p in providers.keys() if p != primary]

    errors = []
    for provider_name in sequence:
        try:
            generator_fn = providers[provider_name]
            logger.info(f"Attempting to generate answer using {provider_name}...")
            answer = generator_fn(question, evidence)
            logger.info(f"Successfully generated answer using {provider_name}.")
            return answer
        except Exception as e:
            logger.warning(f"Provider {provider_name} failed: {e}")
            errors.append(f"{provider_name}: {e}")

    # Fallback if everything fails
    first_line = evidence.split("\n")[0] if evidence else ""
    return (
        f"[Offline mode — LLM generation failed across all providers: {', '.join(errors)}] "
        f"Based on top evidence for '{question}': {first_line}"
    )
