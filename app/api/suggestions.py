"""Grounded Gemini question suggestions for an uploaded meeting."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from google.genai.errors import ClientError

from app import config
from app.models.schemas import QuestionSuggestionsResponse
from app.services.meeting_store import get_meeting_store

router = APIRouter(prefix="/meetings", tags=["suggestions"])
logger = logging.getLogger(__name__)

QUESTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


@router.get("/{meeting_id}/suggested-questions", response_model=QuestionSuggestionsResponse)
async def suggest_questions(meeting_id: str) -> QuestionSuggestionsResponse:
    """Return three decision-oriented questions answerable from this transcript."""
    record = get_meeting_store().get(meeting_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    if not config.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is missing.")

    transcript = "\n".join(
        f"[{segment.start_time:.1f}-{segment.end_time:.1f}s] {segment.text}"
        for segment in record.segments
    )
    # Enough context to represent the meeting without needlessly sending an entire long recording.
    transcript = transcript[:24000]
    prompt = (
        "You generate suggested questions for LISTEN, an explainable meeting QA system. "
        "Use ONLY the transcript below. Return JSON only as {\"questions\":[...]} with exactly three "
        "short questions that are explicitly answerable from the transcript. The questions must focus on: "
        "(1) a decision or conclusion, (2) a reason, concern, or trade-off behind that decision, and "
        "(3) a concrete action, owner, or next step. Do not ask generic questions, do not invent people "
        "or topics, and do not produce questions whose answer is absent or merely implied.\n\n"
        f"Transcript:\n{transcript}"
    )
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=256,
                temperature=0.1,
                response_mime_type="application/json",
                response_json_schema=QUESTION_RESPONSE_SCHEMA,
            ),
        )
        questions = json.loads(response.text or "{}").get("questions", [])
        questions = [str(question).strip() for question in questions if str(question).strip()]
    except ClientError as exc:
        logger.warning("Gemini suggested-question request failed for meeting %s: %s", meeting_id, exc)
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini quota is temporarily exhausted. Please retry shortly.",
            ) from exc
        raise HTTPException(status_code=502, detail="Gemini could not generate grounded questions.") from exc
    except Exception as exc:
        logger.exception("Gemini suggested-question generation failed for meeting %s", meeting_id)
        raise HTTPException(status_code=502, detail="Gemini could not generate grounded questions.") from exc
    if len(questions) != 3:
        raise HTTPException(status_code=502, detail="Gemini returned an invalid suggested-question response.")
    return QuestionSuggestionsResponse(questions=questions)
