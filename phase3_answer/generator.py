"""
LISTEN Phase 3 — LLM Answer Generation (Google Gemini)

Takes the top-weighted evidence segments from the GAT and synthesizes
a coherent answer using Google's Gemini API.

The prompt is structured to:
  1. Present evidence segments ranked by GAT importance weight
  2. Instruct the LLM to cite specific segments in its answer
  3. Generate a concise, grounded answer (no hallucination beyond evidence)
"""

from __future__ import annotations

from typing import List, Optional

from google import genai
from google.genai import types

import config
from schemas import EvidenceNode


# ──────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────

ANSWER_PROMPT_TEMPLATE = """You are a meeting analyst. Answer the question below based ONLY on the provided evidence segments from a meeting transcript.

**Rules:**
- Use ONLY information from the evidence segments to answer.
- Cite segment numbers (e.g., [Seg 1], [Seg 3]) when making claims.
- If the evidence is insufficient, say so explicitly.
- Be concise but thorough — synthesize across multiple segments when needed.
- Do NOT make up information not present in the evidence.

**Question:** {question}

**Evidence Segments (ranked by relevance):**
{evidence_text}

**Answer:**"""


def _format_evidence(nodes: List[EvidenceNode], top_k: int) -> str:
    """Format evidence nodes into a numbered text block for the prompt.

    Args:
        nodes: EvidenceNode list, pre-sorted by weight (descending).
        top_k: Maximum number of segments to include.

    Returns:
        Formatted string with numbered evidence segments.
    """
    lines = []
    for i, node in enumerate(nodes[:top_k]):
        lines.append(
            f"[Seg {i+1}] (Speaker: {node.speaker}, "
            f"Time: {node.start_time:.1f}s – {node.end_time:.1f}s, "
            f"Relevance: {node.weight:.2f})\n"
            f"  \"{node.text}\""
        )
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────
# Generator Class
# ──────────────────────────────────────────────────

class AnswerGenerator:
    """Generates answers from evidence segments using Google Gemini API.

    This is the final stage of Person B's pipeline:
      GAT-weighted segments → structured prompt → Gemini → answer text
    """

    def __init__(
        self,
        api_key: str = config.GEMINI_API_KEY,
        model_name: str = config.GEMINI_MODEL,
        temperature: float = config.GEMINI_TEMPERATURE,
        max_tokens: int = config.GEMINI_MAX_TOKENS,
        top_k_for_answer: int = config.TOP_K_FOR_ANSWER,
    ):
        """
        Args:
            api_key: Google Gemini API key.
            model_name: Gemini model identifier.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in generated answer.
            top_k_for_answer: How many top-weighted segments to include in prompt.
        """
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Set it via:\n"
                "  export GEMINI_API_KEY='your-key-here'  (Linux/Mac)\n"
                "  $env:GEMINI_API_KEY='your-key-here'    (PowerShell)\n"
                "Or set config.GEMINI_API_KEY directly."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k_for_answer

    def generate(
        self,
        question: str,
        evidence_nodes: List[EvidenceNode],
        top_k: Optional[int] = None,
    ) -> str:
        """Generate an answer from evidence segments.

        Args:
            question: The user's question.
            evidence_nodes: List of EvidenceNode objects, sorted by weight (descending).
            top_k: Override default top_k for this call.

        Returns:
            Generated answer string.
        """
        k = top_k or self.top_k

        # Format evidence into prompt
        evidence_text = _format_evidence(evidence_nodes, k)
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            evidence_text=evidence_text,
        )

        # Call Gemini via the new google.genai SDK
        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
            )
            return response.text.strip()

        except Exception as e:
            # Graceful fallback: return a structured summary without LLM
            return self._fallback_answer(question, evidence_nodes, k, error=str(e))

    def _fallback_answer(
        self,
        question: str,
        evidence_nodes: List[EvidenceNode],
        top_k: int,
        error: str = "",
    ) -> str:
        """Generate a basic answer without LLM (fallback if API fails).

        Returns a structured summary of the top evidence segments.
        """
        lines = [f"[LLM unavailable: {error}]", "", f"Question: {question}", ""]
        lines.append("Top evidence segments:")
        for i, node in enumerate(evidence_nodes[:top_k]):
            lines.append(
                f"  {i+1}. [{node.speaker}, {node.start_time:.1f}s] "
                f"(weight={node.weight:.2f}): {node.text}"
            )
        return "\n".join(lines)
