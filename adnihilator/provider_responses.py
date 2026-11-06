"""Require a completed, unblocked provider decision before parsing its text."""

from typing import Any


def completed_openai_text(response: Any) -> str:
    """Accept one stopped, non-refused chat completion with actual text."""
    choices = getattr(response, "choices", None)
    if not choices or len(choices) != 1:
        raise ValueError("OpenAI response must contain one completed decision")
    choice = choices[0]
    if getattr(choice, "finish_reason", None) != "stop":
        raise ValueError("OpenAI response is incomplete, filtered, or missing a successful stop status")
    message = getattr(choice, "message", None)
    if (getattr(message, "refusal", None) or getattr(message, "tool_calls", None)
            or getattr(message, "function_call", None)):
        raise ValueError("OpenAI response refused the decision or returned an unexpected tool call")
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OpenAI response has no decision text")
    return text


def _status_name(value: Any) -> str:
    """Normalize SDK string enums without depending on an SDK import."""
    return str(getattr(value, "value", value) or "")


def completed_gemini_text(response: Any) -> str:
    """Accept one explicitly stopped candidate without prompt or safety blocks."""
    feedback = getattr(response, "prompt_feedback", None)
    block = _status_name(getattr(feedback, "block_reason", None))
    if block not in ("", "BLOCKED_REASON_UNSPECIFIED"):
        raise ValueError("Gemini response blocked by prompt feedback")
    candidates = getattr(response, "candidates", None)
    if not candidates or len(candidates) != 1:
        raise ValueError("Gemini response must contain one completed decision")
    candidate = candidates[0]
    if _status_name(getattr(candidate, "finish_reason", None)) != "STOP":
        raise ValueError("Gemini response blocked, incomplete, or missing a successful STOP status")
    ratings = (getattr(feedback, "safety_ratings", None) or []) + (getattr(candidate, "safety_ratings", None) or [])
    if any(getattr(rating, "blocked", False) for rating in ratings):
        raise ValueError("Gemini response blocked by safety feedback")
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty text in Gemini response")
    return text
