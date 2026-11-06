"""Shared Gemini API helpers (google-genai SDK, generateContent).

Centralizes the default model, SDK client construction, and per-request
configuration so the single-shot and chunked audio paths stay consistent.

Gemini 3.x request rules (Gemini 3.6 Flash / 3.5 Flash-Lite migration guide):
- ``temperature``/``top_p``/``top_k`` are deprecated and must not be sent.
- ``candidate_count`` is unsupported.
- Requests must not end with a prefilled ``model`` turn.
- Thinking is controlled with ``thinking_level`` (not ``thinking_budget``).
"""

import re
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_THINKING_LEVEL = "minimal"
VALID_THINKING_LEVELS = ("minimal", "low", "medium", "high")

_GEMINI_MAJOR_RE = re.compile(r"^(?:models/)?gemini-(\d+)")


def supports_thinking_level(model: str) -> bool:
    """Return True if the model accepts ``thinking_level`` (Gemini 3 and later).

    Older overrides such as ``gemini-2.5-flash`` use ``thinking_budget`` and are
    left on their API defaults so explicit model overrides keep working.
    """
    match = _GEMINI_MAJOR_RE.match(model.strip().lower())
    return bool(match) and int(match.group(1)) >= 3


def validate_thinking_level(level: str) -> str:
    """Normalize and validate a configured thinking level."""
    normalized = str(level).strip().lower()
    if normalized not in VALID_THINKING_LEVELS:
        raise ValueError(
            f"Invalid gemini.thinking_level {level!r}; "
            f"expected one of {', '.join(VALID_THINKING_LEVELS)}"
        )
    return normalized


def create_genai_client(api_key: str) -> Any:
    """Create a google-genai client for the Gemini Developer API."""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai is required for Gemini detection. "
            "Run: pip install google-genai"
        ) from e
    return genai.Client(api_key=api_key)


def build_generate_config(model: str, thinking_level: str = DEFAULT_THINKING_LEVEL) -> Any:
    """Build the generateContent config for an audio ad-detection request.

    Deliberately omits sampling parameters and candidate_count. Returns None
    for models that predate ``thinking_level`` so their defaults are unchanged.
    """
    if not supports_thinking_level(model):
        return None

    from google.genai import types

    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=validate_thinking_level(thinking_level),
        ),
    )


def file_state_name(file_obj: Any) -> str:
    """Return an uploaded file's state name (e.g. PROCESSING, ACTIVE, FAILED)."""
    state = getattr(file_obj, "state", None)
    return str(getattr(state, "name", state) or "")


def extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a generateContent response.

    Output tokens include thinking tokens because Gemini bills thinking at the
    output-token rate.
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}

    input_tokens = getattr(meta, "prompt_token_count", None) or 0
    candidate_tokens = getattr(meta, "candidates_token_count", None) or 0
    thinking_tokens = getattr(meta, "thoughts_token_count", None) or 0
    total_tokens = getattr(meta, "total_token_count", None) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": candidate_tokens + thinking_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
    }
