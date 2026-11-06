"""Classify provider outages and redact secrets before crossing error boundaries."""

import re
from collections.abc import Iterable

from .models import ProviderFailureInfo
from .redaction import redact_secrets

_STATUS = re.compile(r"(?:^\s*|Error code:\s*|HttpError\s+)([45]\d{2})(?=\s|[.:])", re.IGNORECASE)


def classify_provider_error(error: BaseException | str, extra_secrets: Iterable[str | None] = ()) -> ProviderFailureInfo:
    """Prefer SDK status fields, with a fallback for historical exception text."""
    raw = str(error)
    status = None
    response = getattr(error, "response", None)
    for value in (getattr(error, "code", None), getattr(error, "status_code", None), getattr(response, "status_code", None)):
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            status = value
            break
    if status is None:
        match = _STATUS.search(raw)
        if match:
            status = int(match.group(1))
    text = raw.lower()
    category = "unknown"
    if status == 402 or ("depleted" in text and ("credit" in text or "prepay" in text)):
        category = "billing"
    elif status in (401, 403) or "api_key_invalid" in text or "api key not valid" in text or "invalid api key" in text:
        category = "authentication"
    elif status == 404 and "model" in text:
        category = "model"
    elif status == 429:
        category = "rate_limit"
    elif status in (500, 502, 503, 504) or any(word in text for word in ("timed out", "timeout", "deadline exceeded", "overloaded")):
        category = "transient"
    return ProviderFailureInfo(
        category=category, status_code=status,
        global_failure=category in ("billing", "authentication", "model"),
        # Broader transient/JSON retries are a separate owner policy decision.
        retryable=category == "rate_limit", message=redact_secrets(raw, extra_secrets),
    )


def describe_provider_failure(failure: ProviderFailureInfo) -> str:
    """Put an actionable permanent-outage reason before long provider diagnostics."""
    actions = {
        "billing": "Gemini credits are depleted; restore provider credits before retrying.",
        "authentication": "Gemini authentication failed; check the provider key.",
        "model": "Gemini model is unavailable; check the model setting.",
    }
    action = actions.get(failure.category)
    if action is None:
        return failure.message
    status = f" (HTTP {failure.status_code})" if failure.status_code is not None else ""
    return f"{action}{status} {failure.message}"
