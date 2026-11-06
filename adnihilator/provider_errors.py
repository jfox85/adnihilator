"""Classify provider outages and redact secrets before crossing error boundaries."""

import re
from json import JSONDecodeError
from collections.abc import Iterable

import httpx
from pydantic import ValidationError

from .models import ProviderFailureInfo
from .redaction import redact_secrets

_STATUS = re.compile(r"(?:^\s*|Error code:\s*|HttpError\s+)([45]\d{2})(?=\s|[.:])", re.IGNORECASE)


class InvalidProviderDecision(ValueError):
    """Provider data failed validation; its text is not an SDK outage signal."""


def classify_provider_error(error: BaseException | str, extra_secrets: Iterable[str | None] = ()) -> ProviderFailureInfo:
    """Prefer SDK status fields, with a fallback for historical exception text."""
    structured = getattr(error, "failure", None)
    if isinstance(structured, ProviderFailureInfo):
        return structured.model_copy(update={"message": redact_secrets(structured.message, extra_secrets)})
    if isinstance(error, (InvalidProviderDecision, ValidationError, JSONDecodeError)):
        return ProviderFailureInfo(category="unknown", message=redact_secrets(str(error), extra_secrets))
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
    # SDK account-quota metadata differs from ordinary HTTP 429 throttling.
    quota_codes = {getattr(error, field, None) for field in ("code", "type")
                   if isinstance(getattr(error, field, None), str)}
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            quota_codes.update(details.get(field) for field in ("code", "type")
                               if isinstance(details.get(field), str))
    insufficient_quota = "insufficient_quota" in quota_codes
    category = "unknown"
    if insufficient_quota or status == 402 or ("depleted" in text and ("credit" in text or "prepay" in text)):
        category = "billing"
    elif status in (401, 403) or "api_key_invalid" in text or "api key not valid" in text or "invalid api key" in text:
        category = "authentication"
    elif status == 404 and "model" in text:
        category = "model"
    elif status == 429:
        category = "rate_limit"
    elif status in (500, 502, 503, 504) or isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)) or any(word in text for word in ("timed out", "timeout", "deadline exceeded", "overloaded")):
        category = "transient"
    return ProviderFailureInfo(
        category=category, status_code=status,
        global_failure=category in ("billing", "authentication", "model"),
        # Owner D3: permanent failures remain excluded before this branch.
        retryable=category in ("rate_limit", "transient"), message=redact_secrets(raw, extra_secrets),
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


class DetectionSafetyError(RuntimeError):
    """Required detection could not be safely completed."""


class ProviderOutageError(DetectionSafetyError):
    """A required provider decision failed for a global credit/auth/model reason."""

    def __init__(self, provider: str, failure: ProviderFailureInfo, message: str | None = None) -> None:
        self.provider = provider
        self.failure = failure
        super().__init__(message or describe_provider_failure(failure).replace("Gemini", "OpenAI" if provider == "openai" else "Gemini"))
