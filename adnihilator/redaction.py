"""Redact credential values and recognizable secret shapes in diagnostics."""

import os
import re
from collections.abc import Iterable

_SECRET_NAME = re.compile(r"KEY|TOKEN|PASSWORD|SECRET", re.IGNORECASE)
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:key|api[_-]?key|access_token|token|signature|x-amz-signature|x-amz-credential|x-goog-signature|x-goog-credential)=)[^&\s\"'<>]+"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)(\b(?:x-goog-api-key|x-api-key|api[_-]?key|access[_-]?key|secret[_-]?key|password|token|key|authorization)[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+)?)[^\s,}\"'&<>]+"
)


def redact_secrets(message: str, extra_secrets: Iterable[str | None] = ()) -> str:
    """Remove known values, credential assignments, bearer tokens and key shapes."""
    values = {value for name, value in os.environ.items() if _SECRET_NAME.search(name) and value}
    values.update(value for value in extra_secrets if isinstance(value, str) and value)
    for value in sorted(values, key=len, reverse=True):
        if len(value) >= 6:
            message = message.replace(value, "[REDACTED]")
        else:
            # A one-character fake key must not destroy ordinary words such as
            # "blocked". Credential assignments are also masked below.
            message = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", "[REDACTED]", message)
    message = _QUERY_SECRET.sub(r"\1[REDACTED]", message)
    message = _ASSIGNED_SECRET.sub(r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", message)
    message = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+)", "[REDACTED]", message)
    return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", message)
