"""Redact credential values and recognizable secret shapes in diagnostics."""

import os
import re
from collections.abc import Iterable

_SECRET_NAME = re.compile(r"KEY|TOKEN|PASSWORD|SECRET", re.IGNORECASE)
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:key|api[_-]?key|access_token|token|signature|x-amz-signature|x-amz-credential|x-goog-signature|x-goog-credential)=)[^&\s\"'<>]+"
)
_ASSIGNMENT_PREFIX = re.compile(
    r"""(?i)(?<![\w-])(?:--?)?(?P<name>[a-z_][a-z0-9_-]*)["']?\s*[:=]\s*"""
)
_ASSIGNMENT_VALUE = re.compile(
    r""""(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?|[^\s,}"'&<>]+""",
    re.DOTALL,
)
_CREDENTIAL_NAMES = {
    "key", "api_key", "apikey", "access_key", "secret_key", "token",
    "password", "passwd", "secret", "authorization", "signature", "credential",
}
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_AUTH_SCHEMES = {"basic", "bearer", "token", "digest", "negotiate", "aws4-hmac-sha256"}


def _credential_name(name: str) -> bool:
    """Recognize whole credential fields and namespaced suffixes, not counters."""
    normalized = _CAMEL_BOUNDARY.sub("_", name).lower().replace("-", "_")
    return normalized in _CREDENTIAL_NAMES or any(
        normalized.endswith("_" + suffix) for suffix in _CREDENTIAL_NAMES
    )


def _masked_value(name: str, value: str) -> str:
    """Preserve quote delimiters and known schemes, never an unknown raw key."""
    quote = value[0] if value[0] in "\"'" else ""
    closing_quote = quote if len(value) > 1 and value[-1] == quote else ""
    inner = value[1:-1] if closing_quote else value[1:] if quote else value
    scheme = re.match(r"([A-Za-z][A-Za-z0-9_-]*[ \t]+)", inner)
    scheme_prefix = ""
    if (
        name.lower().endswith("authorization")
        and scheme
        and scheme.group(0).strip().lower() in _AUTH_SCHEMES
    ):
        scheme_prefix = scheme.group(0)
    return quote + scheme_prefix + "[REDACTED]" + closing_quote


def _redact_assignments(message: str) -> str:
    """Scan prefixes without consuming ordinary diagnostic wrapper values."""
    parts = []
    covered_end = 0
    for match in _ASSIGNMENT_PREFIX.finditer(message):
        if match.start() < covered_end or not _credential_name(match.group("name")):
            continue
        start = match.end()
        value = _ASSIGNMENT_VALUE.match(message, start)
        if value is None:
            continue
        end = value.end()
        name = match.group("name")
        if name.lower().endswith("authorization") and message[start] not in "\"'":
            # Unquoted headers may contain spaces, commas and parameters.
            # Preserve following diagnostic lines, not an ambiguous suffix.
            end = len(message)
            for newline in ("\r", "\n"):
                boundary = message.find(newline, start)
                if boundary >= 0:
                    end = min(end, boundary)
        parts.append(message[covered_end:start])
        parts.append(_masked_value(name, message[start:end]))
        covered_end = end
    parts.append(message[covered_end:])
    return "".join(parts)


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
    message = _redact_assignments(message)
    message = re.sub(r"(?i)(\bBearer[ \t]+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", message)
    message = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+)", "[REDACTED]", message)
    message = re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "[REDACTED]", message)
    return re.sub(r"(?i)(https?://)[^/?#@\s]+@", r"\1[REDACTED]@", message)
