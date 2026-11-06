"""Redact credential values and recognizable secret shapes in diagnostics."""

import os
import re
from collections.abc import Iterable

_SECRET_NAME = re.compile(r"KEY|TOKEN|PASSWORD|SECRET", re.IGNORECASE)
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:key|api[_-]?key|access_token|token|signature|x-amz-signature|x-amz-credential|x-goog-signature|x-goog-credential)=)[^&\s\"'<>]+"
)
_ASSIGNED_SECRET = re.compile(
    r"""(?ix)(?<![\w-])
    (?P<prefix>(?P<name>[a-z_][a-z0-9_-]*)["']?\s*[:=]\s*)
    (?P<value>"(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?|[^\s,}"'&<>]+)"""
)
_AUTH_HEADER = re.compile(
    r"""(?im)((?<![\w-])(?:proxy[-_])?authorization["']?[ \t]*[:=][ \t]*)([^ \t\r\n"'][^\r\n]*)"""
)
_CREDENTIAL_NAMES = {
    "key", "api_key", "apikey", "access_key", "secret_key", "token",
    "password", "passwd", "secret", "authorization", "signature", "credential",
}


def _credential_name(name: str) -> bool:
    """Recognize whole credential fields and namespaced suffixes, not counters."""
    normalized = name.lower().replace("-", "_")
    return normalized in _CREDENTIAL_NAMES or any(
        normalized.endswith("_" + suffix) for suffix in _CREDENTIAL_NAMES
    )


def _redact_assignment(match: re.Match[str]) -> str:
    """Mask a whole quoted value while retaining its diagnostic field name."""
    name = match.group("name")
    if not _credential_name(name):
        return match.group(0)
    value = match.group("value")
    quote = value[0] if value[0] in "\"'" else ""
    closing_quote = quote if len(value) > 1 and value[-1] == quote else ""
    if not quote and _AUTH_HEADER.match(match.string, match.start()) is not None:
        # A raw header may contain spaces, commas and quoted parameters. Its
        # dedicated pass masks the entire value rather than only the scheme.
        return match.group(0)
    inner = value[1:-1] if closing_quote else value[1:] if quote else value
    scheme = re.match(r"([A-Za-z][A-Za-z0-9_-]*[ \t]+)", inner)
    scheme_prefix = scheme.group(0) if name.lower().endswith("authorization") and scheme else ""
    return match.group("prefix") + quote + scheme_prefix + "[REDACTED]" + closing_quote


def _redact_auth_header(match: re.Match[str]) -> str:
    """A raw authorization header's entire line is credential-bearing."""
    scheme = re.match(r"([A-Za-z][A-Za-z0-9_-]*[ \t]+)", match.group(2))
    return match.group(1) + (scheme.group(0) if scheme else "") + "[REDACTED]"


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
    message = _ASSIGNED_SECRET.sub(_redact_assignment, message)
    message = _AUTH_HEADER.sub(_redact_auth_header, message)
    message = re.sub(r"(?i)(\bBearer[ \t]+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", message)
    message = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+)", "[REDACTED]", message)
    message = re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "[REDACTED]", message)
    return re.sub(r"(?i)(https?://)[^/?#@\s]+@", r"\1[REDACTED]@", message)
