"""Shared safe presentation of processing failures."""

import re
from typing import Any

from adnihilator.redaction import redact_secrets

_HTTP_STATUS_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r'^\s*(401|402|403|404|503)\b',
    r'\b(?:http(?:error|statuserror)?|status(?:[_\s-]?code)?|error\s+code|failed)\s*[:=]?\s*[\'\"]?(401|402|403|404|503)\b',
    r'\b(?:gemini|openai|anthropic)\s+(?:error\s*[:=]?\s*)?(401|402|403|404|503)\b',
    r'[\'\"](?:code|status_code)[\'\"]\s*:\s*(401|402|403|404|503)\b',
))
_PROVIDER_CONTEXT = re.compile(r'\b(?:gemini|openai|anthropic|genai|llm|ai provider)\b', re.IGNORECASE)
_DOWNLOAD_CONTEXT = re.compile(r'\b(?:audio|rss|feed|curl|download(?:ing)?)\b', re.IGNORECASE)


def episode_error(message: str | None) -> dict[str, str]:
    """Give a kind summary, next action and sanitized optional diagnostics."""
    details = redact_secrets(message) if message else ''
    lower = details.lower()
    provider = bool(_PROVIDER_CONTEXT.search(lower))
    status = next((match.group(1) for pattern in _HTTP_STATUS_PATTERNS if (match := pattern.search(lower))), None)
    # Audio/feed hosts can use the same HTTP codes without implicating an AI
    # account. Parse offsets and episode numbers are not HTTP statuses at all.
    if _DOWNLOAD_CONTEXT.search(lower) and not provider:
        status = None
    if status == '402' or ('credit' in lower and any(word in lower for word in ('depleted', 'exhausted', 'insufficient', 'used up'))):
        summary = 'AI provider credits are used up'
        action = 'Ask the administrator to add credits, then re-queue the episode.'
    elif status in ('401', '403') or any(term in lower for term in (
        'invalid api key', 'invalid_api_key', 'api_key_invalid', 'api key not valid', 'unauthenticated',
    )):
        summary = 'AI provider credentials need attention'
        action = 'Ask the administrator to check the provider credentials before retrying.'
    elif status == '404' and 'model' in lower:
        summary = 'The configured AI model is unavailable'
        action = 'Ask the administrator to check the model setting before retrying.'
    elif status == '503' or (provider and 'service unavailable' in lower):
        summary = 'The AI provider is temporarily unavailable'
        action = 'Try again later. Contact the administrator if it continues.'
    else:
        summary = 'Processing failed'
        action = 'Retry the episode, or contact the administrator if it keeps failing.'
    return {'summary': summary, 'action': action, 'details': details}


def register_error_filters(templates: Any) -> None:
    """Keep both template registries consistent without coupling their routes."""
    templates.env.filters['episode_error'] = episode_error
