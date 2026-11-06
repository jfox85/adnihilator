"""Shared safe presentation of processing failures."""

import re
from typing import Any

from adnihilator.redaction import redact_secrets


def episode_error(message: str | None) -> dict[str, str]:
    """Give a kind summary, next action and sanitized optional diagnostics."""
    details = redact_secrets(message) if message else ''
    lower = details.lower()
    if re.search(r'\b402\b', lower) or ('credit' in lower and any(word in lower for word in ('depleted', 'exhausted', 'insufficient', 'used up'))):
        summary = 'AI provider credits are used up'
        action = 'Ask the administrator to add credits, then re-queue the episode.'
    elif re.search(r'\b(?:401|403)\b', lower) or any(term in lower for term in (
        'invalid api key', 'invalid_api_key', 'api_key_invalid', 'api key not valid', 'unauthenticated',
    )):
        summary = 'AI provider credentials need attention'
        action = 'Ask the administrator to check the provider credentials before retrying.'
    elif re.search(r'\b404\b', lower) and 'model' in lower:
        summary = 'The configured AI model is unavailable'
        action = 'Ask the administrator to check the model setting before retrying.'
    elif re.search(r'\b503\b', lower) or 'service unavailable' in lower:
        summary = 'The AI provider is temporarily unavailable'
        action = 'Try again later. Contact the administrator if it continues.'
    else:
        summary = 'Processing failed'
        action = 'Retry the episode, or contact the administrator if it keeps failing.'
    return {'summary': summary, 'action': action, 'details': details}


def register_error_filters(templates: Any) -> None:
    """Keep both template registries consistent without coupling their routes."""
    templates.env.filters['episode_error'] = episode_error
