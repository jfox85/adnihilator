"""Bounded Gemini operation and decision retries authorized by the owner."""

import concurrent.futures
import re
import time
from collections.abc import Callable
from threading import Event
from typing import Any, TypeVar

from .gemini_common import extract_usage
from .provider_errors import InvalidProviderDecision, ProviderOutageError, classify_provider_error

T = TypeVar("T")
MAX_ATTEMPTS = 3


class InvalidGeminiDecision(InvalidProviderDecision):
    """A completed response has malformed JSON, schema or detection bounds."""


def attempt_limit(requested: int = MAX_ATTEMPTS) -> int:
    """Keep the existing lower test/caller limit without exceeding owner policy."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("Gemini attempt limit must be a positive integer")
    return min(requested, MAX_ATTEMPTS)


def check_cancel(cancel_event: Event | None) -> None:
    """Never issue another operation after a sibling's permanent failure."""
    if cancel_event is not None and cancel_event.is_set():
        raise concurrent.futures.CancelledError()


def wait_before_retry(attempt: int, cancel_event: Event | None) -> None:
    """Back off 30 then 60 seconds, checking global cancellation on both sides."""
    check_cancel(cancel_event)
    time.sleep(min(30 * (2 ** attempt), 60))
    check_cancel(cancel_event)


def retry_operation(operation: Callable[[], T], *, max_attempts: int = MAX_ATTEMPTS,
                    cancel_event: Event | None = None) -> T:
    """Retry one upload/status operation within its finite transport budget."""
    limit = attempt_limit(max_attempts)
    for attempt in range(limit):
        check_cancel(cancel_event)
        try:
            return operation()
        except Exception as error:
            if not classify_provider_error(error).retryable or attempt + 1 == limit:
                raise
            wait_before_retry(attempt, cancel_event)
    raise AssertionError("unreachable retry state")


def retry_decision(operation: Callable[[bool], Any], parse: Callable[[Any], T], *,
                   max_attempts: int = MAX_ATTEMPTS, cancel_event: Event | None = None,
                   usage_totals: dict[str, int] | None = None) -> T:
    """Share three model attempts across transport retries and one JSON re-ask."""
    limit = attempt_limit(max_attempts)
    reasked = False
    for attempt in range(limit):
        check_cancel(cancel_event)
        try:
            response = operation(reasked)
        except Exception as error:
            failure = classify_provider_error(error)
            detail = str(getattr(error, "message", error)).lower()
            missing_file = re.search(r"\bfile\s+[\"']?files/", detail) and any(
                word in detail for word in ("not found", "not_found", "expired")
            )
            if failure.status_code == 404 and not missing_file:
                failure = failure.model_copy(update={"category": "model", "global_failure": True, "retryable": False})
                raise ProviderOutageError("gemini", failure) from None
            if not failure.retryable or attempt + 1 == limit:
                raise
            wait_before_retry(attempt, cancel_event)
            continue
        if usage_totals is not None:
            for key, value in extract_usage(response).items():
                usage_totals[key] = usage_totals.get(key, 0) + value
        try:
            return parse(response)
        except InvalidGeminiDecision:
            if reasked or attempt + 1 == limit:
                raise
            reasked = True
    raise AssertionError("unreachable decision retry state")
