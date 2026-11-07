"""Shared UTC-representable transcript date bounds for REST, UI and MCP."""

from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator


def utc_naive(value: datetime) -> datetime:
    """Normalize a valid bound, rejecting UTC underflow/overflow safely."""
    try:
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    except (OverflowError, ValueError):
        raise ValueError("Date is outside the supported UTC range") from None


def _bounded_date(value: datetime) -> datetime:
    """Validate representability while retaining the caller's date/time value."""
    utc_naive(value)
    return value


BoundedDate = Annotated[datetime, AfterValidator(_bounded_date)]
