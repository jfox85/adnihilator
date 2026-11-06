"""Safe numeric views of historical data, without changing stored values."""

import math
from typing import Any


def finite_nonnegative_number(value: Any, maximum: float | None = None) -> int | float | None:
    """Return an available valid number, preserving zero and hiding invalid data."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        return None
    return value if isinstance(value, int) else number
