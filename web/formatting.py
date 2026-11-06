"""Consistent user-facing duration and cost formats."""

import math
from decimal import Decimal, ROUND_HALF_UP, localcontext


def format_duration(value: int | float | Decimal | None) -> str:
    """Format elapsed seconds as HH:MM:SS, or an em dash if unavailable."""
    if value is None or not math.isfinite(value) or value < 0:
        return "—"
    seconds = int(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_cost(value: int | float | Decimal | None) -> str:
    """Show USD costs to four decimal places consistently across pages."""
    if value is None or not math.isfinite(value) or value < 0:
        return "—"
    amount = Decimal(str(value))
    with localcontext() as context:
        context.prec = max(28, len(amount.as_tuple().digits) + max(amount.adjusted(), 0) + 6)
        rounded = amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return f"${rounded:,.4f}"
