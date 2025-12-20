"""Extract ad timestamps from podcast episode descriptions.

Handles various timestamp formats:
- (00:00 - 00:30) Sponsors
- [0:00-0:30] Ad Read
- 00:00 Sponsor: Company Name
- 1:02:03 - Ad Break (hours format)
- 0:30–1:10 (en-dash)
- 00.30 (dots instead of colons)
- 1h02m (hour/minute notation)
"""

import re
from typing import Optional

from .models import AdTimestamp


def extract_ad_timestamps(description: str) -> list[AdTimestamp]:
    """Extract ad timestamps from episode description.

    Args:
        description: Episode description text (plain text or HTML)

    Returns:
        List of AdTimestamp objects sorted by start time
    """
    if not description:
        return []

    # Check for "no ads" disclaimers first
    if _has_no_ads_disclaimer(description):
        return []

    timestamps: list[AdTimestamp] = []

    # Try all patterns
    timestamps.extend(_parse_time_range_pattern(description))
    timestamps.extend(_parse_chapter_markers(description))
    timestamps.extend(_parse_single_timestamp_ads(description))

    # Merge overlapping spans
    timestamps = _merge_overlapping_timestamps(timestamps)

    # Clamp to reasonable values (0 to 10 hours)
    timestamps = [t for t in timestamps if 0 <= t.start < 36000 and t.end <= 36000]

    return sorted(timestamps, key=lambda t: t.start)


def _parse_timestamp(time_str: str) -> Optional[float]:
    """Parse various timestamp formats to seconds.

    Formats:
    - HH:MM:SS, MM:SS, M:SS
    - HH.MM.SS (dots)
    - 1h02m30s, 1h2m, 2m30s
    """
    time_str = time_str.strip()

    # Hour/minute notation: 1h02m30s
    hour_min_pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.fullmatch(hour_min_pattern, time_str, re.IGNORECASE)
    if match and any(match.groups()):
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return h * 3600 + m * 60 + s

    # Colon or dot notation: HH:MM:SS or HH.MM.SS
    parts = re.split(r"[:.]", time_str)
    if len(parts) == 2:  # MM:SS
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    elif len(parts) == 3:  # HH:MM:SS
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            return None

    return None


def _parse_time_range_pattern(description: str) -> list[AdTimestamp]:
    """Pattern: (MM:SS - MM:SS) or [HH:MM:SS-HH:MM:SS]

    Handles: -, –, —, to, through
    """
    # Match various separators: dash, en-dash, em-dash, "to"
    pattern = (
        r"[(\[]?\s*"
        r"(\d{1,2}[:.]?\d{2}(?:[:.]?\d{2})?)"  # Start time
        r"\s*(?:-|–|—|to|through)\s*"  # Separator
        r"(\d{1,2}[:.]?\d{2}(?:[:.]?\d{2})?)"  # End time
        r"\s*[)\]]?"
        r"\s*[:\-]?\s*"
        r"(.{0,100})"  # Label (max 100 chars)
    )

    timestamps = []
    for match in re.finditer(pattern, description, re.IGNORECASE):
        start_str, end_str, label = match.groups()

        start = _parse_timestamp(start_str)
        end = _parse_timestamp(end_str)

        if start is None or end is None or start >= end:
            continue

        # Check for ad keywords in label
        has_ad_keyword = bool(
            re.search(
                r"\b(ad|ads|sponsor|sponsors|commercial|promo|break|advertisement)\b",
                label,
                re.IGNORECASE,
            )
        )

        confidence = 0.95 if has_ad_keyword else 0.6

        timestamps.append(
            AdTimestamp(
                start=start,
                end=end,
                label=label.strip() or None,
                confidence=confidence,
                extraction_method="time_range",
            )
        )

    return timestamps


def _parse_chapter_markers(description: str) -> list[AdTimestamp]:
    """Pattern: MM:SS - Chapter Name (check for ad keywords)"""
    pattern = (
        r"(\d{1,2}[:.]?\d{2}(?:[:.]?\d{2})?)"  # Timestamp
        r"\s*[:\-–—]\s*"  # Separator
        r"(.{1,100}?)"  # Label
        r"(?=\n|\d{1,2}[:.]?\d{2}|$)"  # Lookahead for end
    )

    timestamps = []
    for match in re.finditer(pattern, description, re.IGNORECASE):
        time_str, label = match.groups()

        start = _parse_timestamp(time_str)
        if start is None:
            continue

        # Check for ad keywords with word boundaries
        has_ad_keyword = bool(
            re.search(
                r"\b(ad|ads|sponsor|sponsors|commercial|promo|break|brought to you|advertisement)\b",
                label,
                re.IGNORECASE,
            )
        )

        if not has_ad_keyword:
            continue

        # Estimate 30-second duration for chapter markers
        end = start + 30

        timestamps.append(
            AdTimestamp(
                start=start,
                end=end,
                label=label.strip(),
                confidence=0.8,
                extraction_method="chapter_marker",
            )
        )

    return timestamps


def _parse_single_timestamp_ads(description: str) -> list[AdTimestamp]:
    """Pattern: MM:SS Sponsor/Ad (no end time, estimate 30s duration)"""
    # Match timestamp followed by ad keyword
    pattern = (
        r"(\d{1,2}[:.]?\d{2}(?:[:.]?\d{2})?)"  # Timestamp
        r"\s+"
        r"(?:ad|sponsor|commercial)(?:\b|:)"  # Ad keyword
    )

    timestamps = []
    for match in re.finditer(pattern, description, re.IGNORECASE):
        time_str = match.group(1)
        start = _parse_timestamp(time_str)

        if start is None:
            continue

        timestamps.append(
            AdTimestamp(
                start=start,
                end=start + 30,  # Default 30s duration
                label="Ad",
                confidence=0.7,
                extraction_method="single_timestamp",
            )
        )

    return timestamps


def _has_no_ads_disclaimer(description: str) -> bool:
    """Check if description explicitly states no ads."""
    patterns = [
        r"\bad[ -]?free\b",
        r"\bno ads\b",
        r"\bno commercials\b",
        r"\bwithout ads\b",
        r"\bcommercial[ -]?free\b",
    ]

    for pattern in patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return True

    return False


def _merge_overlapping_timestamps(timestamps: list[AdTimestamp]) -> list[AdTimestamp]:
    """Merge overlapping or adjacent timestamp spans."""
    if not timestamps:
        return []

    # Sort by start time
    sorted_ts = sorted(timestamps, key=lambda t: t.start)

    merged = [sorted_ts[0]]

    for current in sorted_ts[1:]:
        previous = merged[-1]

        # Merge if overlapping or within 5 seconds
        if current.start <= previous.end + 5:
            merged[-1] = AdTimestamp(
                start=previous.start,
                end=max(previous.end, current.end),
                label=previous.label,  # Keep first label
                confidence=max(previous.confidence, current.confidence),
                extraction_method=previous.extraction_method,
            )
        else:
            merged.append(current)

    return merged
