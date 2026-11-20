"""Remap chapter timestamps after ad segments are removed from audio.

Two places carry chapter times that become wrong once ads are cut:
- ID3 chapter frames embedded in the MP3 (podcast app "bookmarks")
- The chapter outline in the episode description/show notes

Both are remapped from original-audio time to processed-audio time using the
list of kept segments produced by the splicer.
"""

import re

KeepSegments = list[tuple[float, float]]

# A chapter whose kept audio is shorter than this was removed with the ads.
MIN_KEPT_CHAPTER_SECONDS = 1.0


def remap_time(t: float, keep: KeepSegments) -> float:
    """Map an original-audio time to processed-audio time.

    Times inside a removed span map to the point where that span was cut.
    """
    out = 0.0
    for start, end in keep:
        if t < start:
            return out
        if t <= end:
            return out + (t - start)
        out += end - start
    return out


def kept_seconds(start: float, end: float, keep: KeepSegments) -> float:
    """Seconds of [start, end) that survive the cut."""
    return sum(max(0.0, min(end, k_end) - max(start, k_start)) for k_start, k_end in keep)


def remap_chapters(
    chapters: list[tuple[float, float, str]], keep: KeepSegments
) -> list[tuple[float, float, str]]:
    """Remap (start, end, title) chapters; drop chapters that were cut out."""
    if not keep:
        return []
    new_duration = sum(end - start for start, end in keep)
    survivors = [
        (remap_time(start, keep), title)
        for start, end, title in sorted(chapters)
        if kept_seconds(start, end, keep) >= MIN_KEPT_CHAPTER_SECONDS
    ]
    result = []
    for i, (start, title) in enumerate(survivors):
        end = survivors[i + 1][0] if i + 1 < len(survivors) else new_duration
        if end > start:
            result.append((start, end, title))
    return result


# Lines are separated by <br>, newlines, or closing block tags.
_LINE_SPLIT = re.compile(r"(<br\s*/?>|\n|</p>|</li>)", re.IGNORECASE)
_SOFT_SEPARATOR = re.compile(r"<br\s*/?>|\n", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
# An outline line: optional leading tags, optional bracket, then a time.
_OUTLINE_LINE = re.compile(
    r"^(?P<prefix>\s*(?:<[^>]+>\s*)*[(\[]?\s*)(?P<time>\d{1,2}:\d{2}(?::\d{2})?)(?=[)\]\s])"
)


def _parse_time(text: str) -> int:
    total = 0
    for part in text.split(":"):
        total = total * 60 + int(part)
    return total


def _format_like(seconds: float, original: str) -> str:
    """Format seconds in the same style as the original timestamp."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = original.split(":")
    if len(parts) == 3 and len(parts[0]) == 2:
        return f"{h:02d}:{m:02d}:{s:02d}"  # zero-padded HH:MM:SS style
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    width = len(parts[1]) if len(parts) == 3 else len(parts[0])
    return f"{m:0{max(width, 2)}d}:{s:02d}"


def remap_description_timestamps(description: str | None, keep: KeepSegments) -> str | None:
    """Rewrite a description's chapter outline to match processed audio.

    Only a coherent outline (two or more line-leading times, strictly
    increasing) is rewritten. Chapters removed with the ads are dropped and
    the remaining times are shifted. Everything else is left byte-for-byte.
    """
    if not description or not keep:
        return description

    parts = _LINE_SPLIT.split(description)  # text, sep, text, sep, ...
    entries = []  # (part index, match, seconds)
    for i in range(0, len(parts), 2):
        match = _OUTLINE_LINE.match(parts[i])
        if match:
            entries.append((i, match, _parse_time(match.group("time"))))

    times = [seconds for _, _, seconds in entries]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        return description

    original_end = keep[-1][1]
    changed = False
    drop: set[int] = set()
    for n, (i, match, seconds) in enumerate(entries):
        chapter_end = times[n + 1] if n + 1 < len(times) else max(original_end, seconds)
        if chapter_end > seconds and kept_seconds(seconds, chapter_end, keep) < MIN_KEPT_CHAPTER_SECONDS:
            drop.add(i)
            changed = True
            continue
        new_seconds = remap_time(seconds, keep)
        if int(round(new_seconds)) != seconds:
            token = _format_like(new_seconds, match.group("time"))
            parts[i] = parts[i][: match.start("time")] + token + parts[i][match.end("time"):]
            changed = True

    if not changed:
        return description

    out = []
    i = 0
    while i < len(parts):
        if i in drop:
            # Drop the line's text and the soft line breaks after it, but keep
            # its markup (e.g. an opening <ul><li>) and block closers such as
            # </p> so the HTML stays balanced.
            tags = "".join(_TAG.findall(parts[i]))
            i += 1
            while i < len(parts) and (parts[i] == "" or _SOFT_SEPARATOR.fullmatch(parts[i])):
                i += 1
            if tags:
                out.append(tags)
            continue
        out.append(parts[i])
        i += 1
    return "".join(out)
