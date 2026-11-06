"""Audio enclosure selection and queue admission predicates."""

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_AUDIO_EXTENSIONS = {".mp3", ".mp2", ".mpga", ".m4a", ".m4b", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac", ".aif", ".aiff", ".weba", ".wma"}


def is_usable_audio_url(value: str | None) -> bool:
    """Accept HTTP(S) URLs with a host, without imposing a host trust policy."""
    if not isinstance(value, str) or not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        # Accessing port validates malformed/out-of-range ports too.
        parsed.port
        return parsed.scheme.lower() in {"http", "https"} and bool(
            host and host.rstrip(".") and not any(char in host for char in "\\/?#@")
        )
    except ValueError:
        return False


def select_audio_enclosure(enclosures: Iterable[Mapping[str, Any]]) -> str | None:
    """Prefer audio MIME, then absent/generic-binary MIME with an audio extension."""
    fallback = None
    for enclosure in enclosures:
        url = enclosure.get("href")
        if not is_usable_audio_url(url):
            continue
        mime = enclosure.get("type", "")
        mime = mime.split(";", 1)[0].strip().lower() if isinstance(mime, str) else ""
        if mime.startswith("audio/") and len(mime) > len("audio/"):
            return url
        if mime in {"", "application/octet-stream", "binary/octet-stream"} and fallback is None:
            path = urlsplit(url).path.lower()
            if any(path.endswith(extension) for extension in _AUDIO_EXTENSIONS):
                fallback = url
    return fallback
