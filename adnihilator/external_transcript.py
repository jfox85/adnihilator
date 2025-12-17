"""External transcript fetching for podcasts that provide their own transcripts.

Currently supports:
- Substack podcasts (JSON transcripts via CloudFront signed URLs)
- Lex Fridman podcasts (HTML transcripts on lexfridman.com)
"""

import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .models import TranscriptSegment, WordTimestamp


# Regex to find Substack transcription.json URLs in page HTML
SUBSTACK_TRANSCRIPT_PATTERN = re.compile(
    r'https://substackcdn\.com/[^"]*transcription\.json[^"]*'
)

# Timeout for HTTP requests (seconds)
FETCH_TIMEOUT = 10.0


def parse_lexfridman_transcript(html: str) -> list[TranscriptSegment] | None:
    """Parse Lex Fridman transcript HTML into TranscriptSegment list.

    Lex Fridman transcripts have the structure:
    - Each speaker turn is: "Speaker Name [(HH:MM:SS)](youtube-link) Text..."
    - Timestamps are in YouTube links with t=SECONDS parameter
    - Segment-level timestamps only (no word-level)

    Args:
        html: Raw HTML content of the transcript page

    Returns:
        List of TranscriptSegment objects, or None if parsing fails
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Find all timestamp links (YouTube links with t= parameter)
        timestamp_links = soup.find_all('a', href=lambda x: x and 'youtube.com' in x and 't=' in x)

        if not timestamp_links:
            return None

        segments = []

        for link in timestamp_links:
            # Extract seconds from YouTube link
            href = link.get('href', '')
            match = re.search(r't=(\d+)', href)
            if not match:
                continue

            start = float(match.group(1))

            # Get the parent element which contains the full text
            # Structure is: <span><a>timestamp</a></span> is inside a line with speaker + text
            parent = link.parent
            if parent and parent.parent:
                # The grandparent typically contains the full speaker line
                full_text = parent.parent.get_text().strip()
            else:
                continue

            # Remove the timestamp from the text
            # Pattern: "Speaker Name [(HH:MM:SS)] Text..." or "[HH:MM:SS]"
            # Handle both brackets and parentheses
            text = re.sub(r'\s*[\[\(]\d{2}:\d{2}:\d{2}[\]\)]\s*', ' ', full_text).strip()

            # Skip empty segments
            if not text:
                continue

            # We don't have end times, so we'll set them later based on next segment
            segments.append(TranscriptSegment(
                index=len(segments),
                start=start,
                end=start,  # Will be updated below
                text=text,
                words=None,  # No word-level timestamps
            ))

        # Set end times based on next segment's start
        for i in range(len(segments) - 1):
            segments[i].end = segments[i + 1].start

        # For the last segment, estimate 60 seconds to avoid truncating content
        # (Lex Fridman episodes often end with long closing remarks)
        if segments:
            segments[-1].end = segments[-1].start + 60.0

        return segments if segments else None

    except Exception:
        # Any parsing error means we can't use this transcript
        return None


def extract_transcript_url(html: str) -> str | None:
    """Extract Substack transcription.json URL from page HTML.

    Args:
        html: Raw HTML content of the episode page

    Returns:
        The transcription.json URL if found, None otherwise
    """
    match = SUBSTACK_TRANSCRIPT_PATTERN.search(html)
    if match:
        # URL may have HTML entities, decode them
        url = match.group(0)
        url = url.replace("&amp;", "&")
        # Strip trailing backslash (from JSON-escaped strings in page source)
        url = url.rstrip("\\")
        return url
    return None


def parse_substack_transcript(data: list[dict[str, Any]]) -> list[TranscriptSegment]:
    """Parse Substack transcript JSON into TranscriptSegment list.

    Args:
        data: Raw JSON data from Substack transcription.json

    Returns:
        List of TranscriptSegment objects
    """
    segments = []

    for idx, item in enumerate(data):
        # Extract basic segment info
        start = item.get("start", 0.0)
        end = item.get("end", 0.0)
        text = item.get("text", "").strip()

        # Skip empty segments
        if not text:
            continue

        # Parse word-level timestamps if available
        words = None
        if "words" in item and item["words"]:
            words = []
            for w in item["words"]:
                # Skip words missing timestamps (Substack sometimes omits them)
                if "start" not in w or "end" not in w:
                    continue
                words.append(WordTimestamp(
                    word=w.get("word", ""),
                    start=w["start"],
                    end=w["end"],
                    probability=w.get("score", 1.0),  # Substack uses "score"
                ))

            # If all words were filtered out, set to None
            if not words:
                words = None

        segments.append(TranscriptSegment(
            index=len(segments),  # Sequential index
            start=start,
            end=end,
            text=text,
            words=words,
        ))

    return segments


def fetch_lexfridman_transcript(episode_page_url: str) -> list[TranscriptSegment] | None:
    """Fetch transcript from a Lex Fridman episode page.

    Lex Fridman provides transcripts on lexfridman.com with the format:
    Episode URL: https://lexfridman.com/GUEST-NAME
    Transcript URL: https://lexfridman.com/GUEST-NAME-transcript

    Args:
        episode_page_url: URL to the episode's web page (from RSS <link>)

    Returns:
        List of TranscriptSegment if successful, None otherwise
    """
    if not episode_page_url or 'lexfridman.com' not in episode_page_url:
        return None

    try:
        # Build transcript URL from episode URL
        # Episode: https://lexfridman.com/irving-finkel
        # Transcript: https://lexfridman.com/irving-finkel-transcript
        base_url = episode_page_url.rstrip('/')
        if base_url.endswith('-transcript'):
            transcript_url = base_url
        else:
            transcript_url = f"{base_url}-transcript"

        # Fetch the transcript page
        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            response = client.get(transcript_url)
            response.raise_for_status()
            html = response.text

        # Parse the HTML transcript
        segments = parse_lexfridman_transcript(html)

        if not segments:
            return None

        return segments

    except (httpx.HTTPError, ValueError, KeyError):
        # Any error means we fall back to Whisper
        return None


def fetch_substack_transcript(episode_page_url: str) -> list[TranscriptSegment] | None:
    """Fetch transcript from a Substack episode page.

    Attempts to extract and fetch the transcription.json from the episode's
    web page. Returns None if transcript is not available, allowing caller
    to fall back to Whisper transcription.

    Args:
        episode_page_url: URL to the episode's web page (from RSS <link>)

    Returns:
        List of TranscriptSegment if successful, None otherwise
    """
    if not episode_page_url:
        return None

    try:
        # Fetch the episode page
        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            response = client.get(episode_page_url)
            response.raise_for_status()
            html = response.text

        # Extract transcript URL
        transcript_url = extract_transcript_url(html)
        if not transcript_url:
            return None

        # Fetch the transcript JSON
        with httpx.Client(timeout=FETCH_TIMEOUT) as client:
            response = client.get(transcript_url)
            response.raise_for_status()
            data = response.json()

        # Parse into our format
        if not isinstance(data, list):
            return None

        segments = parse_substack_transcript(data)

        # Validate we got something useful
        if not segments:
            return None

        return segments

    except (httpx.HTTPError, ValueError, KeyError) as e:
        # Any error means we fall back to Whisper
        # Could log here for debugging
        return None


def fetch_external_transcript(episode_page_url: str) -> list[TranscriptSegment] | None:
    """Fetch external transcript from any supported podcast platform.

    Currently supports:
    - Lex Fridman podcasts (lexfridman.com)
    - Substack podcasts (substackcdn.com)

    Args:
        episode_page_url: URL to the episode's web page (from RSS <link>)

    Returns:
        List of TranscriptSegment if successful, None otherwise
    """
    if not episode_page_url:
        return None

    # Try Lex Fridman first if URL matches
    if 'lexfridman.com' in episode_page_url:
        segments = fetch_lexfridman_transcript(episode_page_url)
        if segments:
            return segments

    # Try Substack for all other URLs
    segments = fetch_substack_transcript(episode_page_url)
    if segments:
        return segments

    return None
