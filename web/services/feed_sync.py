"""Feed synchronization service."""

from collections.abc import Mapping
from datetime import datetime
from email.message import Message
from hashlib import sha256
import re
from time import mktime
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request
from xml.sax import SAXParseException
from calendar import timegm

import feedparser
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from adnihilator.redaction import redact_secrets

from ..models import Episode, EpisodeStatus, Podcast
from .audio import is_usable_audio_url, select_audio_enclosure
from .feed_document import FeedDocumentCapture


class FeedSyncError(RuntimeError):
    """A failed feed update with safe diagnostics and a rolled-back session."""


class _HTTPOnlyRedirect(HTTPRedirectHandler):
    """Allow private HTTP(S) redirects while refusing every other scheme."""

    # feedparser adds its own redirect/auth handler at order 490. Our guard
    # must run first; the default urllib redirect order (500) is too late.
    handler_order = 400

    def http_error_302(
        self, req: Request, fp: object, code: int, msg: str, headers: Message,
    ) -> object:
        """Reject unsupported targets before urllib's own redirect handling."""
        target = headers.get("location") or headers.get("uri")
        if target and not is_usable_audio_url(urljoin(req.full_url, target)):
            raise FeedSyncError("Feed redirect must use an HTTP(S) URL with a host")
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302

    def redirect_request(
        self, req: Request, fp: object, code: int, msg: str,
        headers: Message, newurl: str,
    ) -> Request | None:
        """Validate the resolved redirect destination before opening it."""
        if not is_usable_audio_url(newurl):
            raise FeedSyncError("Feed redirect must use an HTTP(S) URL with a host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def episode_identity(entry: dict, audio_url: str | None) -> str | None:
    """Prefer GUID, then entry link, then a query-preserving enclosure identity.

    The enclosure fallback hashes scheme/host/port/path and the full opaque
    query, excluding userinfo and fragment. Unknown query values may select
    different episodes, so they must not be discarded. Existing rows are never
    backfilled or rewritten; rotating unknown tokens may create new entries.
    """
    for field in ("id", "link"):
        value = entry.get(field) if isinstance(entry, Mapping) else getattr(entry, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if audio_url is None:
        return None
    parsed = urlsplit(audio_url)
    host = parsed.hostname.lower()
    host = f"[{host}]" if ":" in host else host
    port = parsed.port
    if port is not None and (parsed.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    canonical = urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", parsed.query, ""))
    return "enclosure-sha256:" + sha256(canonical.encode("utf-8")).hexdigest()


def _sync_failure(db: Session, podcast_id: str, error: Exception) -> FeedSyncError:
    """Persist a safe feed failure after rolling back its incomplete import."""
    db.rollback()
    message = f"Feed sync failed ({type(error).__name__}): {redact_secrets(str(error))}"
    try:
        podcast = db.get(Podcast, podcast_id)
        if podcast is not None:
            podcast.last_sync_error = message
            db.commit()
    except Exception:
        # A database outage may also prevent saving diagnostics. Preserve the
        # original failure and leave the caller's session usable regardless.
        db.rollback()
    return FeedSyncError(message)


def feed_error_message(diagnostic: str | None) -> str | None:
    """Present an actionable feed failure without exposing internal diagnostics.

    Read-time formatting also covers previously saved errors, leaving their
    safe diagnostic detail unchanged for administrative API clients.
    """
    if not diagnostic:
        return None
    reason = re.sub(r"^Feed sync failed(?: \([^)]*\))?:\s*", "", redact_secrets(diagnostic))
    http_status = re.match(r"Feed server returned HTTP (\d{3})\b", reason)
    if http_status:
        message = f"The feed server returned HTTP {http_status.group(1)}."
    elif reason.startswith("Could not connect to the feed server"):
        message = "Could not connect to the feed server."
    elif reason.startswith("Response is not a recognized RSS or Atom feed"):
        message = "The server response is not an RSS or Atom feed."
    elif reason.startswith("Response is not a complete"):
        message = "The feed response is incomplete or malformed."
    elif reason.startswith("Feed redirect must use"):
        message = "The feed redirected to an unsupported address. Only HTTP and HTTPS are allowed."
    elif reason.startswith("Feed URL must use"):
        message = "The feed address must use HTTP or HTTPS."
    else:
        message = "The feed could not be updated."
    return f"{message} Check the feed URL or choose Retry."


def parse_duration(duration_str: str | None) -> int | None:
    """Parse duration string to seconds.

    Supports formats like:
    - "3600" (seconds)
    - "01:00:00" (HH:MM:SS)
    - "60:00" (MM:SS)
    """
    if not duration_str:
        return None

    try:
        # Try parsing as integer seconds
        return int(duration_str)
    except ValueError:
        pass

    # Try parsing as HH:MM:SS or MM:SS
    parts = duration_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass

    return None


def sync_podcast(db: Session, podcast: Podcast) -> int:
    """Sync and commit one podcast, rolling back the session on failure.

    Retry once only when another sync inserted the same episode GUID after
    this sync's snapshot. Other database and feed errors are not retried.

    Args:
        db: Database session
        podcast: Podcast to sync

    Returns:
        Number of new episodes added
    """
    podcast_id = podcast.id
    try:
        return _sync_podcast_entries(db, podcast)
    except Exception as error:
        db.rollback()
        guid_conflict = (
            isinstance(error, IntegrityError)
            and "UNIQUE constraint failed: episodes.podcast_id, episodes.guid" in str(error.orig)
        )
        if not guid_conflict:
            raise _sync_failure(db, podcast_id, error) from None

    # Another process may have committed the same GUID after our snapshot.
    # Reload once after rollback; unrelated constraints must never be retried.
    try:
        return _sync_podcast_entries(db, podcast)
    except Exception as error:
        raise _sync_failure(db, podcast_id, error) from None


def _sync_podcast_entries(db: Session, podcast: Podcast) -> int:
    """Apply one feed, leaving rollback on failure to the public sync boundary."""
    if not is_usable_audio_url(podcast.source_rss_url):
        raise FeedSyncError("Feed URL must use HTTP(S) with a host")
    document = FeedDocumentCapture()
    feed = feedparser.parse(podcast.source_rss_url, handlers=[_HTTPOnlyRedirect(), document])
    response_status = feed.get("status")
    if isinstance(response_status, int) and response_status >= 400:
        raise FeedSyncError(f"Feed server returned HTTP {response_status}")
    parse_error = feed.get("bozo_exception")
    if isinstance(parse_error, FeedSyncError):
        raise parse_error
    if isinstance(parse_error, HTTPError):
        raise FeedSyncError(f"Feed server returned HTTP {parse_error.code}") from None
    if isinstance(parse_error, URLError):
        raise FeedSyncError("Could not connect to the feed server") from None
    if not feed.get("version"):
        raise FeedSyncError("Response is not a recognized RSS or Atom feed")

    # feedparser resolves page links but leaves RSS enclosure hrefs relative.
    # Its response href contains the final URL after redirects.
    base_url = feed.get("href")
    if not isinstance(base_url, str) or not is_usable_audio_url(base_url):
        base_url = document.url or podcast.source_rss_url
    source_enclosures = None
    if document.url:
        try:
            source_enclosures = document.validate_structure(feed.entries)
        except Exception:
            raise FeedSyncError("Response is not a complete RSS or Atom feed") from None
        if source_enclosures is not None and len(source_enclosures) != len(feed.entries):
            raise FeedSyncError("Atom enclosure metadata could not be read reliably")
    elif isinstance(parse_error, SAXParseException):
        # No captured document can establish that an incomplete parse recovered.
        raise FeedSyncError("Response is not a complete RSS or Atom feed") from None

    # Update podcast metadata
    if hasattr(feed.feed, "title"):
        podcast.title = feed.feed.title
    if hasattr(feed.feed, "summary"):
        podcast.description = feed.feed.summary
    elif hasattr(feed.feed, "subtitle"):
        podcast.description = feed.feed.subtitle

    # Get artwork URL - prefer itunes:image (larger) over regular image (often favicon)
    # feedparser stores itunes:image as href in a dict with 'href' key
    if hasattr(feed.feed, "itunes_image"):
        # itunes_image can be a dict with 'href' or the href directly
        itunes_img = feed.feed.itunes_image
        if isinstance(itunes_img, dict) and "href" in itunes_img:
            podcast.artwork_url = itunes_img["href"]
        elif isinstance(itunes_img, str):
            podcast.artwork_url = itunes_img
    elif hasattr(feed.feed, "image") and hasattr(feed.feed.image, "href"):
        # Fall back to regular image (may be small favicon)
        podcast.artwork_url = feed.feed.image.href

    # Get existing episode GUIDs
    existing_guids = set(
        guid for (guid,) in db.query(Episode.guid).filter_by(podcast_id=podcast.id).all()
    )

    added = 0
    for index, entry in enumerate(feed.entries):
        enclosures = [dict(enclosure) for enclosure in getattr(entry, "enclosures", [])]
        if source_enclosures is not None:
            # The observer reproduces and verifies the exact original entries;
            # its indices identify parser events, not guessed link positions.
            source_entry = source_enclosures[index]
            if not source_entry.actual_namespace:
                continue
            enclosures = []
            for original in source_entry.enclosures:
                raw_href = original.href
                try:
                    raw_valid = (
                        all(
                            isinstance(href, str) and bool(href)
                            and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in href)
                            and is_usable_audio_url(urljoin(original.base_url, href))
                            and (not urlsplit(href).scheme or is_usable_audio_url(href))
                            for href in (raw_href, original.guard_href)
                        )
                    )
                except ValueError:
                    raw_valid = False
                href = original.reference.get("href") if raw_valid else None
                if isinstance(href, str) and href:
                    try:
                        if not urlsplit(href).scheme:
                            href = urljoin(original.base_url, href)
                    except ValueError:
                        href = None
                candidate = {"href": href}
                if "type" in original.attributes:
                    candidate["type"] = original.attributes["type"]
                enclosures.append(candidate)
        for enclosure in enclosures:
            href = enclosure.get("href")
            if isinstance(href, str):
                # Resolve only actual relative references. An explicitly
                # malformed absolute URL (e.g. http:///audio.mp3) must not be
                # repaired into a usable URL on this feed's host.
                try:
                    relative = bool(href) and not urlsplit(href).scheme and not any(
                        char.isspace() or ord(char) < 32 or ord(char) == 127 for char in href
                    )
                except ValueError:
                    relative = False
                enclosure["href"] = urljoin(base_url, href) if relative else href
        audio_url = select_audio_enclosure(enclosures)
        guid = episode_identity(entry, audio_url)
        if not guid or guid in existing_guids:
            continue

        # Parse published date
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published_at = datetime.utcfromtimestamp(timegm(entry.published_parsed))
            except (ValueError, OverflowError):
                pass

        # Get episode description
        description = getattr(entry, "summary", None) or getattr(entry, "description", None)

        # Get episode duration (itunes:duration)
        duration = parse_duration(getattr(entry, "itunes_duration", None))

        # Get episode source URL (web page link for transcripts)
        source_url = getattr(entry, "link", None)

        episode = Episode(
            podcast_id=podcast.id,
            guid=guid,
            title=getattr(entry, "title", None),
            description=description,
            duration=duration,
            original_audio_url=audio_url,
            source_url=source_url,
            published_at=published_at,
            status=EpisodeStatus.PENDING.value if podcast.auto_process and audio_url else EpisodeStatus.SKIPPED.value,
        )
        db.add(episode)
        existing_guids.add(guid)
        added += 1

    podcast.last_sync_error = None
    # A concurrent failed refresh may have saved an error after this instance
    # loaded NULL. Include NULL in the same successful UPDATE even when the
    # in-memory value appears unchanged, so the last success really clears it.
    flag_modified(podcast, "last_sync_error")
    podcast.last_synced_at = datetime.utcnow()
    db.commit()
    return added


def sync_all_podcasts(db: Session) -> dict:
    """Sync all podcasts.

    Returns:
        Dict mapping podcast IDs to number of new episodes
    """
    results = {}
    podcast_ids = [podcast_id for (podcast_id,) in db.query(Podcast.id).all()]

    for podcast_id in podcast_ids:
        try:
            podcast = db.get(Podcast, podcast_id)
            if podcast is None:
                continue
            added = sync_podcast(db, podcast)
            results[podcast_id] = {"added": added, "error": None}
        except Exception as e:
            db.rollback()
            results[podcast_id] = {"added": 0, "error": redact_secrets(str(e))}

    return results
