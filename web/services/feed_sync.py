"""Feed synchronization service."""

from datetime import datetime
from time import mktime

import feedparser
from sqlalchemy.orm import Session

from ..models import Episode, EpisodeStatus, Podcast


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
    """Sync episodes from the source RSS feed.

    Args:
        db: Database session
        podcast: Podcast to sync

    Returns:
        Number of new episodes added
    """
    feed = feedparser.parse(podcast.source_rss_url)

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
    for entry in feed.entries:
        guid = getattr(entry, "id", None) or getattr(entry, "link", None)
        if not guid or guid in existing_guids:
            continue

        # Extract audio URL from enclosures
        audio_url = None
        if hasattr(entry, "enclosures") and entry.enclosures:
            for enc in entry.enclosures:
                if hasattr(enc, "href"):
                    audio_url = enc.href
                    break

        # Parse published date
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published_at = datetime.fromtimestamp(mktime(entry.published_parsed))
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
            status=EpisodeStatus.PENDING.value if podcast.auto_process else EpisodeStatus.SKIPPED.value,
        )
        db.add(episode)
        added += 1

    podcast.last_synced_at = datetime.utcnow()
    db.commit()
    return added


def sync_all_podcasts(db: Session) -> dict:
    """Sync all podcasts.

    Returns:
        Dict mapping podcast IDs to number of new episodes
    """
    results = {}
    podcasts = db.query(Podcast).all()

    for podcast in podcasts:
        try:
            added = sync_podcast(db, podcast)
            results[podcast.id] = {"added": added, "error": None}
        except Exception as e:
            results[podcast.id] = {"added": 0, "error": str(e)}

    return results
